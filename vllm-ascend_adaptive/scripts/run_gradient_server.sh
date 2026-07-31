#!/usr/bin/env bash
set -euo pipefail

PLAN="${PLAN:-gradient_plan.csv}"
RESULT_ROOT="${RESULT_ROOT:-gradient_results}"
TRACE_FILE="${TRACE_FILE:-/workspace/vllm_ascend_pp_trace.jsonl}"
TRACE_ENABLED="${TRACE_ENABLED:-1}"
ASCEND_DEVICES="${ASCEND_DEVICES:-0,1}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-60}"
SERVER_RESTART_COOLDOWN="${SERVER_RESTART_COOLDOWN:-15}"

# Adapt these to the actual fixed Micro-Batched PP switch in your branch.
# Examples:
#   MICROBATCH_ENV_NAME=VLLM_PP_MICRO_BATCHES
#   MICROBATCH_ARG_TEMPLATE='--num-micro-batches {micro}'
#   MICROBATCH_ARG_TEMPLATE='--fixed-micro-batch-size {micro}'
MICROBATCH_ENV_NAME="${MICROBATCH_ENV_NAME:-VLLM_ASCEND_PP_MICROBATCH}"
MICROBATCH_ARG_TEMPLATE="${MICROBATCH_ARG_TEMPLATE:-}"

mkdir -p "$RESULT_ROOT"

if [[ ! -f "$PLAN" ]]; then
  echo "Plan not found: $PLAN"
  echo "Run: bash scripts/generate_gradient_plan.sh $PLAN"
  exit 1
fi

cleanup_server() {
  if [[ -z "${SERVER_PID:-}" ]]; then
    return
  fi

  if kill -0 "$SERVER_PID" 2>/dev/null; then
    # vLLM spawns EngineCore and WorkerProc children. Terminate the whole
    # process group so stale workers do not keep NPU memory between runs.
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true

    local start
    start="$(date +%s)"
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      if (( "$(date +%s)" - start > SERVER_STOP_TIMEOUT )); then
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
        break
      fi
      sleep 1
    done
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT

wait_for_ready() {
  local host="$1"
  local port="$2"
  local timeout="${SERVER_READY_TIMEOUT:-900}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    if (( "$(date +%s)" - start > timeout )); then
      return 1
    fi
    sleep 2
  done
}

tail -n +2 "$PLAN" | while IFS=, read -r run_id scenario_id dataset scenario model_label model request_rate burstiness seed num_prompts micro_batch max_model_len host port pp_size tp_size dtype dataset_name dataset_path dataset_extra_args; do
  run_dir="$RESULT_ROOT/$run_id"
  mkdir -p "$run_dir"
  rm -f "$run_dir/server.ready" "$run_dir/server.failed" "$run_dir/client.done"
  trace_dir="$(dirname "$TRACE_FILE")"
  if [[ -n "$trace_dir" && "$trace_dir" != "." ]]; then
    mkdir -p "$trace_dir"
  fi
  rm -f "$TRACE_FILE"

  echo "[$run_id] Starting server: model=$model micro_batch=$micro_batch max_model_len=$max_model_len"

  micro_args=()
  if [[ -n "$MICROBATCH_ARG_TEMPLATE" && "$micro_batch" =~ ^[0-9]+$ && "$micro_batch" != "0" ]]; then
    micro_arg="${MICROBATCH_ARG_TEMPLATE//\{micro\}/$micro_batch}"
    read -r -a micro_args <<< "$micro_arg"
  fi

  env_args=(
    "ASCEND_LAUNCH_BLOCKING=1"
    "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_DEVICES"
    "VLLM_USE_MODELSCOPE=True"
    "PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256"
    "VLLM_ASCEND_PP_TRACE=$TRACE_ENABLED"
    "VLLM_ASCEND_PP_TRACE_FILE=$TRACE_FILE"
  )
  if [[ -n "$MICROBATCH_ENV_NAME" && "$micro_batch" =~ ^[0-9]+$ ]]; then
    if [[ "$MICROBATCH_ENV_NAME" == "VLLM_ASCEND_PP_MICROBATCH" ]]; then
      env_args+=("${MICROBATCH_ENV_NAME}=1")
    else
      env_args+=("${MICROBATCH_ENV_NAME}=$micro_batch")
    fi
  fi
  # Also set the micro-batch count when micro_batch > 0.
  # VLLM_ASCEND_PP_MICROBATCH is the on/off switch;
  # VLLM_ASCEND_PP_MICROBATCH_NUM controls how many micro-batches.
  if [[ "$micro_batch" =~ ^[0-9]+$ && "$micro_batch" != "0" ]]; then
    env_args+=("VLLM_ASCEND_PP_MICROBATCH_NUM=$micro_batch")
  fi

  {
    echo "run_id=$run_id"
    echo "scenario_id=$scenario_id"
    echo "dataset=$dataset"
    echo "scenario=$scenario"
    echo "model_label=$model_label"
    echo "model=$model"
    echo "request_rate=$request_rate"
    echo "burstiness=$burstiness"
    echo "seed=$seed"
    echo "num_prompts=$num_prompts"
    echo "micro_batch=$micro_batch"
    echo "max_model_len=$max_model_len"
    echo "dataset_name=$dataset_name"
    echo "dataset_path=$dataset_path"
    echo "trace_enabled=$TRACE_ENABLED"
    echo "trace_file=$TRACE_FILE"
    echo "extra_server_args=$EXTRA_SERVER_ARGS"
    echo "microbatch_env_name=$MICROBATCH_ENV_NAME"
    echo "microbatch_arg_template=$MICROBATCH_ARG_TEMPLATE"
  } > "$run_dir/server_config.txt"

  setsid env "${env_args[@]}" vllm serve "$model" \
    --host 0.0.0.0 \
    --port "$port" \
    --pipeline-parallel-size "$pp_size" \
    --tensor-parallel-size "$tp_size" \
    --dtype "$dtype" \
    --enforce-eager \
    --no-async-scheduling \
    --max-model-len "$max_model_len" \
    "${micro_args[@]}" \
    $EXTRA_SERVER_ARGS \
    > "$run_dir/server.log" 2>&1 &

  SERVER_PID="$!"
  echo "$SERVER_PID" > "$run_dir/server.pid"

  if wait_for_ready "$host" "$port"; then
    date -Is > "$run_dir/server.ready"
    echo "[$run_id] Server ready. Waiting for client.done."
  else
    date -Is > "$run_dir/server.failed"
    echo "[$run_id] Server failed or timed out. See $run_dir/server.log"
    cleanup_server
    continue
  fi

  while [[ ! -f "$run_dir/client.done" ]]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[$run_id] Server exited before client.done."
      date -Is > "$run_dir/server.failed"
      break
    fi
    sleep 2
  done

  cleanup_server
  date -Is > "$run_dir/server.stopped"
  echo "[$run_id] Server stopped."
  sleep "$SERVER_RESTART_COOLDOWN"
done

echo "Server sweep finished."
