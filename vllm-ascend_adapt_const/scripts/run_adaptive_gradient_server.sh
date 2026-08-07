#!/usr/bin/env bash
set -euo pipefail

PLAN="${PLAN:-gradient_plan.csv}"
RESULT_ROOT="${RESULT_ROOT:-adaptive_gradient_results}"
TRACE_FILE="${TRACE_FILE:-/workspace/vllm_ascend_pp_trace.jsonl}"
TRACE_ENABLED="${TRACE_ENABLED:-0}"
ASCEND_DEVICES="${ASCEND_DEVICES:-0,1}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-60}"
SERVER_RESTART_COOLDOWN="${SERVER_RESTART_COOLDOWN:-15}"
# The July 29 best result used uniform grouping. compute_aware and scom remain
# selectable for the composition-aware experiments through this environment
# variable.
MICROBATCH_GROUPING="${MICROBATCH_GROUPING:-uniform}"
COMPUTE_AWARE_MIN_TOKENS="${COMPUTE_AWARE_MIN_TOKENS:-512}"
COMPUTE_AWARE_MIN_GAIN_PCT="${COMPUTE_AWARE_MIN_GAIN_PCT:-5}"
COMPUTE_AWARE_QUANTUM="${COMPUTE_AWARE_QUANTUM:-8}"
SCOM_MIN_GAIN_PCT="${SCOM_MIN_GAIN_PCT:-3}"
SCOM_SHAPE_BUCKETS="${SCOM_SHAPE_BUCKETS:-128,256,512,1024,2048,4096,8192}"
SCOM_OPTIMIZE_CAPACITIES="${SCOM_OPTIMIZE_CAPACITIES:-1}"
SCOM_ALLOW_BUCKET_CROSSING="${SCOM_ALLOW_BUCKET_CROSSING:-0}"
SCOM_CAPACITY_QUANTUM="${SCOM_CAPACITY_QUANTUM:-64}"
SCOM_MAX_CAPACITY_CANDIDATES="${SCOM_MAX_CAPACITY_CANDIDATES:-8}"
SCOM_MAX_SWAPS="${SCOM_MAX_SWAPS:-4}"
ADAPTIVE_UBATCH_MAX_SIZE="${ADAPTIVE_UBATCH_MAX_SIZE:-4}"
ADAPTIVE_UBATCH_MIN_GAIN_PCT="${ADAPTIVE_UBATCH_MIN_GAIN_PCT:-5}"
ADAPTIVE_UBATCH_PREFILL_THRESHOLD_PCT="${ADAPTIVE_UBATCH_PREFILL_THRESHOLD_PCT:-85}"
ADAPTIVE_UBATCH_WARMUP_STEPS="${ADAPTIVE_UBATCH_WARMUP_STEPS:-8}"
ADAPTIVE_UBATCH_MIN_OBSERVATIONS="${ADAPTIVE_UBATCH_MIN_OBSERVATIONS:-8}"
ADAPTIVE_UBATCH_EXPLORE_PCT="${ADAPTIVE_UBATCH_EXPLORE_PCT:-5}"
ADAPTIVE_UBATCH_ENABLE_EXPLORATION="${ADAPTIVE_UBATCH_ENABLE_EXPLORATION:-0}"
ADAPTIVE_UBATCH_SWITCH_THRESHOLD_PCT="${ADAPTIVE_UBATCH_SWITCH_THRESHOLD_PCT:-5}"
ADAPTIVE_UBATCH_BAD_THRESHOLD_PCT="${ADAPTIVE_UBATCH_BAD_THRESHOLD_PCT:-20}"
ADAPTIVE_UBATCH_COOLDOWN_STEPS="${ADAPTIVE_UBATCH_COOLDOWN_STEPS:-32}"
ADAPTIVE_UBATCH_FAILURE_COOLDOWN_STEPS="${ADAPTIVE_UBATCH_FAILURE_COOLDOWN_STEPS:-32}"
ADAPTIVE_UBATCH_EWMA_ALPHA="${ADAPTIVE_UBATCH_EWMA_ALPHA:-0.2}"
ADAPTIVE_DISABLE_M4_FOR_LARGE_MODEL="${ADAPTIVE_DISABLE_M4_FOR_LARGE_MODEL:-0}"
ADAPTIVE_USE_ANALYTICAL_PRIOR="${ADAPTIVE_USE_ANALYTICAL_PRIOR:-1}"
ADAPTIVE_UBATCH_MIN_TOKENS_M2="${ADAPTIVE_UBATCH_MIN_TOKENS_M2:-128}"
ADAPTIVE_UBATCH_MIN_TOKENS_M4="${ADAPTIVE_UBATCH_MIN_TOKENS_M4:-512}"
ADAPTIVE_UBATCH_MIN_PREFILL_RATIO_M4="${ADAPTIVE_UBATCH_MIN_PREFILL_RATIO_M4:-0.85}"
ADAPTIVE_UBATCH_MODE="${ADAPTIVE_UBATCH_MODE:-contextual_safe}"
ADAPTIVE_UBATCH_RISK_KAPPA="${ADAPTIVE_UBATCH_RISK_KAPPA:-1.0}"
ADAPTIVE_UBATCH_MAX_UNCERTAINTY_RATIO="${ADAPTIVE_UBATCH_MAX_UNCERTAINTY_RATIO:-0.15}"
ADAPTIVE_UBATCH_COLD_START_PENALTY_RATIO="${ADAPTIVE_UBATCH_COLD_START_PENALTY_RATIO:-0.15}"
ADAPTIVE_UBATCH_MAX_CORRECTION_RATIO="${ADAPTIVE_UBATCH_MAX_CORRECTION_RATIO:-0.3}"
ADAPTIVE_UBATCH_MAX_CALIBRATION_SCALE="${ADAPTIVE_UBATCH_MAX_CALIBRATION_SCALE:-8.0}"
ADAPTIVE_UBATCH_MIN_HOLD_STEPS="${ADAPTIVE_UBATCH_MIN_HOLD_STEPS:-4}"
ADAPTIVE_UBATCH_SWITCH_CONFIRMATIONS="${ADAPTIVE_UBATCH_SWITCH_CONFIRMATIONS:-2}"
ADAPTIVE_UBATCH_FEEDBACK_INTERVAL_STEPS="${ADAPTIVE_UBATCH_FEEDBACK_INTERVAL_STEPS:-64}"
ADAPTIVE_UBATCH_CANDIDATE_CALIBRATION_OBSERVATIONS="${ADAPTIVE_UBATCH_CANDIDATE_CALIBRATION_OBSERVATIONS:-3}"
ADAPTIVE_UBATCH_SAFE_M="${ADAPTIVE_UBATCH_SAFE_M:-1}"
ADAPTIVE_UBATCH_EXPLORATION_INTERVAL_STEPS="${ADAPTIVE_UBATCH_EXPLORATION_INTERVAL_STEPS:-16}"
ADAPTIVE_UBATCH_EXPLORATION_STABLE_STEPS="${ADAPTIVE_UBATCH_EXPLORATION_STABLE_STEPS:-8}"
ADAPTIVE_UBATCH_MAX_EXPLORATION_REGRET_PCT="${ADAPTIVE_UBATCH_MAX_EXPLORATION_REGRET_PCT:-5}"
ADAPTIVE_UBATCH_QUEUE_SAFETY_ENABLED="${ADAPTIVE_UBATCH_QUEUE_SAFETY_ENABLED:-1}"
ADAPTIVE_UBATCH_QUEUE_GROWTH_THRESHOLD="${ADAPTIVE_UBATCH_QUEUE_GROWTH_THRESHOLD:-2}"
ADAPTIVE_UBATCH_REGRET_BUDGET_PCT="${ADAPTIVE_UBATCH_REGRET_BUDGET_PCT:-2}"
ADAPTIVE_UBATCH_REGRET_WINDOW_STEPS="${ADAPTIVE_UBATCH_REGRET_WINDOW_STEPS:-64}"
ADAPTIVE_UBATCH_CONTEXT_MIN_OBSERVATIONS="${ADAPTIVE_UBATCH_CONTEXT_MIN_OBSERVATIONS:-3}"
ADAPTIVE_UBATCH_CONTEXT_FORGETTING_FACTOR="${ADAPTIVE_UBATCH_CONTEXT_FORGETTING_FACTOR:-0.98}"
ADAPTIVE_UBATCH_CONTEXT_CHANGE_THRESHOLD="${ADAPTIVE_UBATCH_CONTEXT_CHANGE_THRESHOLD:-0.12}"
ADAPTIVE_DECISION_TRACE_ENABLED="${ADAPTIVE_DECISION_TRACE_ENABLED:-1}"

case " $EXTRA_SERVER_ARGS " in
  *" --enable-prefix-caching "*)
    echo "Adaptive gradient experiments require prefix caching to be disabled." >&2
    exit 1
    ;;
  *" --no-enable-prefix-caching "*) ;;
  *) EXTRA_SERVER_ARGS="--no-enable-prefix-caching $EXTRA_SERVER_ARGS" ;;
esac

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
  adaptive_id="$run_id"
  run_dir="$RESULT_ROOT/$adaptive_id"
  mkdir -p "$run_dir"
  adaptive_trace_file="$run_dir/adaptive_ubatch_decisions.jsonl"
  rm -f "$adaptive_trace_file"
  rm -f "$run_dir/server.ready" "$run_dir/server.failed" "$run_dir/client.done"
  trace_dir="$(dirname "$TRACE_FILE")"
  if [[ -n "$trace_dir" && "$trace_dir" != "." ]]; then
    mkdir -p "$trace_dir"
  fi
  rm -f "$TRACE_FILE"

  env_args=(
    "ASCEND_LAUNCH_BLOCKING=1"
    "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_DEVICES"
    "VLLM_USE_MODELSCOPE=True"
    "PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256"
    "VLLM_ASCEND_PP_TRACE=$TRACE_ENABLED"
    "VLLM_ASCEND_PP_TRACE_FILE=$TRACE_FILE"
    "VLLM_ASCEND_PP_MICROBATCH_GROUPING=$MICROBATCH_GROUPING"
    "VLLM_ASCEND_PP_COMPUTE_AWARE_MIN_TOKENS=$COMPUTE_AWARE_MIN_TOKENS"
    "VLLM_ASCEND_PP_COMPUTE_AWARE_MIN_GAIN_PCT=$COMPUTE_AWARE_MIN_GAIN_PCT"
    "VLLM_ASCEND_PP_COMPUTE_AWARE_QUANTUM=$COMPUTE_AWARE_QUANTUM"
    "VLLM_ASCEND_PP_SCOM_MIN_GAIN_PCT=$SCOM_MIN_GAIN_PCT"
    "VLLM_ASCEND_PP_SCOM_SHAPE_BUCKETS=$SCOM_SHAPE_BUCKETS"
    "VLLM_ASCEND_PP_SCOM_OPTIMIZE_CAPACITIES=$SCOM_OPTIMIZE_CAPACITIES"
    "VLLM_ASCEND_PP_SCOM_ALLOW_BUCKET_CROSSING=$SCOM_ALLOW_BUCKET_CROSSING"
    "VLLM_ASCEND_PP_SCOM_CAPACITY_QUANTUM=$SCOM_CAPACITY_QUANTUM"
    "VLLM_ASCEND_PP_SCOM_MAX_CAPACITY_CANDIDATES=$SCOM_MAX_CAPACITY_CANDIDATES"
    "VLLM_ASCEND_PP_SCOM_MAX_SWAPS=$SCOM_MAX_SWAPS"
  )

  server_mode="adaptive"
  server_args=()
  if [[ "$scenario" == "capacity_probe" ]]; then
    # Capacity C must come from a stable fixed-M=1 baseline, not from the
    # controller being evaluated by the following formal rows.
    server_mode="capacity_fixed_m1"
    env_args+=(
      "VLLM_ASCEND_PP_MICROBATCH=1"
      "VLLM_ASCEND_PP_MICROBATCH_NUM=1"
    )
  fi
  echo "[$adaptive_id] Starting server: mode=$server_mode model=$model max_model_len=$max_model_len"

  analytical_prior_arg="--adaptive-ubatch-use-analytical-prior"
  if [[ "$ADAPTIVE_USE_ANALYTICAL_PRIOR" == "0" ]]; then
    analytical_prior_arg="--no-adaptive-ubatch-use-analytical-prior"
  fi

  large_model_m4_arg="--adaptive-ubatch-disable-m4-for-large-model"
  if [[ "$ADAPTIVE_DISABLE_M4_FOR_LARGE_MODEL" == "0" ]]; then
    large_model_m4_arg="--no-adaptive-ubatch-disable-m4-for-large-model"
  fi
  exploration_arg="--adaptive-ubatch-enable-exploration"
  if [[ "$ADAPTIVE_UBATCH_ENABLE_EXPLORATION" == "0" ]]; then
    exploration_arg="--no-adaptive-ubatch-enable-exploration"
  fi
  queue_safety_arg="--adaptive-ubatch-queue-safety-enabled"
  if [[ "$ADAPTIVE_UBATCH_QUEUE_SAFETY_ENABLED" == "0" ]]; then
    queue_safety_arg="--no-adaptive-ubatch-queue-safety-enabled"
  fi
  adaptive_trace_args=()
  adaptive_trace_config_path=""
  if [[ "$ADAPTIVE_DECISION_TRACE_ENABLED" == "1" ]]; then
    adaptive_trace_config_path="$adaptive_trace_file"
    adaptive_trace_args=(
      --adaptive-ubatch-trace-path "$adaptive_trace_file"
    )
  fi

  if [[ "$server_mode" == "adaptive" ]]; then
    server_args=(
      --enable-adaptive-ubatch
      --adaptive-ubatch-max-size "$ADAPTIVE_UBATCH_MAX_SIZE"
      --adaptive-ubatch-min-gain-pct "$ADAPTIVE_UBATCH_MIN_GAIN_PCT"
      "$analytical_prior_arg"
      --adaptive-ubatch-prefill-threshold-pct "$ADAPTIVE_UBATCH_PREFILL_THRESHOLD_PCT"
      --adaptive-ubatch-warmup-steps "$ADAPTIVE_UBATCH_WARMUP_STEPS"
      --adaptive-ubatch-min-observations "$ADAPTIVE_UBATCH_MIN_OBSERVATIONS"
      --adaptive-ubatch-explore-pct "$ADAPTIVE_UBATCH_EXPLORE_PCT"
      "$exploration_arg"
      --adaptive-ubatch-switch-threshold-pct "$ADAPTIVE_UBATCH_SWITCH_THRESHOLD_PCT"
      --adaptive-ubatch-bad-threshold-pct "$ADAPTIVE_UBATCH_BAD_THRESHOLD_PCT"
      --adaptive-ubatch-cooldown-steps "$ADAPTIVE_UBATCH_COOLDOWN_STEPS"
      --adaptive-ubatch-failure-cooldown-steps "$ADAPTIVE_UBATCH_FAILURE_COOLDOWN_STEPS"
      --adaptive-ubatch-ewma-alpha "$ADAPTIVE_UBATCH_EWMA_ALPHA"
      --adaptive-ubatch-min-tokens-m2 "$ADAPTIVE_UBATCH_MIN_TOKENS_M2"
      --adaptive-ubatch-min-tokens-m4 "$ADAPTIVE_UBATCH_MIN_TOKENS_M4"
      --adaptive-ubatch-min-prefill-ratio-m4 "$ADAPTIVE_UBATCH_MIN_PREFILL_RATIO_M4"
      --adaptive-ubatch-mode "$ADAPTIVE_UBATCH_MODE"
      --adaptive-ubatch-risk-kappa "$ADAPTIVE_UBATCH_RISK_KAPPA"
      --adaptive-ubatch-max-uncertainty-ratio "$ADAPTIVE_UBATCH_MAX_UNCERTAINTY_RATIO"
      --adaptive-ubatch-cold-start-penalty-ratio "$ADAPTIVE_UBATCH_COLD_START_PENALTY_RATIO"
      --adaptive-ubatch-max-correction-ratio "$ADAPTIVE_UBATCH_MAX_CORRECTION_RATIO"
      --adaptive-ubatch-max-calibration-scale "$ADAPTIVE_UBATCH_MAX_CALIBRATION_SCALE"
      --adaptive-ubatch-min-hold-steps "$ADAPTIVE_UBATCH_MIN_HOLD_STEPS"
      --adaptive-ubatch-switch-confirmations "$ADAPTIVE_UBATCH_SWITCH_CONFIRMATIONS"
      --adaptive-ubatch-feedback-interval-steps "$ADAPTIVE_UBATCH_FEEDBACK_INTERVAL_STEPS"
      --adaptive-ubatch-candidate-calibration-observations "$ADAPTIVE_UBATCH_CANDIDATE_CALIBRATION_OBSERVATIONS"
      --adaptive-ubatch-safe-m "$ADAPTIVE_UBATCH_SAFE_M"
      --adaptive-ubatch-exploration-interval-steps "$ADAPTIVE_UBATCH_EXPLORATION_INTERVAL_STEPS"
      --adaptive-ubatch-exploration-stable-steps "$ADAPTIVE_UBATCH_EXPLORATION_STABLE_STEPS"
      --adaptive-ubatch-max-exploration-regret-pct "$ADAPTIVE_UBATCH_MAX_EXPLORATION_REGRET_PCT"
      "$queue_safety_arg"
      --adaptive-ubatch-queue-growth-threshold "$ADAPTIVE_UBATCH_QUEUE_GROWTH_THRESHOLD"
      --adaptive-ubatch-regret-budget-pct "$ADAPTIVE_UBATCH_REGRET_BUDGET_PCT"
      --adaptive-ubatch-regret-window-steps "$ADAPTIVE_UBATCH_REGRET_WINDOW_STEPS"
      --adaptive-ubatch-context-min-observations "$ADAPTIVE_UBATCH_CONTEXT_MIN_OBSERVATIONS"
      --adaptive-ubatch-context-forgetting-factor "$ADAPTIVE_UBATCH_CONTEXT_FORGETTING_FACTOR"
      --adaptive-ubatch-context-change-threshold "$ADAPTIVE_UBATCH_CONTEXT_CHANGE_THRESHOLD"
      "${adaptive_trace_args[@]}"
      "$large_model_m4_arg"
    )
  fi

  {
    echo "run_id=$adaptive_id"
    echo "source_run_id=$run_id"
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
    echo "server_mode=$server_mode"
    echo "max_model_len=$max_model_len"
    echo "dataset_name=$dataset_name"
    echo "dataset_path=$dataset_path"
    echo "trace_enabled=$TRACE_ENABLED"
    echo "trace_file=$TRACE_FILE"
    echo "microbatch_grouping=$MICROBATCH_GROUPING"
    echo "compute_aware_min_tokens=$COMPUTE_AWARE_MIN_TOKENS"
    echo "compute_aware_min_gain_pct=$COMPUTE_AWARE_MIN_GAIN_PCT"
    echo "compute_aware_quantum=$COMPUTE_AWARE_QUANTUM"
    echo "scom_min_gain_pct=$SCOM_MIN_GAIN_PCT"
    echo "scom_shape_buckets=$SCOM_SHAPE_BUCKETS"
    echo "scom_optimize_capacities=$SCOM_OPTIMIZE_CAPACITIES"
    echo "scom_allow_bucket_crossing=$SCOM_ALLOW_BUCKET_CROSSING"
    echo "scom_capacity_quantum=$SCOM_CAPACITY_QUANTUM"
    echo "scom_max_capacity_candidates=$SCOM_MAX_CAPACITY_CANDIDATES"
    echo "scom_max_swaps=$SCOM_MAX_SWAPS"
    echo "adaptive_ubatch_max_size=$ADAPTIVE_UBATCH_MAX_SIZE"
    echo "adaptive_ubatch_min_gain_pct=$ADAPTIVE_UBATCH_MIN_GAIN_PCT"
    echo "adaptive_ubatch_prefill_threshold_pct=$ADAPTIVE_UBATCH_PREFILL_THRESHOLD_PCT"
    echo "adaptive_ubatch_warmup_steps=$ADAPTIVE_UBATCH_WARMUP_STEPS"
    echo "adaptive_ubatch_min_observations=$ADAPTIVE_UBATCH_MIN_OBSERVATIONS"
    echo "adaptive_ubatch_explore_pct=$ADAPTIVE_UBATCH_EXPLORE_PCT"
    echo "adaptive_ubatch_enable_exploration=$ADAPTIVE_UBATCH_ENABLE_EXPLORATION"
    echo "adaptive_ubatch_switch_threshold_pct=$ADAPTIVE_UBATCH_SWITCH_THRESHOLD_PCT"
    echo "adaptive_ubatch_bad_threshold_pct=$ADAPTIVE_UBATCH_BAD_THRESHOLD_PCT"
    echo "adaptive_ubatch_cooldown_steps=$ADAPTIVE_UBATCH_COOLDOWN_STEPS"
    echo "adaptive_ubatch_failure_cooldown_steps=$ADAPTIVE_UBATCH_FAILURE_COOLDOWN_STEPS"
    echo "adaptive_ubatch_ewma_alpha=$ADAPTIVE_UBATCH_EWMA_ALPHA"
    echo "adaptive_use_analytical_prior=$ADAPTIVE_USE_ANALYTICAL_PRIOR"
    echo "adaptive_disable_m4_for_large_model=$ADAPTIVE_DISABLE_M4_FOR_LARGE_MODEL"
    echo "adaptive_ubatch_min_tokens_m2=$ADAPTIVE_UBATCH_MIN_TOKENS_M2"
    echo "adaptive_ubatch_min_tokens_m4=$ADAPTIVE_UBATCH_MIN_TOKENS_M4"
    echo "adaptive_ubatch_min_prefill_ratio_m4=$ADAPTIVE_UBATCH_MIN_PREFILL_RATIO_M4"
    echo "adaptive_ubatch_mode=$ADAPTIVE_UBATCH_MODE"
    echo "adaptive_ubatch_risk_kappa=$ADAPTIVE_UBATCH_RISK_KAPPA"
    echo "adaptive_ubatch_max_uncertainty_ratio=$ADAPTIVE_UBATCH_MAX_UNCERTAINTY_RATIO"
    echo "adaptive_ubatch_cold_start_penalty_ratio=$ADAPTIVE_UBATCH_COLD_START_PENALTY_RATIO"
    echo "adaptive_ubatch_max_correction_ratio=$ADAPTIVE_UBATCH_MAX_CORRECTION_RATIO"
    echo "adaptive_ubatch_max_calibration_scale=$ADAPTIVE_UBATCH_MAX_CALIBRATION_SCALE"
    echo "adaptive_ubatch_min_hold_steps=$ADAPTIVE_UBATCH_MIN_HOLD_STEPS"
    echo "adaptive_ubatch_switch_confirmations=$ADAPTIVE_UBATCH_SWITCH_CONFIRMATIONS"
    echo "adaptive_ubatch_feedback_interval_steps=$ADAPTIVE_UBATCH_FEEDBACK_INTERVAL_STEPS"
    echo "adaptive_ubatch_candidate_calibration_observations=$ADAPTIVE_UBATCH_CANDIDATE_CALIBRATION_OBSERVATIONS"
    echo "adaptive_ubatch_feedback_scope=full_worker_step"
    echo "adaptive_ubatch_safe_m=$ADAPTIVE_UBATCH_SAFE_M"
    echo "adaptive_ubatch_exploration_interval_steps=$ADAPTIVE_UBATCH_EXPLORATION_INTERVAL_STEPS"
    echo "adaptive_ubatch_exploration_stable_steps=$ADAPTIVE_UBATCH_EXPLORATION_STABLE_STEPS"
    echo "adaptive_ubatch_max_exploration_regret_pct=$ADAPTIVE_UBATCH_MAX_EXPLORATION_REGRET_PCT"
    echo "adaptive_ubatch_queue_safety_enabled=$ADAPTIVE_UBATCH_QUEUE_SAFETY_ENABLED"
    echo "adaptive_ubatch_queue_growth_threshold=$ADAPTIVE_UBATCH_QUEUE_GROWTH_THRESHOLD"
    echo "adaptive_ubatch_regret_budget_pct=$ADAPTIVE_UBATCH_REGRET_BUDGET_PCT"
    echo "adaptive_ubatch_regret_window_steps=$ADAPTIVE_UBATCH_REGRET_WINDOW_STEPS"
    echo "adaptive_ubatch_context_min_observations=$ADAPTIVE_UBATCH_CONTEXT_MIN_OBSERVATIONS"
    echo "adaptive_ubatch_context_forgetting_factor=$ADAPTIVE_UBATCH_CONTEXT_FORGETTING_FACTOR"
    echo "adaptive_ubatch_context_change_threshold=$ADAPTIVE_UBATCH_CONTEXT_CHANGE_THRESHOLD"
    echo "adaptive_decision_trace_enabled=$ADAPTIVE_DECISION_TRACE_ENABLED"
    echo "adaptive_ubatch_trace_path=$adaptive_trace_config_path"
    echo "extra_server_args=$EXTRA_SERVER_ARGS"
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
    "${server_args[@]}" \
    $EXTRA_SERVER_ARGS \
    > "$run_dir/server.log" 2>&1 &

  SERVER_PID="$!"
  echo "$SERVER_PID" > "$run_dir/server.pid"

  if wait_for_ready "$host" "$port"; then
    date -Is > "$run_dir/server.ready"
    echo "[$adaptive_id] Server ready. Waiting for client.done."
  else
    date -Is > "$run_dir/server.failed"
    echo "[$adaptive_id] Server failed or timed out. See $run_dir/server.log"
    cleanup_server
    continue
  fi

  while [[ ! -f "$run_dir/client.done" ]]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[$adaptive_id] Server exited before client.done."
      date -Is > "$run_dir/server.failed"
      break
    fi
    sleep 2
  done

  cleanup_server
  date -Is > "$run_dir/server.stopped"
  echo "[$adaptive_id] Server stopped."
  sleep "$SERVER_RESTART_COOLDOWN"
done

echo "Adaptive server sweep finished."
