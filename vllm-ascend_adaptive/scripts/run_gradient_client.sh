#!/usr/bin/env bash
set -euo pipefail

PLAN="${PLAN:-gradient_plan.csv}"
RESULT_ROOT="${RESULT_ROOT:-gradient_results}"
TRACE_FILE="${TRACE_FILE:-/workspace/vllm_ascend_pp_trace.jsonl}"
TRACE_SUMMARY_SCRIPT="${TRACE_SUMMARY_SCRIPT:-pp_trace_summary.py}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
TOP_SLOW="${TOP_SLOW:-10}"
SKIP_STEPS="${SKIP_STEPS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

mkdir -p "$RESULT_ROOT"

if [[ ! -f "$PLAN" ]]; then
  echo "Plan not found: $PLAN"
  echo "Run: bash scripts/generate_gradient_plan.sh $PLAN"
  exit 1
fi

SUMMARY_CSV="$RESULT_ROOT/summary.csv"
printf 'run_id,scenario_id,dataset,scenario,model_label,model,request_rate,burstiness,seed,num_prompts,micro_batch,dataset_name,dataset_path,success,failed,duration_s,req_s,out_tok_s,total_tok_s,mean_ttft_ms,p99_ttft_ms,mean_tpot_ms,p99_tpot_ms,mean_itl_ms,p99_itl_ms,pp_makespan_ms,bubble_pct,overlap_pct,idle_pct\n' > "$SUMMARY_CSV"

extract_metric() {
  local file="$1"
  local pattern="$2"
  python - "$file" "$pattern" <<'PY'
import re
import sys

path, pattern = sys.argv[1], sys.argv[2]
text = open(path, "r", encoding="utf-8", errors="ignore").read()
m = re.search(pattern, text)
print(m.group(1) if m else "")
PY
}

wait_for_server_ready_file() {
  local run_dir="$1"
  local timeout="$2"
  local start
  start="$(date +%s)"
  while [[ ! -f "$run_dir/server.ready" ]]; do
    if [[ -f "$run_dir/server.failed" ]]; then
      return 1
    fi
    if (( "$(date +%s)" - start > timeout )); then
      return 1
    fi
    sleep 2
  done
}

wait_for_http() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start > timeout )); then
      return 1
    fi
    sleep 2
  done
}

run_benchmark_once() {
  local model="$1"
  local request_rate="$2"
  local burstiness="$3"
  local seed="$4"
  local num_prompts="$5"
  local host="$6"
  local port="$7"
  local dataset_name="$8"
  local dataset_path="$9"
  local dataset_extra_args="${10}"
  local output_file="${11}"

  cmd=(
    vllm bench serve
    --backend vllm \
    --model "$model" \
    --endpoint /v1/completions \
    --dataset-name "$dataset_name" \
    --num-prompts "$num_prompts" \
    --request-rate "$request_rate" \
    --burstiness "$burstiness" \
    --seed "$seed" \
    --host "$host" \
    --port "$port"
  )
  if [[ -n "$dataset_path" ]]; then
    cmd+=(--dataset-path "$dataset_path")
  fi

  # dataset_extra_args and EXTRA_BENCH_ARGS are intentionally shell-split so
  # callers can pass dataset-specific flags such as "--hf-split test".
  "${cmd[@]}" $dataset_extra_args $EXTRA_BENCH_ARGS > "$output_file" 2>&1
}

tail -n +2 "$PLAN" | while IFS=, read -r run_id scenario_id dataset scenario model_label model request_rate burstiness seed num_prompts micro_batch max_model_len host port pp_size tp_size dtype dataset_name dataset_path dataset_extra_args; do
  run_dir="$RESULT_ROOT/$run_id"
  mkdir -p "$run_dir"
  rm -f "$run_dir/client.done" "$run_dir/client.failed"

  echo "[$run_id] Waiting for server."
  if ! wait_for_server_ready_file "$run_dir" "$SERVER_READY_TIMEOUT"; then
    echo "[$run_id] Server ready file missing or failed."
    date -Is > "$run_dir/client.failed"
    touch "$run_dir/client.done"
    continue
  fi
  if ! wait_for_http "$host" "$port" "$SERVER_READY_TIMEOUT"; then
    echo "[$run_id] Server HTTP readiness timed out."
    date -Is > "$run_dir/client.failed"
    touch "$run_dir/client.done"
    continue
  fi

  echo "[$run_id] Running benchmark."
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
    echo "dataset_name=$dataset_name"
    echo "dataset_path=$dataset_path"
    echo "dataset_extra_args=$dataset_extra_args"
    echo "host=$host"
    echo "port=$port"
    echo "warmup_runs=$WARMUP_RUNS"
  } > "$run_dir/client_config.txt"

  if (( WARMUP_RUNS > 0 )); then
    for warmup_idx in $(seq 1 "$WARMUP_RUNS"); do
      echo "[$run_id] Running warmup benchmark $warmup_idx/$WARMUP_RUNS."
      set +e
      run_benchmark_once "$model" "$request_rate" "$burstiness" "$seed" \
        "$num_prompts" "$host" "$port" "$dataset_name" "$dataset_path" \
        "$dataset_extra_args" "$run_dir/warmup_${warmup_idx}.txt"
      warmup_status="$?"
      set -e

      rm -f "$TRACE_FILE"

      if [[ "$warmup_status" -ne 0 ]]; then
        echo "Warmup benchmark $warmup_idx failed with exit code $warmup_status" \
          > "$run_dir/client.failed"
        date -Is > "$run_dir/client.done"
        echo "[$run_id] Warmup failed."
        continue 2
      fi
    done
  fi

  echo "[$run_id] Running measured benchmark."
  set +e
  run_benchmark_once "$model" "$request_rate" "$burstiness" "$seed" \
    "$num_prompts" "$host" "$port" "$dataset_name" "$dataset_path" \
    "$dataset_extra_args" "$run_dir/benchmark.txt"
  bench_status="$?"
  set -e

  if [[ -f "$TRACE_FILE" ]]; then
    cp "$TRACE_FILE" "$run_dir/pp_trace.jsonl"
    python "$TRACE_SUMMARY_SCRIPT" "$TRACE_FILE" --skip "$SKIP_STEPS" --top-stages "$TOP_SLOW" \
      > "$run_dir/pp_trace_summary.txt" 2>&1 || true
    rm -f "$TRACE_FILE"
  else
    echo "Trace file not found: $TRACE_FILE" > "$run_dir/pp_trace_summary.txt"
  fi

  if [[ "$bench_status" -ne 0 ]]; then
    echo "Benchmark failed with exit code $bench_status" > "$run_dir/client.failed"
  fi

  benchmark="$run_dir/benchmark.txt"
  trace_summary="$run_dir/pp_trace_summary.txt"
  success="$(extract_metric "$benchmark" 'Successful requests:\s+([0-9.]+)')"
  failed="$(extract_metric "$benchmark" 'Failed requests:\s+([0-9.]+)')"
  duration_s="$(extract_metric "$benchmark" 'Benchmark duration \(s\):\s+([0-9.]+)')"
  req_s="$(extract_metric "$benchmark" 'Request throughput \(req/s\):\s+([0-9.]+)')"
  out_tok_s="$(extract_metric "$benchmark" 'Output token throughput \(tok/s\):\s+([0-9.]+)')"
  total_tok_s="$(extract_metric "$benchmark" 'Total token throughput \(tok/s\):\s+([0-9.]+)')"
  mean_ttft_ms="$(extract_metric "$benchmark" 'Mean TTFT \(ms\):\s+([0-9.]+)')"
  p99_ttft_ms="$(extract_metric "$benchmark" 'P99 TTFT \(ms\):\s+([0-9.]+)')"
  mean_tpot_ms="$(extract_metric "$benchmark" 'Mean TPOT \(ms\):\s+([0-9.]+)')"
  p99_tpot_ms="$(extract_metric "$benchmark" 'P99 TPOT \(ms\):\s+([0-9.]+)')"
  mean_itl_ms="$(extract_metric "$benchmark" 'Mean ITL \(ms\):\s+([0-9.]+)')"
  p99_itl_ms="$(extract_metric "$benchmark" 'P99 ITL \(ms\):\s+([0-9.]+)')"
  pp_makespan_ms="$(extract_metric "$trace_summary" 'makespan avg/p50/p95\s+:\s+([0-9.]+) ms')"
  bubble_pct="$(extract_metric "$trace_summary" 'bubble ratio avg/p95\s+:\s+([0-9.]+)%')"
  overlap_pct="$(extract_metric "$trace_summary" 'compute overlap avg/max\s+:\s+([0-9.]+)%')"
  idle_pct="$(extract_metric "$trace_summary" 'idle ratio avg/p95\s+:\s+([0-9.]+)%')"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" "$scenario_id" "$dataset" "$scenario" "$model_label" "$model" \
    "$request_rate" "$burstiness" "$seed" "$num_prompts" "$micro_batch" \
    "$dataset_name" "$dataset_path" \
    "$success" "$failed" "$duration_s" "$req_s" "$out_tok_s" "$total_tok_s" \
    "$mean_ttft_ms" "$p99_ttft_ms" "$mean_tpot_ms" "$p99_tpot_ms" \
    "$mean_itl_ms" "$p99_itl_ms" "$pp_makespan_ms" "$bubble_pct" "$overlap_pct" "$idle_pct" \
    >> "$SUMMARY_CSV"

  date -Is > "$run_dir/client.done"
  echo "[$run_id] Done."
done

echo "Client sweep finished. Summary: $SUMMARY_CSV"
