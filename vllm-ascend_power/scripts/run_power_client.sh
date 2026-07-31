#!/usr/bin/env bash
set -euo pipefail

PLAN="${PLAN:-gradient_plan.csv}"
RESULT_ROOT="${RESULT_ROOT:-gradient_results}"
TRACE_FILE="${TRACE_FILE:-/workspace/vllm_ascend_pp_trace.jsonl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_SUMMARY_SCRIPT="${TRACE_SUMMARY_SCRIPT:-$SCRIPT_DIR/pp_trace_summary.py}"
POWER_MONITOR_SCRIPT="${POWER_MONITOR_SCRIPT:-$SCRIPT_DIR/power_monitor.py}"
POWER_METRICS_ENABLED="${POWER_METRICS_ENABLED:-1}"
POWER_METRICS_INTERVAL_MS="${POWER_METRICS_INTERVAL_MS:-500}"
POWER_METRICS_NPU_IDS="${POWER_METRICS_NPU_IDS:-0,1}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
TOP_SLOW="${TOP_SLOW:-10}"
SKIP_STEPS="${SKIP_STEPS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"
POWER_MONITOR_PID=""

mkdir -p "$RESULT_ROOT"

if [[ ! -f "$PLAN" ]]; then
  echo "Plan not found: $PLAN"
  echo "Run: bash scripts/generate_gradient_plan.sh $PLAN"
  exit 1
fi

SUMMARY_CSV="$RESULT_ROOT/summary_power.csv"
printf 'run_id,model,input_len,output_len,request_rate,num_prompts,micro_batch,success,failed,duration_s,req_s,out_tok_s,total_tok_s,mean_ttft_ms,p99_ttft_ms,mean_tpot_ms,p99_tpot_ms,mean_itl_ms,p99_itl_ms,pp_makespan_ms,bubble_pct,overlap_pct,idle_pct,avg_power_w,max_power_w,energy_j,energy_per_request_j,energy_per_output_token_j,tokens_per_joule,output_tokens_per_second_per_watt\n' > "$SUMMARY_CSV"

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

json_metric() {
  local file="$1"
  local key="$2"
  python - "$file" "$key" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
except Exception:
    print("")
    raise SystemExit(0)
value = data.get(key, "")
print("" if value is None else value)
PY
}

stop_power_monitor() {
  if [[ -n "${POWER_MONITOR_PID:-}" ]]; then
    kill "$POWER_MONITOR_PID" > /dev/null 2>&1 || true
    wait "$POWER_MONITOR_PID" > /dev/null 2>&1 || true
    POWER_MONITOR_PID=""
  fi
}
trap stop_power_monitor EXIT

start_power_monitor() {
  local power_trace="$1"
  POWER_MONITOR_PID=""

  if [[ "$POWER_METRICS_ENABLED" == "0" ]]; then
    return 0
  fi
  if ! command -v npu-smi > /dev/null 2>&1; then
    echo "npu-smi not found, skip power metrics."
    return 0
  fi

  mkdir -p "$(dirname "$power_trace")"
  rm -f "$power_trace"

  monitor_args=(
    "$POWER_MONITOR_SCRIPT"
    "sample"
    "--output" "$power_trace"
    "--interval-ms" "$POWER_METRICS_INTERVAL_MS"
  )
  if [[ -n "$POWER_METRICS_NPU_IDS" ]]; then
    monitor_args+=("--npu-ids" "$POWER_METRICS_NPU_IDS")
  fi

  python "${monitor_args[@]}" &
  POWER_MONITOR_PID=$!
  echo "Power monitor started: pid=$POWER_MONITOR_PID trace=$power_trace"
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
  local input_len="$2"
  local output_len="$3"
  local num_prompts="$4"
  local request_rate="$5"
  local host="$6"
  local port="$7"
  local output_file="$8"
  local result_dir="${9:-}"
  local result_filename="${10:-}"

  cmd=(
    vllm bench serve
    --backend vllm
    --model "$model"
    --endpoint /v1/completions
    --dataset-name random
    --random-input-len "$input_len"
    --random-output-len "$output_len"
    --num-prompts "$num_prompts"
    --request-rate "$request_rate"
    --host "$host"
    --port "$port"
  )
  if [[ -n "$result_dir" && -n "$result_filename" ]]; then
    cmd+=(
      --save-result
      --result-dir "$result_dir"
      --result-filename "$result_filename"
    )
  fi

  # EXTRA_BENCH_ARGS is intentionally split by the shell for compatibility
  # with the original run_gradient_client.sh usage.
  "${cmd[@]}" $EXTRA_BENCH_ARGS > "$output_file" 2>&1
}

tail -n +2 "$PLAN" | while IFS=, read -r run_id model input_len output_len request_rate num_prompts micro_batch max_model_len host port pp_size tp_size dtype; do
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
    echo "model=$model"
    echo "input_len=$input_len"
    echo "output_len=$output_len"
    echo "request_rate=$request_rate"
    echo "num_prompts=$num_prompts"
    echo "micro_batch=$micro_batch"
    echo "host=$host"
    echo "port=$port"
    echo "warmup_runs=$WARMUP_RUNS"
    echo "power_metrics_enabled=$POWER_METRICS_ENABLED"
    echo "power_metrics_interval_ms=$POWER_METRICS_INTERVAL_MS"
    echo "power_metrics_npu_ids=$POWER_METRICS_NPU_IDS"
  } > "$run_dir/client_config.txt"

  if (( WARMUP_RUNS > 0 )); then
    for warmup_idx in $(seq 1 "$WARMUP_RUNS"); do
      echo "[$run_id] Running warmup benchmark $warmup_idx/$WARMUP_RUNS."
      set +e
      run_benchmark_once "$model" "$input_len" "$output_len" "$num_prompts" \
        "$request_rate" "$host" "$port" "$run_dir/warmup_${warmup_idx}.txt"
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

  echo "[$run_id] Running measured benchmark with power sampling."
  power_trace="$run_dir/power.jsonl"
  benchmark_json="$run_dir/benchmark.json"
  rm -f "$benchmark_json" "$run_dir/power_summary.txt"

  start_power_monitor "$power_trace"
  set +e
  run_benchmark_once "$model" "$input_len" "$output_len" "$num_prompts" \
    "$request_rate" "$host" "$port" "$run_dir/benchmark.txt" \
    "$run_dir" "benchmark.json"
  bench_status="$?"
  stop_power_monitor
  set -e

  if [[ -s "$power_trace" && -f "$benchmark_json" ]]; then
    python "$POWER_MONITOR_SCRIPT" report \
      --input "$power_trace" \
      --benchmark-json "$benchmark_json" \
      --output "$benchmark_json" \
      > "$run_dir/power_summary.txt" 2>&1 || true
  else
    echo "Power trace or benchmark json missing." > "$run_dir/power_summary.txt"
  fi

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
  success="$(json_metric "$benchmark_json" completed)"
  failed="$(json_metric "$benchmark_json" failed)"
  duration_s="$(json_metric "$benchmark_json" duration)"
  req_s="$(json_metric "$benchmark_json" request_throughput)"
  out_tok_s="$(json_metric "$benchmark_json" output_throughput)"
  total_tok_s="$(json_metric "$benchmark_json" total_token_throughput)"
  mean_ttft_ms="$(json_metric "$benchmark_json" mean_ttft_ms)"
  p99_ttft_ms="$(json_metric "$benchmark_json" p99_ttft_ms)"
  mean_tpot_ms="$(json_metric "$benchmark_json" mean_tpot_ms)"
  p99_tpot_ms="$(json_metric "$benchmark_json" p99_tpot_ms)"
  mean_itl_ms="$(json_metric "$benchmark_json" mean_itl_ms)"
  p99_itl_ms="$(json_metric "$benchmark_json" p99_itl_ms)"

  success="${success:-$(extract_metric "$benchmark" 'Successful requests:\s+([0-9.]+)')}"
  failed="${failed:-$(extract_metric "$benchmark" 'Failed requests:\s+([0-9.]+)')}"
  duration_s="${duration_s:-$(extract_metric "$benchmark" 'Benchmark duration \(s\):\s+([0-9.]+)')}"
  req_s="${req_s:-$(extract_metric "$benchmark" 'Request throughput \(req/s\):\s+([0-9.]+)')}"
  out_tok_s="${out_tok_s:-$(extract_metric "$benchmark" 'Output token throughput \(tok/s\):\s+([0-9.]+)')}"
  total_tok_s="${total_tok_s:-$(extract_metric "$benchmark" 'Total token throughput \(tok/s\):\s+([0-9.]+)')}"
  mean_ttft_ms="${mean_ttft_ms:-$(extract_metric "$benchmark" 'Mean TTFT \(ms\):\s+([0-9.]+)')}"
  p99_ttft_ms="${p99_ttft_ms:-$(extract_metric "$benchmark" 'P99 TTFT \(ms\):\s+([0-9.]+)')}"
  mean_tpot_ms="${mean_tpot_ms:-$(extract_metric "$benchmark" 'Mean TPOT \(ms\):\s+([0-9.]+)')}"
  p99_tpot_ms="${p99_tpot_ms:-$(extract_metric "$benchmark" 'P99 TPOT \(ms\):\s+([0-9.]+)')}"
  mean_itl_ms="${mean_itl_ms:-$(extract_metric "$benchmark" 'Mean ITL \(ms\):\s+([0-9.]+)')}"
  p99_itl_ms="${p99_itl_ms:-$(extract_metric "$benchmark" 'P99 ITL \(ms\):\s+([0-9.]+)')}"

  pp_makespan_ms="$(extract_metric "$trace_summary" 'makespan avg/p50/p95\s+:\s+([0-9.]+) ms')"
  bubble_pct="$(extract_metric "$trace_summary" 'bubble ratio avg/p95\s+:\s+([0-9.]+)%')"
  overlap_pct="$(extract_metric "$trace_summary" 'compute overlap avg/max\s+:\s+([0-9.]+)%')"
  idle_pct="$(extract_metric "$trace_summary" 'idle ratio avg/p95\s+:\s+([0-9.]+)%')"
  avg_power_w="$(json_metric "$benchmark_json" avg_power_w)"
  max_power_w="$(json_metric "$benchmark_json" max_power_w)"
  energy_j="$(json_metric "$benchmark_json" energy_j)"
  energy_per_request_j="$(json_metric "$benchmark_json" energy_per_request_j)"
  energy_per_output_token_j="$(json_metric "$benchmark_json" energy_per_output_token_j)"
  tokens_per_joule="$(json_metric "$benchmark_json" tokens_per_joule)"
  output_tokens_per_second_per_watt="$(json_metric "$benchmark_json" output_tokens_per_second_per_watt)"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" "$model" "$input_len" "$output_len" "$request_rate" "$num_prompts" "$micro_batch" \
    "$success" "$failed" "$duration_s" "$req_s" "$out_tok_s" "$total_tok_s" \
    "$mean_ttft_ms" "$p99_ttft_ms" "$mean_tpot_ms" "$p99_tpot_ms" \
    "$mean_itl_ms" "$p99_itl_ms" "$pp_makespan_ms" "$bubble_pct" "$overlap_pct" "$idle_pct" \
    "$avg_power_w" "$max_power_w" "$energy_j" "$energy_per_request_j" \
    "$energy_per_output_token_j" "$tokens_per_joule" "$output_tokens_per_second_per_watt" \
    >> "$SUMMARY_CSV"

  date -Is > "$run_dir/client.done"
  echo "[$run_id] Done."
done

echo "Power client sweep finished. Summary: $SUMMARY_CSV"
