#!/usr/bin/env bash
set -euo pipefail

PLAN="${PLAN:-gradient_plan.csv}"
RESULT_ROOT="${RESULT_ROOT:-gradient_results}"
TRACE_FILE="${TRACE_FILE:-/workspace/vllm_ascend_pp_trace.jsonl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_SUMMARY_SCRIPT="${TRACE_SUMMARY_SCRIPT:-$SCRIPT_DIR/pp_trace_summary.py}"
POWER_MONITOR_SCRIPT="${POWER_MONITOR_SCRIPT:-$SCRIPT_DIR/power_monitor.py}"
CAPACITY_EXCHANGE_SCRIPT="${CAPACITY_EXCHANGE_SCRIPT:-$SCRIPT_DIR/capacity_exchange.py}"
POWER_METRICS_ENABLED="${POWER_METRICS_ENABLED:-1}"
POWER_METRICS_INTERVAL_MS="${POWER_METRICS_INTERVAL_MS:-500}"
POWER_METRICS_NPU_IDS="${POWER_METRICS_NPU_IDS:-0,1}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
TOP_SLOW="${TOP_SLOW:-10}"
SKIP_STEPS="${SKIP_STEPS:-5}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"
CAPACITY_FILE="${CAPACITY_FILE:-$RESULT_ROOT/capacity_qps.csv}"
LOW_QPS_FACTOR="${LOW_QPS_FACTOR:-0.35}"
MEDIUM_QPS_FACTOR="${MEDIUM_QPS_FACTOR:-1.0}"
HIGH_QPS_FACTOR="${HIGH_QPS_FACTOR:-2.0}"
CAPACITY_SOURCE="${CAPACITY_SOURCE:-local}"
CAPACITY_EXCHANGE_DIR="${CAPACITY_EXCHANGE_DIR:-}"
CAPACITY_SESSION="${CAPACITY_SESSION:-}"
CAPACITY_WAIT_TIMEOUT="${CAPACITY_WAIT_TIMEOUT:-7200}"
CAPACITY_WAIT_INTERVAL="${CAPACITY_WAIT_INTERVAL:-2}"
POWER_MONITOR_PID=""

mkdir -p "$RESULT_ROOT"

if [[ ! -f "$PLAN" ]]; then
  echo "Plan not found: $PLAN"
  echo "Run: bash scripts/generate_gradient_plan.sh $PLAN"
  exit 1
fi

SUMMARY_CSV="$RESULT_ROOT/summary_power.csv"
printf 'run_id,scenario_id,dataset,scenario,model_label,model,request_rate,burstiness,seed,num_prompts,micro_batch,dataset_name,dataset_path,success,failed,duration_s,req_s,out_tok_s,total_tok_s,mean_ttft_ms,p99_ttft_ms,mean_tpot_ms,p99_tpot_ms,mean_itl_ms,p99_itl_ms,pp_makespan_ms,bubble_pct,overlap_pct,idle_pct,avg_power_w,max_power_w,energy_j,energy_per_request_j,energy_per_output_token_j,tokens_per_joule,output_tokens_per_second_per_watt\n' > "$SUMMARY_CSV"
case "$CAPACITY_SOURCE" in
  local)
    mkdir -p "$(dirname "$CAPACITY_FILE")"
    printf 'model_label,dataset,capacity_qps,probe_rate,num_prompts,run_id\n' > "$CAPACITY_FILE"
    ;;
  shared)
    if [[ -z "$CAPACITY_EXCHANGE_DIR" || -z "$CAPACITY_SESSION" ]]; then
      echo "CAPACITY_SOURCE=shared requires CAPACITY_EXCHANGE_DIR and CAPACITY_SESSION." >&2
      exit 1
    fi
    if grep -q ',capacity_probe,' "$PLAN"; then
      echo "CAPACITY_SOURCE=shared requires a 297-row formal-only baseline plan." >&2
      exit 1
    fi
    ;;
  *)
    echo "CAPACITY_SOURCE must be either local or shared." >&2
    exit 1
    ;;
esac

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

resolve_request_rate() {
  local request_rate_spec="$1"
  local model_label="$2"
  local dataset="$3"

  python - "$CAPACITY_FILE" "$model_label" "$dataset" "$request_rate_spec" \
    "$LOW_QPS_FACTOR" "$MEDIUM_QPS_FACTOR" "$HIGH_QPS_FACTOR" <<'PY'
import csv
import sys

path, model_label, dataset, spec = sys.argv[1:5]
factors = {
    "auto-low": float(sys.argv[5]),
    "auto-medium": float(sys.argv[6]),
    "auto-high": float(sys.argv[7]),
}
if spec not in factors:
    print(spec)
    raise SystemExit(0)

try:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
except OSError as exc:
    print(f"capacity file unavailable: {exc}", file=sys.stderr)
    raise SystemExit(1)

for row in reversed(rows):
    if row["model_label"] == model_label and row["dataset"] == dataset:
        value = float(row["capacity_qps"]) * factors[spec]
        if value <= 0:
            break
        print(f"{value:.6g}")
        raise SystemExit(0)

print(
    f"no capacity result for model={model_label} dataset={dataset}",
    file=sys.stderr,
)
raise SystemExit(1)
PY
}

wait_for_shared_request_rate() {
  local request_rate_spec="$1"
  local model_label="$2"
  local dataset="$3"
  local level="${request_rate_spec#auto-}"

  python "$CAPACITY_EXCHANGE_SCRIPT" wait \
    --exchange-dir "$CAPACITY_EXCHANGE_DIR" \
    --session "$CAPACITY_SESSION" \
    --model-label "$model_label" \
    --dataset "$dataset" \
    --level "$level" \
    --timeout "$CAPACITY_WAIT_TIMEOUT" \
    --interval "$CAPACITY_WAIT_INTERVAL"
}

record_capacity() {
  local benchmark_json="$1"
  local model_label="$2"
  local dataset="$3"
  local probe_rate="$4"
  local num_prompts="$5"
  local run_id="$6"

  python - "$CAPACITY_FILE" "$benchmark_json" "$model_label" "$dataset" \
    "$probe_rate" "$num_prompts" "$run_id" <<'PY'
import csv
import json
import os
import sys
import tempfile

path, benchmark_path, model_label, dataset, probe_rate, num_prompts, run_id = sys.argv[1:]
with open(benchmark_path, encoding="utf-8-sig") as f:
    benchmark = json.load(f)

capacity = float(benchmark.get("request_throughput", 0))
completed = int(benchmark.get("completed", 0))
failed = int(benchmark.get("failed", 0))
if capacity <= 0 or completed <= 0 or failed != 0:
    raise SystemExit(
        f"invalid capacity probe: throughput={capacity}, "
        f"completed={completed}, failed={failed}"
    )

fieldnames = [
    "model_label",
    "dataset",
    "capacity_qps",
    "probe_rate",
    "num_prompts",
    "run_id",
]
rows = []
if os.path.exists(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
rows = [
    row
    for row in rows
    if not (row["model_label"] == model_label and row["dataset"] == dataset)
]
rows.append(
    {
        "model_label": model_label,
        "dataset": dataset,
        "capacity_qps": f"{capacity:.9g}",
        "probe_rate": probe_rate,
        "num_prompts": num_prompts,
        "run_id": run_id,
    }
)

directory = os.path.dirname(os.path.abspath(path))
fd, temporary_path = tempfile.mkstemp(prefix="capacity_qps.", dir=directory)
try:
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)
except BaseException:
    try:
        os.unlink(temporary_path)
    except OSError:
        pass
    raise

print(f"Recorded capacity C={capacity:.6g} req/s for {model_label}/{dataset}")
PY
}

validate_dataset() {
  local dataset_name="$1"
  local dataset_path="$2"

  if [[ "$dataset_name" != "custom" ]]; then
    return 0
  fi
  if [[ -z "$dataset_path" || ! -f "$dataset_path" ]]; then
    echo "Custom dataset file not found: $dataset_path" >&2
    return 1
  fi

  python - "$dataset_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8-sig") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or "prompt" not in item:
            raise SystemExit(f"{path}:{line_number}: custom JSONL row must contain a prompt field")
        if not str(item["prompt"]).strip():
            raise SystemExit(f"{path}:{line_number}: prompt must not be empty")
        break
    else:
        raise SystemExit(f"{path}: custom JSONL file is empty")
PY
}

stop_power_monitor() {
  if [[ -n "${POWER_MONITOR_PID:-}" ]]; then
    kill "$POWER_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$POWER_MONITOR_PID" >/dev/null 2>&1 || true
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
  if ! command -v npu-smi >/dev/null 2>&1; then
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
  local result_dir="${12:-}"
  local result_filename="${13:-}"

  cmd=(
    vllm bench serve
    --backend vllm
    --model "$model"
    --endpoint /v1/completions
    --dataset-name "$dataset_name"
    --num-prompts "$num_prompts"
    --request-rate "$request_rate"
    --burstiness "$burstiness"
    --seed "$seed"
    --host "$host"
    --port "$port"
  )
  if [[ -n "$dataset_path" ]]; then
    cmd+=(--dataset-path "$dataset_path")
  fi
  if [[ -n "$result_dir" && -n "$result_filename" ]]; then
    cmd+=(
      --save-result
      --result-dir "$result_dir"
      --result-filename "$result_filename"
    )
  fi

  # dataset_extra_args and EXTRA_BENCH_ARGS are intentionally shell-split so
  # callers can pass dataset-specific flags such as "--hf-split test".
  "${cmd[@]}" $dataset_extra_args $EXTRA_BENCH_ARGS > "$output_file" 2>&1
}

tail -n +2 "$PLAN" | while IFS=, read -r run_id scenario_id dataset scenario model_label model request_rate burstiness seed num_prompts micro_batch max_model_len host port pp_size tp_size dtype dataset_name dataset_path dataset_extra_args; do
  run_dir="$RESULT_ROOT/$run_id"
  mkdir -p "$run_dir"
  rm -f "$run_dir/client.done" "$run_dir/client.failed"

  if ! validate_dataset "$dataset_name" "$dataset_path" > "$run_dir/dataset_check.txt" 2>&1; then
    echo "[$run_id] Dataset validation failed."
    date -Is > "$run_dir/client.failed"
    touch "$run_dir/client.done"
    continue
  fi

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

  request_rate_spec="$request_rate"
  if [[ "$request_rate_spec" == auto-* ]]; then
    set +e
    if [[ "$CAPACITY_SOURCE" == "shared" ]]; then
      request_rate="$(wait_for_shared_request_rate \
        "$request_rate_spec" "$model_label" "$dataset")"
    else
      request_rate="$(resolve_request_rate \
        "$request_rate_spec" "$model_label" "$dataset")"
    fi
    resolve_status="$?"
    set -e
    if [[ "$resolve_status" -ne 0 || -z "$request_rate" ]]; then
      if [[ "$CAPACITY_SOURCE" == "shared" ]]; then
        echo "[$run_id] Could not resolve $request_rate_spec from shared session $CAPACITY_SESSION."
      else
        echo "[$run_id] Could not resolve $request_rate_spec from $CAPACITY_FILE."
      fi
      date -Is > "$run_dir/client.failed"
      date -Is > "$run_dir/client.done"
      continue
    fi
    echo "[$run_id] Resolved request rate: $request_rate_spec -> $request_rate req/s."
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
    echo "request_rate_spec=$request_rate_spec"
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
    echo "power_metrics_enabled=$POWER_METRICS_ENABLED"
    echo "power_metrics_interval_ms=$POWER_METRICS_INTERVAL_MS"
    echo "power_metrics_npu_ids=$POWER_METRICS_NPU_IDS"
    echo "capacity_source=$CAPACITY_SOURCE"
    echo "capacity_exchange_dir=$CAPACITY_EXCHANGE_DIR"
    echo "capacity_session=$CAPACITY_SESSION"
    echo "capacity_wait_timeout=$CAPACITY_WAIT_TIMEOUT"
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

  echo "[$run_id] Running measured benchmark with power sampling."
  power_trace="$run_dir/power.jsonl"
  benchmark_json="$run_dir/benchmark.json"
  rm -f "$benchmark_json" "$run_dir/power_summary.txt"

  start_power_monitor "$power_trace"
  set +e
  run_benchmark_once "$model" "$request_rate" "$burstiness" "$seed" \
    "$num_prompts" "$host" "$port" "$dataset_name" "$dataset_path" \
    "$dataset_extra_args" "$run_dir/benchmark.txt" "$run_dir" "benchmark.json"
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

  if [[ "$scenario" == "capacity_probe" && "$bench_status" -eq 0 ]]; then
    if ! record_capacity "$benchmark_json" "$model_label" "$dataset" \
      "$request_rate" "$num_prompts" "$run_id"; then
      echo "Capacity probe result was invalid; automatic QPS cannot be resolved." \
        > "$run_dir/client.failed"
    fi
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

  pp_makespan_ms="$(extract_metric "$trace_summary" 'makespan avg/p50/p95:\s+([0-9.]+) ms')"
  bubble_pct="$(extract_metric "$trace_summary" 'bubble ratio avg/p50/p95:\s+([0-9.]+)%')"
  overlap_pct="$(extract_metric "$trace_summary" 'compute overlap avg/max:\s+([0-9.]+)%')"
  idle_pct="$(extract_metric "$trace_summary" 'idle ratio avg/p95\s+:\s+([0-9.]+)%')"
  avg_power_w="$(json_metric "$benchmark_json" avg_power_w)"
  max_power_w="$(json_metric "$benchmark_json" max_power_w)"
  energy_j="$(json_metric "$benchmark_json" energy_j)"
  energy_per_request_j="$(json_metric "$benchmark_json" energy_per_request_j)"
  energy_per_output_token_j="$(json_metric "$benchmark_json" energy_per_output_token_j)"
  tokens_per_joule="$(json_metric "$benchmark_json" tokens_per_joule)"
  output_tokens_per_second_per_watt="$(json_metric "$benchmark_json" output_tokens_per_second_per_watt)"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" "$scenario_id" "$dataset" "$scenario" "$model_label" "$model" \
    "$request_rate" "$burstiness" "$seed" "$num_prompts" "$micro_batch" \
    "$dataset_name" "$dataset_path" \
    "$success" "$failed" "$duration_s" "$req_s" "$out_tok_s" "$total_tok_s" \
    "$mean_ttft_ms" "$p99_ttft_ms" "$mean_tpot_ms" "$p99_tpot_ms" \
    "$mean_itl_ms" "$p99_itl_ms" "$pp_makespan_ms" "$bubble_pct" "$overlap_pct" "$idle_pct" \
    "$avg_power_w" "$max_power_w" "$energy_j" "$energy_per_request_j" \
    "$energy_per_output_token_j" "$tokens_per_joule" "$output_tokens_per_second_per_watt" \
    >> "$SUMMARY_CSV"

  date -Is > "$run_dir/client.done"
  echo "[$run_id] Done."
done

echo "Gradient power client sweep finished. Summary: $SUMMARY_CSV"
