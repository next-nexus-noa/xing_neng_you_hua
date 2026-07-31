# Adaptive Micro-Batched PP Gradient Experiments

This folder uses the same workload matrix as the current fixed-M gradient plan:

- datasets: GovReport-GAO (1K/2K/3K/3.5K/3.8K), LongBench-QMSum
  (1K/3K/3.8K), and STELLA-Corpus (1K/3K/3.8K)
- models: Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B
- capacity calibration: one saturated fixed-M=1 probe for every dataset/model
  combination
- scenarios: low/medium/high Poisson traffic at 0.35C/1.0C/2.0C, where C is
  the measured fixed-M=1 request throughput for that exact combination
- micro batch mode: `micro_batch=adaptive`
- metrics: vLLM benchmark latency/throughput, PP trace summary, and NPU power/energy

The generator emits 33 capacity probes followed by 99 adaptive experiments:
11 datasets x 3 models x (1 probe + 3 traffic levels). Every probe is placed
immediately before the three formal rows that consume its result.

## Generate The Plan

```bash
bash scripts/generate_gradient_plan.sh gradient_plan.csv
```

The capacity probe submits at 10 req/s by default so that the measured
throughput is the saturated capacity rather than the offered rate. It uses the
same number of prompts as the formal rows. Both can be overridden:

```bash
export CAPACITY_PROBE_RATE=10
export CAPACITY_NUM_PROMPTS=200
```

Override the common dataset directory or benchmark arguments when needed:

```bash
export DATASET_ROOT=/workspace/DATASET/vllm_custom
export DATASET_NAME=custom
export DATASET_EXTRA_ARGS="--custom-output-len 128 --skip-chat-template"
```

## Run Adaptive Server

```bash
export PLAN=gradient_plan.csv
export RESULT_ROOT=adaptive_gradient_results
export ASCEND_DEVICES=0,1
export TRACE_ENABLED=1
export TRACE_FILE=/workspace/vllm_ascend_pp_trace.jsonl
bash scripts/run_adaptive_gradient_server.sh
```

Prefix caching is forcibly disabled in this server script, so capacity warmup
cannot leak prompt KV cache into the measured probe. The script prepends
`--no-enable-prefix-caching` to additional `EXTRA_SERVER_ARGS` and rejects an
explicit `--enable-prefix-caching`.

Adaptive runs do not use a fixed `micro_batch` value. The plan writes
`micro_batch=adaptive` and generates one row per dataset/scenario/model.

Useful adaptive knobs:

```bash
export ADAPTIVE_UBATCH_MAX_SIZE=4
export ADAPTIVE_UBATCH_MIN_GAIN_PCT=5
export ADAPTIVE_UBATCH_PREFILL_THRESHOLD_PCT=85
export ADAPTIVE_UBATCH_WARMUP_STEPS=8
export ADAPTIVE_UBATCH_EXPLORE_PCT=5
export ADAPTIVE_UBATCH_SWITCH_THRESHOLD_PCT=5
export ADAPTIVE_UBATCH_BAD_THRESHOLD_PCT=8
export ADAPTIVE_UBATCH_COOLDOWN_STEPS=64
export ADAPTIVE_UBATCH_EWMA_ALPHA=0.2
export ADAPTIVE_UBATCH_MAX_CALIBRATION_SCALE=8.0
export ADAPTIVE_UBATCH_SWITCH_CONFIRMATIONS=2
```

`MAX_CALIBRATION_SCALE` bounds multiplicative online correction to
`[1 / scale, scale]`. `SWITCH_CONFIRMATIONS` requires the same candidate to
win consecutive comparable-bucket decisions before changing M.

## Run Adaptive Client

```bash
export PLAN=gradient_plan.csv
export RESULT_ROOT=adaptive_gradient_results
export TRACE_FILE=/workspace/vllm_ascend_pp_trace.jsonl
export WARMUP_RUNS=1
export POWER_METRICS_NPU_IDS=0,1
export POWER_METRICS_INTERVAL_MS=500
bash scripts/run_adaptive_gradient_client.sh
```

The client writes `$RESULT_ROOT/capacity_qps.csv`. Formal plan rows contain
`auto-low`, `auto-medium`, or `auto-high`; the client resolves them just before
each benchmark using these defaults:

```bash
export LOW_QPS_FACTOR=0.35
export MEDIUM_QPS_FACTOR=1.0
export HIGH_QPS_FACTOR=2.0
```

For example, a measured capacity of `C=0.9 req/s` produces formal request
rates of `0.315`, `0.9`, and `1.8 req/s`. The resolved numeric rate is written
to `client_config.txt` and `summary_power.csv`; `request_rate_spec` preserves
the original automatic level in `client_config.txt`.

If a capacity probe fails, reports failed requests, or has no positive request
throughput, its following automatic rows are marked failed instead of silently
falling back to a hard-coded rate.

The measured benchmark writes:

- `benchmark.txt`
- `benchmark.json`
- `power.jsonl`
- `power_summary.txt`
- `pp_trace.jsonl`
- `pp_trace_summary.txt`
- aggregate `adaptive_gradient_results/summary_power.csv`

`summary_power.csv` includes the same core metrics as the power sweep:
success/failed requests, duration, request and token throughput, TTFT/TPOT/ITL,
PP makespan/bubble/overlap/idle, average/max power, energy, energy per request,
energy per output token, tokens per joule, and output tokens per second per watt.

## Fixed Baseline Scripts

`run_gradient_server.sh` and `run_gradient_client.sh` were kept as fixed-M
baseline entrypoints and can read the same 20-column dataset plan format. When
the plan says `micro_batch=adaptive`, the fixed server does not pass fixed-M
micro-batch environment variables.
