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
export ADAPTIVE_UBATCH_ENABLE_EXPLORATION=1
export ADAPTIVE_UBATCH_EXPLORATION_INTERVAL_STEPS=32
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

## SCOM and compute-aware micro-batch grouping

The `adapt_const` copy defaults to bucket-safe SCOM grouping:

```bash
export MICROBATCH_GROUPING=scom
```

The server scripts pass this as
`VLLM_ASCEND_PP_MICROBATCH_GROUPING=scom`. The previous scalar planner is
retained as an ablation with `compute_aware`, while `uniform` retains the
original contiguous equal-token slices:

```bash
export MICROBATCH_GROUPING=compute_aware
export MICROBATCH_GROUPING=uniform
```

The compute-aware safety gate defaults to:

```bash
export COMPUTE_AWARE_MIN_TOKENS=512
export COMPUTE_AWARE_MIN_GAIN_PCT=5
export COMPUTE_AWARE_QUANTUM=8
```

SCOM defaults to:

```bash
export SCOM_MIN_GAIN_PCT=3
export SCOM_SHAPE_BUCKETS=128,256,512,1024,2048,4096,8192
export SCOM_OPTIMIZE_CAPACITIES=1
export SCOM_ALLOW_BUCKET_CROSSING=0
export SCOM_CAPACITY_QUANTUM=64
export SCOM_MAX_CAPACITY_CANDIDATES=8
export SCOM_MAX_SWAPS=4
```

SCOM evaluates the complete two-stage asynchronous Broadcast pipeline rather
than only the maximum scalar group cost. Capacity search starts from uniform
capacities and only generates candidates that remain in the uniform
baseline's NPU shape bucket. Saturated `1024+1024` and `512x4` steps therefore
remain uniform; SCOM optimizes their member composition and ordered slots
without forcing a harmful ratio. Unsaturated steps may use non-uniform
capacities inside the same shape bucket.

The validated equal-capacity compute-aware composition is retained as a
guardrail. If it clears `SCOM_MIN_GAIN_PCT`, SCOM applies that composition
even when the initial analytical pipeline model does not rank it above the
contiguous layout. The more experimental shape-safe capacity search replaces
the guardrail only when it predicts a larger gain. The baseline capacity is
not searched twice, avoiding the previous rejected-plan overhead.

Each model runner also keeps a 64-entry exact-input LRU plan cache. Repeated
warmup/formal step shapes reuse the complete plan without rerunning candidate
search. Trace metadata records `selection_source` and `cache_hit`.

The initial implementation identifies its predictor as
`analytical_shape_aware_with_compute_guardrail`. It is intended for
correctness checks and
representative pilot runs. Before treating SCOM results as final paper data,
calibrate the stage and Broadcast cost tables from NPU profiles and validate
their prediction error in shadow mode.

The server scripts map these values to the centralized Ascend environment
variables `VLLM_ASCEND_PP_COMPUTE_AWARE_MIN_TOKENS`,
`VLLM_ASCEND_PP_COMPUTE_AWARE_MIN_GAIN_PCT`, and
`VLLM_ASCEND_PP_COMPUTE_AWARE_QUANTUM`, and record the resolved values in
`server_config.txt`.

There are no fixed `45/55` or ratio-candidate modes. Compute-aware grouping
keeps the same equal-token shape as uniform M=2/M=4, but fills each
micro-batch with token blocks from different requests to reduce predicted
critical-path cost. The planner:

1. estimates token cost from scheduled query position and existing context;
2. assigns small request-token blocks to equal-token micro-batches;
3. preserves increasing token order within every request so later prefill
   blocks cannot execute before their earlier KV blocks;
4. gathers input ids, positions, slot mappings, block tables, sequence
   metadata, and PP intermediate tensors using one synchronized plan;
5. restores the original flattened token order on the last PP stage before
   logits and sampling.

Steps with fewer than 512 scheduled tokens bypass planning entirely. Larger
steps apply a plan only when predicted
critical-path gain is at least 5%. Otherwise the step uses the original
contiguous uniform path, avoiding permutation overhead when the estimated
benefit is too small. The defaults target the measured NPU workload, where
sub-512-token decode steps had about 2% predicted gain while large prefill
steps averaged substantially higher gain.

Adaptive M selection and compute-aware/SCOM grouping are deterministic and
are mirrored on every PP rank from the same scheduler output. They do not add
a CPU object or tensor broadcast to each scheduler step. Runtime calibration
uses a sampled cross-stage MAX reduction at the configured feedback interval
and explicit calibration/exploration steps. M=1-only control runs skip
feedback because there is no alternative candidate to learn. Only the first
PP rank owns the optional adaptive decision trace.

PP trace step metadata records `scom_grouping` or
`compute_aware_grouping`, including whether the plan was applied, its reason,
predicted gain, per-group segments, capacity candidates, shape buckets, and
the relevant objective values. Below-threshold plans omit full segment
payloads to reduce trace size. Per-step diagnostic messages use DEBUG level
instead of INFO, so normal benchmark server logs remain lightweight.
Benchmark client logs and metric columns are unchanged.

For final performance measurements, disable PP tracing for both compute-aware
and uniform control runs:

```bash
export TRACE_ENABLED=0
export ADAPTIVE_DECISION_TRACE_ENABLED=0
```

Both are now disabled by default. Use trace-enabled runs only for a smaller
diagnostic subset. To retain per-step adaptive M decisions for such a run:

```bash
export ADAPTIVE_DECISION_TRACE_ENABLED=1
```

Decision JSONL output is buffered and flushed in batches to keep diagnostic
logging out of the per-step critical path as much as possible.

The first implementation intentionally falls back to uniform grouping for
speculative decoding, pooling, encoder inputs, context parallelism, compressed
attention, GDN/Mamba paths, and KV-sharing fast prefill. The Qwen decoder-only
PP experiments used by this project remain in the supported path.
