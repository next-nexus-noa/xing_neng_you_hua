#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-gradient_plan.csv}"
PLAN_MODE="${PLAN_MODE:-all}"

case "$PLAN_MODE" in
  all|probe|formal) ;;
  *)
    echo "PLAN_MODE must be one of: all, probe, formal." >&2
    exit 1
    ;;
esac

# In all/probe mode, each model/dataset combination gets a saturated fixed-M=1
# capacity probe. In formal mode, the 54 adaptive rows reuse previously
# published capacities and no probes are generated.
MODELS=(
  "3B|Qwen/Qwen2.5-3B|0"
  "7B|Qwen/Qwen2.5-7B|1"
  "14B|Qwen/Qwen2.5-14B|2"
)

# dataset label | file below DATASET_ROOT | first scenario id
DATASETS=(
  "GovReport-GAO-1K|gov_report_gao_1K.jsonl|1"
  "GovReport-GAO-2K|gov_report_gao_2K.jsonl|2"
  "GovReport-GAO-3K|gov_report_gao_3K.jsonl|3"
  "GovReport-GAO-3.5K|gov_report_gao_3.5K.jsonl|4"
  "GovReport-GAO-3.8K|gov_report_gao_3.8K.jsonl|5"
  "LongBench-QMSum-1K|longbench_qmsum_1K.jsonl|20"
  "LongBench-QMSum-3K|longbench_qmsum_3K.jsonl|21"
  "LongBench-QMSum-3.8K|longbench_qmsum_3.8K.jsonl|22"
  "STELLA-Corpus-1K|stella_corpus_merged_1K.jsonl|30"
)

MICRO_BATCH="${MICRO_BATCH:-adaptive}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-0}"
CAPACITY_PROBE_RATE="${CAPACITY_PROBE_RATE:-10}"
CAPACITY_NUM_PROMPTS="${CAPACITY_NUM_PROMPTS:-$NUM_PROMPTS}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/DATASET/vllm_custom}"
DATASET_NAME="${DATASET_NAME:-custom}"
DATASET_EXTRA_ARGS="${DATASET_EXTRA_ARGS:---custom-output-len 128 --skip-chat-template}"

printf 'run_id,scenario_id,dataset,scenario,model_label,model,request_rate,burstiness,seed,num_prompts,micro_batch,max_model_len,host,port,pp_size,tp_size,dtype,dataset_name,dataset_path,dataset_extra_args\n' > "$OUT"

# Keep every capacity probe at the front of the plan. This lets the adaptive
# container publish all 27 model/dataset capacities as early as possible while
# the baseline container consumes them from the shared exchange directory.
i=0
capacity_i=0
formal_i=0

if [[ "$PLAN_MODE" == "all" || "$PLAN_MODE" == "probe" ]]; then
  # Phase 1: 9 datasets x 3 models = 27 fixed-M=1 capacity probes.
  for dataset_spec in "${DATASETS[@]}"; do
    IFS='|' read -r dataset dataset_file scenario_base <<< "$dataset_spec"
    dataset_path="$DATASET_ROOT/$dataset_file"

    for model_spec in "${MODELS[@]}"; do
      IFS='|' read -r model_label model model_offset <<< "$model_spec"

      capacity_i=$((capacity_i + 1))
      i=$((i + 1))
      capacity_scenario_id="$(printf 'C%03d' "$capacity_i")"
      run_id="$(printf '%04d' "$i")_${capacity_scenario_id}_${dataset}_capacity_probe_${model_label}_qps-${CAPACITY_PROBE_RATE}_burst-1_seed-${SEED}_mb-1"
      run_id="${run_id//\//_}"

      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$run_id" "$capacity_scenario_id" "$dataset" "capacity_probe" \
        "$model_label" "$model" "$CAPACITY_PROBE_RATE" "1" "$SEED" \
        "$CAPACITY_NUM_PROMPTS" "1" "$MAX_MODEL_LEN" "$HOST" "$PORT" \
        "$PP_SIZE" "$TP_SIZE" "$DTYPE" "$DATASET_NAME" "$dataset_path" \
        "$DATASET_EXTRA_ARGS" >> "$OUT"
    done
  done
fi

# Phase 2: 9 datasets x 3 models x 2 load levels = 54 adaptive runs.
workload_specs=(
  "medium_poisson|3|auto-medium"
  "high_poisson|6|auto-high"
)

if [[ "$PLAN_MODE" == "all" || "$PLAN_MODE" == "formal" ]]; then
  for dataset_spec in "${DATASETS[@]}"; do
    IFS='|' read -r dataset dataset_file scenario_base <<< "$dataset_spec"
    dataset_path="$DATASET_ROOT/$dataset_file"

    for model_spec in "${MODELS[@]}"; do
      IFS='|' read -r model_label model model_offset <<< "$model_spec"

      for workload_spec in "${workload_specs[@]}"; do
        IFS='|' read -r scenario scenario_offset request_rate <<< "$workload_spec"
        scenario_id="$(printf 'E%02d' "$((scenario_base + model_offset + scenario_offset))")"

        i=$((i + 1))
        formal_i=$((formal_i + 1))
        run_id="$(printf '%04d' "$i")_${scenario_id}_${dataset}_${scenario}_${model_label}_qps-${request_rate}_burst-1_seed-${SEED}_mb-${MICRO_BATCH}"
        run_id="${run_id//\//_}"

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "$run_id" "$scenario_id" "$dataset" "$scenario" "$model_label" "$model" \
          "$request_rate" "1" "$SEED" "$NUM_PROMPTS" "$MICRO_BATCH" \
          "$MAX_MODEL_LEN" "$HOST" "$PORT" "$PP_SIZE" "$TP_SIZE" "$DTYPE" \
          "$DATASET_NAME" "$dataset_path" "$DATASET_EXTRA_ARGS" >> "$OUT"
      done
    done
  done
fi

echo "Wrote $OUT in PLAN_MODE=$PLAN_MODE with $capacity_i capacity probes and $formal_i adaptive experiments ($i rows total)."
