#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-gradient_plan_representative_8.csv}"

MICRO_BATCH="${MICRO_BATCH:-adaptive}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-0}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/DATASET/vllm_custom}"
DATASET_NAME="${DATASET_NAME:-custom}"
DATASET_EXTRA_ARGS="${DATASET_EXTRA_ARGS:---custom-output-len 128 --skip-chat-template}"

# scenario id | dataset label | dataset file | scenario | model label |
# model | request rate
#
# The first six rows target the long-input/high-load failures observed in the
# full adaptive sweep. The final two rows are short-input controls used to
# detect regressions caused by a more aggressive adaptive policy.
REPRESENTATIVE_SCENARIOS=(
  "E09|GovReport-GAO-2K|gov_report_gao_2K.jsonl|high_poisson|7B|Qwen/Qwen2.5-7B|auto-high"
  "E09|GovReport-GAO-3K|gov_report_gao_3K.jsonl|high_poisson|3B|Qwen/Qwen2.5-3B|auto-high"
  "E10|GovReport-GAO-3K|gov_report_gao_3K.jsonl|high_poisson|7B|Qwen/Qwen2.5-7B|auto-high"
  "E08|GovReport-GAO-3K|gov_report_gao_3K.jsonl|medium_poisson|14B|Qwen/Qwen2.5-14B|auto-medium"
  "E10|GovReport-GAO-3.8K|gov_report_gao_3.8K.jsonl|medium_poisson|14B|Qwen/Qwen2.5-14B|auto-medium"
  "E28|LongBench-QMSum-3K|longbench_qmsum_3K.jsonl|high_poisson|7B|Qwen/Qwen2.5-7B|auto-high"
  "E04|GovReport-GAO-1K|gov_report_gao_1K.jsonl|medium_poisson|3B|Qwen/Qwen2.5-3B|auto-medium"
  "E08|GovReport-GAO-1K|gov_report_gao_1K.jsonl|high_poisson|7B|Qwen/Qwen2.5-7B|auto-high"
)

printf 'run_id,scenario_id,dataset,scenario,model_label,model,request_rate,burstiness,seed,num_prompts,micro_batch,max_model_len,host,port,pp_size,tp_size,dtype,dataset_name,dataset_path,dataset_extra_args\n' > "$OUT"

i=0
for scenario_spec in "${REPRESENTATIVE_SCENARIOS[@]}"; do
  IFS='|' read -r scenario_id dataset dataset_file scenario model_label model request_rate \
    <<< "$scenario_spec"

  i=$((i + 1))
  dataset_path="$DATASET_ROOT/$dataset_file"
  run_id="$(printf '%04d' "$i")_${scenario_id}_${dataset}_${scenario}_${model_label}_qps-${request_rate}_burst-1_seed-${SEED}_mb-${MICRO_BATCH}"
  run_id="${run_id//\//_}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" "$scenario_id" "$dataset" "$scenario" "$model_label" "$model" \
    "$request_rate" "1" "$SEED" "$NUM_PROMPTS" "$MICRO_BATCH" \
    "$MAX_MODEL_LEN" "$HOST" "$PORT" "$PP_SIZE" "$TP_SIZE" "$DTYPE" \
    "$DATASET_NAME" "$dataset_path" "$DATASET_EXTRA_ARGS" >> "$OUT"
done

echo "Wrote $OUT with $i representative adaptive experiments and no capacity probes."
