#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-gradient_plan_compute_aware_pilot.csv}"

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

# Four scenarios covering model size, load, input length, and dataset family.
# scenario id | dataset label | dataset file | scenario | model label |
# model | request rate
PILOT_SCENARIOS=(
  "E09|GovReport-GAO-2K|gov_report_gao_2K.jsonl|high_poisson|7B|Qwen/Qwen2.5-7B|auto-high"
  "E09|GovReport-GAO-3K|gov_report_gao_3K.jsonl|high_poisson|3B|Qwen/Qwen2.5-3B|auto-high"
  "E08|GovReport-GAO-3K|gov_report_gao_3K.jsonl|medium_poisson|14B|Qwen/Qwen2.5-14B|auto-medium"
  "E28|LongBench-QMSum-3K|longbench_qmsum_3K.jsonl|high_poisson|7B|Qwen/Qwen2.5-7B|auto-high"
)

MICRO_BATCHES=("2" "4")

printf 'run_id,scenario_id,dataset,scenario,model_label,model,request_rate,burstiness,seed,num_prompts,micro_batch,max_model_len,host,port,pp_size,tp_size,dtype,dataset_name,dataset_path,dataset_extra_args\n' > "$OUT"

i=0
for scenario_spec in "${PILOT_SCENARIOS[@]}"; do
  IFS='|' read -r scenario_id dataset dataset_file scenario model_label model request_rate \
    <<< "$scenario_spec"
  dataset_path="$DATASET_ROOT/$dataset_file"

  for micro_batch in "${MICRO_BATCHES[@]}"; do
    i=$((i + 1))
    run_id="$(printf '%04d' "$i")_${scenario_id}_${dataset}_${scenario}_${model_label}_qps-${request_rate}_burst-1_seed-${SEED}_mb-${micro_batch}"
    run_id="${run_id//\//_}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$run_id" "$scenario_id" "$dataset" "$scenario" "$model_label" "$model" \
      "$request_rate" "1" "$SEED" "$NUM_PROMPTS" "$micro_batch" \
      "$MAX_MODEL_LEN" "$HOST" "$PORT" "$PP_SIZE" "$TP_SIZE" "$DTYPE" \
      "$DATASET_NAME" "$dataset_path" "$DATASET_EXTRA_ARGS" >> "$OUT"
  done
done

echo "Wrote $OUT with $i compute-aware fixed-M pilot experiments."
