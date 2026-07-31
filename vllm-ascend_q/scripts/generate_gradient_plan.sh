#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-gradient_plan.csv}"

MODELS=(
  "3B:Qwen/Qwen2.5-3B"
  "7B:Qwen/Qwen2.5-7B"
  "14B:Qwen/Qwen2.5-14B"
)

# Micro-batch experiment mode.
#   queue: run the M4 Queue-Aware baseline only; M is selected online from
#          queue length by the server-side scheduler.
#   fixed: run the fixed-M baselines M0/M1/M2 with M in {1,2,4}.
#   both : run fixed-M plus the queue-aware baseline.
MICRO_BATCH_MODE="${MICRO_BATCH_MODE:-queue}"
case "$MICRO_BATCH_MODE" in
  queue)
    MICRO_BATCHES=("queue")
    ;;
  fixed)
    MICRO_BATCHES=("1" "2" "4")
    ;;
  both)
    MICRO_BATCHES=("1" "2" "4" "queue")
    ;;
  *)
    echo "Unknown MICRO_BATCH_MODE=$MICRO_BATCH_MODE; expected queue, fixed, or both." >&2
    exit 1
    ;;
esac

NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-float16}"

# Use only the prepared vLLM custom datasets.
CUSTOM_DATA_ROOT="${CUSTOM_DATA_ROOT:-/workspace/DATASET/vllm_custom}"
CUSTOM_OUTPUT_LEN="${CUSTOM_OUTPUT_LEN:-128}"
CUSTOM_EXTRA_ARGS="${CUSTOM_EXTRA_ARGS:---custom-output-len $CUSTOM_OUTPUT_LEN --skip-chat-template}"

DATASETS=(
  "GovReport-GAO-1K:E01:gov_report_gao_1K.jsonl"
  "GovReport-GAO-2K:E02:gov_report_gao_2K.jsonl"
  "GovReport-GAO-3K:E03:gov_report_gao_3K.jsonl"
  "GovReport-GAO-3.5K:E04:gov_report_gao_3.5K.jsonl"
  "GovReport-GAO-3.8K:E05:gov_report_gao_3.8K.jsonl"
  "LongBench-QMSum-1K:E20:longbench_qmsum_1K.jsonl"
  "LongBench-QMSum-3K:E21:longbench_qmsum_3K.jsonl"
  "LongBench-QMSum-3.8K:E22:longbench_qmsum_3.8K.jsonl"
  "STELLA-Corpus-1K:E30:stella_corpus_merged_1K.jsonl"
)

LOAD_SPECS=(
  # Resolved by run_gradient_power_client.sh as:
  # auto-low=0.35*C, auto-medium=1.0*C, auto-high=2.0*C,
  # where C is the measured/requested capacity QPS for this dataset/model.
  "medium|auto-medium"
  "high|auto-high"
)

printf 'run_id,scenario_id,dataset,scenario,model_label,model,request_rate,burstiness,seed,num_prompts,micro_batch,max_model_len,host,port,pp_size,tp_size,dtype,dataset_name,dataset_path,dataset_extra_args\n' > "$OUT"

i=0
for dataset_spec in "${DATASETS[@]}"; do
  IFS=: read -r dataset scenario_base dataset_file <<< "$dataset_spec"
  dataset_name="custom"
  dataset_path="$CUSTOM_DATA_ROOT/$dataset_file"
  dataset_extra_args="$CUSTOM_EXTRA_ARGS"

  for model_spec in "${MODELS[@]}"; do
    IFS=: read -r model_label model <<< "$model_spec"
    case "$model_label" in
      3B) model_offset=0 ;;
      7B) model_offset=1 ;;
      14B) model_offset=2 ;;
      *)
        echo "Unknown model label: $model_label" >&2
        exit 1
        ;;
    esac

    for qps_idx in "${!LOAD_SPECS[@]}"; do
      IFS='|' read -r load_level request_rate <<< "${LOAD_SPECS[$qps_idx]}"
      scenario="${load_level}_poisson"
      burstiness="1"
      scenario_num="${scenario_base#E}"
      scenario_id="$(printf 'E%02d' "$((10#$scenario_num + model_offset + qps_idx * 3))")"

      for micro_batch in "${MICRO_BATCHES[@]}"; do
        i=$((i + 1))
        run_id="$(printf '%04d' "$i")_${scenario_id}_${dataset}_${scenario}_${model_label}_qps-${request_rate}_burst-${burstiness}_seed-0_mb-${micro_batch}"
        run_id="${run_id//\//_}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "$run_id" "$scenario_id" "$dataset" "$scenario" "$model_label" "$model" \
          "$request_rate" "$burstiness" "0" "$NUM_PROMPTS" "$micro_batch" \
          "$MAX_MODEL_LEN" "$HOST" "$PORT" "$PP_SIZE" "$TP_SIZE" "$DTYPE" \
          "$dataset_name" "$dataset_path" "$dataset_extra_args" >> "$OUT"
      done
    done
  done
done

echo "Wrote $OUT with $i ${MICRO_BATCH_MODE} micro-batch experiments."
