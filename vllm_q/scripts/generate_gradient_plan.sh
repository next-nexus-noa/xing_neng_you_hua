#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-gradient_plan.csv}"

MODELS=(
  "3B:Qwen/Qwen2.5-3B"
  "7B:Qwen/Qwen2.5-7B"
  "14B:Qwen/Qwen2.5-14B"
)

# Fixed-M experiments only. Adaptive runs are handled by the adaptive repos.
MICRO_BATCHES=(
  "1"
  "2"
  "4"
)

NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-float16}"

# Override these when the datasets have been downloaded or preprocessed locally.
# dataset_name must be one accepted by `vllm bench serve`, e.g. hf/sharegpt/custom.
SHAREGPT_DATASET_NAME="${SHAREGPT_DATASET_NAME:-custom}"
SHAREGPT_DATASET_PATH="${SHAREGPT_DATASET_PATH:-RyokoAI/ShareGPT52K}"
SHAREGPT_EXTRA_ARGS="${SHAREGPT_EXTRA_ARGS:-}"

APBENCH_DATASET_NAME="${APBENCH_DATASET_NAME:-custom}"
APBENCH_DATASET_PATH="${APBENCH_DATASET_PATH:-woodywu/APBench}"
APBENCH_EXTRA_ARGS="${APBENCH_EXTRA_ARGS:-}"

ASTRO_QA_DATASET_NAME="${ASTRO_QA_DATASET_NAME:-custom}"
ASTRO_QA_DATASET_PATH="${ASTRO_QA_DATASET_PATH:-ACMISLab/Astro-QA}"
ASTRO_QA_EXTRA_ARGS="${ASTRO_QA_EXTRA_ARGS:-}"

SCENARIOS=(
  "ShareGPT:E01:low_poisson:1:1"
  "ShareGPT:E04:medium_poisson:2:1"
  "ShareGPT:E07:medium_burst:2:0.5"
  "ShareGPT:E10:strong_burst:2:0.25"
  "APBench:E13:low_poisson:0.25:1"
  "APBench:E16:medium_poisson:1:1"
  "APBench:E19:medium_burst:1:0.5"
  "APBench:E22:strong_burst:1:0.25"
  "Astro-QA:E25:low_poisson:0.5:1"
  "Astro-QA:E28:medium_poisson:2:1"
  "Astro-QA:E31:medium_burst:2:0.5"
  "Astro-QA:E34:strong_burst:2:0.25"
)

dataset_config() {
  local dataset="$1"
  case "$dataset" in
    ShareGPT)
      printf '%s,%s,%s' "$SHAREGPT_DATASET_NAME" "$SHAREGPT_DATASET_PATH" "$SHAREGPT_EXTRA_ARGS"
      ;;
    APBench)
      printf '%s,%s,%s' "$APBENCH_DATASET_NAME" "$APBENCH_DATASET_PATH" "$APBENCH_EXTRA_ARGS"
      ;;
    Astro-QA)
      printf '%s,%s,%s' "$ASTRO_QA_DATASET_NAME" "$ASTRO_QA_DATASET_PATH" "$ASTRO_QA_EXTRA_ARGS"
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      return 1
      ;;
  esac
}

printf 'run_id,scenario_id,dataset,scenario,model_label,model,request_rate,burstiness,seed,num_prompts,micro_batch,max_model_len,host,port,pp_size,tp_size,dtype,dataset_name,dataset_path,dataset_extra_args\n' > "$OUT"

i=0
for scenario_spec in "${SCENARIOS[@]}"; do
  IFS=: read -r dataset scenario_base scenario request_rate burstiness <<< "$scenario_spec"
  IFS=, read -r dataset_name dataset_path dataset_extra_args <<< "$(dataset_config "$dataset")"

  for model_spec in "${MODELS[@]}"; do
    IFS=: read -r model_label model <<< "$model_spec"
    scenario_num="${scenario_base#E}"
    case "$model_label" in
      3B) model_offset=0 ;;
      7B) model_offset=1 ;;
      14B) model_offset=2 ;;
      *)
        echo "Unknown model label: $model_label" >&2
        exit 1
        ;;
    esac
    scenario_id="$(printf 'E%02d' "$((10#$scenario_num + model_offset))")"

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

echo "Wrote $OUT with $i fixed-M experiments."
