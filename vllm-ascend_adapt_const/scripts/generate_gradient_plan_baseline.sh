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
  "2"
  "4"
)

NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-float16}"

# The prepared dataset directory on the server.
DATASET_ROOT="${DATASET_ROOT:-/workspace/DATASET}"
CUSTOM_DATA_ROOT="${CUSTOM_DATA_ROOT:-$DATASET_ROOT/vllm_custom}"
# Default to using existing vllm_custom/*.jsonl files. Set PREPARE_DATASETS=1
# only when the raw downloaded repos also exist under DATASET_ROOT.
PREPARE_DATASETS="${PREPARE_DATASETS:-0}"
CUSTOM_SAMPLE_LIMIT="${CUSTOM_SAMPLE_LIMIT:-200}"
# Leave empty by default so offline servers do not try to reach Hugging Face.
# Set TOKENIZER_MODEL to a local tokenizer/model path for more accurate token trimming.
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"

GOV_REPORT_SOURCE="${GOV_REPORT_SOURCE:-$DATASET_ROOT/gov_report/data/gao_train.jsonl}"
LONGBENCH_QMSUM_SOURCE="${LONGBENCH_QMSUM_SOURCE:-$DATASET_ROOT/LongBench/data/qmsum.jsonl}"
STELLA_CORPUS_SOURCE="${STELLA_CORPUS_SOURCE:-$DATASET_ROOT/STELLA/corpus.jsonl}"

CUSTOM_OUTPUT_LEN="${CUSTOM_OUTPUT_LEN:-128}"
CUSTOM_EXTRA_ARGS="${CUSTOM_EXTRA_ARGS:---custom-output-len $CUSTOM_OUTPUT_LEN --skip-chat-template}"

# Main switching-point experiment: one corpus, controlled input lengths.
GOV_REPORT_LENGTHS=(
  "1K:1024"
  "2K:2048"
  "3K:3072"
  "3.5K:3584"
  "3.8K:3840"
)

# Cross-task and aerospace-domain validation.
LONGBENCH_QMSUM_LENGTHS=(
  "1K:1024"
  "3K:3072"
  "3.8K:3840"
)

STELLA_LENGTHS=(
  "1K:1024"
)

prepare_custom_datasets() {
  mkdir -p "$CUSTOM_DATA_ROOT"
  python - "$GOV_REPORT_SOURCE" "$LONGBENCH_QMSUM_SOURCE" "$STELLA_CORPUS_SOURCE" \
    "$CUSTOM_DATA_ROOT" "$CUSTOM_SAMPLE_LIMIT" "$TOKENIZER_MODEL" <<'PY'
import json
import re
import sys
from pathlib import Path

gov_src, qmsum_src, stella_src, out_root, sample_limit, tokenizer_model = sys.argv[1:]
out_root = Path(out_root)
sample_limit = int(sample_limit)

LENGTHS_MAIN = [("1K", 1024), ("2K", 2048), ("3K", 3072), ("3.5K", 3584), ("3.8K", 3840)]
LENGTHS_VALIDATE = [("1K", 1024), ("3K", 3072), ("3.8K", 3840)]

try:
    if not tokenizer_model:
        raise RuntimeError("TOKENIZER_MODEL is empty")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
except Exception as exc:
    tokenizer = None
    print(f"Tokenizer unavailable ({exc}); falling back to whitespace token approximation.", file=sys.stderr)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def token_count(text):
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def trim_to_tokens(text, budget):
    if budget <= 0:
        return ""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(ids[:budget], skip_special_tokens=True)
    return " ".join(text.split()[:budget])


def make_prompt(task, document, target_tokens):
    document = " ".join(str(document).split())
    if task == "gov_report":
        prefix = "Summarize the following government report.\n\nReport:\n"
    elif task == "qmsum":
        prefix = "Answer the meeting-summary request using the transcript.\n\nTranscript:\n"
    elif task == "stella":
        prefix = "Analyze the following aerospace technical document and provide a concise technical summary.\n\nDocument:\n"
    else:
        raise ValueError(task)
    suffix = "\n\nResponse:"
    budget = target_tokens - token_count(prefix) - token_count(suffix)
    prompt = prefix + trim_to_tokens(document, budget) + suffix
    return prompt


def flatten_report_sections(value):
    parts = []
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, list):
        for item in value:
            parts.extend(flatten_report_sections(item))
    elif isinstance(value, dict):
        title = value.get("section_title")
        if title:
            parts.append(str(title))
        parts.extend(flatten_report_sections(value.get("paragraphs", [])))
        parts.extend(flatten_report_sections(value.get("subsections", [])))
    return parts


def write_dataset(filename, task, docs, target_tokens):
    path = out_root / filename
    prompts = []
    for doc in docs:
        prompt = make_prompt(task, doc, target_tokens)
        if token_count(prompt) < max(64, target_tokens - 64):
            continue
        prompts.append(prompt)
        if len(prompts) >= sample_limit:
            break
    if not prompts:
        print(f"Wrote 0 prompts: {path}")
        return

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for idx in range(sample_limit):
            prompt = prompts[idx % len(prompts)]
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    count = sample_limit
    print(f"Wrote {count} prompts: {path}")


def gov_docs():
    for item in read_jsonl(gov_src):
        title = item.get("title", "")
        report = "\n".join(flatten_report_sections(item.get("report", [])))
        if report:
            yield f"Title: {title}\n\n{report}"


def qmsum_docs():
    for item in read_jsonl(qmsum_src):
        question = item.get("input", "")
        context = item.get("context", "")
        if context:
            yield f"Request: {question}\n\n{context}"


def stella_doc_id(raw_id):
    stem = str(raw_id).rsplit("/", 1)[-1]
    return re.sub(r"_[0-9]+\.txt$", "", stem)


def write_stella_datasets():
    targets = LENGTHS_VALIDATE
    handles = {}
    counts = {}
    try:
        for label, _length in targets:
            path = out_root / f"stella_corpus_merged_{label}.jsonl"
            handles[label] = path.open("w", encoding="utf-8", newline="\n")
            counts[label] = 0

        # STELLA corpus shards for the same document are not adjacent. Keep a
        # bounded in-memory accumulator and emit a document as soon as it is
        # long enough for each target length.
        docs = {}
        emitted = {}
        approx_counts = {}

        def maybe_emit(doc_id):
            doc = "\n".join(docs.get(doc_id, []))
            if not doc.strip():
                return
            emitted.setdefault(doc_id, set())
            for label, target_tokens in targets:
                if counts[label] >= sample_limit or label in emitted[doc_id]:
                    continue
                if approx_counts.get(doc_id, 0) < target_tokens:
                    continue
                prompt = make_prompt("stella", doc, target_tokens)
                if token_count(prompt) < max(64, target_tokens - 64):
                    continue
                handles[label].write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
                counts[label] += 1
                emitted[doc_id].add(label)
            if len(emitted[doc_id]) == len(targets):
                docs.pop(doc_id, None)
                emitted.pop(doc_id, None)
                approx_counts.pop(doc_id, None)

        for item in read_jsonl(stella_src):
            if all(count >= sample_limit for count in counts.values()):
                break
            doc_id = stella_doc_id(item.get("_id", ""))
            text = item.get("text", "")
            if not text:
                continue
            title = item.get("title", "")
            chunk = f"{title}\n{text}" if title else text
            docs.setdefault(doc_id, []).append(chunk)
            approx_counts[doc_id] = approx_counts.get(doc_id, 0) + len(chunk.split())
            maybe_emit(doc_id)

        for doc_id in list(docs):
            maybe_emit(doc_id)
    finally:
        for handle in handles.values():
            handle.close()

    for label, _length in targets:
        print(f"Wrote {counts[label]} prompts: {out_root / f'stella_corpus_merged_{label}.jsonl'}")


for src in (gov_src, qmsum_src, stella_src):
    if not Path(src).is_file():
        raise FileNotFoundError(src)

out_root.mkdir(parents=True, exist_ok=True)

for label, length in LENGTHS_MAIN:
    write_dataset(f"gov_report_gao_{label}.jsonl", "gov_report", gov_docs(), length)

for label, length in LENGTHS_VALIDATE:
    write_dataset(f"longbench_qmsum_{label}.jsonl", "qmsum", qmsum_docs(), length)

write_stella_datasets()
PY
}

append_dataset_specs() {
  local -n out_ref="$1"
  local prefix="$2"
  local scenario_base="$3"
  local filename_prefix="$4"
  shift 4
  local lengths=("$@")

  for length_spec in "${lengths[@]}"; do
    IFS=: read -r length_label target_tokens <<< "$length_spec"
    out_ref+=("${prefix}-${length_label}:${scenario_base}:${target_tokens}:${CUSTOM_DATA_ROOT}/${filename_prefix}_${length_label}.jsonl")
    scenario_num="${scenario_base#E}"
    scenario_base="$(printf 'E%02d' "$((10#$scenario_num + 1))")"
  done
}

if [[ "$PREPARE_DATASETS" == "1" ]]; then
  prepare_custom_datasets
fi

DATASETS=()
append_dataset_specs DATASETS "GovReport-GAO" "E01" "gov_report_gao" "${GOV_REPORT_LENGTHS[@]}"
append_dataset_specs DATASETS "LongBench-QMSum" "E20" "longbench_qmsum" "${LONGBENCH_QMSUM_LENGTHS[@]}"
append_dataset_specs DATASETS "STELLA-Corpus" "E30" "stella_corpus_merged" "${STELLA_LENGTHS[@]}"

LOAD_SPECS=(
  "medium|auto-medium|3"
  "high|auto-high|6"
)

printf 'run_id,scenario_id,dataset,scenario,model_label,model,request_rate,burstiness,seed,num_prompts,micro_batch,max_model_len,host,port,pp_size,tp_size,dtype,dataset_name,dataset_path,dataset_extra_args\n' > "$OUT"

i=0
formal_i=0
for dataset_spec in "${DATASETS[@]}"; do
  IFS=: read -r dataset scenario_base target_tokens dataset_path <<< "$dataset_spec"
  dataset_name="custom"
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

    for load_spec in "${LOAD_SPECS[@]}"; do
      IFS='|' read -r load_level request_rate scenario_offset <<< "$load_spec"
      scenario="${load_level}_poisson"
      burstiness="1"
      scenario_num="${scenario_base#E}"
      scenario_id="$(printf 'E%02d' "$((10#$scenario_num + model_offset + scenario_offset))")"

      for micro_batch in "${MICRO_BATCHES[@]}"; do
        i=$((i + 1))
        formal_i=$((formal_i + 1))
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

echo "Wrote $OUT with $formal_i fixed-M experiments ($i rows total; capacities come from the shared exchange)."
