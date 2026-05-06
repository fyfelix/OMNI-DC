#!/usr/bin/env bash

set -euo pipefail

export OPENCV_IO_ENABLE_OPENEXR=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash evaluation/run_dreds.sh [checkpoint] [variant=all]

Arguments:
  checkpoint       OMNI-DC checkpoint. Default: CHECKPOINT or ckpts/modelv1.1_best_72epochs.pt
  variant          catknown | catnovel | all. Default: all

Environment overrides:
  CHECKPOINT          OMNI-DC checkpoint. Default: ckpts/modelv1.1_best_72epochs.pt
  CKPT_DIR            Directory for dependency checkpoints. Default: ckpts
  DREDS_KNOWN_JSONL   DREDS catknown JSONL. Default: data/DREDS/test_std_catknown.jsonl
  DREDS_NOVEL_JSONL   DREDS catnovel JSONL. Default: data/DREDS/test_std_catnovel.jsonl
  INTRINSICS_PATH     Camera intrinsics path. Default: data/HAMMER/intrinsics.txt
  OUTPUT_DIR          Output directory for variant=catknown or variant=catnovel only.
  OUTPUT_ROOT         Output root for default per-variant directories. Default: checkpoint directory
  BATCH_SIZE          Kept for compatibility; inference runs one image at a time. Default: 1
  NUM_WORKERS         Kept for compatibility. Default: 0
  SAVE_VIS            Save visualization grids. true/false. Default: true
  CLEANUP_NPY         Remove predictions/*.npy after evaluation. true/false. Default: false
  MAX_SAMPLES         Sample limit. 0 evaluates all samples. Default: 0
  PYTHON_BIN          Python executable. Default: python

DREDS uses EXR floating-point depth in meters. raw-type is passed as d435 only
to satisfy the shared Python CLI and is ignored by the DREDS dataset loader.
EOF
}

resolve_path() {
    local input="$1"
    if [[ "${input}" = /* ]]; then
        printf '%s\n' "${input}"
    else
        printf '%s\n' "${PROJECT_ROOT}/${input}"
    fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ckpt_dir="${CKPT_DIR:-ckpts}"
checkpoint="${1:-${CHECKPOINT:-${ckpt_dir}/modelv1.1_best_72epochs.pt}}"
variant="${2:-all}"
camera_type="d435"
dreds_known_jsonl="${DREDS_KNOWN_JSONL:-data/DREDS/test_std_catknown.jsonl}"
dreds_novel_jsonl="${DREDS_NOVEL_JSONL:-data/DREDS/test_std_catnovel.jsonl}"
intrinsics_path="${INTRINSICS_PATH:-data/HAMMER/intrinsics.txt}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
save_vis="${SAVE_VIS:-true}"
cleanup_npy="${CLEANUP_NPY:-false}"
max_samples="${MAX_SAMPLES:-0}"

if [[ "${variant}" == "all" && -n "${OUTPUT_DIR:-}" ]]; then
    echo "OUTPUT_DIR can only be used with variant=catknown or variant=catnovel; use OUTPUT_ROOT for variant=all." >&2
    exit 2
fi

checkpoint="$(resolve_path "${checkpoint}")"
ckpt_dir="$(resolve_path "${ckpt_dir}")"
dreds_known_jsonl="$(resolve_path "${dreds_known_jsonl}")"
dreds_novel_jsonl="$(resolve_path "${dreds_novel_jsonl}")"
intrinsics_path="$(resolve_path "${intrinsics_path}")"

model_name="$(basename "${checkpoint}")"
model_stub="${model_name%%.*}"
model_dir="$(dirname "${checkpoint}")"
output_root="${OUTPUT_ROOT:-${model_dir}}"
output_root="$(resolve_path "${output_root}")"

save_vis_arg=()
if [[ "${save_vis}" == "false" || "${save_vis}" == "0" ]]; then
    save_vis_arg=(--no-save-vis)
else
    save_vis_arg=(--save-vis)
fi

run_one_variant() {
    local label="$1"
    local jsonl_path="$2"
    local output_dir

    if [[ -n "${OUTPUT_DIR:-}" ]]; then
        output_dir="$(resolve_path "${OUTPUT_DIR}")"
    else
        output_dir="${output_root}/dreds_${label}_${model_stub}"
    fi

    echo "[${label}] project root: ${PROJECT_ROOT}"
    echo "[${label}] model: OMNI-DC OGNIDC v1.1"
    echo "[${label}] checkpoint: ${checkpoint}"
    echo "[${label}] ckpt dir: ${ckpt_dir}"
    echo "[${label}] dataset: DREDS"
    echo "[${label}] dataset path: ${jsonl_path}"
    echo "[${label}] camera type: ${camera_type}"
    echo "[${label}] intrinsics path: ${intrinsics_path}"
    echo "[${label}] output dir: ${output_dir}"
    echo "[${label}] save vis: ${save_vis}"
    echo "[${label}] cleanup npy: ${cleanup_npy}"
    echo "[${label}] max samples: ${max_samples}"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
        --model-path "${checkpoint}" \
        --ckpt-dir "${ckpt_dir}" \
        --dataset "${jsonl_path}" \
        --intrinsics-path "${intrinsics_path}" \
        --raw-type "${camera_type}" \
        --output "${output_dir}" \
        --batch-size "${batch_size}" \
        --num-workers "${num_workers}" \
        --max-samples "${max_samples}" \
        "${save_vis_arg[@]}"

    echo "[${label}] evaluating the model on DREDS"
    time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
        --encoder vitl \
        --model-path "${checkpoint}" \
        --dataset "${jsonl_path}" \
        --output "${output_dir}" \
        --raw-type "${camera_type}" \
        --max-samples "${max_samples}"

    if [[ "${cleanup_npy}" == "true" || "${cleanup_npy}" == "1" ]]; then
        echo "[${label}] cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
        if [[ -d "${output_dir}/predictions" ]]; then
            find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
        fi
    fi
}

case "${variant}" in
    catknown)
        run_one_variant catknown "${dreds_known_jsonl}"
        ;;
    catnovel)
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    all)
        run_one_variant catknown "${dreds_known_jsonl}"
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    *)
        echo "unknown DREDS variant: ${variant} (expected: catknown | catnovel | all)" >&2
        exit 2
        ;;
esac
