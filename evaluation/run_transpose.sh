#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash evaluation/run_transpose.sh [checkpoint] [camera_type=l515]

Arguments:
  checkpoint       OMNI-DC checkpoint. Default: CHECKPOINT or ckpts/modelv1.1_best_72epochs.pt
  camera_type      Raw depth source: l515. Default: RAW_TYPE or l515

Environment overrides:
  CHECKPOINT       OMNI-DC checkpoint. Default: ckpts/modelv1.1_best_72epochs.pt
  CKPT_DIR         Directory for dependency checkpoints. Default: ckpts
  DATASET_PATH     TRansPose JSONL path. Default: data/TRansPose/sequences/dc_testset.jsonl
  INTRINSICS_PATH  Camera intrinsics path. Default: data/TRansPose/sequences/intrinsics.txt
  OUTPUT_DIR       Prediction/evaluation output directory.
  BATCH_SIZE       Kept for compatibility; inference runs one image at a time. Default: 1
  NUM_WORKERS      Kept for compatibility. Default: 0
  SAVE_VIS         Save 3x2 visualization grids. true/false. Default: true
  CLEANUP_NPY      Remove predictions/*.npy after evaluation. true/false. Default: false
  MAX_SAMPLES      Sample limit. 0 evaluates all samples. Default: 0
  PYTHON_BIN       Python executable. Default: python

TRansPose is fixed to raw-type=l515.
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
camera_type="${2:-${RAW_TYPE:-l515}}"
dataset_path="${DATASET_PATH:-data/TRansPose/sequences/dc_testset.jsonl}"
intrinsics_path="${INTRINSICS_PATH:-data/TRansPose/sequences/intrinsics.txt}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
save_vis="${SAVE_VIS:-true}"
cleanup_npy="${CLEANUP_NPY:-false}"
max_samples="${MAX_SAMPLES:-0}"

case "${camera_type}" in
    l515)
        ;;
    *)
        echo "unknown TRansPose camera_type: ${camera_type} (expected: l515)" >&2
        exit 2
        ;;
esac

checkpoint="$(resolve_path "${checkpoint}")"
ckpt_dir="$(resolve_path "${ckpt_dir}")"
dataset_path="$(resolve_path "${dataset_path}")"
intrinsics_path="$(resolve_path "${intrinsics_path}")"

model_name="$(basename "${checkpoint}")"
model_stub="${model_name%%.*}"
model_dir="$(dirname "${checkpoint}")"
output_dir="${OUTPUT_DIR:-${model_dir}/transpose_${model_stub}_data_${camera_type}}"
output_dir="$(resolve_path "${output_dir}")"

save_vis_arg=()
if [[ "${save_vis}" == "false" || "${save_vis}" == "0" ]]; then
    save_vis_arg=(--no-save-vis)
else
    save_vis_arg=(--save-vis)
fi

echo "project root: ${PROJECT_ROOT}"
echo "model: OMNI-DC OGNIDC v1.1"
echo "checkpoint: ${checkpoint}"
echo "ckpt dir: ${ckpt_dir}"
echo "dataset: TRansPose"
echo "dataset path: ${dataset_path}"
echo "camera type: ${camera_type}"
echo "intrinsics path: ${intrinsics_path}"
echo "output dir: ${output_dir}"
echo "save vis: ${save_vis}"
echo "cleanup npy: ${cleanup_npy}"
echo "max samples: ${max_samples}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
    --model-path "${checkpoint}" \
    --ckpt-dir "${ckpt_dir}" \
    --dataset "${dataset_path}" \
    --intrinsics-path "${intrinsics_path}" \
    --raw-type "${camera_type}" \
    --output "${output_dir}" \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --max-samples "${max_samples}" \
    "${save_vis_arg[@]}"

echo "evaluating the model on TRansPose"
time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
    --encoder vitl \
    --model-path "${checkpoint}" \
    --dataset "${dataset_path}" \
    --output "${output_dir}" \
    --raw-type "${camera_type}" \
    --max-samples "${max_samples}"

if [[ "${cleanup_npy}" == "true" || "${cleanup_npy}" == "1" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
    if [[ -d "${output_dir}/predictions" ]]; then
        find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
    fi
fi
