#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  ./evaluation/run_eval.sh [checkpoint]

Environment overrides:
  CHECKPOINT       OMNI-DC checkpoint. Default: ckpts/modelv1.1_best_72epochs.pt
  CKPT_DIR         Directory for dependency checkpoints. Default: ckpts
  DATASET_PATH     HAMMER JSONL path. Default: data/HAMMER/test.jsonl
  INTRINSICS_PATH  HAMMER intrinsics path. Default: data/HAMMER/intrinsics.txt
  OUTPUT_DIR       Prediction/evaluation output directory. Default: evaluation/output
  RAW_TYPE         HAMMER raw depth source: d435, l515, tof. Default: d435
  BATCH_SIZE       Kept for compatibility; inference runs one image at a time. Default: 1
  NUM_WORKERS      Kept for compatibility. Default: 0
  SAVE_VIS         Save visualization grids by default. true/false. Default: true
  CLEANUP_NPY      Remove prediction .npy files after evaluation. Default: false
  MAX_SAMPLES      Optional smoke-test sample limit.
  PYTHON_BIN       Python executable. Default: python

Required default files under ckpts/:
  modelv1.1_best_72epochs.pt
  resnet34.pth
  pvt.pth
  depth_anything_v2_vitl.pth
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
dataset_path="${DATASET_PATH:-data/HAMMER/test.jsonl}"
intrinsics_path="${INTRINSICS_PATH:-data/HAMMER/intrinsics.txt}"
output_dir="${OUTPUT_DIR:-evaluation/output}"
raw_type="${RAW_TYPE:-d435}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
save_vis="${SAVE_VIS:-true}"
cleanup_npy="${CLEANUP_NPY:-false}"
max_samples="${MAX_SAMPLES:-}"

checkpoint="$(resolve_path "${checkpoint}")"
ckpt_dir="$(resolve_path "${ckpt_dir}")"
dataset_path="$(resolve_path "${dataset_path}")"
intrinsics_path="$(resolve_path "${intrinsics_path}")"
output_dir="$(resolve_path "${output_dir}")"

echo "project root: ${PROJECT_ROOT}"
echo "model: OMNI-DC OGNIDC v1.1"
echo "checkpoint: ${checkpoint}"
echo "ckpt dir: ${ckpt_dir}"
echo "dataset path: ${dataset_path}"
echo "intrinsics path: ${intrinsics_path}"
echo "raw type: ${raw_type}"
echo "output dir: ${output_dir}"
echo "save vis: ${save_vis}"
echo "cleanup npy: ${cleanup_npy}"

infer_args=(
    "${SCRIPT_DIR}/infer.py"
    --model-path "${checkpoint}"
    --ckpt-dir "${ckpt_dir}"
    --dataset "${dataset_path}"
    --intrinsics-path "${intrinsics_path}"
    --raw-type "${raw_type}"
    --output "${output_dir}"
    --batch-size "${batch_size}"
    --num-workers "${num_workers}"
)

eval_args=(
    "${SCRIPT_DIR}/eval.py"
    --encoder vitl
    --model-path "${checkpoint}"
    --dataset "${dataset_path}"
    --output "${output_dir}"
    --raw-type "${raw_type}"
)

if [[ -n "${max_samples}" ]]; then
    infer_args+=(--max-samples "${max_samples}")
    eval_args+=(--max-samples "${max_samples}")
fi

if [[ "${save_vis}" == "false" || "${save_vis}" == "0" ]]; then
    infer_args+=(--no-save-vis)
else
    infer_args+=(--save-vis)
fi

"${PYTHON_BIN}" "${infer_args[@]}"

echo "evaluating the model on HAMMER"
time "${PYTHON_BIN}" "${eval_args[@]}"

if [[ "${cleanup_npy}" == "true" || "${cleanup_npy}" == "1" ]]; then
    echo "CLEANUP_NPY is enabled, removing generated .npy files under ${output_dir}"
    find "${output_dir}" -maxdepth 1 -type f -name '*.npy' -delete
fi
