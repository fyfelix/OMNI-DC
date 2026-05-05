#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

usage() {
    printf '%s\n' \
"Usage:" \
"  ./evaluation_ibims/run_all.sh [model_path=ckpts/modelv1.1_best_72epochs.pt] [extra run_all.py args...]" \
"" \
"Environment overrides:" \
"  PYTHON_BIN    Python executable. Default: .venv/bin/python if present, otherwise python"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
    model_path="${1}"
    shift
else
    model_path="ckpts/modelv1.1_best_72epochs.pt"
fi

cd "${PROJECT_ROOT}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_all.py" \
    --model-path "${model_path}" \
    "$@"
