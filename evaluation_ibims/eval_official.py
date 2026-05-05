#!/usr/bin/env python3
"""Prepare an iBims official eval workspace and run the bundled evaluator."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_SHAPE = (480, 640)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data/ibims1 official iBims evaluation on *_results.mat predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ibims-root", default="data/ibims1", help="iBims dataset root")
    parser.add_argument("--pred-dir", required=True, help="Directory containing *_results.mat predictions")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Official eval workspace; defaults to <run-dir>/official_eval/<level>/workspace",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Sample limit from imagelist.txt. 0 evaluates all samples",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and validate the workspace without running the official script",
    )
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be 0 or a positive integer")
    return args


def resolve_root(path: str | Path) -> Path:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path.cwd() / path_obj
    return path_obj.resolve()


def default_workspace(pred_dir: Path) -> Path:
    if pred_dir.parent.name == "predictions":
        run_dir = pred_dir.parent.parent
        return run_dir / "official_eval" / pred_dir.name / "workspace"
    return pred_dir / "_official_eval_workspace"


def read_image_names(imagelist_path: Path, max_samples: int) -> list[str]:
    with open(imagelist_path, "r", encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]
    if max_samples > 0:
        names = names[:max_samples]
    if not names:
        raise ValueError(f"No image names found in: {imagelist_path}")
    return names


def link_or_copy(src: Path, dst: Path) -> None:
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src:
            return
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def validate_prediction(path: Path) -> None:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise SystemExit("scipy is required to validate official iBims .mat predictions.") from exc

    mat = loadmat(path)
    if "pred_depths" not in mat:
        raise ValueError(f"Missing pred_depths variable: {path}")
    pred = mat["pred_depths"]
    if pred.shape != EXPECTED_SHAPE:
        raise ValueError(f"{path} has shape {pred.shape}, expected {EXPECTED_SHAPE}")


def prepare_workspace(
    ibims_root: Path,
    pred_dir: Path,
    workspace: Path,
    max_samples: int,
) -> tuple[Path, list[str]]:
    imagelist_path = ibims_root / "imagelist.txt"
    mat_root = ibims_root / "ibims1_core_mat"
    eval_script = ibims_root / "evaluation_scripts" / "evaluate_ibims.py"

    for required_path in (imagelist_path, mat_root, eval_script, pred_dir):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required path: {required_path}")

    names = read_image_names(imagelist_path, max_samples)
    workspace.mkdir(parents=True, exist_ok=True)

    with open(workspace / "imagelist.txt", "w", encoding="utf-8") as file:
        for name in names:
            file.write(f"{name}\n")

    for name in names:
        gt_mat = mat_root / f"{name}.mat"
        pred_mat = pred_dir / f"{name}_results.mat"
        if not gt_mat.exists():
            raise FileNotFoundError(f"Missing iBims GT mat: {gt_mat}")
        if not pred_mat.exists():
            raise FileNotFoundError(f"Missing prediction mat: {pred_mat}")
        validate_prediction(pred_mat)
        link_or_copy(gt_mat, workspace / gt_mat.name)
        link_or_copy(pred_mat, workspace / pred_mat.name)

    return eval_script, names


def run_official_eval(
    eval_script: Path,
    workspace: Path,
    log_path: Path,
    *,
    check: bool = True,
    echo: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    script_dir = str(eval_script.parent)
    env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(eval_script)],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as file:
        file.write(result.stdout)
        if result.stderr:
            file.write("\n[stderr]\n")
            file.write(result.stderr)

    if echo and result.stdout:
        print(result.stdout, end="")
    if echo and result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def main() -> None:
    args = parse_args()
    ibims_root = resolve_root(args.ibims_root)
    pred_dir = resolve_root(args.pred_dir)
    workspace = resolve_root(args.workspace) if args.workspace else default_workspace(pred_dir).resolve()

    eval_script, names = prepare_workspace(ibims_root, pred_dir, workspace, args.max_samples)
    print(f"Prepared official iBims eval workspace: {workspace}")
    print(f"Validated {len(names)} predictions from: {pred_dir}")

    if args.prepare_only:
        return

    log_path = workspace.parent / "official_eval_stdout.txt"
    run_official_eval(eval_script, workspace, log_path)
    print(f"Official eval log saved to: {log_path}")


if __name__ == "__main__":
    main()
