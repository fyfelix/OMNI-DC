#!/usr/bin/env python3
"""One-shot OMNI-DC iBims inference and official evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
for path in (PROJECT_ROOT, PIPELINE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from eval_official import prepare_workspace, resolve_root, run_official_eval  # noqa: E402
from infer_to_mat import (  # noqa: E402
    ALL_LEVELS,
    SYNTHETIC_RAW_DIR_NAME,
    load_model,
    resolve_inference_args,
    resolve_project_path,
    run_manifest_inference,
    str2bool,
)


RESULT_METRIC_KEYS = [
    "rel",
    "sq_rel",
    "rms",
    "log10",
    "thr1",
    "thr2",
    "thr3",
    "dde_0",
    "dde_p",
    "dde_m",
    "pe_fla",
    "pe_ori",
    "dbe_acc",
    "dbe_com",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OMNI-DC iBims inference and official eval across difficulty levels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        "--checkpoint",
        dest="model_path",
        default=None,
        help="Path to OMNI-DC checkpoint. Defaults to <ckpt-dir>/modelv1.1_best_72epochs.pt",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="ckpts",
        help="Directory containing OMNI-DC and dependency checkpoints",
    )
    parser.add_argument("--ibims-root", default="data/ibims1", help="iBims dataset root")
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=ALL_LEVELS,
        default=list(ALL_LEVELS),
        help="Difficulty levels to process",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Output root directory; defaults to evaluation_ibims/output/ibims_omnidc_<model>_<timestamp>",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Kept for wrapper compatibility; OMNI-DC inference runs one image at a time",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Sample limit per level. 0 processes all samples",
    )
    parser.add_argument(
        "--intrinsics-path",
        default=None,
        help="Optional global camera intrinsics override. Defaults to ibims1_core_raw/calib/<sample_id>.txt",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Raw depth scale; defaults to each manifest row depth_scale",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Depth clamp for raw input; defaults to each manifest row depth-range max",
    )
    parser.add_argument(
        "--load-dav2",
        type=str2bool,
        default=True,
        help="Use OMNI-DC v1.1 Depth Anything V2 auxiliary depth",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip inference and use existing predictions in --run-dir",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip official evaluation and only run inference",
    )
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be 0 or a positive integer")
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than 0")
    return args


def default_model_path(args: argparse.Namespace) -> Path:
    ckpt_dir = resolve_project_path(args.ckpt_dir)
    return resolve_project_path(args.model_path) if args.model_path else ckpt_dir / "modelv1.1_best_72epochs.pt"


def default_run_dir(model_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return PIPELINE_DIR / "output" / f"ibims_omnidc_{model_path.stem}_{timestamp}"


def manifest_for_level(ibims_root: Path, level: str) -> Path:
    return ibims_root / SYNTHETIC_RAW_DIR_NAME / "manifests" / f"ibims_{level}.jsonl"


def parse_eval_stdout(text: str) -> dict[str, float]:
    results = {}
    in_block = False
    for line in text.splitlines():
        if not in_block:
            if line.strip() == "Results:":
                in_block = True
            continue
        if line.strip() == "":
            continue
        match = re.match(r"(\S+)\s*=\s*([\d.eE+\-]+)", line.strip())
        if match:
            results[match.group(1)] = float(match.group(2))
        else:
            break
    return results


def run_inference(args: argparse.Namespace, run_dir: Path) -> None:
    args = resolve_inference_args(args)
    ibims_root = Path(args.ibims_root)
    model = load_model(args)
    print(f"Model: OMNI-DC OGNIDC v1.1 ({args.model_path})")
    print("Device: cuda")

    for level in args.levels:
        manifest_path = manifest_for_level(ibims_root, level)
        if not manifest_path.is_file():
            print(f"[skip infer] manifest not found: {manifest_path}")
            continue

        pred_dir = run_dir / "predictions" / level
        stats = run_manifest_inference(
            manifest_path,
            pred_dir,
            model,
            ibims_root=ibims_root,
            batch_size=args.batch_size,
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
            intrinsics_path=args.intrinsics_path,
            run_metadata={
                **vars(args),
                "model_path": str(args.model_path),
                "device_resolved": "cuda",
                "resolved_model_module": "model.ognidc",
                "resolved_model_class": "OGNIDC",
                "output_kind": "metric_depth_meter",
                "alignment": "none",
            },
        )
        print(f"[infer] {level}: wrote {stats['num_predictions']} predictions to {pred_dir}")


def run_evaluation(args: argparse.Namespace, run_dir: Path) -> None:
    ibims_root = resolve_root(args.ibims_root)
    all_metrics: dict[str, dict[str, float]] = {}

    for level in args.levels:
        pred_dir = run_dir / "predictions" / level
        if not pred_dir.is_dir():
            print(f"[skip eval] prediction dir not found: {pred_dir}")
            continue

        workspace = run_dir / "official_eval" / level / "workspace"
        log_path = run_dir / "official_eval" / level / "official_eval_stdout.txt"
        print(f"[eval] {level}: preparing workspace {workspace}")
        eval_script, names = prepare_workspace(ibims_root, pred_dir, workspace, args.max_samples)
        print(f"[eval] {level}: validated {len(names)} predictions")
        print(f"[eval] {level}: running official eval")

        result = run_official_eval(eval_script, workspace, log_path, check=False, echo=False)
        if result.returncode != 0:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            print(f"[eval] {level}: official eval failed, log saved to {log_path}", file=sys.stderr)
            raise SystemExit(result.returncode)

        metrics = parse_eval_stdout(result.stdout)
        all_metrics[level] = metrics
        if metrics:
            print(f"[eval] {level}: extracted {len(metrics)} metrics")
        else:
            print(f"[eval] {level}: WARNING - no metrics parsed from output")
            print(result.stdout[-500:] if result.stdout else "(empty stdout)")

    if all_metrics:
        summary_path = run_dir / "eval_summary.csv"
        with open(summary_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["level"] + RESULT_METRIC_KEYS)
            writer.writeheader()
            for level in [item for item in ALL_LEVELS if item in all_metrics]:
                writer.writerow(
                    {"level": level, **{key: all_metrics[level].get(key) for key in RESULT_METRIC_KEYS}}
                )
        with open(run_dir / "eval_summary.json", "w", encoding="utf-8") as file:
            json.dump(all_metrics, file, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"Eval summary saved to: {summary_path}")
        print_metrics_table(all_metrics)
    else:
        print("[eval] No metrics collected.")


def print_metrics_table(all_metrics: dict[str, dict[str, float]]) -> None:
    levels = [level for level in ALL_LEVELS if level in all_metrics]
    all_keys = []
    for metrics in all_metrics.values():
        for key in metrics:
            if key not in all_keys:
                all_keys.append(key)

    col_width = 10
    header = f"{'metric':<12}" + "".join(f"{level:>{col_width}}" for level in levels)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for key in all_keys:
        row = f"{key:<12}"
        for level in levels:
            value = all_metrics[level].get(key)
            row += f"{value:{col_width}.4f}" if value is not None else f"{'-':>{col_width}}"
        print(row)
    print(sep)


def main() -> None:
    args = parse_args()
    model_path = default_model_path(args)
    run_dir = resolve_project_path(args.run_dir) if args.run_dir else default_run_dir(model_path).resolve()
    args.ibims_root = str(resolve_project_path(args.ibims_root))

    if args.skip_infer and not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist (needed when --skip-infer): {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    if not args.skip_infer:
        run_inference(args, run_dir)

    if not args.skip_eval:
        run_evaluation(args, run_dir)


if __name__ == "__main__":
    main()
