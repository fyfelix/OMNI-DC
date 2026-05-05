#!/usr/bin/env python3
"""Run OMNI-DC inference for iBims and save official *_results.mat files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


EVALUATION_IBIMS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_IBIMS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (str(SRC_DIR), str(SRC_DIR / "model")):
    if path not in sys.path:
        sys.path.insert(0, path)


IBIMS_DEPTH_MAX_M = 50.0
IBIMS_DEPTH_SCALE = 65535.0 / IBIMS_DEPTH_MAX_M
SYNTHETIC_RAW_DIR_NAME = "ibims1_synthetic_raw_depth"
EXPECTED_SHAPE = (480, 640)
ALL_LEVELS = ("easy", "medium", "hard", "extreme")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "t", "yes", "y"):
        return True
    if value in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def torch_load_compatible(path: str | Path, **kwargs: Any) -> Any:
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OMNI-DC iBims inference writer for official MAT evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="iBims synthetic JSONL manifest path")
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
        "--output-dir",
        default=None,
        help="Prediction directory; defaults to evaluation_ibims/output/ibims_omnidc_<model>_<timestamp>/predictions/<level>",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Kept for wrapper compatibility; OMNI-DC inference runs one image at a time",
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
        "--intrinsics-path",
        default=None,
        help="Optional global camera intrinsics override. Defaults to ibims1_core_raw/calib/<sample_id>.txt",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Sample limit. 0 processes all samples",
    )
    parser.add_argument(
        "--load-dav2",
        type=str2bool,
        default=True,
        help="Use OMNI-DC v1.1 Depth Anything V2 auxiliary depth",
    )
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be 0 or a positive integer")
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than 0")
    return args


def default_model_args(load_dav2: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        model="OGNIDC",
        load_dav2=bool(load_dav2),
        num_resolution=3,
        multi_resolution_learnable_gradients_weights="uniform",
        multi_resolution_learnable_input_weights=0,
        backbone_mode="rgbd",
        backbone="cformer",
        pred_confidence_input=1,
        pred_context_feature=True,
        pred_depth=False,
        depth_activation_format="exp",
        whiten_sparse_depths=1,
        GRU_iters=1,
        gru_internal_whiten_method="median",
        gru_hidden_dim=64,
        gru_context_dim=64,
        optim_layer_input_clamp=1.0,
        integration_alpha=5.0,
        max_depth=300.0,
        prop_time=6,
        prop_kernel=3,
        spn_type="dyspn",
        conf_prop=True,
        conf_min=1.0,
        preserve_input=False,
        affinity="TGASS",
        affinity_gamma=0.5,
        backbone_output_downsample_rate=4,
        depth_downsample_method="min",
        training_depth_random_shift_range=0.0,
        backbone_pattern_condition_format="none",
        num_pattern_types=3,
        loss="1.0*SeqL1+1.0*SeqL2",
    )


def required_weight_links(ckpt_dir: Path, load_dav2: bool) -> list[tuple[Path, Path]]:
    links = [
        (ckpt_dir / "resnet34.pth", SRC_DIR / "pretrained" / "resnet34.pth"),
        (ckpt_dir / "pvt.pth", SRC_DIR / "pretrained" / "pvt.pth"),
    ]
    if load_dav2:
        links.append(
            (
                ckpt_dir / "depth_anything_v2_vitl.pth",
                SRC_DIR
                / "depth_models"
                / "depth_anything_v2"
                / "checkpoints"
                / "depth_anything_v2_vitl.pth",
            )
        )
    return links


def prepare_weight_links(ckpt_dir: Path, load_dav2: bool) -> list[Path]:
    missing = []
    for source, target in required_weight_links(ckpt_dir, load_dav2):
        if target.exists() or target.is_symlink():
            continue
        if not source.exists():
            missing.append(source)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
    return missing


def resolve_inference_args(args: argparse.Namespace) -> argparse.Namespace:
    if hasattr(args, "manifest") and args.manifest is not None:
        args.manifest = str(resolve_project_path(args.manifest))
    args.ibims_root = str(resolve_project_path(args.ibims_root))
    args.ckpt_dir = str(resolve_project_path(args.ckpt_dir))
    if hasattr(args, "output_dir"):
        args.output_dir = str(resolve_project_path(args.output_dir)) if args.output_dir else None
    args.intrinsics_path = (
        str(resolve_project_path(args.intrinsics_path)) if args.intrinsics_path else None
    )

    if args.model_path is None:
        args.model_path = str(Path(args.ckpt_dir) / "modelv1.1_best_72epochs.pt")
    else:
        args.model_path = str(resolve_project_path(args.model_path))

    missing = []
    required_paths = [args.ibims_root, args.model_path]
    if hasattr(args, "manifest") and args.manifest is not None:
        required_paths.insert(0, args.manifest)
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
    if args.intrinsics_path and not Path(args.intrinsics_path).exists():
        missing.append(args.intrinsics_path)
    if not args.intrinsics_path:
        calib_dir = Path(args.ibims_root) / "ibims1_core_raw" / "calib"
        if not calib_dir.is_dir():
            missing.append(calib_dir)

    link_missing = prepare_weight_links(Path(args.ckpt_dir), args.load_dav2)
    missing.extend(str(path) for path in link_missing)

    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {display_path(path)}")
        print("\nPrepare these default files under ckpts/:")
        print("  - ckpts/modelv1.1_best_72epochs.pt")
        print("  - ckpts/resnet34.pth")
        print("  - ckpts/pvt.pth")
        if args.load_dav2:
            print("  - ckpts/depth_anything_v2_vitl.pth")
        raise SystemExit(1)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this OMNI-DC iBims adapter.")

    return args


def load_model(args: argparse.Namespace):
    from model.ognidc import OGNIDC

    model_args = default_model_args(load_dav2=args.load_dav2)
    checkpoint_path = Path(args.model_path)

    cwd = Path.cwd()
    os.chdir(SRC_DIR)
    try:
        model = OGNIDC(model_args)
    finally:
        os.chdir(cwd)

    checkpoint = torch_load_compatible(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        states = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        states = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        states = checkpoint["state_dict"]
    else:
        states = checkpoint

    states = {key[7:] if key.startswith("module.") else key: value for key, value in states.items()}
    load_result = model.load_state_dict(states, strict=False)

    if load_result.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing required OMNI-DC keys:\n"
            + "\n".join(f"  - {key}" for key in load_result.missing_keys[:50])
        )
    if load_result.unexpected_keys:
        print("Warning: unexpected checkpoint keys ignored:")
        for key in load_result.unexpected_keys[:50]:
            print(f"  - {key}")

    model.cuda()
    model.eval()
    return model


def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("dataset") != "ibims":
                raise ValueError(f"{manifest_path}:{line_number} is not an iBims row")
            for key in ("sample_id", "rgb", "raw_depth"):
                if key not in row:
                    raise ValueError(f"{manifest_path}:{line_number} missing required key: {key}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def infer_difficulty(manifest_path: Path, rows: list[dict[str, Any]]) -> str:
    difficulty = rows[0].get("difficulty")
    if difficulty:
        return str(difficulty)
    stem = manifest_path.stem
    return stem[len("ibims_") :] if stem.startswith("ibims_") else stem


def default_output_dir(manifest_path: Path, rows: list[dict[str, Any]], model_path: Path) -> Path:
    difficulty = infer_difficulty(manifest_path, rows)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        EVALUATION_IBIMS_DIR
        / "output"
        / f"ibims_omnidc_{model_path.stem}_{timestamp}"
        / "predictions"
        / difficulty
    )


def row_depth_scale(row: dict[str, Any], cli_depth_scale: float | None) -> float:
    if cli_depth_scale is not None:
        return cli_depth_scale
    return float(row.get("depth_scale", IBIMS_DEPTH_SCALE))


def row_max_depth(row: dict[str, Any], cli_max_depth: float | None) -> float:
    if cli_max_depth is not None:
        return cli_max_depth
    depth_range = row.get("depth-range", [0.01, IBIMS_DEPTH_MAX_M])
    return float(depth_range[1])


def load_intrinsics_file(intrinsics_path: str | Path) -> np.ndarray:
    text = Path(intrinsics_path).read_text(encoding="utf-8")
    cleaned_lines = []
    key_values = {}

    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cleaned_lines.append(line)
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([-+0-9.eE]+)\s*$", line)
        if match:
            key_values[match.group(1).lower()] = float(match.group(2))

    if {"fx", "fy", "cx", "cy"}.issubset(key_values):
        fx = key_values["fx"]
        fy = key_values["fy"]
        cx = key_values["cx"]
        cy = key_values["cy"]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    numeric_text = "\n".join(cleaned_lines).replace(",", " ")
    values = [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", numeric_text)]

    if len(values) == 4:
        fx, fy, cx, cy = values
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if len(values) == 9:
        return np.array(values, dtype=np.float32).reshape(3, 3)
    if len(values) == 16:
        return np.array(values, dtype=np.float32).reshape(4, 4)[:3, :3]

    raise ValueError(
        f"Unsupported intrinsics format in {intrinsics_path}. "
        "Expected 3x3 matrix, 4 values 'fx fy cx cy', or fx/fy/cx/cy key-value lines."
    )


def sample_intrinsics_path(row: dict[str, Any], ibims_root: Path) -> Path:
    sample_id = str(row["sample_id"])
    return ibims_root / "ibims1_core_raw" / "calib" / f"{sample_id}.txt"


def load_sample_intrinsics(
    row: dict[str, Any],
    ibims_root: Path,
    override_intrinsics_path: str | Path | None,
) -> np.ndarray:
    intrinsics_path = Path(override_intrinsics_path) if override_intrinsics_path else sample_intrinsics_path(row, ibims_root)
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"Missing iBims intrinsics file: {intrinsics_path}")
    return load_intrinsics_file(intrinsics_path)


def load_rgb_tensor(rgb_path: str | Path) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(rgb_path).convert("RGB")
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transform(image).unsqueeze(0).cuda(), np.asarray(image)


def load_depth_meters(depth_path: str | Path, depth_scale: float, max_depth: float) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not read depth: {depth_path}")
    depth = np.asarray(depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth / float(depth_scale)
    invalid = ~np.isfinite(depth) | (depth <= 0.0) | (depth > max_depth)
    depth[invalid] = 0.0
    return depth.astype(np.float32)


def infer_one(model: torch.nn.Module, rgb_tensor: torch.Tensor, dep_tensor: torch.Tensor, intrinsics: torch.Tensor) -> np.ndarray:
    sample = {
        "rgb": rgb_tensor,
        "dep": dep_tensor,
        "K": intrinsics,
        "pattern": 0,
    }

    rgb = sample["rgb"]
    dep = sample["dep"]
    rgb_raw = torch.clone(rgb)
    dep_raw = torch.clone(dep)

    _, _, height, width = rgb.shape
    model_args = model.module.args if hasattr(model, "module") else model.args
    divisor = int(4 * 2 ** (model_args.num_resolution - 1))
    if not height % divisor == 0:
        height_new = (height // divisor + 1) * divisor
        height_pad = height_new - height
        rgb = torch.nn.functional.pad(rgb, (0, 0, 0, height_pad))
        dep = torch.nn.functional.pad(dep, (0, 0, 0, height_pad))
    else:
        height_new = height
        height_pad = 0

    if not width % divisor == 0:
        width_new = (width // divisor + 1) * divisor
        width_pad = width_new - width
        rgb = torch.nn.functional.pad(rgb, (0, width_pad, 0, 0))
        dep = torch.nn.functional.pad(dep, (0, width_pad, 0, 0))
    else:
        width_new = width
        width_pad = 0

    sample["rgb"] = rgb
    sample["dep"] = dep

    with torch.no_grad():
        output = model(sample)

    pred = output["pred"][..., : height_new - height_pad, : width_new - width_pad]
    sample["rgb"] = rgb_raw
    sample["dep"] = dep_raw
    return pred.squeeze().detach().cpu().numpy().astype(np.float32)


def normalize_prediction(pred_depth: Any, target_shape: tuple[int, int]) -> np.ndarray:
    pred = np.asarray(pred_depth, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Expected HxW prediction, got shape {pred.shape}")
    if pred.shape != target_shape:
        pred = cv2.resize(pred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = pred.astype(np.float32, copy=False)
    invalid = ~np.isfinite(pred) | (pred <= 0.0)
    pred[invalid] = np.nan
    return pred


@torch.inference_mode()
def run_manifest_inference(
    manifest_path: str | Path,
    output_dir: str | Path,
    model: torch.nn.Module,
    *,
    ibims_root: str | Path,
    batch_size: int = 1,
    depth_scale: float | None = None,
    max_depth: float | None = None,
    max_samples: int = 0,
    intrinsics_path: str | Path | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")
    if max_samples < 0:
        raise ValueError("max_samples must be 0 or a positive integer")
    if batch_size != 1:
        print("Warning: OMNI-DC adapter runs one image at a time; --batch-size is accepted for compatibility only.")

    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise SystemExit("scipy is required to write official iBims .mat predictions.") from exc

    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    ibims_root = Path(ibims_root).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = load_manifest(manifest_path)
    if max_samples > 0:
        rows = rows[:max_samples]
    difficulty = infer_difficulty(manifest_path, rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    progress = tqdm(total=len(rows), desc=f"iBims {difficulty} inference")
    try:
        for row in rows:
            sample_id = str(row["sample_id"])
            rgb_path = resolve_path(manifest_path.parent, row["rgb"])
            raw_depth_path = resolve_path(manifest_path.parent, row["raw_depth"])

            rgb_tensor, _ = load_rgb_tensor(rgb_path)
            raw_depth = load_depth_meters(
                raw_depth_path,
                row_depth_scale(row, depth_scale),
                row_max_depth(row, max_depth),
            )
            intrinsics_np = load_sample_intrinsics(row, ibims_root, intrinsics_path)
            intrinsics = torch.from_numpy(intrinsics_np).reshape(1, 3, 3).cuda()
            dep_tensor = torch.from_numpy(raw_depth).unsqueeze(0).unsqueeze(0).cuda()

            pred_depth = infer_one(model, rgb_tensor, dep_tensor, intrinsics)
            pred_depth = normalize_prediction(pred_depth, raw_depth.shape)
            if pred_depth.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"{sample_id}: expected prediction shape {EXPECTED_SHAPE}, got {pred_depth.shape}"
                )

            savemat(
                output_dir / f"{sample_id}_results.mat",
                {"pred_depths": pred_depth.astype(np.float32, copy=False)},
            )
            written += 1
            progress.update(1)
    finally:
        progress.close()

    stats = {
        "difficulty": difficulty,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "num_predictions": written,
    }
    metadata = dict(run_metadata or {})
    metadata.update(stats)
    metadata.setdefault(
        "intrinsics_source",
        str(intrinsics_path) if intrinsics_path else "ibims1_core_raw/calib/<sample_id>.txt",
    )
    with open(output_dir / "infer_args.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, sort_keys=True, default=str)

    return stats


def main() -> None:
    args = resolve_inference_args(parse_args())
    manifest_path = Path(args.manifest)
    rows = load_manifest(manifest_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(manifest_path, rows, Path(args.model_path))
    )
    model = load_model(args)

    stats = run_manifest_inference(
        manifest_path,
        output_dir,
        model,
        ibims_root=args.ibims_root,
        batch_size=args.batch_size,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        intrinsics_path=args.intrinsics_path,
        run_metadata={
            **vars(args),
            "model_path": str(args.model_path),
            "resolved_model_module": "model.ognidc",
            "resolved_model_class": "OGNIDC",
            "output_kind": "metric_depth_meter",
            "alignment": "none",
        },
    )
    print(f"Wrote {stats['num_predictions']} official iBims predictions to: {output_dir}")


if __name__ == "__main__":
    main()
