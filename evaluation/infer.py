#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (str(EVALUATION_DIR), str(SRC_DIR), str(SRC_DIR / "model")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset import HAMMERDataset


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "t", "yes", "y"):
        return True
    if value in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def torch_load_compatible(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="OMNI-DC OGNIDC v1.1 inference for HAMMER evaluation",
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
    parser.add_argument(
        "--dataset",
        default="data/HAMMER/test_filled_d435.jsonl",
        help="HAMMER JSONL dataset path",
    )
    parser.add_argument(
        "--output",
        default="evaluation/output",
        help="Run metadata output directory. Prediction/visualization directories fall back here when omitted",
    )
    parser.add_argument(
        "--prediction-dir",
        default=None,
        help="Directory for prediction .npy files. Defaults to --output",
    )
    parser.add_argument(
        "--raw-type",
        default="d435",
        choices=("d435", "l515", "tof"),
        help="HAMMER raw depth source",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Depth scale used to convert uint depth maps to meters",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=300.0,
        help="Maximum raw input depth in meters; larger raw values are set to 0",
    )
    parser.add_argument(
        "--intrinsics-path",
        default="data/HAMMER/intrinsics.txt",
        help="Path to HAMMER camera intrinsics. Supports 3x3 matrix, fx fy cx cy, or key=value text",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Kept for wrapper compatibility; OMNI-DC inference runs one image at a time",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Kept for wrapper compatibility",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Sample limit. 0 evaluates all samples",
    )
    parser.add_argument(
        "--encoder",
        default="vitl",
        help="Kept for compatibility; OMNI-DC v1.1 uses Depth Anything V2 Large internally",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Kept for compatibility; OMNI-DC v1.1 uses 518 internally for DAV2",
    )
    parser.add_argument(
        "--image-min",
        type=float,
        default=0.1,
        help="Minimum depth for visualization colorization",
    )
    parser.add_argument(
        "--image-max",
        type=float,
        default=5.0,
        help="Maximum depth for visualization colorization",
    )
    parser.add_argument(
        "--save-vis",
        dest="save_vis",
        action="store_true",
        help="Save visualization grids",
    )
    parser.add_argument(
        "--no-save-vis",
        dest="save_vis",
        action="store_false",
        help="Disable visualization grids",
    )
    parser.add_argument(
        "--visualization-dir",
        "--vis-dir",
        dest="visualization_dir",
        default=None,
        help="Visualization directory. Defaults to --output",
    )
    parser.add_argument(
        "--load-dav2",
        type=str2bool,
        default=True,
        help="Use OMNI-DC v1.1 Depth Anything V2 auxiliary depth",
    )
    parser.set_defaults(save_vis=True)
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be 0 or a positive integer")
    return args


def default_model_args(load_dav2=True):
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


def required_weight_links(ckpt_dir, load_dav2):
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


def prepare_weight_links(ckpt_dir, load_dav2):
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


def validate_inputs(args):
    args.dataset = str(resolve_project_path(args.dataset))
    args.output = str(resolve_project_path(args.output))
    args.ckpt_dir = str(resolve_project_path(args.ckpt_dir))
    if args.prediction_dir is None:
        args.prediction_dir = args.output
    else:
        args.prediction_dir = str(resolve_project_path(args.prediction_dir))
    if args.visualization_dir is None:
        args.visualization_dir = args.output
    else:
        args.visualization_dir = str(resolve_project_path(args.visualization_dir))

    if args.model_path is None:
        args.model_path = str(Path(args.ckpt_dir) / "modelv1.1_best_72epochs.pt")
    else:
        args.model_path = str(resolve_project_path(args.model_path))
    args.intrinsics_path = str(resolve_project_path(args.intrinsics_path))

    missing = []
    if not Path(args.dataset).exists():
        missing.append(args.dataset)
    if not Path(args.model_path).exists():
        missing.append(args.model_path)
    if not Path(args.intrinsics_path).exists():
        missing.append(args.intrinsics_path)

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
        print("  - data/HAMMER/intrinsics.txt")
        raise SystemExit(1)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this OMNI-DC evaluation adapter.")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.prediction_dir).mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        Path(args.visualization_dir).mkdir(parents=True, exist_ok=True)

    return args


def load_model(args):
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


def sample_id_from_rgb_path(rgb_path):
    parts = str(rgb_path).split("/")
    scene_name = parts[-4]
    return scene_name + "#" + Path(rgb_path).stem


def load_rgb_tensor(rgb_path):
    image = Image.open(rgb_path).convert("RGB")
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transform(image).unsqueeze(0).cuda(), np.asarray(image)


def load_depth_meters(depth_path, depth_scale, max_depth):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not read depth: {depth_path}")
    depth = np.asarray(depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth / float(depth_scale)
    invalid = ~np.isfinite(depth) | (depth <= 0.0) | (depth > max_depth)
    depth[invalid] = 0.0
    return depth.astype(np.float32)


def load_intrinsics(intrinsics_path):
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


def load_gt_depth_for_vis(depth_path, depth_scale):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not read GT depth: {depth_path}")
    depth = np.asarray(depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / float(depth_scale)


def colorize_depth(depth, image_min, image_max):
    import matplotlib.pyplot as plt

    depth = np.asarray(depth).squeeze()
    valid = np.isfinite(depth) & (depth > 0.0)
    denom = max(image_max - image_min, 1e-6)
    normalized = np.clip((depth - image_min) / denom, 0.0, 1.0)
    colored = plt.get_cmap("Spectral")(normalized, bytes=True)[..., :3]
    colored[~valid] = 0
    return colored.astype(np.uint8)


def save_visualization(rgb, raw_depth, pred_depth, gt_depth, output_path, image_min, image_max):
    panels = [
        rgb.astype(np.uint8),
        colorize_depth(raw_depth, image_min, image_max),
        colorize_depth(pred_depth, image_min, image_max),
        colorize_depth(gt_depth, image_min, image_max),
    ]
    height, width = panels[0].shape[:2]
    resized = [
        np.asarray(Image.fromarray(panel).resize((width, height), resample=Image.BILINEAR))
        for panel in panels
    ]
    top = np.concatenate(resized[:2], axis=1)
    bottom = np.concatenate(resized[2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    Image.fromarray(grid).save(output_path)


def infer_one(model, rgb_tensor, dep_tensor, intrinsics):
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


def inference(args):
    args = validate_inputs(args)

    if args.batch_size != 1:
        print("Warning: OMNI-DC adapter runs one image at a time; --batch-size is accepted for compatibility only.")

    if "hammer" not in args.dataset.lower():
        raise ValueError(f"Invalid dataset: {args.dataset}")

    dataset = HAMMERDataset(args.dataset, args.raw_type)
    total = len(dataset) if args.max_samples == 0 else min(len(dataset), args.max_samples)
    model = load_model(args)
    intrinsics_np = load_intrinsics(args.intrinsics_path)
    intrinsics = torch.from_numpy(intrinsics_np).reshape(1, 3, 3).cuda()

    with open(Path(args.output) / "args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    for idx in tqdm(range(total), desc="OMNI-DC inference"):
        rgb_path, raw_depth_path, gt_depth_path = dataset[idx]
        name = sample_id_from_rgb_path(rgb_path)

        rgb_tensor, rgb_np = load_rgb_tensor(rgb_path)
        raw_depth = load_depth_meters(raw_depth_path, args.depth_scale, args.max_depth)
        dep_tensor = torch.from_numpy(raw_depth).unsqueeze(0).unsqueeze(0).cuda()

        pred = infer_one(model, rgb_tensor, dep_tensor, intrinsics)
        np.save(Path(args.prediction_dir) / f"{name}.npy", pred)

        if args.save_vis:
            gt_depth = load_gt_depth_for_vis(gt_depth_path, args.depth_scale)
            save_visualization(
                rgb_np,
                raw_depth,
                pred,
                gt_depth,
                Path(args.visualization_dir) / f"{name}_promptda_vis.jpg",
                args.image_min,
                args.image_max,
            )


if __name__ == "__main__":
    inference(parse_arguments())
