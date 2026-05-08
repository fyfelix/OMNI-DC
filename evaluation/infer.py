#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

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

from dataset import load_test_dataset, sample_name_for_sample


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
        description=(
            "OMNI-DC OGNIDC v1.1 inference for "
            "HAMMER/ClearPose/DREDS/TRansPose evaluation"
        ),
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
        help="HAMMER, ClearPose, DREDS, or TRansPose JSONL dataset path",
    )
    parser.add_argument(
        "--output",
        default="evaluation/output",
        help="Run metadata output directory. Prediction/visualization subdirectories are created here when omitted",
    )
    parser.add_argument(
        "--prediction-dir",
        default=None,
        help="Directory for prediction .npy files. Defaults to <output>/predictions",
    )
    parser.add_argument(
        "--raw-type",
        default="d435",
        choices=("d435", "l515", "tof"),
        help=(
            "Raw depth source. ClearPose only supports d435; "
            "TRansPose only supports l515"
        ),
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
        help="Path to camera intrinsics. Supports 3x3 matrix, fx fy cx cy, or key=value text",
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
        help="Visualization directory. Defaults to <output>/visualizations",
    )
    parser.add_argument(
        "--pc-rot-x-deg",
        type=float,
        default=25.0,
        help="Point cloud view rotation around X axis in degrees",
    )
    parser.add_argument(
        "--pc-rot-y-deg",
        type=float,
        default=15.0,
        help="Point cloud view rotation around Y axis in degrees",
    )
    parser.add_argument(
        "--pc-knn-k",
        type=int,
        default=16,
        help="KNN neighbors for predicted point cloud floater filtering",
    )
    parser.add_argument(
        "--pc-knn-std-ratio",
        type=float,
        default=2.0,
        help="Mean-distance std ratio threshold for predicted point cloud filtering",
    )
    parser.add_argument(
        "--disable-pc-knn-filter",
        action="store_true",
        help="Disable KNN filtering for predicted point cloud visualization",
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
    if args.save_vis and args.pc_knn_k < 1:
        parser.error("--pc-knn-k must be greater than 0")
    if args.save_vis and args.pc_knn_std_ratio < 0:
        parser.error("--pc-knn-std-ratio must be non-negative")
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
        args.prediction_dir = str(Path(args.output) / "predictions")
    else:
        args.prediction_dir = str(resolve_project_path(args.prediction_dir))
    if args.visualization_dir is None:
        args.visualization_dir = str(Path(args.output) / "visualizations")
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
        print("  - data/HAMMER/intrinsics.txt or data/TRansPose/sequences/intrinsics.txt")
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
        raise ValueError(f"Could not load GT depth from {depth_path}")
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


def image_grid(imgs, rows, cols):
    if not imgs:
        return None
    if len(imgs) != rows * cols:
        raise ValueError(f"Expected {rows * cols} images, got {len(imgs)}")

    height, width = imgs[0].shape[:2]
    grid = Image.new("RGB", size=(cols * width, rows * height))
    for idx, img in enumerate(imgs):
        col_idx = idx % cols
        row_idx = idx // cols
        panel = np.asarray(img).astype(np.uint8)
        if panel.ndim == 2:
            panel = np.repeat(panel[:, :, None], 3, axis=2)
        if panel.shape[:2] != (height, width):
            panel = np.asarray(
                Image.fromarray(panel).resize((width, height), resample=Image.BILINEAR)
            )
        grid.paste(Image.fromarray(panel), box=(col_idx * width, row_idx * height))
    return np.asarray(grid)


def scale_intrinsics(intrinsics, orig_hw, new_hw):
    sy = new_hw[0] / orig_hw[0]
    sx = new_hw[1] / orig_hw[1]
    scaled = intrinsics.copy()
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    return scaled


def resize_to(image, target_hw):
    target_h, target_w = target_hw
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def filter_pointcloud_knn(points, colors, k=16, std_ratio=2.0):
    if k < 1 or points.shape[0] <= k:
        return points, colors

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return points, colors

    neighbor_count = min(k + 1, points.shape[0])
    try:
        tree = cKDTree(points)
        try:
            distances, _ = tree.query(points, k=neighbor_count, workers=-1)
        except TypeError:
            distances, _ = tree.query(points, k=neighbor_count)
    except Exception:
        return points, colors

    if distances.ndim == 1:
        return points, colors

    mean_distances = distances[:, 1:].mean(axis=1)
    finite = np.isfinite(mean_distances)
    if not finite.any():
        return points, colors

    valid_mean_distances = mean_distances[finite]
    threshold = valid_mean_distances.mean() + std_ratio * valid_mean_distances.std()
    keep = finite & (mean_distances <= threshold)
    if not keep.any():
        return points, colors
    return points[keep], colors[keep]


def render_pointcloud_reproject(
    depth_map,
    intrinsics,
    rgb_img,
    rot_x_deg=25.0,
    rot_y_deg=15.0,
    bg_color=(255, 255, 255),
    knn_filter=True,
    knn_k=16,
    knn_std_ratio=2.0,
):
    depth_map = np.asarray(depth_map, dtype=np.float32).squeeze()
    height, width = depth_map.shape
    rgb_img = resize_to(np.asarray(rgb_img), (height, width))
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid = (depth_map > 1e-8) & np.isfinite(depth_map)
    if not valid.any():
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    z = depth_map[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)
    colors = rgb_img[valid]
    if knn_filter:
        points, colors = filter_pointcloud_knn(
            points,
            colors,
            k=knn_k,
            std_ratio=knn_std_ratio,
        )

    center = points.mean(axis=0)
    points_centered = points - center

    rx = np.radians(rot_x_deg)
    ry = np.radians(rot_y_deg)
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)

    x1 = points_centered[:, 0]
    y1 = points_centered[:, 1] * cos_x - points_centered[:, 2] * sin_x
    z1 = points_centered[:, 1] * sin_x + points_centered[:, 2] * cos_x
    x2 = x1 * cos_y + z1 * sin_y
    y2 = y1
    z2 = -x1 * sin_y + z1 * cos_y
    points_rot = np.stack([x2, y2, z2], axis=-1) + center
    z_new = points_rot[:, 2]
    keep = z_new > 1e-4
    if not keep.any():
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    u_proj = points_rot[keep, 0] * fx / z_new[keep] + cx
    v_proj = points_rot[keep, 1] * fy / z_new[keep] + cy
    z_buf = z_new[keep]
    c_buf = colors[keep]

    pad = int(max(height, width) * 0.3)
    canvas_h, canvas_w = height + 2 * pad, width + 2 * pad
    ui = np.round(u_proj + pad).astype(np.int32)
    vi = np.round(v_proj + pad).astype(np.int32)

    in_bounds = (ui >= 0) & (ui < canvas_w) & (vi >= 0) & (vi < canvas_h)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z_buf = z_buf[in_bounds]
    c_buf = c_buf[in_bounds]
    if ui.size == 0:
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    order = np.argsort(-z_buf)
    ui = ui[order]
    vi = vi[order]
    c_buf = c_buf[order]

    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    canvas[vi, ui] = c_buf

    filled = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    filled[vi, ui] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    filled_dilated = cv2.dilate(filled, kernel, iterations=1)
    holes = (filled_dilated > 0) & (filled == 0)
    if holes.any():
        for channel_idx in range(3):
            blurred = cv2.blur(canvas[:, :, channel_idx].astype(np.float32), (3, 3))
            canvas[:, :, channel_idx][holes] = blurred[holes].astype(np.uint8)

    rows = np.any(filled_dilated > 0, axis=1)
    cols = np.any(filled_dilated > 0, axis=0)
    if rows.any() and cols.any():
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        margin = 10
        row_min = max(0, row_min - margin)
        row_max = min(canvas_h - 1, row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(canvas_w - 1, col_max + margin)
        canvas = canvas[row_min : row_max + 1, col_min : col_max + 1]

    return resize_to(canvas, (height, width)).astype(np.uint8)


def create_visualizationv2(
    rgb,
    raw_depth,
    pred_depth,
    gt_depth,
    image_min,
    image_max,
    intrinsics,
    pc_rot_x_deg=25.0,
    pc_rot_y_deg=15.0,
    pc_knn_k=16,
    pc_knn_std_ratio=2.0,
    disable_pc_knn_filter=False,
):
    rgb_display = np.asarray(rgb)
    if rgb_display.dtype != np.uint8:
        if np.nanmax(rgb_display) <= 1.0:
            rgb_display = rgb_display * 255.0
        rgb_display = np.clip(rgb_display, 0, 255).astype(np.uint8)

    target_hw = rgb_display.shape[:2]
    raw_depth = resize_to(np.asarray(raw_depth, dtype=np.float32), target_hw)
    pred_depth = resize_to(np.asarray(pred_depth, dtype=np.float32), target_hw)
    gt_depth = resize_to(np.asarray(gt_depth, dtype=np.float32), target_hw)

    pred_pointcloud = render_pointcloud_reproject(
        pred_depth,
        intrinsics,
        rgb_display,
        rot_x_deg=pc_rot_x_deg,
        rot_y_deg=pc_rot_y_deg,
        knn_filter=not disable_pc_knn_filter,
        knn_k=pc_knn_k,
        knn_std_ratio=pc_knn_std_ratio,
    )
    gt_pointcloud = render_pointcloud_reproject(
        gt_depth,
        intrinsics,
        rgb_display,
        rot_x_deg=pc_rot_x_deg,
        rot_y_deg=pc_rot_y_deg,
        knn_filter=False,
    )

    return image_grid(
        [
            rgb_display,
            colorize_depth(raw_depth, image_min, image_max),
            colorize_depth(pred_depth, image_min, image_max),
            colorize_depth(gt_depth, image_min, image_max),
            pred_pointcloud,
            gt_pointcloud,
        ],
        3,
        2,
    )


def save_visualization(
    rgb,
    raw_depth,
    pred_depth,
    gt_depth,
    output_path,
    image_min,
    image_max,
    intrinsics,
    pc_rot_x_deg=25.0,
    pc_rot_y_deg=15.0,
    pc_knn_k=16,
    pc_knn_std_ratio=2.0,
    disable_pc_knn_filter=False,
):
    scaled_intrinsics = scale_intrinsics(
        intrinsics,
        rgb.shape[:2],
        rgb.shape[:2],
    )
    grid = create_visualizationv2(
        rgb,
        raw_depth,
        pred_depth,
        gt_depth,
        image_min,
        image_max,
        scaled_intrinsics,
        pc_rot_x_deg=pc_rot_x_deg,
        pc_rot_y_deg=pc_rot_y_deg,
        pc_knn_k=pc_knn_k,
        pc_knn_std_ratio=pc_knn_std_ratio,
        disable_pc_knn_filter=disable_pc_knn_filter,
    )
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

    dataset, dataset_kind = load_test_dataset(args.dataset, args.raw_type)
    args.dataset_kind = dataset_kind
    if hasattr(dataset, "depth_scale"):
        args.depth_scale = dataset.depth_scale

    total = len(dataset) if args.max_samples == 0 else min(len(dataset), args.max_samples)
    model = load_model(args)
    intrinsics_np = load_intrinsics(args.intrinsics_path)
    intrinsics = torch.from_numpy(intrinsics_np).reshape(1, 3, 3).cuda()

    with open(Path(args.output) / "args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    for idx in tqdm(range(total), desc="OMNI-DC inference"):
        sample = dataset[idx]
        rgb_path, raw_depth_path, gt_depth_path = sample[:3]
        name = sample_name_for_sample(dataset_kind, sample)

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
                intrinsics_np,
                pc_rot_x_deg=args.pc_rot_x_deg,
                pc_rot_y_deg=args.pc_rot_y_deg,
                pc_knn_k=args.pc_knn_k,
                pc_knn_std_ratio=args.pc_knn_std_ratio,
                disable_pc_knn_filter=args.disable_pc_knn_filter,
            )


if __name__ == "__main__":
    inference(parse_arguments())
