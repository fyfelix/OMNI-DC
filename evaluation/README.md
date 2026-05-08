# OMNI-DC HAMMER / ClearPose / DREDS / TRansPose Evaluation

这个目录提供 OMNI-DC `OGNIDC` v1.1 的轻量评估入口。目录结构与 `eval_pipeline_cdm` 对齐，但模型加载和推理链路保留本项目实现，不使用 CDM 的 `RGBDDepth`、`resize_method`、`is_disp` 或 `model_registry.yaml` 分支。

```text
evaluation/
  run_hammer.sh
  run_clearpose.sh
  run_dreds.sh
  run_transpose.sh
  infer.py
  eval.py
  dataset.py
  requirements.txt
  utils/
```

推理和评估边界：

1. `infer.py` 读取 dataset JSONL 和 OMNI-DC checkpoint，逐样本写出 `predictions/*.npy`。
2. `eval.py` 从输出目录读取 `predictions/*.npy`，计算指标并写出 CSV/JSON。
3. 四个 `run_*.sh` wrapper 只负责选择数据集、组织输出目录，并可选删除逐样本 `.npy`。

## 准备文件

默认所有权重放在项目根目录 `ckpts/`：

```text
ckpts/
  modelv1.1_best_72epochs.pt
  resnet34.pth
  pvt.pth
  depth_anything_v2_vitl.pth
```

`modelv1.1_best_72epochs.pt` 是 OMNI-DC 主 checkpoint；其余三个权重用于 OMNI-DC backbone 和 Depth Anything V2 Large。`infer.py` 会尝试通过软链接让原始代码找到这些依赖权重，不会覆盖已有目标文件。

相机内参默认读取：

```text
data/HAMMER/intrinsics.txt
```

支持 `fx fy cx cy`、`fx=...` 风格 key-value，或 3x3 矩阵。HAMMER、ClearPose、DREDS、TRansPose 四条路线都会把该内参作为 `sample["K"]` 传给 OGNIDC；如使用其他数据集路径，请通过 `INTRINSICS_PATH` 覆盖。TRansPose wrapper 的默认内参路径是 `data/TRansPose/sequences/intrinsics.txt`。

## 数据集格式

HAMMER 是逐样本 JSONL，每行包含：

```text
rgb
d435_depth
l515_depth
tof_depth
depth
depth-range
```

`raw_type` 决定读取 `d435_depth`、`l515_depth` 或 `tof_depth`。raw depth 和 GT depth 按 `uint_depth / 1000.0` 转成 meter。

ClearPose 是序列展开 JSONL，每行包含：

```text
rgb
rgb-suffix
raw_depth-suffix
depth-suffix
depth-range
```

`dataset.py` 会在 `rgb` 目录下按 suffix glob 展开逐帧 RGB、raw depth 和 GT depth。ClearPose 固定 `raw_type=d435`，`depth_scale=1000.0`。

DREDS 使用 `test_std_catknown` / `test_std_catnovel` 风格 JSONL，每行包含：

```text
seq_name
rgb
rgb-suffix
raw_depth-suffix
depth-suffix
depth-range
video
```

DREDS raw / GT depth 是 EXR float，单位已经是 meter，因此 `depth_scale=1.0`。脚本会设置 `OPENCV_IO_ENABLE_OPENEXR=1` 以启用 OpenCV EXR 读取。

TRansPose L515 是逐样本 JSONL，每行包含：

```text
rgb
l515_depth
depth
seq_name  # optional
depth-range  # optional
```

TRansPose 固定 `raw_type=l515`，`l515_depth` 和 `depth` 均按毫米 PNG 读取并除以 `1000.0`。如果 JSONL 行提供 `seq_name`，推理会保存 `predictions/<seq_name>.npy`，评估也用同一个 `<seq_name>.npy` 查找预测；没有 `seq_name` 时按 RGB 路径兜底生成样本名。默认有效深度范围为 `0.1-6.0m`，可由 JSONL 的 `depth-range` 覆盖。

## 运行路线

### HAMMER

```bash
DATASET_PATH=data/HAMMER/test_filled_d435.jsonl \
INTRINSICS_PATH=data/HAMMER/intrinsics.txt \
bash evaluation/run_hammer.sh ckpts/modelv1.1_best_72epochs.pt d435
```

参数：

```text
bash evaluation/run_hammer.sh [checkpoint] [camera_type=d435]
```

`camera_type` 支持 `d435`、`l515`、`tof`，也可以通过 `RAW_TYPE` 设置。

### ClearPose

```bash
DATASET_PATH=data/clearpose/test.jsonl \
INTRINSICS_PATH=data/HAMMER/intrinsics.txt \
bash evaluation/run_clearpose.sh ckpts/modelv1.1_best_72epochs.pt
```

参数：

```text
bash evaluation/run_clearpose.sh [checkpoint]
```

ClearPose 固定使用 `raw_type=d435`。

### DREDS

```bash
DREDS_KNOWN_JSONL=data/DREDS/test_std_catknown.jsonl \
DREDS_NOVEL_JSONL=data/DREDS/test_std_catnovel.jsonl \
OUTPUT_ROOT=evaluation/output_dreds \
INTRINSICS_PATH=data/HAMMER/intrinsics.txt \
bash evaluation/run_dreds.sh ckpts/modelv1.1_best_72epochs.pt all
```

参数：

```text
bash evaluation/run_dreds.sh [checkpoint] [variant=all]
```

`variant` 支持 `catknown`、`catnovel`、`all`。`all` 会顺序运行 known 和 novel，此时请使用 `OUTPUT_ROOT`；如果同时设置 `OUTPUT_DIR`，脚本会退出报错。

### TRansPose

```bash
DATASET_PATH=data/TRansPose/sequences/dc_testset.jsonl \
INTRINSICS_PATH=data/TRansPose/sequences/intrinsics.txt \
bash evaluation/run_transpose.sh ckpts/modelv1.1_best_72epochs.pt l515
```

参数：

```text
bash evaluation/run_transpose.sh [checkpoint] [camera_type=l515]
```

TRansPose 固定使用 `raw_type=l515`。默认 JSONL 为 `data/TRansPose/sequences/dc_testset.jsonl`，默认内参为 `data/TRansPose/sequences/intrinsics.txt`，默认输出目录为 `<checkpoint_dir>/transpose_<checkpoint_stub>_data_l515/`。`SAVE_VIS=true` 时会保存 3x2 可视化网格。

## 常用环境变量

```text
CHECKPOINT          主 checkpoint，默认 ckpts/modelv1.1_best_72epochs.pt
CKPT_DIR            依赖权重目录，默认 ckpts
DATASET_PATH        HAMMER / ClearPose / TRansPose JSONL 路径
DREDS_KNOWN_JSONL   DREDS catknown JSONL 路径
DREDS_NOVEL_JSONL   DREDS catnovel JSONL 路径
INTRINSICS_PATH     相机内参路径，TRansPose wrapper 默认 data/TRansPose/sequences/intrinsics.txt
OUTPUT_DIR          单数据集输出目录
OUTPUT_ROOT         DREDS all 模式的输出根目录
BATCH_SIZE          兼容参数；OMNI-DC 当前逐张推理，默认 1
NUM_WORKERS         兼容参数，默认 0
SAVE_VIS            是否保存可视化，默认 true
CLEANUP_NPY         是否在评估后删除 predictions/*.npy，默认 false
MAX_SAMPLES         样本上限，0 表示全量，默认 0
PYTHON_BIN          Python 可执行文件，默认 python
```

## 输出目录

如果未显式设置 `OUTPUT_DIR` / `OUTPUT_ROOT`，默认写到 checkpoint 同级目录：

```text
<checkpoint_dir>/hammer_<checkpoint_stub>_data_<camera_type>/
<checkpoint_dir>/clearpose_<checkpoint_stub>_data_d435/
<checkpoint_dir>/dreds_catknown_<checkpoint_stub>/
<checkpoint_dir>/dreds_catnovel_<checkpoint_stub>/
<checkpoint_dir>/transpose_<checkpoint_stub>_data_l515/
```

输出内容：

```text
args.json
eval_args.json
predictions/*.npy
visualizations/*_promptda_vis.jpg
all_metrics_<timestamp>_False.csv
mean_metrics_<timestamp>_False.json
```

`*.npy` 是 `HxW float32` metric depth，单位 meter。`eval.py` 默认优先读取 `predictions/*.npy`，如果不存在，会 fallback 到旧版根目录 `*.npy`。

## 关键约定

- 模型固定为 OMNI-DC `OGNIDC` v1.1；`--encoder` 和 `--input-size` 仅为兼容参数，DAV2 实际固定使用 `vitl` 和 518。
- RGB 预处理使用 `ToTensor + ImageNet Normalize`；raw depth 使用 `cv2.IMREAD_UNCHANGED` 读取并按 dataset `depth_scale` 转 meter。
- 推理时按 OGNIDC 原逻辑 pad 到 `4 * 2 ** (num_resolution - 1)` 的倍数，输出后 crop 回原图大小。
- 模型输出 `output["pred"]` 已按 meter 表示，不做 disparity/inverse depth 转换，也不新增 alignment 评估模式。
- DREDS 评估允许 prediction shape 与 GT shape 不一致，并用 nearest resize 对齐；HAMMER / ClearPose 遇到 shape mismatch 会直接报错。
- TRansPose 推理和评估都通过 `seq_name` 查找同一个 `<seq_name>.npy`，避免按 RGB 路径推导名称导致不一致。
- `--save-vis` 会生成 RGB、raw depth、prediction、GT depth、prediction point cloud、GT point cloud 的 3x2 网格。prediction 点云默认启用 KNN floater 过滤，依赖 `scipy`；GT 点云不启用 KNN 过滤。
- OMNI-DC 推理始终需要有效内参文件，因为模型输入会使用 `sample["K"]`；`SAVE_VIS=true` 只额外启用点云可视化参数。
- 端到端推理需要 CUDA、checkpoint、依赖权重、数据集和内参文件齐备；Mac 本地通常只适合做 help、导入和语法检查。
