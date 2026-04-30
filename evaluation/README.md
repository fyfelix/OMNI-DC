# OMNI-DC HAMMER Evaluation

这个目录是在 OMNI-DC 外部项目根目录中适配的 HAMMER 评估入口。评估代码复用 `run_bs_eval_pipeline` 的 `dataset.py`、`eval.py` 和 `utils/metric.py`，模型推理侧固定适配 OMNI-DC README 推荐的 `OGNIDC` v1.1 RGB-D depth completion 配置。

## 需要准备的文件

默认所有权重放在项目根目录 `ckpts/`：

```text
ckpts/
  modelv1.1_best_72epochs.pt
  resnet34.pth
  pvt.pth
  depth_anything_v2_vitl.pth
```

含义：

- `modelv1.1_best_72epochs.pt`：OMNI-DC v1.1 主 checkpoint。
- `resnet34.pth`：OMNI-DC PVT/backbone 初始化所需 ResNet34 权重。
- `pvt.pth`：OMNI-DC PVT 初始化权重。
- `depth_anything_v2_vitl.pth`：`load_dav2=1` 时需要的 Depth Anything V2 Large 权重。

HAMMER 数据默认放在：

```text
data/HAMMER/test.jsonl
```

JSONL 中引用的 `rgb`、`d435_depth`、`l515_depth`、`tof_depth`、`depth` 文件需要能相对 `data/HAMMER/` 访问。raw depth 和 GT depth 默认按 `uint_depth / 1000.0` 转成 meter。

## 运行

默认运行：

```bash
./evaluation/run_eval.sh
```

也可以显式传 checkpoint：

```bash
./evaluation/run_eval.sh /path/to/modelv1.1_best_72epochs.pt
```

常用环境变量：

```text
CHECKPOINT       主 checkpoint，默认 ckpts/modelv1.1_best_72epochs.pt
CKPT_DIR         依赖权重目录，默认 ckpts
DATASET_PATH     HAMMER JSONL，默认 data/HAMMER/test.jsonl
OUTPUT_DIR       输出目录，默认 evaluation/output
RAW_TYPE         d435/l515/tof，默认 d435
BATCH_SIZE       保留兼容参数；当前适配逐张推理，默认 1
NUM_WORKERS      保留兼容参数，默认 0
SAVE_VIS         是否保存可视化，默认 true
CLEANUP_NPY      是否在评估后删除 .npy，默认 false
MAX_SAMPLES      可选 smoke-test 样本数
PYTHON_BIN       Python 可执行文件，默认 python
```

示例：

```bash
RAW_TYPE=l515 \
DATASET_PATH=data/HAMMER/test.jsonl \
OUTPUT_DIR=evaluation/output_l515 \
./evaluation/run_eval.sh
```

## 输出

默认输出到 `evaluation/output`：

```text
args.json
eval_args.json
*.npy
visualizations/*_grid.jpg
all_metrics_*.csv
mean_metrics_*.json
```

其中 `*.npy` 是 `HxW float32` metric depth，单位 meter，供 `evaluation/eval.py` 直接读取。默认会保留预测 `.npy`，并保存可视化 grid；如需关闭可视化：

```bash
SAVE_VIS=false ./evaluation/run_eval.sh
```

## 当前适配细节

- 模型固定为 OMNI-DC `OGNIDC` v1.1。
- 输入为 RGB-D depth completion；RGB 使用官方 demo 的 ImageNet normalize，raw depth 使用 HAMMER 的 `d435/l515/tof` 字段。
- 默认参数包括 `load_dav2=1`、`num_resolution=3`、`backbone_mode=rgbd`、`depth_activation_format=exp`、`whiten_sparse_depths=1`。
- 推理时按官方 demo/test 逻辑 pad 到 `4 * 2 ** (num_resolution - 1)` 的倍数，输出后 crop 回原图大小。
- 模型输出 `output["pred"]` 已按 meter 表示；本适配不做 disparity/inverse depth 转换，也不做 alignment。
- 脚本只使用 CUDA。没有 CUDA 时会直接退出；MacBook 仅适合做参数、导入和文件检查。
- `infer.py` 会优先用软链接让 OMNI-DC 原代码按硬编码路径找到 `ckpts/` 中的同名权重；如果目标路径已存在，不会覆盖。
