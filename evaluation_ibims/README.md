# OMNI-DC iBims 官方评估

`evaluation_ibims/` 是当前项目的 iBims 官方评估适配目录。它消费已有
synthetic raw depth manifest，使用 OMNI-DC `OGNIDC` v1.1 做 RGB-D depth
completion 推理，并把结果保存为 iBims 官方 evaluator 需要的 MAT 格式。

本目录不生成 synthetic raw depth；需要先准备好：

```text
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_easy.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_medium.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_hard.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_extreme.jsonl
```

完整官方评估还需要 iBims 数据集自带文件：

```text
data/ibims1/imagelist.txt
data/ibims1/ibims1_core_mat/
data/ibims1/ibims1_core_raw/calib/
data/ibims1/evaluation_scripts/evaluate_ibims.py
```

## 权重和依赖

默认权重放在项目根目录 `ckpts/`：

```text
ckpts/modelv1.1_best_72epochs.pt
ckpts/resnet34.pth
ckpts/pvt.pth
ckpts/depth_anything_v2_vitl.pth
```

`run_all.sh` 会优先使用项目 `.venv/bin/python`。服务器 conda 环境可用
`PYTHON_BIN` 覆盖：

```bash
PYTHON_BIN=python ./evaluation_ibims/run_all.sh
```

iBims MAT 读写需要 `scipy`；官方 evaluator 还需要 `scikit-image` 和
`scikit-learn`。

## 一站式运行

在项目根目录运行：

```bash
./evaluation_ibims/run_all.sh
```

等价于：

```bash
./evaluation_ibims/run_all.sh ckpts/modelv1.1_best_72epochs.pt
```

小样本 smoke：

```bash
./evaluation_ibims/run_all.sh ckpts/modelv1.1_best_72epochs.pt \
  --levels easy \
  --max-samples 1 \
  --skip-eval
```

## Python 入口

```bash
.venv/bin/python evaluation_ibims/run_all.py \
  --model-path ckpts/modelv1.1_best_72epochs.pt \
  --ckpt-dir ckpts \
  --ibims-root data/ibims1 \
  --levels easy medium hard extreme
```

常用参数：

```text
--run-dir <dir>          指定输出根目录
--max-samples <N>       每档只跑前 N 个样本；0 表示全量
--intrinsics-path <file> 使用一个全局 K 覆盖逐样本 calib
--depth-scale <scale>   覆盖 manifest 中的 depth_scale
--max-depth <meters>    覆盖 manifest depth-range 上限
--load-dav2 true/false  是否加载 Depth Anything V2 辅助深度
--skip-infer            跳过推理，使用 --run-dir 下已有 predictions
--skip-eval             跳过官方评估，只生成 MAT prediction
```

## 输出结构

默认输出目录：

```text
evaluation_ibims/output/ibims_omnidc_<checkpoint_stem>_<YYYY-mm-dd_HH-MM-SS>/
```

主要内容：

```text
predictions/<level>/<sample>_results.mat
predictions/<level>/infer_args.json
official_eval/<level>/workspace/
official_eval/<level>/official_eval_stdout.txt
eval_summary.csv
eval_summary.json
```

每个 prediction MAT 包含变量 `pred_depths`：

```text
shape: 480x640
dtype: float32
unit: meter
invalid prediction: NaN
```

## 推理处理约定

- RGB 使用 PIL 读取并按 OMNI-DC demo/evaluation 的 ImageNet 统计量 normalize。
- raw depth 使用 manifest 中的 `depth_scale`，默认 `65535 / 50`，转换为 meter。
- raw depth 中非有限值、`<= 0`、超过 `depth-range` 上限的点会置为 `0`。
- 每个样本默认读取 `data/ibims1/ibims1_core_raw/calib/<sample_id>.txt`，格式为
  `[fx fy cx cy]`，并构造 `3x3 K` 传给 `sample["K"]`。
- 模型固定为当前项目 `model.ognidc.OGNIDC` v1.1 配置。
- 推理时 pad 到模型分辨率要求的倍数，输出后 crop 回原图大小。
- 输出是 metric depth，单位 meter；不做 disparity/inverse depth 转换，也不做 alignment。
