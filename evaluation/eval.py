import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from utils.metric import abs_relative_difference, rmse_linear, delta1_acc, mae_linear, delta4_acc_105, delta5_acc110
from os.path import exists
from dataset import load_dataset_for_eval, resolve_sample_name

from datetime import datetime
from os.path import join

import json
from os.path import exists


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="RGBD Depth Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="Model encoder type",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="HAMMER or ClearPose JSONL dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_dir",
        help="Directory for metrics and eval_args.json",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="Directory containing prediction .npy files. Defaults to --output",
    )
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Raw type. ClearPose only supports d435",
    )
    parser.add_argument(
        "--input-size", type=int, default=518, help="Input size for inference"
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for depth values",
    )
    parser.add_argument(
        "--max-depth", type=float, default=6.0, help="Maximum valid depth value"
    )
    parser.add_argument(
        "--image-min", type=float, default=0.1, help="Minimum valid depth value"
    )
    parser.add_argument(
        "--image-max", type=float, default=5.0, help="Maximum valid depth value"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Sample limit. 0 evaluates all samples",
    )
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be 0 or a positive integer")
    return args


def load_gt_depth(depth_path, depth_scale, max_depth,min_depth):
    depth_GT = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_GT = np.asarray(depth_GT).astype(np.float32) / depth_scale
    valid_mask = (depth_GT >= min_depth) & (depth_GT <= max_depth)
    depth_GT[~valid_mask] = min_depth
    return depth_GT, valid_mask


class EvalDataset(Dataset):
    def __init__(self, dataset, prediction_path, args, depth_scale, align=False):
        self.dataset = dataset
        self.prediction_path = prediction_path
        self.args = args
        self.depth_scale = depth_scale
        self.align = align

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        depth_GT, valid_mask = load_gt_depth(sample[2], self.depth_scale, self.args.max_depth, self.args.min_depth)

        name = resolve_sample_name(sample[0], self.args.dataset)

        pred = np.load(join(self.prediction_path, name+'.npy'))
        
        pred_invalid_mask = np.logical_or(np.isnan(pred), np.isinf(pred))
        if pred_invalid_mask.sum() > 0:
            # print(f"Invalid mask: {name} {pred_invalid_mask.sum()}")
            valid_mask = valid_mask & ~pred_invalid_mask

        if self.align:
            depth_GT_reshaped = depth_GT[valid_mask].reshape((-1, 1))
            pred_reshaped = pred[valid_mask].reshape((-1, 1))

            _ones = np.ones_like(pred_reshaped)
            A = np.concatenate([pred_reshaped, _ones], axis=-1)
            X = np.linalg.lstsq(A, depth_GT_reshaped, rcond=None)[0]
            scale, shift = X 
            pred_reshaped = scale * pred_reshaped + shift
            pred_reshaped = np.clip(pred_reshaped, a_min=self.args.min_depth, a_max=None) 
            
            # For ALIGN=True, shapes are variable (N_valid, 1), cannot simple stack in default collate
            # We return them as is, but batch_size should be 1 or custom collate used
            return {
                'name': name,
                'pred': pred_reshaped.astype(np.float32),
                'gt': depth_GT_reshaped.astype(np.float32),
                'mask': np.ones_like(pred_reshaped, dtype=bool),
                'is_aligned': True
            }
        else:
            return {
                'name': name,
                'pred': pred.astype(np.float32),
                'gt': depth_GT.astype(np.float32),
                'mask': valid_mask.astype(bool),
                'is_aligned': False
            }


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


args = parse_arguments()

import pandas as pd


from os import makedirs


output_path = args.output
prediction_path = args.prediction_dir or args.output
args.prediction_dir = prediction_path
makedirs(output_path, exist_ok=True)

depth_scale = 1000.0

dataset = load_dataset_for_eval(args.dataset, args.raw_type)
depth_range = dataset.depth_range

if args.max_samples > 0:
    dataset = Subset(dataset, range(min(len(dataset), args.max_samples)))


with open(join(output_path, 'eval_args.json'), 'w') as f:
    json.dump(vars(args), f)


min_depth = depth_range[0]
max_depth = depth_range[1]

args.min_depth = min_depth
args.max_depth = max_depth

print('min depth is updated and set to ',min_depth, 'and max depth is updated and set to ',max_depth)


all_metrics = []

ALIGN = False

# Use DataLoader for acceleration
eval_dataset = EvalDataset(dataset, prediction_path, args, depth_scale, align=ALIGN)

# If ALIGN is True, we can't batch variable sized tensors easily without padding. 
# Since ALIGN=False is default and target for optimization, we use batch > 1 only when ALIGN=False.
batch_size = 1 if ALIGN else 32 
num_workers = 0 if ALIGN else 8

loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

for batch in tqdm(loader):
    names = batch['name']
    
    # Move to GPU
    pred_depth_ts = batch['pred'].cuda()
    gt_depth_ts = batch['gt'].cuda()
    mask_ts = batch['mask'].cuda()
    
    # Compute metrics with reduction='none' to get per-sample results
    # All these return (B,) tensors
    l1 = mae_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
    rmse = rmse_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
    abs_rel = abs_relative_difference(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
    d4 = delta4_acc_105(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
    d5 = delta5_acc110(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
    d1 = delta1_acc(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
    
    # Transfer back to CPU only once per batch
    # We stack them into a dict of lists or process them
    
    # We can iterate over the batch dimension on CPU side to reconstruct the list of dicts
    # or just use vectorization
    
    batch_len = len(names)
    l1_cpu = l1.detach().cpu().numpy()
    rmse_cpu = rmse.detach().cpu().numpy()
    abs_rel_cpu = abs_rel.detach().cpu().numpy()
    d4_cpu = d4.detach().cpu().numpy()
    d5_cpu = d5.detach().cpu().numpy()
    d1_cpu = d1.detach().cpu().numpy()
    for i in range(batch_len):
        metrics = {
            'name': names[i],
            'L1': l1_cpu[i],
            'rmse_linear': rmse_cpu[i],
            'abs_relative_difference': abs_rel_cpu[i],
            'delta4_acc_105': d4_cpu[i],
            'delta5_acc110': d5_cpu[i],
            'delta1_acc': d1_cpu[i],
            
        }
        all_metrics.append(metrics)

all_metrics = pd.DataFrame(all_metrics)

all_metrics_mean = all_metrics.mean(numeric_only=True).to_frame().T



all_metrics.to_csv(join(output_path,f'all_metrics_{current_time}_{ALIGN}.csv'), index=False)
all_metrics_mean.to_json(join(output_path, f'mean_metrics_{current_time}_{ALIGN}.json'), orient='records', lines=True, force_ascii=False)
from loguru import logger
logger.info(f'save dir: {output_path}')
logger.info(f'prediction dir: {prediction_path}')
