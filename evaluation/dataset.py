import json
from glob import glob
from os.path import dirname, join

from torch.utils.data import Dataset


class HAMMERDataset(Dataset):
    def __init__(self, jsonl_path, raw_type="d435"):
        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                self.data.append(json.loads(line))

        self.raw_type = raw_type
        self.depth_range = self.data[0]["depth-range"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        rgb = join(self.root, item["rgb"])
        raw_type = self.raw_type.lower()

        if raw_type == "d435":
            raw_depth = join(self.root, item["d435_depth"])
        elif raw_type == "l515":
            raw_depth = join(self.root, item["l515_depth"])
        elif raw_type == "tof":
            raw_depth = join(self.root, item["tof_depth"])
        else:
            raise ValueError(f"Invalid raw type: {self.raw_type}")

        gt_depth = join(self.root, item["depth"])
        return rgb, raw_depth, gt_depth


class ClearPoseDataset(Dataset):
    def __init__(self, jsonl_path, max_length_each_sequence=300):
        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.data = []
        self.rgbs = []
        self.raw_depths = []
        self.gt_depths = []

        depth_range = None

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                if depth_range is None:
                    depth_range = item["depth-range"]

                rgb = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["rgb-suffix"]))
                )[:max_length_each_sequence]
                raw_depth = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["raw_depth-suffix"]))
                )[:max_length_each_sequence]
                gt_depth = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["depth-suffix"]))
                )[:max_length_each_sequence]

                self.rgbs.extend(rgb)
                self.raw_depths.extend(raw_depth)
                self.gt_depths.extend(gt_depth)
                self.data.append(item)

        self.depth_range = depth_range

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        return self.rgbs[idx], self.raw_depths[idx], self.gt_depths[idx]


def load_dataset_for_eval(dataset_path, raw_type):
    dataset_lower = dataset_path.lower()
    if "clearpose" in dataset_lower:
        if raw_type.lower() != "d435":
            raise ValueError("ClearPose dataset only supports raw-type=d435")
        return ClearPoseDataset(dataset_path)
    if "hammer" in dataset_lower:
        return HAMMERDataset(dataset_path, raw_type)
    raise ValueError(f"Invalid dataset: {dataset_path}")


def resolve_sample_name(rgb_path, dataset_path):
    parts = rgb_path.split("/")
    dataset_lower = dataset_path.lower()

    if "hammer" in dataset_lower:
        scene_name = parts[-4]
        return scene_name + "#" + parts[-1].split(".")[0]

    if "clearpose" in dataset_lower:
        return "#".join(parts[-3:-1]) + "#" + parts[-1].split(".")[0]

    raise ValueError(f"Invalid dataset: {dataset_path}")
