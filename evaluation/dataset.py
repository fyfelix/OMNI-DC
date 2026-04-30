from os.path import dirname, join
import json

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
