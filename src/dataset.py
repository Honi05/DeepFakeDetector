import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _jpeg_compress(img: Image.Image) -> Image.Image:
    quality = random.randint(60, 95)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    result = Image.open(buf)
    result.load()
    return result.convert('RGB')


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.Lambda(_jpeg_compress),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DeepfakeDataset(Dataset):
    def __init__(self, crops_dir: str, split_file: str, split: str, transform=None):
        self.crops_dir = Path(crops_dir)
        self.transform = transform or (TRAIN_TRANSFORMS if split == "train" else VAL_TRANSFORMS)

        with open(split_file) as f:
            assignments = json.load(f)

        self.samples = []   # list of (img_path_str, label_int)
        self.video_ids = [] # parallel list of "real/vid_001" for video-level eval

        for rel_path, info in assignments.items():
            if info["split"] != split:
                continue
            video_dir = self.crops_dir / rel_path
            if not video_dir.exists():
                continue
            for img_path in sorted(video_dir.glob("*.jpg")):
                self.samples.append((str(img_path), info["label"]))
                self.video_ids.append(rel_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32)
