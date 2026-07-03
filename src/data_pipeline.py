# src/data_pipeline.py
import cv2
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import torch
import random
import json


def extract_frames(video_path: str, output_dir: str, n_frames: int = 15) -> list:
    """Sample n_frames uniformly from video. Returns list of saved JPEG paths."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = list(dict.fromkeys(int(i * total / n_frames) for i in range(n_frames)))
    video_id = Path(video_path).stem
    out_dir = Path(output_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            path = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(str(path))
    finally:
        cap.release()

    return saved


_mtcnn_instance = None


def get_mtcnn(device: str = 'cpu'):
    global _mtcnn_instance
    if _mtcnn_instance is None:
        _mtcnn_instance = MTCNN(image_size=224, margin=20, keep_all=False, device=device)
    return _mtcnn_instance


def detect_and_crop(frame_path: str, output_dir: str, device: str = 'cpu') -> str | None:
    """Detect primary face in frame, crop to 224x224, save JPEG. Returns path or None."""
    img = Image.open(frame_path).convert('RGB')
    mtcnn = get_mtcnn(device)
    crop = mtcnn(img)  # FloatTensor (3, 224, 224) in [-1, 1] or None

    if crop is None:
        return None

    # Convert from [-1, 1] to [0, 255] uint8
    crop_np = ((crop.permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    pil_crop = Image.fromarray(crop_np)

    out_path = Path(output_dir) / Path(frame_path).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_crop.save(str(out_path))
    return str(out_path)


def split_by_video(
    crops_dir: str,
    output_path: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Assign each video subfolder in crops_dir/real/ and crops_dir/fake/ to train/val/test.
    Saves JSON to output_path. Returns assignment dict:
      {"real/vid_001": {"split": "train", "label": 0}, ...}
    """
    random.seed(seed)
    assignments = {}

    for label_name, label in [("real", 0), ("fake", 1)]:
        label_dir = Path(crops_dir) / label_name
        if not label_dir.exists():
            continue
        video_ids = sorted([d.name for d in label_dir.iterdir() if d.is_dir()])
        random.shuffle(video_ids)
        n = len(video_ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        for i, vid in enumerate(video_ids):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignments[f"{label_name}/{vid}"] = {"split": split, "label": label}

    with open(output_path, "w") as f:
        json.dump(assignments, f, indent=2)

    return assignments
