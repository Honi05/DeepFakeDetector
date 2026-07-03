# src/ablations.py
"""
Run ablation experiments. Execute after downloading data and running at least one
full training run to verify the pipeline works.

Usage:
    python src/ablations.py --crops_dir data/crops --split_file data/split.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DeepfakeDataset, VAL_TRANSFORMS
from model import DeepfakeClassifier
from train import FocalLoss, train
from evaluate import evaluate, video_level_auc
import torch.nn.functional as F
import torch.nn as nn


NO_AUG_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BCELoss(nn.Module):
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)


ABLATIONS = [
    {
        "name": "baseline",
        "backbone": "b4",
        "freeze_blocks": 5,
        "loss": "focal",
        "train_transform": None,  # use default TRAIN_TRANSFORMS
    },
    {
        "name": "no_augmentation",
        "backbone": "b4",
        "freeze_blocks": 5,
        "loss": "focal",
        "train_transform": NO_AUG_TRANSFORMS,
    },
    {
        "name": "efficientnet_b0",
        "backbone": "b0",
        "freeze_blocks": 5,
        "loss": "focal",
        "train_transform": None,
    },
    {
        "name": "bce_loss",
        "backbone": "b4",
        "freeze_blocks": 5,
        "loss": "bce",
        "train_transform": None,
    },
    {
        "name": "fully_finetuned",
        "backbone": "b4",
        "freeze_blocks": 0,
        "loss": "focal",
        "train_transform": None,
    },
]


def run_ablation(cfg: dict, crops_dir: str, split_file: str, device: str, epochs: int = 10):
    name = cfg["name"]
    print(f"\n{'='*50}\nAblation: {name}\n{'='*50}")

    train_ds = DeepfakeDataset(
        crops_dir, split_file, "train",
        transform=cfg["train_transform"]
    )
    val_ds  = DeepfakeDataset(crops_dir, split_file, "val")
    test_ds = DeepfakeDataset(crops_dir, split_file, "test")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=2)

    model = DeepfakeClassifier(freeze_blocks=cfg["freeze_blocks"], backbone=cfg["backbone"])
    criterion = FocalLoss() if cfg["loss"] == "focal" else BCELoss()

    checkpoint = f"checkpoints/ablation_{name}.pt"
    best_auc = train(
        model, train_loader, val_loader,
        epochs=epochs,
        checkpoint_path=checkpoint,
        device=device,
        criterion=criterion,
        wandb_project="deepfake-ablations",
        run_name=name,
    )

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    metrics = evaluate(model, test_loader, device=device)
    metrics["val_best_auc"] = best_auc
    metrics["video_auc"] = video_level_auc(model, test_loader, test_ds.video_ids, device=device)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops_dir",  default="data/crops")
    parser.add_argument("--split_file", default="data/split.json")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--output",     default="checkpoints/ablation_results.json")
    args = parser.parse_args()

    results = {}
    for cfg in ABLATIONS:
        results[cfg["name"]] = run_ablation(
            cfg, args.crops_dir, args.split_file, args.device, args.epochs
        )
        print(f"  AUC: {results[cfg['name']]['auc']:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
