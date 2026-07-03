# train_main.py
"""
Step 3: Full training run.
Run AFTER split_data.py.

Usage:
    python train_main.py [--device cuda] [--epochs 20] [--wandb_project deepfake-detection]
"""
import argparse
import torch
from torch.utils.data import DataLoader

from src.dataset import DeepfakeDataset
from src.model import DeepfakeClassifier
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops_dir",      default="data/crops")
    parser.add_argument("--split_file",     default="data/split.json")
    parser.add_argument("--checkpoint",     default="checkpoints/best_model.pt")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--epochs",         type=int, default=20)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--num_workers",    type=int, default=0)
    parser.add_argument("--wandb_project",  default="deepfake-detection")
    parser.add_argument("--run_name",       default="efficientnet-b4-focal-loss")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    train_ds = DeepfakeDataset(args.crops_dir, args.split_file, "train")
    val_ds   = DeepfakeDataset(args.crops_dir, args.split_file, "val")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    model = DeepfakeClassifier(freeze_blocks=5)
    best_auc = train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        device=args.device,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )
    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
