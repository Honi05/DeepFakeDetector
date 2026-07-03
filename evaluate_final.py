# evaluate_final.py
"""
Step 4: Evaluate the trained model on the test set.
Run AFTER train_main.py.

Usage:
    python evaluate_final.py [--checkpoint checkpoints/best_model.pt]
"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from src.dataset import DeepfakeDataset
from src.model import DeepfakeClassifier
from src.evaluate import evaluate, video_level_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops_dir",  default="data/crops")
    parser.add_argument("--split_file", default="data/split.json")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    test_ds = DeepfakeDataset(args.crops_dir, args.split_file, "test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = DeepfakeClassifier(freeze_blocks=5)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))

    metrics = evaluate(model, test_loader, device=args.device)
    vid_auc = video_level_auc(model, test_loader, test_ds.video_ids, device=args.device)
    metrics["video_auc"] = vid_auc

    print("\n=== Test Set Results ===")
    print(f"Frame AUC:        {metrics['auc']:.4f}")
    print(f"Video AUC:        {metrics['video_auc']:.4f}")
    print(f"Frame Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Frame F1:         {metrics['f1']:.4f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")

    with open("checkpoints/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved to checkpoints/test_metrics.json")


if __name__ == "__main__":
    main()
