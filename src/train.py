# src/train.py
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device) -> float:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    if len(set(all_labels)) < 2:
        return 0.5  # undefined AUC with single class — return chance level
    return roc_auc_score(all_labels, all_probs)


def train(
    model,
    train_loader,
    val_loader,
    epochs: int = 20,
    lr_backbone: float = 1e-4,
    lr_head: float = 5e-4,
    patience: int = 5,
    checkpoint_path: str = "checkpoints/best_model.pt",
    device: str = "cuda",
    criterion=None,
    wandb_project: str = None,
    run_name: str = None,
) -> float:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if criterion is None:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

    backbone_params = [p for n, p in model.named_parameters()
                       if 'features' in n and p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_auc, patience_counter = 0.0, 0
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # W&B init
    run = None
    if _WANDB_AVAILABLE and wandb_project:
        run = wandb.init(project=wandb_project, name=run_name, config={
            "epochs": epochs, "lr_backbone": lr_backbone, "lr_head": lr_head,
            "patience": patience, "backbone": getattr(model, 'backbone', 'b4'),
        })

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = eval_epoch(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        if run:
            run.log({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if run:
        run.finish()

    return best_auc
