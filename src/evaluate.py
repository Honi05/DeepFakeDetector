# src/evaluate.py
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


def evaluate(model, loader, device='cpu') -> dict:
    """Frame-level metrics: AUC, accuracy, F1, confusion matrix."""
    device = torch.device(device)
    model.eval().to(device)
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    auc = 0.5 if len(set(all_labels.tolist())) < 2 else roc_auc_score(all_labels, all_probs)
    return {
        "auc": auc,
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, preds).tolist(),
    }


def video_level_auc(model, loader, video_ids: list, device='cpu') -> float:
    """
    Aggregate frame-level predictions to video level by mean probability.
    video_ids: list of video ID strings, same order and length as all samples in loader.
    """
    device = torch.device(device)
    model.eval().to(device)
    all_probs, all_labels, all_vids = [], [], []
    idx = 0

    with torch.no_grad():
        for imgs, labels in loader:
            bs = len(imgs)
            logits = model(imgs.to(device)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_vids.extend(video_ids[idx:idx + bs])
            idx += bs

    vid_probs: dict = defaultdict(list)
    vid_labels: dict = {}
    for p, l, v in zip(all_probs, all_labels, all_vids):
        vid_probs[v].append(p)
        vid_labels[v] = l

    if len(set(vid_labels.values())) < 2:
        return 0.5  # single-class guard

    mean_probs = [np.mean(vid_probs[v]) for v in vid_probs]
    true_labels = [vid_labels[v] for v in vid_probs]
    return roc_auc_score(true_labels, mean_probs)
