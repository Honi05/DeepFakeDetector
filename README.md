# DeepFake Detection with Explainable AI

Bachelor project — AI & Applied Computer Science

Detects deepfake videos using a fine-tuned EfficientNet-B4 classifier with Grad-CAM explainability and a deterministic forensic report generator.

## Results

| Metric | Score |
|---|---|
| Frame-level AUC | 0.9933 |
| Video-level AUC | 0.9990 |
| Frame Accuracy | 97.46% |
| Frame F1 | 98.55% |

## Project Structure

```
src/                  # Core modules
  data_pipeline.py    # Frame extraction, MTCNN face detection, dataset split
  dataset.py          # PyTorch Dataset with augmentation
  model.py            # EfficientNet-B4 classifier
  train.py            # Focal loss, training loop, W&B logging
  evaluate.py         # Frame-level and video-level AUC
  gradcam.py          # Grad-CAM heatmaps and facial zone mapping
  forensic_text.py    # Deterministic forensic report template
  ablations.py        # Ablation experiment runner
demo/
  app.py              # Gradio two-tab interface
```

## Setup (RunPod GPU)

```bash
git clone https://github.com/Honi05/DeepFakeDetector.git
cd DeepFakeDetector
cp .env.example .env   # fill in your credentials
bash setup_runpod.sh
bash train_runpod.sh
```

## Dataset

Celeb-DF v2 — 590 real + 5,639 fake celebrity videos. Split by video ID (not frame) to avoid data leakage.

## Model

EfficientNet-B4 pretrained on ImageNet, first 5 blocks frozen, custom classification head. Trained with Focal Loss (γ=2, α=0.25) to handle 5:1 class imbalance.

## Explainability

Grad-CAM activations on the last convolutional block, mapped to 5 facial zones (forehead, eyes, nose, jaw, hairline). A template engine converts confidence + activated zones into a structured forensic report.
