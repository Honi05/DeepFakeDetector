# DeepFake Detector

<div align="center">

**EfficientNet-B4 · Grad-CAM · Forensic Reports · Gradio Demo**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/Model-honi05%2Fdeepfake--detection-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/honi05/deepfake-detection)
[![Frame AUC](https://img.shields.io/badge/Frame%20AUC-99.33%25-DC2626?style=flat-square)](#results)
[![Video AUC](https://img.shields.io/badge/Video%20AUC-99.90%25-DC2626?style=flat-square)](#results)

</div>

---

A deepfake face detector that tells you **what** it found and **where** it found it.

Most deepfake classifiers output a single probability and stop there. This one runs **Grad-CAM** on every prediction to highlight the exact facial region that triggered the alert — forehead, eyes, nose, jaw, or hairline — and generates a **structured forensic report** describing the artefacts detected.

```
Input face  →  EfficientNet-B4  →  Fake / Real
                                          ↓
                              Grad-CAM heatmap (224×224)
                                          ↓
                   5-zone spatial attribution (eyes, jaw ...)
                                          ↓
                         Forensic text report (4 tiers)
```

---

## Results

Trained on **Celeb-DF v2** (590 real + 5,639 fake videos) using an NVIDIA RTX A4000.

| Metric | Score |
|---|---|
| Frame-Level AUC-ROC | **99.33%** |
| Video-Level AUC-ROC | **99.90%** |
| Frame Accuracy | **97.46%** |
| Frame F1 Score | **98.55%** |
| False Negative Rate | **0.44%** |

> **Video-level** scores are computed by averaging frame probabilities per video ID. This mirrors real-world deployment and explains why video AUC exceeds frame AUC.

### Ablation Study

| Configuration | AUC | vs Baseline |
|---|---|---|
| **This model** | **0.9933** | — |
| No augmentation | 0.9701 | −2.32% |
| EfficientNet-B0 | 0.9612 | −3.21% |
| BCE loss (no focal) | 0.9814 | −1.19% |
| Fully fine-tuned | 0.9878 | −0.55% |

---

## Architecture

```
224×224 face crop
    │
    ▼
EfficientNet-B4 (ImageNet pretrained)
    ├── Blocks 0–4  ──  frozen  (spatial feature extraction)
    └── Blocks 5–8  ──  fine-tuned  (LR = 1e-4)
            │
            ▼
    Global Average Pool  →  1792-d feature vector
            │
            ▼
    Dropout(0.4) → Linear(1792→256) → ReLU → Dropout(0.2) → Linear(256→1)
            │
            ▼
       sigmoid ≥ 0.5  →  FAKE
```

**Training:**
- Loss: Focal Loss (α=0.25, γ=2.0) — handles 5:1 class imbalance
- Optimizer: AdamW, differential LR (head 5×10⁻⁴, backbone 1×10⁻⁴)
- Scheduler: CosineAnnealingLR, 20 epochs max, early stopping patience=5
- Augmentation: horizontal flip, colour jitter, rotation ±10°, JPEG quality simulation

---

## Explainability

Grad-CAM registers a hook on the last EfficientNet convolutional block and computes:

```
αᵏ = (1/Z) ΣᵢΣⱼ ∂yᶜ/∂Aᵏᵢⱼ        (gradient weights)
Lᶜ = ReLU( Σₖ αᵏ Aᵏ )             (localisation map)
   → bilinear upsample to 224×224
```

The heatmap is then mapped to **5 facial zones**:

| Zone | Pixel rows | What fakes leave behind |
|---|---|---|
| Forehead | 0 – 60 | Hair boundary blending, skin tone mismatch |
| Eyes | 60 – 100 | Unnatural reflections, pupil shape, lash generation |
| Nose | 100 – 145 | Texture discontinuities, geometric distortion |
| Jaw | 145 – 185 | Blending seam, edge softening |
| Hairline | 185 – 224 | Hair generation artefacts, boundary warping |

The top-2 activated zones feed into a **4-tier forensic report**:

```
HIGH CONFIDENCE FAKE (91.3%)
Eyes region shows unnatural reflection patterns inconsistent with genuine facial
geometry. Jaw area exhibits a visible blending seam characteristic of face-swap
artefacts. Recommend forensic verification.
```

---

## Project Structure

```
src/
  data_pipeline.py    # frame extraction, MTCNN face detection, dataset split
  dataset.py          # PyTorch Dataset + augmentation pipeline
  model.py            # EfficientNet-B4 classifier
  train.py            # Focal Loss, training loop, W&B logging
  evaluate.py         # frame-level + video-level AUC
  gradcam.py          # Grad-CAM + 5-zone spatial attribution
  forensic_text.py    # deterministic forensic report generator
  ablations.py        # ablation study runner
demo/
  app.py              # Gradio two-tab interface (image + video)
train_main.py         # training entry point
evaluate_final.py     # test-set evaluation
run_pipeline.py       # frame extraction + face detection pipeline
split_data.py         # video-stratified train/val/test split
setup_runpod.sh       # environment setup for GPU training
train_runpod.sh       # full automated pipeline (download → train → evaluate)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Honi05/DeepFakeDetector.git
cd DeepFakeDetector
pip install -r requirements.txt
```

### 2. Download the model

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('honi05/deepfake-detection', 'best_model.pt', local_dir='checkpoints/')
"
```

### 3. Run the demo

```bash
python demo/app.py
```

Opens a Gradio interface at `http://localhost:7860` with two tabs — **Image** and **Video**.

### 4. Python API

```python
import torch
from PIL import Image
from torchvision import transforms
from src.model import DeepfakeClassifier
from src.gradcam import GradCAM, get_top_zones
from src.forensic_text import generate_forensic_report

# Load model
model = DeepfakeClassifier(freeze_blocks=5)
state = torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

# Preprocess face crop
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img = Image.open("face.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0)

# Detect + explain
grad_cam = GradCAM(model)
heatmap, confidence = grad_cam.compute(tensor)
zones = get_top_zones(heatmap, top_k=2)
report = generate_forensic_report(confidence, zones[0], zones[1])
overlay = grad_cam.overlay(img, heatmap)

print(f"{'FAKE' if confidence >= 0.5 else 'REAL'} — {confidence:.1%}")
print(report)
overlay.save("heatmap.jpg")
```

---

## Train from Scratch (GPU)

Requires Kaggle API key, W&B account, HuggingFace token. Copy `.env.example` to `.env` and fill in credentials.

```bash
# On a GPU machine (tested on RTX A4000 via RunPod)
bash setup_runpod.sh     # creates venv, installs deps, authenticates
bash train_runpod.sh     # downloads Celeb-DF v2, trains, evaluates, uploads to HF
```

Full pipeline takes ~3 hours on an RTX A4000.

---

## Dataset

**Celeb-DF v2** — Li et al., CVPR 2020  
590 real celebrity videos + 5,639 high-quality deepfake videos (~6 GB)

Split strategy: **by video ID** (80/10/10). All frames from a given video land in exactly one partition — this prevents identity leakage that would inflate test metrics when splitting per-frame.

Face crops: MTCNN detection → 224×224 px, 20 px margin.  
Frame budget: 15 uniformly sampled frames per video.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model + transforms |
| `facenet-pytorch` | MTCNN face detection |
| `opencv-python` | Video frame extraction |
| `gradio` | Interactive demo |
| `wandb` | Training metrics logging |
| `huggingface-hub` | Model hosting |
| `scikit-learn` | AUC, F1 evaluation |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) — Li et al., CVPR 2020
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, ICML 2019
- [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., ICCV 2017
- [Focal Loss](https://arxiv.org/abs/1708.02002) — Lin et al., ICCV 2017
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN implementation
