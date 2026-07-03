#!/bin/bash
# ============================================================
# train_runpod.sh — Full training pipeline on RunPod GPU
# Run AFTER setup_runpod.sh:
#   bash train_runpod.sh
#
# What this does:
#   1. Download Celeb-DF v2 from Kaggle
#   2. Reorganize into data/raw/celeb-df-v2/{real,fake}/
#   3. Extract 15 frames/video + MTCNN face detection (GPU)
#   4. Generate train/val/test split.json
#   5. Train EfficientNet-B4 (20 epochs, early stopping, W&B)
#   6. Evaluate on test set
#   7. Upload checkpoint + demo to HuggingFace
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

source venv/bin/activate
set -a; source .env; set +a

LOG_FILE="training_run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Step 1: Download dataset ────────────────────────────────
echo ""
echo "=== [1/7] Downloading Celeb-DF v2 from Kaggle ==="
mkdir -p data/raw/download

if [ -d "data/raw/celeb-df-v2/real" ] && [ "$(ls -A data/raw/celeb-df-v2/real 2>/dev/null | head -1)" ]; then
    echo "Raw videos already present — skipping download."
else
    kaggle datasets download \
        -d reubensuju/celeb-df-v2 \
        -p data/raw/download/ \
        --unzip \
        --force

    echo "Download complete. Listing top-level contents:"
    ls data/raw/download/

    # ── Reorganize into expected structure ──────────────────
    echo ""
    echo "=== [1b/7] Reorganizing dataset into real/ and fake/ ==="
    mkdir -p data/raw/celeb-df-v2/real data/raw/celeb-df-v2/fake

    # Handle common Kaggle dataset layouts for Celeb-DF v2
    DL="data/raw/download"

    # Layout 1: DL/real/ and DL/fake/
    if [ -d "$DL/real" ] && [ -d "$DL/fake" ]; then
        echo "Detected layout: real/ and fake/ at top level"
        mv "$DL/real/"*.mp4 data/raw/celeb-df-v2/real/ 2>/dev/null || true
        mv "$DL/fake/"*.mp4 data/raw/celeb-df-v2/fake/ 2>/dev/null || true

    # Layout 2: DL/Celeb-real/ and DL/Celeb-synthesis/
    elif [ -d "$DL/Celeb-real" ] || [ -d "$DL/YouTube-real" ]; then
        echo "Detected layout: Celeb-real / YouTube-real / Celeb-synthesis"
        find "$DL" -maxdepth 2 -name "*.mp4" | while read f; do
            parent=$(basename "$(dirname "$f")")
            if [[ "$parent" == *"real"* ]] || [[ "$parent" == *"Real"* ]]; then
                cp "$f" data/raw/celeb-df-v2/real/
            else
                cp "$f" data/raw/celeb-df-v2/fake/
            fi
        done

    # Layout 3: flat directory with naming convention (id0_id1... = fake)
    else
        echo "Unknown layout — searching recursively for .mp4 files..."
        find "$DL" -name "*.mp4" | head -5
        echo ""
        echo "ERROR: Could not auto-detect dataset structure."
        echo "Please manually move .mp4 files into:"
        echo "  data/raw/celeb-df-v2/real/  (real celebrity videos)"
        echo "  data/raw/celeb-df-v2/fake/  (deepfake videos)"
        echo "Then re-run this script with SKIP_DOWNLOAD=1:"
        echo "  SKIP_DOWNLOAD=1 bash train_runpod.sh"
        exit 1
    fi

    REAL_COUNT=$(ls data/raw/celeb-df-v2/real/*.mp4 2>/dev/null | wc -l)
    FAKE_COUNT=$(ls data/raw/celeb-df-v2/fake/*.mp4 2>/dev/null | wc -l)
    echo "Reorganized: $REAL_COUNT real videos, $FAKE_COUNT fake videos"
fi

# ── Step 2: Frame extraction + face detection ───────────────
echo ""
echo "=== [2/7] Extracting frames and detecting faces (GPU) ==="
if [ -d "data/crops/real" ] && [ "$(ls -A data/crops/real 2>/dev/null | head -1)" ]; then
    echo "Crops already present — skipping pipeline."
else
    python run_pipeline.py --device cuda --n_frames 15
fi

REAL_CROPS=$(find data/crops/real -name "*.jpg" 2>/dev/null | wc -l)
FAKE_CROPS=$(find data/crops/fake -name "*.jpg" 2>/dev/null | wc -l)
echo "Face crops: $REAL_CROPS real, $FAKE_CROPS fake"

# ── Step 3: Generate split ──────────────────────────────────
echo ""
echo "=== [3/7] Generating train/val/test split ==="
if [ -f "data/split.json" ]; then
    echo "split.json already exists — skipping."
else
    python split_data.py
fi

# ── Step 4: Train ───────────────────────────────────────────
echo ""
echo "=== [4/7] Training EfficientNet-B4 (20 epochs, early stopping) ==="
python train_main.py \
    --device cuda \
    --epochs 20 \
    --batch_size 32 \
    --num_workers 4 \
    --wandb_project deepfake-detection \
    --run_name "efficientnet-b4-focal-loss-$(date +%Y%m%d)"

# ── Step 5: Evaluate ────────────────────────────────────────
echo ""
echo "=== [5/7] Evaluating on test set ==="
python evaluate_final.py \
    --device cuda \
    --checkpoint checkpoints/best_model.pt

echo ""
echo "Test metrics:"
cat checkpoints/test_metrics.json

# ── Step 6: Run ablations (optional, takes ~4 hrs) ──────────
if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
    echo ""
    echo "=== [6/7] Running ablation experiments ==="
    python src/ablations.py \
        --crops_dir data/crops \
        --split_file data/split.json \
        --device cuda \
        --epochs 10 \
        --output checkpoints/ablation_results.json
else
    echo ""
    echo "=== [6/7] Skipping ablations (set RUN_ABLATIONS=1 to enable) ==="
fi

# ── Step 7: Upload to HuggingFace ───────────────────────────
echo ""
echo "=== [7/7] Uploading to HuggingFace Hub ==="
python upload_to_hub.py \
    --checkpoint checkpoints/best_model.pt \
    --repo_id "Honi05/deepfake-detection" \
    --token "$HUGGINGFACE_TOKEN"

echo ""
echo "======================================================"
echo "  Training pipeline complete!"
echo "  Log saved to: $LOG_FILE"
echo "  Model: checkpoints/best_model.pt"
echo "  HuggingFace: https://huggingface.co/Honi05/deepfake-detection"
echo "======================================================"
