#!/bin/bash
# ============================================================
# setup_runpod.sh — One-time environment setup on RunPod
# Run once after cloning the repo:
#   cd /workspace/deepfakedetection
#   bash setup_runpod.sh
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== [1/5] Creating Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet

echo "=== [2/5] Installing project dependencies ==="
pip install -r requirements.txt

echo "=== [3/5] Verifying GPU is available ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo "=== [4/5] Loading credentials from .env ==="
if [ ! -f .env ]; then
    echo ""
    echo "ERROR: .env file not found. Create it at $REPO_DIR/.env with:"
    echo ""
    echo "  KAGGLE_USERNAME=arorahoni966"
    echo "  KAGGLE_KEY=<your-kaggle-key>"
    echo "  WANDB_API_KEY=<your-wandb-key>"
    echo "  HUGGINGFACE_TOKEN=<your-hf-token>"
    echo ""
    exit 1
fi

set -a
source .env
set +a

# Set up Kaggle credentials file
mkdir -p ~/.kaggle
echo "{\"username\": \"${KAGGLE_USERNAME}\", \"key\": \"${KAGGLE_KEY}\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle credentials written to ~/.kaggle/kaggle.json"

echo "=== [5/5] Logging in to Weights & Biases ==="
wandb login "$WANDB_API_KEY" --relogin

echo ""
echo "Setup complete! Now run:"
echo "  bash train_runpod.sh"
