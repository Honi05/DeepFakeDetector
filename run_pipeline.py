# run_pipeline.py
"""
Step 1: Extract frames from raw videos and detect/crop faces.
Run AFTER downloading Celeb-DF v2 to data/raw/celeb-df-v2/

Usage:
    python run_pipeline.py [--device cuda]
"""
import argparse
from pathlib import Path
from tqdm import tqdm

from src.data_pipeline import extract_frames, detect_and_crop

RAW_DIR    = Path("data/raw/celeb-df-v2")
FRAMES_DIR = Path("data/frames")
CROPS_DIR  = Path("data/crops")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Device for MTCNN (cpu or cuda)")
    parser.add_argument("--n_frames", type=int, default=15)
    args = parser.parse_args()

    for label in ["real", "fake"]:
        label_dir = RAW_DIR / label
        if not label_dir.exists():
            print(f"[WARN] {label_dir} not found — skipping {label}")
            continue

        videos = list(label_dir.glob("*.mp4"))
        print(f"Processing {len(videos)} {label} videos...")

        for video_path in tqdm(videos, desc=label):
            frame_paths = extract_frames(
                str(video_path),
                str(FRAMES_DIR / label),
                n_frames=args.n_frames,
            )
            for fp in frame_paths:
                detect_and_crop(fp, str(CROPS_DIR / label / video_path.stem), device=args.device)

    print("Pipeline complete. Run split_data.py next.")


if __name__ == "__main__":
    main()
