# split_data.py
"""
Step 2: Generate train/val/test split.json from crops directory.
Run AFTER run_pipeline.py.

Usage:
    python split_data.py
"""
from src.data_pipeline import split_by_video

assignments = split_by_video(
    crops_dir="data/crops",
    output_path="data/split.json",
    train_frac=0.8,
    val_frac=0.1,
    seed=42,
)
print(f"Split saved to data/split.json")
print(f"  Train videos: {sum(1 for v in assignments.values() if v['split'] == 'train')}")
print(f"  Val videos:   {sum(1 for v in assignments.values() if v['split'] == 'val')}")
print(f"  Test videos:  {sum(1 for v in assignments.values() if v['split'] == 'test')}")
