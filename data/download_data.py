"""
Download project datasets from Hugging Face.
Run once before starting the app:
    python data/download_data.py
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "AaronJ320/OSCARSmathHacks"
OUT_DIR = os.path.join(os.path.dirname(__file__))

FILES = [
    "currents.nc",
    "microplastics.csv",
]

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for filename in FILES:
        out_path = os.path.join(OUT_DIR, filename)
        if os.path.exists(out_path):
            print(f"Already exists, skipping: {filename}")
            continue
        print(f"Downloading {filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=OUT_DIR,
        )
        print(f"Saved to {out_path}")
    print("\nAll data ready.")
