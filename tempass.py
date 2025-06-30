import os
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm

def compute_pixel_frequencies(mask_folder):
    freq_dict = defaultdict(int)

    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(mask_folder, file)
        mask = Image.open(mask_path).convert("L")  # ensure grayscale
        mask_np = np.array(mask)

        unique, counts = np.unique(mask_np, return_counts=True)
        for cls, count in zip(unique, counts):
            freq_dict[int(cls)] += int(count)

    return dict(sorted(freq_dict.items()))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute pixel frequency per class from grayscale masks.")
    parser.add_argument('--mask_dir', type=str, required=True, help="Path to grayscale mask images directory")
    args = parser.parse_args()

    result = compute_pixel_frequencies(args.mask_dir)

    print("\n[📊] Pixel Frequency Per Class:")
    for k, v in result.items():
        print(f"Class {k:>2}: {v:,} pixels")
