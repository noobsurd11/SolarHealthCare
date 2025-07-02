import pandas as pd
import numpy as np

def compute_mfb_weights(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure the column names are stripped of whitespace
    df.columns = [col.strip() for col in df.columns]

    # Extract class names and pixel frequencies
    class_names = df['Desc'].tolist()
    pixel_freqs = df['Freq'].astype(str).str.replace(',', '').astype(int).to_numpy()

    # Avoid division by zero for empty classes
    pixel_freqs[pixel_freqs == 0] = 1

    # Compute median frequency
    median_freq = np.median(pixel_freqs)

    # MFB formula: weight[c] = median_freq / freq[c]
    weights = median_freq / pixel_freqs

    # Print result
    print("[📊] Median Frequency Balancing Weights:\n")
    for name, w in zip(class_names, weights):
        print(f"{name.strip()}: {w:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute class weights using Median Frequency Balancing.")
    parser.add_argument('--csv', type=str, required=True, help='Path to ListOfClassesAndColorCodes.csv')
    args = parser.parse_args()

    compute_mfb_weights(args.csv)
