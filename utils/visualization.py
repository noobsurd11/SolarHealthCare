import pandas as pd
import numpy as np


def decode_segmap(mask, color_map_csv):
    """
    Convert an index-based mask to an RGB image using the class color map.
    Args:
        mask (H, W) numpy array with class indices
        color_map_csv: path to the dataset's ListOfClassesAndColorCodes.csv
    Returns:
        (H, W, 3) RGB numpy array
    """
    df = pd.read_csv(color_map_csv, header=None)
    index_to_rgb = {}

    for _, row in df.iterrows():
        try:
            index = int(row[6])
            r, g, b = int(row[3]), int(row[4]), int(row[5])
            index_to_rgb[index] = (r, g, b)
        except ValueError:
            continue


    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, color in index_to_rgb.items():
        rgb[mask == idx] = color

    return rgb
