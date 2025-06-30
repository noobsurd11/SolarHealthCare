import torch
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from PIL import Image
from tqdm import tqdm


from sklearn.metrics import confusion_matrix

def compute_metrics(preds, targets, csv_path):
    """
    Compute per-class IoU, mean IoU, pixel accuracy, and mean recall.
    Args:
        preds: [B, C, H, W] logits
        targets: [B, H, W] ground truth indices
        num_classes: total number of classes
    Returns:
        dict with keys: iou_per_class, mean_iou, pixel_acc, mean_recall
    """
    with torch.no_grad():
        preds = torch.argmax(preds, dim=1).cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        df = pd.read_csv(csv_path)
        class_names = df['Desc'].tolist()
        num_classes = len(class_names)

        cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        pixel_acc = TP.sum() / cm.sum()

        iou = TP / (TP + FP + FN + 1e-6)
        recall = TP / (TP + FN + 1e-6)

        mean_iou = np.mean(iou)
        mean_recall = np.mean(recall)

        return {
            'iou_per_class': iou.tolist(),
            'mean_iou': mean_iou,
            'pixel_acc': pixel_acc,
            'mean_recall': mean_recall
        }
    
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