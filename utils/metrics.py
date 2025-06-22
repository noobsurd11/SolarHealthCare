import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_metrics(preds, targets, num_classes):
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