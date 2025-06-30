import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
from utils.metrics import compute_pixel_frequencies

def plot_epoch_metrics(train_losses, val_losses, miou_scores, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.plot(epochs, miou_scores, label='mIoU', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training and Validation Loss with mIoU')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_over_epochs.png'))
    plt.close()

def plot_confusion_matrix(y_pred, y_true, csv_path, output_dir):
    df = pd.read_csv(csv_path)
    class_names = df['Desc'].tolist()
    num_classes = len(class_names)
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized[np.isnan(cm_normalized)] = 0

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', fmt='.2f', 
                xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=90)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_iou_per_class(iou_history, csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    class_names = df['Desc'].tolist()

    iou_history = np.array(iou_history)  # shape: (epochs, num_classes)
    epochs = range(1, len(iou_history) + 1)

    plt.figure(figsize=(14, 8))
    for class_idx in range(iou_history.shape[1]):
        plt.plot(epochs, iou_history[:, class_idx], label=class_names[class_idx])

    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Per-Class IoU over Epochs')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_per_class_over_epochs.png'))
    plt.close()


def plot_classwise_iou_vs_pixel_freq(iou_per_class, mask_dir, class_names, results_dir):
    """
    Plot class-wise IoU vs pixel frequency on test set.
    
    Args:
        iou_per_class (List[float]): List of IoU values for each class.
        mask_dir (str): Path to ground truth grayscale masks.
        class_names (List[str]): Names of the classes (ordered by label index).
        save_path (str): Path to save the plot.
    """
    pixel_freqs = compute_pixel_frequencies(mask_dir)

    # Ensure pixel_freqs align with class indices
    sorted_freqs = [pixel_freqs.get(i, 0) for i in range(len(iou_per_class))]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(range(len(iou_per_class)), iou_per_class, color='skyblue', edgecolor='k')
    
    # Annotate with pixel counts
    for idx, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{sorted_freqs[idx]//1000}k", ha='center', fontsize=8, rotation=90)

    plt.xlabel("Class Index")
    plt.ylabel("IoU")
    plt.title("Per-Class IoU vs Pixel Frequency")
    plt.xticks(ticks=range(len(iou_per_class)), labels=class_names, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    os.makedirs(os.path.dirname(results_dir), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'iou_per_class.png'))
    plt.close()
