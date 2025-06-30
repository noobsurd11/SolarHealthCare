import torch
from torch.utils.data import DataLoader
from datasets.dataloader import SolarDataset
from utils.losses import build_loss_fn
from utils.metrics import compute_metrics
import os
from tqdm import tqdm
import numpy as np
from utils.visualization import decode_segmap
from utils.plots import plot_epoch_metrics, plot_confusion_matrix, plot_iou_per_class, plot_classwise_iou_vs_pixel_freq
from PIL import Image

def train(cfg):
    print(f"[🚀] Starting training: {cfg['experiment']['name']}")

    device = cfg['device']

    # --- Data ---
    print("[📦] Preparing datasets...")
    cfg['data']['split'] = 'train'
    train_dataset = SolarDataset(cfg, split='train')
    cfg['data']['split'] = 'val'
    val_dataset = SolarDataset(cfg, split='val')

    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['workers'])

    # --- Model ---
    from model.smp import build_model
    model = build_model(cfg).to(device)

    # --- Loss & Optimizer ---
    loss_fn = build_loss_fn(cfg, train_dataset.class_names)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['train']['lr']))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['train']['step_size'], gamma=cfg['train']['gamma'])

    best_miou = 0.0
    results_dir = os.path.join(cfg['experiment']['output_dir'], 'train_results')
    os.makedirs(results_dir, exist_ok=True)

    all_train_losses = []
    all_val_losses = []
    all_miou_scores = []
    iou_history = []

    for epoch in range(1, int(cfg['train']['epochs']) + 1):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"[Epoch {epoch}/{cfg['train']['epochs']}] Training")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        all_train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_metrics = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu()
                labels = masks.cpu()

                all_preds.append(preds)
                all_labels.append(labels)

                metrics = compute_metrics(outputs, masks, csv_path=train_dataset.csv_path)
                all_metrics.append(metrics)

                # pred_classes, pred_counts = torch.unique(preds, return_counts=True)
                # gt_classes, gt_counts = torch.unique(labels, return_counts=True)

                # print(f"[DEBUG] Epoch {epoch} Batch {batch_idx} - Predicted class distribution: {dict(zip(pred_classes.tolist(), pred_counts.tolist()))}")
                # print(f"[DEBUG] Epoch {epoch} Batch {batch_idx} - Ground truth class distribution: {dict(zip(gt_classes.tolist(), gt_counts.tolist()))}")

                if epoch == 1 and batch_idx == 0:
                    rgb_pred = decode_segmap(preds[0].numpy(), os.path.join(cfg['data']['root_dir'], cfg['data']['version'], cfg['data']['color_map_csv']))
                    Image.fromarray(rgb_pred).save(f"debug_val_pred_epoch{epoch}.png")

        avg_val_loss = val_loss / len(val_loader)
        all_val_losses.append(avg_val_loss)
        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0]:
            if isinstance(all_metrics[0][key], list):
                per_class = np.array([d[key] for d in all_metrics])
                avg_metrics[key] = np.mean(per_class, axis=0).tolist()
            else:
                avg_metrics[key] = sum(d[key] for d in all_metrics) / len(all_metrics)

        iou_history.append(avg_metrics['iou_per_class'])
        miou = avg_metrics['mean_iou']
        all_miou_scores.append(miou)

        print(f"[DEBUG] mIoU this epoch: {miou:.4f}, Best so far: {best_miou:.4f}")
        print(f"[DEBUG] IoU per class: {avg_metrics['iou_per_class']}")
        print(f"[📊] Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | mIoU={miou:.4f}")

        # --- Save best model ---
        if miou > best_miou and cfg['logging']['save_best_only']:
            torch.save(model.state_dict(), os.path.join(cfg['experiment']['output_dir'], 'best_model.pth'))
            print(f"[💾] Best model saved at epoch {epoch} with mIoU={miou:.4f}")
            best_miou = miou

    plot_epoch_metrics(all_train_losses, all_val_losses, all_miou_scores, results_dir)
    plot_confusion_matrix(torch.cat(all_preds).flatten(), torch.cat(all_labels).flatten(), train_dataset.csv_path, results_dir)
    plot_iou_per_class(iou_history, train_dataset.csv_path, results_dir)
    plot_classwise_iou_vs_pixel_freq(
        iou_per_class=avg_metrics['iou_per_class'],
        mask_dir=train_dataset.mask_dir,
        class_names=train_dataset.class_names,
        results_dir=results_dir
)
