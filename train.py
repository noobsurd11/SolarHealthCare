import torch
from torch.utils.data import DataLoader
from datasets.dataloader import SolarDataset
from utils.losses import build_loss_fn
from utils.metrics import compute_metrics
import os
from tqdm import tqdm


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
    from models.deeplabv3plus import build_model
    model = build_model(cfg).to(device)

    # --- Loss & Optimizer ---
    loss_fn = build_loss_fn(cfg, train_dataset.class_names)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['train']['step_size'], gamma=cfg['train']['gamma'])

    best_miou = 0.0
    os.makedirs(cfg['experiment']['output_dir'], exist_ok=True)

    for epoch in range(1, cfg['train']['epochs'] + 1):
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

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_metrics = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                metrics = compute_metrics(outputs, masks, cfg['data']['num_classes'])
                all_metrics.append(metrics)

        # Aggregate metrics
        avg_metrics = {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        miou = avg_metrics['mean_iou']

        print(f"[📊] Epoch {epoch}: Train Loss={epoch_loss:.4f} | Val Loss={val_loss:.4f} | mIoU={miou:.4f}")

        # --- Save best model ---
        if miou > best_miou and cfg['logging']['save_best_only']:
            torch.save(model.state_dict(), os.path.join(cfg['experiment']['output_dir'], 'best_model.pth'))
            print(f"[💾] Best model saved at epoch {epoch} with mIoU={miou:.4f}")
            best_miou = miou
