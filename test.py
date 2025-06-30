import torch
from torch.utils.data import DataLoader
from datasets.dataloader import SolarDataset
from model.smp import build_model
from utils.losses import build_loss_fn
from utils.metrics import compute_metrics
from utils.visualization import decode_segmap
from utils.plots import plot_confusion_matrix, plot_iou_per_class, plot_classwise_iou_vs_pixel_freq
import os
from PIL import Image
import numpy as np
from tqdm import tqdm



def test(cfg):
    print("[🔍] Running final evaluation on test set...")

    cfg['data']['split'] = 'test'
    dataset = SolarDataset(cfg, split='test')
    loader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['workers'])

    model = build_model(cfg).to(cfg['device'])
    model.load_state_dict(torch.load(os.path.join(cfg['experiment']['output_dir'], 'best_model.pth')))
    model.eval()

    all_metrics = []
    os.makedirs(os.path.join(cfg['experiment']['output_dir'], 'test_preds'), exist_ok=True)
    results_dir = os.path.join(cfg['experiment']['output_dir'], 'test_results')
    os.makedirs(results_dir, exist_ok=True)

    all_preds = []
    test_loss = 0.0
    all_labels = []
    loss_fn = build_loss_fn(cfg, dataset.class_names)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(loader)):
            images, masks = images.to(cfg['device']), masks.to(cfg['device'])
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu()
            labels = masks.cpu()

            all_preds.append(preds)
            all_labels.append(labels)

            # Save RGB prediction
            for i in range(preds.shape[0]):
                rgb_mask = decode_segmap(preds[i].numpy(), os.path.join(cfg['data']['root_dir'], cfg['data']['version'], cfg['data']['color_map_csv']))
                rgb_img = Image.fromarray(rgb_mask.astype(np.uint8))

                img =images[i].cpu()
                if img.shape[0] == 1:
                  img = img.expand(3, -1, -1)

                img_np = img.permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(img_np)
                
                rgb_img.save(os.path.join(cfg['experiment']['output_dir'], 'test_preds', f"{idx}_{i}_pred.png"))
                image.save(os.path.join(cfg['experiment']['output_dir'], 'test_preds', f"{idx}_{i}_EL.png"))

            metrics = compute_metrics(outputs, masks,  csv_path=dataset.csv_path)
            all_metrics.append(metrics)

    avg_test_loss = test_loss / len(loader)

    avg_metrics = {}
    for key in all_metrics[0]:
        if isinstance(all_metrics[0][key], list):
            per_class = np.array([d[key] for d in all_metrics])
            avg_metrics[key] = np.mean(per_class, axis=0).tolist()
        else:
            avg_metrics[key] = sum(d[key] for d in all_metrics) / len(all_metrics)


    print(f"[📊] Test Set Evaluation: {avg_metrics}")
    print(f" Test Loss={avg_test_loss:.4f}")
    plot_confusion_matrix(torch.cat(all_preds).flatten(), torch.cat(all_labels).flatten(), dataset.csv_path, results_dir)
    plot_classwise_iou_vs_pixel_freq(
        iou_per_class=avg_metrics['iou_per_class'],
        mask_dir=cfg['data']['root_dir'] + '/' + cfg['data']['version'] + '/el_masks_test',
        class_names=dataset.class_names,
        results_dir= results_dir)

