import torch
from torch.utils.data import DataLoader
from data.dataset import SolarDataset
from models.deeplabv3plus import build_model
from utils.losses import build_loss_fn
from utils.metrics import compute_metrics
from utils.visualization import decode_segmap
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

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(loader)):
            images, masks = images.to(cfg['device']), masks.to(cfg['device'])
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Save RGB prediction
            for i in range(preds.shape[0]):
                rgb_mask = decode_segmap(preds[i].cpu().numpy(), dataset.index_to_class)
                rgb_img = Image.fromarray(rgb_mask.astype(np.uint8))
                rgb_img.save(os.path.join(cfg['experiment']['output_dir'], 'test_preds', f"sample_{idx}_{i}.png"))

            metrics = compute_metrics(outputs, masks, cfg['data']['num_classes'])
            all_metrics.append(metrics)

    avg_metrics = {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    print(f"[📊] Test Set Evaluation: {avg_metrics}")