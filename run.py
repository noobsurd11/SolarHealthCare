import torch
import numpy as np
from PIL import Image
from model.smp import build_model
from datasets.dataloader import SolarDataset
from utils.visualization import decode_segmap
import torchvision.transforms as T
import yaml
import os


def predict_single_image( image_path, save_path):
    cfg_path = os.path.join('configs/defaults.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = cfg['device']
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg['experiment']['output_dir'], 'best_model.pth'), map_location=device))
    model.eval()

    # Load class color map
    _, class_names = SolarDataset(cfg, split='train').load_class_map(os.path.join(cfg['data']['root_dir'], cfg['data']['version'], cfg['data']['color_map_csv']))

    # Load image
    image = Image.open(image_path).convert("L")
    transform = T.Compose([
        T.Resize((cfg['data']['img_size'], cfg['data']['img_size'])),
        T.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    rgb_mask = decode_segmap(pred, os.path.join(cfg['data']['root_dir'], cfg['data']['version'], cfg['data']['color_map_csv']))
    Image.fromarray(rgb_mask).save(save_path)
    print(f"[✅] Segmentation saved to {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to grayscale EL image')
    parser.add_argument('--save', type=str, required=True, help='Path to save RGB prediction')
    args = parser.parse_args()

    predict_single_image(args.image, args.save)
