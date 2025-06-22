import argparse
import yaml
import os
import traceback

from train.train import train
from data.dataset import SolarDataset
from models.deeplabv3plus import build_model
from utils.losses import build_loss_fn, get_class_weights_from_yaml
from torch.utils.data import DataLoader
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Solar PV Fault Segmentation")
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--device', type=str, help='cuda or cpu')
    parser.add_argument('--experiment', type=str, help='Optional experiment name')
    return parser.parse_args()


def prompt_or_default(field, default):
    try:
        val = input(f"[?] No value for '{field}'. Use default '{default}'? (Y/n): ").strip().lower()
        if val in ('n', 'no'):
            return input(f"Enter value for '{field}': ").strip()
    except Exception:
        print("[!] Failed to read input. Using default.")
    return default


def load_config(args):
    if not args.config or not os.path.exists(args.config):
        raise FileNotFoundError("[!] Config file not provided or does not exist.")

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # CLI overrides with fallback
    cfg['device'] = args.device or prompt_or_default('device', cfg.get('device', 'cuda'))
    cfg['experiment']['name'] = args.experiment or cfg['experiment'].get('name') or \
        prompt_or_default('experiment name', 'default_experiment')

    return cfg


def sanity_check(cfg):
    try:
        print("[✓] Running sanity check on data and model...")
        dataset = SolarDataset(cfg, split='train')
        sample_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=cfg['data']['workers'])
        images, masks = next(iter(sample_loader))

        model = build_model(cfg).to(cfg['device'])
        outputs = model(images.to(cfg['device']))

        assert outputs.shape[0] == images.shape[0], "Batch size mismatch"
        assert outputs.shape[2:] == images.shape[2:], "Spatial dim mismatch"
        assert outputs.shape[1] == cfg['data']['num_classes'], "Class count mismatch"

        print("[✓] Sanity check passed! Ready for training.")

    except Exception as e:
        print("[!] Sanity check failed:")
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    try:
        args = get_args()
        cfg = load_config(args)

        # Set random seed
        torch.manual_seed(cfg['experiment']['seed'])

        # Sanity check (model and data)
        sanity_check(cfg)

        # Training entry point
        train(cfg)

    except Exception as e:
        print("[!] Fatal error during setup:")
        traceback.print_exc()
        exit(1)
