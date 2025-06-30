import segmentation_models_pytorch as smp

def build_model(cfg):
    if cfg['model']['name'] == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            encoder_name=cfg['model']['backbone'],            # e.g., 'resnet50'
            encoder_weights='imagenet' if cfg['model']['pretrained'] else None,
            in_channels=1,                                     # EL images are grayscale
            classes=cfg['data']['num_classes'],                # number of output classes
            activation=None)                                 # raw logits for CE/Dice
    
    elif cfg['model']['name'] == 'unetplusplus':
        return smp.UnetPlusPlus(
            encoder_name=cfg['model']['backbone'],             # e.g., 'resnet50'
            encoder_weights='imagenet' if cfg['model']['pretrained'] else None,
            in_channels=1,                                     # EL images are grayscale
            classes=cfg['data']['num_classes'],                # number of output classes
            activation=None                                    # raw logits (no softmax/sigmoid)
       )
    else:
        raise ValueError(f"Unknown model type: {cfg['model']['name']}")
