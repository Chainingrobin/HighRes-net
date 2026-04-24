import numpy as np
import torch
from DataLoader import ImageSet
from DeepNetworks.HRNet import HRNet
from Evaluator import shift_cPSNR
from utils import collateFunction


def get_sr_and_score(imset, model, min_L=16):
    '''
    Super resolves an imset with a given model.
    Args:
        imset: imageset
        model: HRNet, pytorch model
        min_L: int, pad length
    Returns:
        sr: tensor (1, C_out, W, H), super resolved image
        scPSNR: float, shift cPSNR score
    '''
    
    if imset.__class__ is ImageSet:
        collator = collateFunction(min_L=min_L)
        lrs, alphas, hrs, hr_maps, names = collator([imset])
    elif isinstance(imset, tuple):  # imset is a tuple of batches
        lrs, alphas, hrs, hr_maps, names = imset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lrs = lrs.float().to(device)
    alphas = alphas.float().to(device)
    
    sr = model(lrs, alphas)[:, 0]
    sr = sr.detach().cpu().numpy()[0]

    if len(hrs) > 0:
        scPSNR = shift_cPSNR(sr=np.clip(sr, 0, 1),
                             hr=hrs.numpy()[0],
                             hr_map=hr_maps.numpy()[0])
    else:
        scPSNR = None

    return sr, scPSNR


def load_model(config, checkpoint_file):
    '''
    Loads a pretrained model from disk.
    Args:
        config: dict, configuration file
        checkpoint_file: str, checkpoint filename
    Returns:
        model: HRNet, a pytorch model
    '''
    
#     checkpoint_dir = config["paths"]["checkpoint_dir"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRNet(config["network"]).to(device)
    state_dict = torch.load(checkpoint_file, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(
            "Checkpoint loaded with partial match "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})."
        )
    return model


class Model(object):
    
    def __init__(self, config):
        self.config = config
        
    def load_checkpoint(self, checkpoint_file):
        self.model = load_model(self.config, checkpoint_file)
        
    def __call__(self, imset):
        sr, scPSNR = get_sr_and_score(imset, self.model, min_L=self.config['training']['min_L'])
        return sr, scPSNR
