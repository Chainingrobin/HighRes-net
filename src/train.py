""" Python script to train HRNet + shiftNet for multi frame super resolution (MFSR) """

import json
import os
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.optim as optim
import argparse
from torch import nn
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from DeepNetworks.HRNet import HRNet
from DeepNetworks.ShiftNet import ShiftNet

from DataLoader import TiffPatchDataset, collateFunction
from Evaluator import shift_cPSNR
from tensorboardX import SummaryWriter


def register_batch(shiftNet, lrs, reference):
    """
    Registers images against references.
    Args:
        shiftNet: torch.model
        lrs: tensor (batch size, views, W, H), images to shift
        reference: tensor (batch size, W, H), reference images to shift
    Returns:
        thetas: tensor (batch size, views, 2)
    """
    
    n_views = lrs.size(1)
    thetas = []
    for i in range(n_views):
        theta = shiftNet(torch.cat([reference, lrs[:, i : i + 1]], 1))
        thetas.append(theta)
    thetas = torch.stack(thetas, 1)

    return thetas


def apply_shifts(shiftNet, images, thetas, device):
    """
    Applies sub-pixel translations to images with Lanczos interpolation.
    Args:
        shiftNet: torch.model
        images: tensor (batch size, views, W, H), images to shift
        thetas: tensor (batch size, views, 2), translation params
    Returns:
        new_images: tensor (batch size, views, W, H), warped images
    """
    
    batch_size, n_views, height, width = images.shape
    images = images.view(-1, 1, height, width)
    thetas = thetas.view(-1, 2)
    new_images = shiftNet.transform(thetas, images, device=device)

    return new_images.view(-1, n_views, images.size(2), images.size(3))


def get_loss(srs, hrs, hr_maps, metric='cMSE'):
    """
    Computes ESA loss for each instance in a batch.
    Args:
        srs: tensor (B, W, H), super resolved images
        hrs: tensor (B, W, H), high-res images
        hr_maps: tensor (B, W, H), high-res status maps
    Returns:
        loss: tensor (B), metric for each super resolved image.
    """
    
    # ESA Loss: https://kelvins.esa.int/proba-v-super-resolution/scoring/
    criterion = nn.MSELoss(reduction='none')
    if metric == 'masked_MSE':
        loss = criterion(hr_maps * srs, hr_maps * hrs)
        return torch.mean(loss, dim=(1, 2))
    nclear = torch.sum(hr_maps, dim=(1, 2))  # Number of clear pixels in target image
    bright = torch.sum(hr_maps * (hrs - srs), dim=(1, 2)).clone().detach() / nclear  # Correct for brightness
    loss = torch.sum(hr_maps * criterion(srs + bright.view(-1, 1, 1), hrs), dim=(1, 2)) / nclear  # cMSE(A,B) for each point
    if metric == 'cMSE':
        return loss
    return -10 * torch.log10(loss)  # cPSNR


def get_crop_mask(patch_size, crop_size, upscale=2):
    """
    Computes a mask to crop borders.
    Args:
        patch_size: int, size of LR patches
        crop_size: int, size to crop (border in upscaled space)
        upscale: int, upscaling factor (default 2 for 2x SR)
    Returns:
        torch_mask: tensor (1, 1, upscale*patch_size, upscale*patch_size), mask
    """
    
    sr_size = upscale * patch_size
    mask = np.ones((1, 1, sr_size, sr_size))  # crop_mask for loss (B, C, W, H)
    mask[0, 0, :crop_size, :] = 0
    mask[0, 0, -crop_size:, :] = 0
    mask[0, 0, :, :crop_size] = 0
    mask[0, 0, :, -crop_size:] = 0
    torch_mask = torch.from_numpy(mask).type(torch.FloatTensor)
    return torch_mask


def trainAndGetBestModel(fusion_model, regis_model, optimizer, dataloaders, config):
    """
    Trains HRNet and ShiftNet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        fusion_model: torch.model, HRNet
        regis_model: torch.model, ShiftNet
        optimizer: torch.optim, optimizer to minimize loss
        dataloaders: dict, wraps train and validation dataloaders
        config: dict, configuration file
    """
    np.random.seed(123)  # seed all RNGs for reproducibility
    torch.manual_seed(123)

    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    n_views = config["training"]["n_views"]
    min_L = config["training"]["min_L"]  # minimum number of views

    subfolder_pattern = 'batch_{}_views_{}_min_{}_time_{}'.format(
        batch_size, n_views, min_L, f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}")

    checkpoint_dir_run = os.path.join(config["paths"]["checkpoint_dir"], subfolder_pattern)
    os.makedirs(checkpoint_dir_run, exist_ok=True)

    tb_logging_dir = config['paths']['tb_log_file_dir']
    logging_dir = os.path.join(tb_logging_dir, subfolder_pattern)
    os.makedirs(logging_dir, exist_ok=True)

    writer = SummaryWriter(logging_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_cfg = config["training"]
    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == 'cuda'
    amp_dtype_name = str(train_cfg.get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ("bf16", "bfloat16") else torch.float16
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    best_score = 100

    P = config["training"]["patch_size"]
    UPSCALE = 2  # 2x upsampling for TIFF dataset (128 LR -> 256 SR)
    offset = (UPSCALE * config["training"]["patch_size"] - 128) // 2
    C = config["training"]["crop"]
    torch_mask = get_crop_mask(patch_size=P, crop_size=C, upscale=UPSCALE)
    torch_mask = torch_mask.to(device)  # crop borders (loss)

    fusion_model.to(device)
    regis_model.to(device)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['training']['lr_decay'],
                                               verbose=True, patience=config['training']['lr_step'])

    for epoch in tqdm(range(1, num_epochs + 1)):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        # Train
        fusion_model.train()
        regis_model.train()
        train_loss = 0.0  # monitor train loss

        # Iterate over data.
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (lrs, alphas, hrs, hr_maps, names) in enumerate(tqdm(dataloaders['train']), start=1):

            lrs = lrs.float().to(device)
            alphas = alphas.float().to(device)
            hr_maps = hr_maps.float().to(device)
            hrs = hrs.float().to(device)

            amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp) if device.type == 'cuda' else nullcontext()
            with amp_ctx:
                # torch.autograd.set_detect_anomaly(mode=True)
                srs = fusion_model(lrs, alphas)  # fuse multi frames (B, 1, 3*W, 3*H)

                # Register batch wrt HR
                shifts = register_batch(regis_model,
                                        srs[:, :, offset:(offset + 128), offset:(offset + 128)],
                                        reference=hrs[:, offset:(offset + 128), offset:(offset + 128)].view(-1, 1, 128, 128))
                srs_shifted = apply_shifts(regis_model, srs, shifts, device)[:, 0]

                # Training loss
                cropped_mask = torch_mask[0] * hr_maps  # Compute current mask (Batch size, W, H)
                # srs_shifted = torch.clamp(srs_shifted, min=0.0, max=1.0)  # correct over/under-shoots
                loss = -get_loss(srs_shifted, hrs, cropped_mask, metric='cPSNR')
                loss = torch.mean(loss)
                # Backward-compatible: prefer explicit shift regularization key.
                lambda_shift_reg = config["training"].get("lambda_shift_reg", config["training"].get("lambda", 0.0))
                loss += lambda_shift_reg * torch.mean(shifts)**2

            # Backprop
            loss_for_log = loss.detach()
            loss = loss / grad_accum_steps
            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step_idx % grad_accum_steps == 0 or step_idx == len(dataloaders['train']):
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss = loss_for_log.cpu().numpy() * len(hrs) / len(dataloaders['train'].dataset)
            train_loss += epoch_loss

        # Eval
        fusion_model.eval()
        regis_model.eval()
        val_score = 0.0  # monitor val score
        last_srs = None
        last_hrs = None

        with torch.no_grad():
            for lrs, alphas, hrs, hr_maps, names in dataloaders['val']:
                lrs = lrs.float().to(device)
                alphas = alphas.float().to(device)
                hrs = hrs.numpy()
                hr_maps = hr_maps.numpy()

                amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp) if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    srs = fusion_model(lrs, alphas)[:, 0]  # fuse multi frames (B, 1, 3*W, 3*H)

                # compute validation score (negative cPSNR for loss comparison)
                srs = srs.detach().cpu().numpy()
                last_srs = srs
                last_hrs = hrs
                for i in range(srs.shape[0]):  # batch size
                    val_score -= shift_cPSNR(np.clip(srs[i], 0, 1), hrs[i], hr_maps[i])

        val_score /= len(dataloaders['val'].dataset)

        if best_score > val_score:
            torch.save(fusion_model.state_dict(),
                       os.path.join(checkpoint_dir_run, 'HRNet.pth'))
            torch.save(regis_model.state_dict(),
                       os.path.join(checkpoint_dir_run, 'ShiftNet.pth'))
            best_score = val_score

        if last_srs is not None and last_hrs is not None:
            writer.add_image('SR Image', (last_srs[0] - np.min(last_srs[0])) / np.max(last_srs[0]), epoch, dataformats='HW')
            error_map = last_hrs[0] - last_srs[0]
            writer.add_image('Error Map', error_map, epoch, dataformats='HW')
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/val_loss", val_score, epoch)
        if device.type == 'cuda':
            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            writer.add_scalar("train/peak_vram_gb", peak_vram_gb, epoch)
        scheduler.step(val_score)
    writer.close()


def main(config):
    """
    Given a configuration, trains HRNet and ShiftNet for Multi-Frame Super Resolution (MFSR), and saves best model.
    Args:
        config: dict, configuration file
    """
    
    # Reproducibility options
    np.random.seed(0)  # RNG seeds
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the network based on the network configuration
    fusion_model = HRNet(config["network"])
    regis_model = ShiftNet()

    optimizer = optim.Adam(list(fusion_model.parameters()) + list(regis_model.parameters()), lr=config["training"]["lr"])  # optim
    
    # Load TIFF patch dataset
    data_directory = config["paths"]["prefix"]
    train_dir = os.path.join(data_directory, "train")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    
    # Discover all patch directories
    patch_dirs = sorted([os.path.join(train_dir, d) for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
    
    if len(patch_dirs) == 0:
        raise FileNotFoundError(f"No patch directories found in {train_dir}")
    
    # Split into train/val
    val_proportion = config['training']['val_proportion']
    train_list, val_list = train_test_split(patch_dirs,
                                            test_size=val_proportion,
                                            random_state=1, shuffle=True)

    # Dataloaders
    batch_size = config["training"]["batch_size"]
    n_workers = config["training"]["n_workers"]
    n_views = config["training"]["n_views"]
    min_L = config["training"]["min_L"]

    train_dataset = TiffPatchDataset(patch_dirs=train_list, max_views=n_views)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=n_workers,
                                  collate_fn=collateFunction(min_L=min_L),
                                  pin_memory=True)

    val_dataset = TiffPatchDataset(patch_dirs=val_list, max_views=n_views)
    val_dataloader = DataLoader(val_dataset, batch_size=1,
                                shuffle=False, num_workers=n_workers,
                                collate_fn=collateFunction(min_L=min_L),
                                pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Train model
    torch.cuda.empty_cache()

    trainAndGetBestModel(fusion_model, regis_model, optimizer, dataloaders, config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path of the config file", default='config/config.json')
    parser.add_argument("--lambda_reg", type=float, default=None,
                        help="Override training lambda regularization term.")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override number of epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate.")

    args = parser.parse_args()
    assert os.path.isfile(args.config)

    with open(args.config, "r") as read_file:
        config = json.load(read_file)

    # Optional CLI overrides for fast experiment sweeps without editing config files.
    if args.lambda_reg is not None:
        config["training"]["lambda"] = args.lambda_reg
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    print("\nEffective training hyperparameters:")
    print(f"  lambda: {config['training']['lambda']}")
    print(f"  num_epochs: {config['training']['num_epochs']}")
    print(f"  batch_size: {config['training']['batch_size']}")
    print(f"  lr: {config['training']['lr']}")

    main(config)
