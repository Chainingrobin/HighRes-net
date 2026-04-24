"""
Compare trained weights vs random weights
This will definitively show whether training made a difference
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy import ndimage
import json

from src.DataLoader import SimpleMicroscopyDataset, collateFunction
from src.DeepNetworks.HRNet import HRNet
from src.diagnostics import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = 'config/config.json'

with open(config_path, 'r') as f:
    config = json.load(f)

print("="*70)
print("TRAINED vs RANDOM WEIGHTS COMPARISON")
print("="*70)

# Setup
dataset_root = Path("D:/GUC/Datasets/HighRes input test")
scene_dirs = sorted([str(d) for d in dataset_root.iterdir() if d.is_dir()])

dataset = SimpleMicroscopyDataset(
    imset_dirs=scene_dirs,
    config=config['training'],
    max_views=config['training']['n_views']
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collateFunction(min_L=config['training']['min_L']),
    pin_memory=torch.cuda.is_available()
)

batch = next(iter(dataloader))
lrs, alphas, hrs, hr_maps, names = batch

lrs = lrs.float().to(device)
alphas = alphas.float().to(device)

# Extract HR
hr_true_raw = hrs[0].numpy() if torch.is_tensor(hrs) else hrs[0]
if hr_true_raw.ndim == 3:
    hr_true_raw = np.squeeze(hr_true_raw, axis=0)

print(f"\nTest data: {names[0]}")
print(f"  Ground truth range: [{hr_true_raw.min():.4f}, {hr_true_raw.max():.4f}]")

# TEST 1: Trained weights
print(f"\n{'='*70}")
print("TEST 1: TRAINED WEIGHTS")
print(f"{'='*70}")

model_trained = HRNet(config['network']).to(device)
model_trained.eval()

weights_path = Path("models/weights/HRNet.pth")
print(f"\nLoading: {weights_path}")
state_dict = torch.load(str(weights_path), map_location=device)
model_trained.load_state_dict(state_dict)

with torch.no_grad():
    sr_trained = model_trained(lrs, alphas)[0, 0].cpu().numpy()
    sr_trained = np.clip(sr_trained, 0, 1)

print(f"Output range: [{sr_trained.min():.4f}, {sr_trained.max():.4f}]")
print(f"Output span: {sr_trained.max() - sr_trained.min():.4f}")

# TEST 2: Random weights
print(f"\n{'='*70}")
print("TEST 2: RANDOM WEIGHTS (fresh model, no training)")
print(f"{'='*70}")

torch.manual_seed(42)  # Fixed seed for reproducibility
model_random = HRNet(config['network']).to(device)
model_random.eval()

print(f"Model created with random initialization")

with torch.no_grad():
    sr_random = model_random(lrs, alphas)[0, 0].cpu().numpy()
    sr_random = np.clip(sr_random, 0, 1)

print(f"Output range: [{sr_random.min():.4f}, {sr_random.max():.4f}]")
print(f"Output span: {sr_random.max() - sr_random.min():.4f}")

# Handle shape mismatch
if hr_true_raw.shape != sr_trained.shape:
    scale = sr_trained.shape[0] / hr_true_raw.shape[0]
    hr_true = ndimage.zoom(hr_true_raw, scale, order=3)
else:
    hr_true = hr_true_raw

# COMPARISON
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")

metrics_trained = compute_metrics(hr_true, sr_trained)
metrics_random = compute_metrics(hr_true, sr_random)

print(f"\nTRAINED MODEL:")
print(f"  PSNR: {metrics_trained['psnr']:.2f} dB")
print(f"  SSIM: {metrics_trained['ssim']:.4f}")
print(f"  MSE:  {metrics_trained['mse']:.6f}")

print(f"\nRANDOM MODEL:")
print(f"  PSNR: {metrics_random['psnr']:.2f} dB")
print(f"  SSIM: {metrics_random['ssim']:.4f}")
print(f"  MSE:  {metrics_random['mse']:.6f}")

print(f"\nDIFFERENCE:")
psnr_diff = metrics_trained['psnr'] - metrics_random['psnr']
ssim_diff = metrics_trained['ssim'] - metrics_random['ssim']
mse_diff = metrics_random['mse'] - metrics_trained['mse']

print(f"  PSNR: {psnr_diff:+.2f} dB", end="")
if psnr_diff > 1.0:
    print(" ✓ TRAINED IS BETTER")
elif psnr_diff > 0:
    print(" ⚠️  TRAINED slightly better")
elif psnr_diff > -1.0:
    print(" ⚠️  SIMILAR (suggests training may have failed)")
else:
    print(" ✗ TRAINED IS WORSE")

print(f"  SSIM: {ssim_diff:+.4f}", end="")
if ssim_diff > 0.02:
    print(" ✓ TRAINED IS BETTER")
elif ssim_diff > 0:
    print(" ⚠️  TRAINED slightly better")
elif ssim_diff > -0.02:
    print(" ⚠️  SIMILAR (suggests training may have failed)")
else:
    print(" ✗ TRAINED IS WORSE")

print(f"  MSE: {mse_diff:+.6f}", end="")
if mse_diff > 0.001:
    print(" ✓ TRAINED IS BETTER")
else:
    print(" ⚠️  SIMILAR or WORSE")

# Final verdict
print(f"\n{'='*70}")
if psnr_diff < 0.5:
    print("🔴 VERDICT: Training did NOT work!")
    print("   Trained model performs similar to or worse than random")
    print("   Possible causes:")
    print("   - Weights file from before training")
    print("   - Training crashed and didn't save")
    print("   - Training corrupted the weights somehow")
    print("   - Model collapsed during training")
else:
    print("✓ VERDICT: Training worked!")
    print(f"   Improvement: {psnr_diff:.2f} dB PSNR and {ssim_diff:.4f} SSIM")
print(f"{'='*70}")
