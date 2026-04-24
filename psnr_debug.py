"""
Detailed PSNR/SSIM debugging script
Run this to see exactly what's being compared and why PSNR might be stuck.
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy import ndimage
import json
from dataclasses import dataclass

# Import our modules
from src.DataLoader import SimpleMicroscopyDataset, collateFunction
from src.DeepNetworks.HRNet import HRNet
from src.diagnostics import compute_metrics

# Create PSNR calculation with full debug output
def compute_psnr_debug(img_true, img_pred, name=""):
    """Compute PSNR with detailed debug info"""
    print(f"\n{'='*70}")
    print(f"PSNR Calculation Debug: {name}")
    print(f"{'='*70}")
    
    print(f"\nInput shapes:")
    print(f"  img_true: {img_true.shape} | dtype: {img_true.dtype} | range: [{img_true.min():.4f}, {img_true.max():.4f}]")
    print(f"  img_pred: {img_pred.shape} | dtype: {img_pred.dtype} | range: [{img_pred.min():.4f}, {img_pred.max():.4f}]")
    
    # Clip to [0, 1]
    img_true_clipped = np.clip(img_true, 0, 1)
    img_pred_clipped = np.clip(img_pred, 0, 1)
    
    print(f"\nAfter clipping to [0, 1]:")
    print(f"  img_true: range: [{img_true_clipped.min():.4f}, {img_true_clipped.max():.4f}]")
    print(f"  img_pred: range: [{img_pred_clipped.min():.4f}, {img_pred_clipped.max():.4f}]")
    
    # Compute MSE
    mse = np.mean((img_true_clipped - img_pred_clipped) ** 2)
    print(f"\nMSE = {mse:.8f}")
    print(f"sqrt(MSE) = {np.sqrt(mse):.8f}")
    
    # Compute PSNR (with data_range=1.0)
    if mse == 0:
        psnr = float('inf')
        print(f"PSNR = ∞ (MSE is zero - images identical)")
    else:
        data_range = 1.0
        psnr = 10 * np.log10(data_range ** 2 / mse)
        print(f"PSNR = 10 * log10({data_range}^2 / {mse:.8f})")
        print(f"PSNR = 10 * log10({1.0 / mse:.2f})")
        print(f"PSNR = {psnr:.2f} dB")
    
    # Analyze difference
    diff = np.abs(img_true_clipped - img_pred_clipped)
    print(f"\nDifference analysis:")
    print(f"  max diff: {diff.max():.6f}")
    print(f"  mean diff: {diff.mean():.6f}")
    print(f"  pixels with diff > 0.1: {(diff > 0.1).sum()} / {diff.size} ({100*(diff > 0.1).sum()/diff.size:.1f}%)")
    print(f"  pixels with diff > 0.01: {(diff > 0.01).sum()} / {diff.size} ({100*(diff > 0.01).sum()/diff.size:.1f}%)")
    
    return psnr

# Main
if __name__ == '__main__':
    print("PSNR/SSIM Debugging Script")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = 'config/config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    print("\n1️⃣ Loading model...")
    model = HRNet(config['network'])
    weights_path = Path("models/weights/HRNet.pth")
    
    if weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state_dict)
        print(f"   ✓ Weights loaded from {weights_path}")
    else:
        print(f"   ✗ No weights found at {weights_path} - using random init")
    
    model.to(device)
    model.eval()
    
    # Load data
    print("\n2️⃣ Loading dataset...")
    dataset_root = Path("D:/GUC/Datasets/HighRes input test")
    scene_dirs = sorted([str(d) for d in dataset_root.iterdir() if d.is_dir()])
    
    if not scene_dirs:
        print(f"   ✗ No scenes found in {dataset_root}")
        sys.exit(1)
    
    print(f"   Found {len(scene_dirs)} scenes")
    print(f"   Using scene: {Path(scene_dirs[0]).name}")
    
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
    
    # Get first batch
    print("\n3️⃣ Running inference...")
    batch = next(iter(dataloader))
    lrs, alphas, hrs, hr_maps, names = batch
    
    print(f"   Batch loaded:")
    print(f"   - LR shape: {lrs.shape}")
    print(f"   - HR shape: {hrs.shape}")
    print(f"   - Scene: {names[0]}")
    
    lrs = lrs.float().to(device)
    alphas = alphas.float().to(device)
    
    with torch.no_grad():
        sr_output = model(lrs, alphas)
    
    print(f"   SR output shape: {sr_output.shape}")
    
    # Extract numpy arrays
    sr = sr_output[0, 0].cpu().numpy()
    lr_first = lrs[0, 0].cpu().numpy()
    
    # Extract HR (with proper squeezing)
    hr_true_raw = hrs[0].numpy() if torch.is_tensor(hrs) else hrs[0]
    if hr_true_raw.ndim == 3:
        hr_true_raw = np.squeeze(hr_true_raw, axis=0)
    
    print(f"   SR extracted: {sr.shape}")
    print(f"   HR extracted: {hr_true_raw.shape}")
    
    # Clip SR
    sr = np.clip(sr, 0, 1)
    
    # Handle shape mismatch
    if sr.shape != hr_true_raw.shape:
        print(f"\n4️⃣ Handling shape mismatch...")
        print(f"   SR: {sr.shape} vs HR: {hr_true_raw.shape}")
        
        # Resample HR to SR size
        scale = sr.shape[0] / hr_true_raw.shape[0]
        print(f"   Scale factor: {scale:.4f}")
        hr_true = ndimage.zoom(hr_true_raw, scale, order=3)
        print(f"   HR resampled to: {hr_true.shape}")
    else:
        hr_true = hr_true_raw
        print(f"\n4️⃣ Shapes already match: {sr.shape}")
    
    # Compute PSNR with debug
    print(f"\n5️⃣ Computing metrics...")
    sr_psnr = compute_psnr_debug(hr_true, sr, "HRNet Output")
    
    # Compute bicubic baseline
    print(f"\n6️⃣ Computing bicubic baseline...")
    bicubic = ndimage.zoom(lr_first, 3, order=3)
    if bicubic.shape != hr_true.shape:
        scale = hr_true.shape[0] / bicubic.shape[0]
        bicubic = ndimage.zoom(bicubic, scale, order=3)
    
    bicubic_psnr = compute_psnr_debug(hr_true, bicubic, "Bicubic Baseline")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"HRNet PSNR:  {sr_psnr:.2f} dB")
    print(f"Bicubic PSNR: {bicubic_psnr:.2f} dB")
    
    if sr_psnr > bicubic_psnr + 0.5:
        print(f"✓ HRNet is {sr_psnr - bicubic_psnr:.2f} dB better than bicubic")
    elif sr_psnr > bicubic_psnr - 0.5:
        print(f"⚠️  HRNet is similar to bicubic (within 0.5 dB)")
        print(f"   This suggests the model may not be properly trained")
    else:
        print(f"✗ HRNet is WORSE than bicubic by {bicubic_psnr - sr_psnr:.2f} dB")
        print(f"   Something is seriously wrong!")
    
    print(f"{'='*70}\n")

