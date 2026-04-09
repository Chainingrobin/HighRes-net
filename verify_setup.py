#!/usr/bin/env python
"""
Quick verification script to test HighRes-Net setup for microscopy inference
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

print("=" * 70)
print("HighRes-Net Microscopy Setup Verification")
print("=" * 70)

# Step 1: Check Python and PyTorch
print("\n[1/5] Checking Python and PyTorch...")
print(f"  Python version: {sys.version.split()[0]}")

try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    - GPU: {torch.cuda.get_device_name(0)}")
        print(f"    - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError as e:
    print(f"  ✗ PyTorch import failed: {e}")
    sys.exit(1)

# Step 2: Check required packages
print("\n[2/5] Checking required packages...")
packages = ['numpy', 'scipy', 'skimage', 'tensorboardX', 'matplotlib']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} (missing)")

# Step 3: Check config
print("\n[3/5] Loading configuration...")
try:
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    print(f"  ✓ Config loaded")
    print(f"    - n_views: {config['training']['n_views']}")
    print(f"    - min_L: {config['training']['min_L']}")
    print(f"    - batch_size: {config['training']['batch_size']}")
except Exception as e:
    print(f"  ✗ Config loading failed: {e}")
    sys.exit(1)

# Step 4: Check DataLoader
print("\n[4/5] Testing SimpleMicroscopyDataset...")
try:
    from src.DataLoader import SimpleMicroscopyDataset, read_microscopy_imageset, collateFunction
    print(f"  ✓ SimpleMicroscopyDataset imported")
    print(f"  ✓ read_microscopy_imageset imported")
    print(f"  ✓ collateFunction imported")
except Exception as e:
    print(f"  ✗ DataLoader import failed: {e}")
    sys.exit(1)

# Step 5: Check model
print("\n[5/5] Testing HRNet model...")
try:
    from src.DeepNetworks.HRNet import HRNet
    print(f"  ✓ HRNet imported")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRNet(config['network'])
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ HRNet instantiated on {device}")
    print(f"    - Total parameters: {total_params:,}")
    
    # Quick forward pass test
    batch_size = config['training']['batch_size']
    n_views = config['training']['n_views']
    dummy_lrs = torch.randn(batch_size, n_views, 64, 64).to(device)
    dummy_alphas = torch.ones(batch_size, n_views).to(device)
    
    with torch.no_grad():
        output = model(dummy_lrs, dummy_alphas)
    
    print(f"  ✓ Forward pass successful")
    print(f"    - Input shape: {dummy_lrs.shape}")
    print(f"    - Output shape: {output.shape}")
    
except Exception as e:
    print(f"  ✗ HRNet test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All checks passed! Setup is ready.")
print("=" * 70)
print("\nNext steps:")
print("1. Prepare your microscopy dataset in structure:")
print("   your_dataset/scene_1/LR_1.png, LR_2.png, ..., LR_7.png, HR.png")
print("2. Update the dataset_root path in notebooks/inference_microscopy.ipynb")
print("3. Run the inference notebook to test on your data")
print("=" * 70)
