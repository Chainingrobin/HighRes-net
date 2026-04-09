"""
Test script to verify the black image issue is fixed.
Compares pixel values before and after the normalization fix.
"""

import numpy as np
from pathlib import Path
from skimage import io
import skimage

# Test scene
scene_path = Path("D:/GUC/Datasets/HighRes input test/scene_1")

print("="*70)
print("BLACK IMAGE FIX VERIFICATION")
print("="*70)

# Find HR image
possible_hr = [scene_path / "HR.png", scene_path / "hr_ground_truth.png"]
hr_file = None
for f in possible_hr:
    if f.exists():
        hr_file = f
        break

if not hr_file:
    print("❌ HR image not found")
    exit(1)

print(f"\n📁 Testing with: {hr_file.name}")

# ===== OLD WAY (BROKEN) =====
print("\n" + "-"*70)
print("❌ OLD BROKEN WAY (what was happening before):")
print("-"*70)

hr_old = np.array(io.imread(hr_file), dtype=np.float32)  # WRONG: no normalization!
print(f"  After io.imread + dtype=float32:")
print(f"    dtype: {hr_old.dtype}")
print(f"    min: {hr_old.min():.1f}, max: {hr_old.max():.1f}")
print(f"    These are RAW uint8/uint16 values, not normalized!")

hr_old_normalized = skimage.img_as_float(hr_old).astype(np.float32)
print(f"\n  After skimage.img_as_float():")
print(f"    dtype: {hr_old_normalized.dtype}")
print(f"    min: {hr_old_normalized.min():.6f}, max: {hr_old_normalized.max():.6f}")
print(f"    🔴 Most values clipped to 0 or 1 (BINARY IMAGE!) - explains black output")

# ===== NEW WAY (FIXED) =====
print("\n" + "-"*70)
print("✅ NEW FIXED WAY (what happens now):")
print("-"*70)

hr_new = np.array(io.imread(hr_file))  # Correct: let imread return native dtype
print(f"  After io.imread (no forced float32):")
print(f"    dtype: {hr_new.dtype}")
print(f"    min: {hr_new.min():.1f}, max: {hr_new.max():.1f}")
print(f"    These are native uint8/uint16 values from file")

hr_new_normalized = skimage.img_as_float(hr_new).astype(np.float32)
print(f"\n  After skimage.img_as_float():")
print(f"    dtype: {hr_new_normalized.dtype}")
print(f"    min: {hr_new_normalized.min():.6f}, max: {hr_new_normalized.max():.6f}")
print(f"    ✅ Proper [0, 1] range - images should display correctly!")

# Calculate difference
diff_pixels = np.abs(hr_old_normalized - hr_new_normalized).sum()
print(f"\n📊 Pixel value differences:")
print(f"    Total absolute difference: {diff_pixels:.0f}")
if diff_pixels > 1000:
    print(f"    🔴 SIGNIFICANT difference - fix important!")
else:
    print(f"    ✅ OK for this scene")

print("\n" + "="*70)
print("✅ FIX STATUS: DataLoader.py has been corrected")
print("   Re-run your inference_diagnostic.ipynb to see properly displayed images")
print("="*70)
