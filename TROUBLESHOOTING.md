# Comprehensive Troubleshooting Guide

## Your Issues & Solutions

### Issue 1: GPU Not Available + Storage Error

**Symptoms:**

```
PyTorch: 2.11.0+cpu
CUDA available: False
OSError: [Errno 28] No space left on device
```

**Root Cause:**

- PyTorch CPU version (not GPU)
- GPU wheel ~2.8 GB needs to download to C: drive
- C: drive is full, download fails

**Solution:** Follow [GPU_INSTALLATION.md](GPU_INSTALLATION.md)

- Set TMP, TEMP, PIP_CACHE_DIR to D: drive
- Install CUDA 11.8 PyTorch to D: drive
- **Time needed:** 10 minutes
- **Expected result:** `CUDA available: True`

---

### Issue 2: SR Output Looks Bad (Grainy, Dark, Artifacts)

**Visual symptoms:**

- Output looks noisier than input
- Darker than expected
- Visible scan lines / artifacts
- Worse quality than simple bicubic upsampling

**Root Cause:**

- **Model is using RANDOM weights (untrained)**
- Random initialization = random noise output
- Nothing to learn from, no meaningful features

**Proof:**

```python
# This shows the issue:
import torch
from pathlib import Path

weights_path = Path("models/weights/HRNet.pth")
print(f"Weights file exists: {weights_path.exists()}")

# If False, that's your problem!
```

**Solution:** Choose one option:

#### Option A: Quick - Use Bicubic Baseline (No Training)

```python
from scipy import ndimage
import numpy as np

lr_image = ...  # Your LR image
bicubic_sr = ndimage.zoom(lr_image, 3, order=3)

# This will look better than random NN weights
# Gives you baseline to compare against
```

**Time:** 10 minutes  
**Quality:** 24-25 dB PSNR (reference baseline)

#### Option B: Best - Train the Model

```python
# Follow DOCUMENTATION.md → Training Guide section
python src/train.py --config config/config.json
```

**Time:** 1-2 hours (with GPU)  
**Quality:** 25-28 dB PSNR (better than bicubic)  
**Result:** Model learns actual features from your data

#### Option C: Search for Pre-trained Weights

- Look for microscopy/MFSR pre-trained models
- Download `.pth` file
- Place in `models/weights/HRNet.pth`

**Recommendation:** Option B (training) - you have the data!

---

### Issue 3: HR Ground Truth Shows as Black

**Symptoms:**

```
HR image shape: (512, 512)
HR value range: [0, 0]    # All zeros!
HR is completely black
```

**Root Cause:**
Multiple possibilities:

1. File actually IS black (image data was lost)
2. File won't load with scikit-image
3. Normalization issue with uint8→float conversion

**Diagnosis - Run this:**

```python
from pathlib import Path
from skimage import io
from PIL import Image
import numpy as np

hr_path = Path("D:/GUC/Datasets/HighRes input test/scene_1/HR.png")

# Check file size first
print(f"File exists: {hr_path.exists()}")
print(f"File size: {hr_path.stat().st_size} bytes")

if hr_path.exists():
    # Method 1: scikit-image
    hr_ski = io.imread(str(hr_path))
    print(f"scikit-image: shape={hr_ski.shape}, min={hr_ski.min()}, max={hr_ski.max()}")

    # Method 2: PIL
    hr_pil = np.array(Image.open(str(hr_path)))
    print(f"PIL: shape={hr_pil.shape}, min={hr_pil.min()}, max={hr_pil.max()}")

    # Check if file is corrupted
    if hr_ski.max() == 0 and hr_pil.max() == 0:
        print("🔴 Both methods return black - file is corrupted or all zeros")
```

**Solutions:**

1. **File is actually black:** Regenerate HR image properly

   ```python
   # If synthetically generating:
   from scipy import ndimage
   from skimage import io

   lr = io.imread("LR_1.png")  # Load one LR
   hr = ndimage.zoom(lr, 3, order=3)  # 4x upsampling
   io.imsave("HR.png", hr)
   ```

2. **File won't load:** Try different loader

   ```python
   from PIL import Image
   hr = np.array(Image.open(str(hr_path)))
   # Now use this instead
   ```

3. **Normalization issue:** Fix in data loading
   ```python
   from skimage import img_as_float
   hr_float = img_as_float(hr_uint8)
   # Check range - should be [0, 1] now
   ```

**Run diagnostic to find exact issue:**

```python
# In your notebook:
from pathlib import Path
import sys
sys.path.insert(0, '../src')

from diagnostics import check_image_loading

results = check_image_loading(Path("D:/GUC/Datasets/HighRes input test/scene_1"))
```

---

### Issue 4: Unsure About 3x vs 4x Upsampling

**Symptom:**

```
LR size: 128×128
HR size: 512×512
Scale: 512/128 = 4x

But model expects 3x (stride=3 in config)
```

**Solutions:**

#### Option A: Resize HR to 3x (quick, no retraining)

```python
from scipy import ndimage
from skimage import io

lr = io.imread("LR_1.png")  # 128×128
hr_4x_current = io.imread("HR.png")  # 512×512

# Resize to 3x
hr_3x = ndimage.zoom(hr_4x_current, (384/512, 384/512), order=3)
# Now hr_3x is 384×384 (3x upsampling)

io.imsave("HR.png", hr_3x)  # Overwrite
```

- **Pro:** Model works as-is
- **Con:** You're downsampling your data

#### Option B: Retrain with 4x stride (proper, needs retraining)

```json
// In config/config.json:
{
  "decoder": {
    "deconv": {
      "stride": 4 // Changed from 3
    }
  }
}
```

Then retrain: `python src/train.py --config config/config.json`

- **Pro:** Uses full resolution
- **Con:** Model trained differently than original code

**Recommendation:** Option A (quick fix) unless you have >20 scenes for retraining

---

## Decision Tree: What to Do First

```
START
  │
  ├─→ Do you have GPU access? (Verify with GPU_INSTALLATION.md)
  │   NO  → Install GPU support first (1 hour)
  │   YES → Continue
  │
  ├─→ Can you visualize HR.png (not black)?
  │   NO  → Fix image loading (Issue 3 diagnostic)
  │   YES → Continue
  │
  ├─→ Do you have trained model weights?
  │   YES → Skip to inference testing
  │   NO  → Choose below:
  │
  │   OPTION 1: Quick test (30 min)
  │   └─→ Run bicubic baseline + metrics
  │       → Gives you reference PSNR ~24 dB
  │
  │   OPTION 2: Proper training (2-3 hours)
  │   └─→ Follow DOCUMENTATION.md → Training Guide
  │       → Model learns, PSNR →25-28 dB
  │
  │   OPTION 3: Find pre-trained (variable)
  │   └─→ Search online for microscopy MFSR model
  │       → Use if available
```

---

## Performance Checklist

### Before Training

- [ ] GPU installed (`CUDA available: True`)
- [ ] Dataset organized (7 LR per scene + HR)
- [ ] HR images load correctly (range [0, 255] or [0, 1])
- [ ] Scale is 3x or 4x consistent
- [ ] `verify_setup.py` passes all checks

### After Training

- [ ] Training loss decreases in first epoch
- [ ] Validation PSNR improves over epochs
- [ ] No NaN or Inf values
- [ ] Checkpoint saved to `models/weights/HRNet.pth`
- [ ] Inference tests produce 25-28 dB PSNR

### Production

- [ ] PSNR > baseline bicubic (24 dB)
- [ ] Visual inspection acceptable (no artifacts)
- [ ] GPU memory < 8 GB during inference
- [ ] Inference speed acceptable (>5 images/sec with GPU)

---

## Quick Reference: Running Diagnostics

### 1. Image Loading

```python
from src.diagnostics import check_image_loading
check_image_loading(Path("D:/GUC/Datasets/HighRes input test/scene_1"))
```

### 2. Model Weights Status

```python
from src.diagnostics import check_model_weights
check_model_weights(Path("models/weights/HRNet.pth"))
```

### 3. Full Diagnostic (Recommended)

Run notebook: `notebooks/inference_diagnostic.ipynb`

- Checks everything
- Computes PSNR/SSIM
- Shows visualizations
- Compares with bicubic baseline

### 4. System Check

```bash
python verify_setup.py
```

---

## Timeline to Production

| Task             | Time       | Dependencies          |
| ---------------- | ---------- | --------------------- |
| Install GPU      | 15 min     | Disk space (D:)       |
| Diagnostic run   | 5 min      | GPU installed         |
| Fix HR loading   | 10-30 min  | Diagnostic output     |
| Bicubic baseline | 10 min     | Working data          |
| **Train model**  | **90 min** | **Data organized**    |
| **Evaluate**     | **10 min** | **(trained weights)** |
| Deploy           | 5 min      | Checkpoint saved      |

**Fastest path to >25 dB PSNR:** ~2 hours (GPU + training)

---

## Recommended Order

1. **Install GPU** (GPU_INSTALLATION.md)

   ```powershell
   # 15 minutes
   $env:TMP = "D:\pip_temp"
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Run Diagnostics** (inference_diagnostic.ipynb)

   ```python
   # 5 minutes, identifies issues
   ```

3. **Fix Data Issues** (if any from diagnostics)

   ```python
   # 10-30 minutes, makes HR readable, organizes for training
   ```

4. **Train Model** (See DOCUMENTATION.md)

   ```bash
   # 90 minutes with GPU, produces better model
   python src/train.py --config config/config.json
   ```

5. **Evaluate** (inference_diagnostic.ipynb again)
   ```python
   # 10 minutes, shows improvement vs random weights
   ```

---

## When You're Stuck

If something doesn't work:

1. **Check the error message** - What exact error?
2. **Read diagnostic output** - Run `check_image_loading()` or `check_model_weights()`
3. **Check file existence** - Are files actually there?
4. **Verify configuration** - Is config.json valid JSON?
5. **Test on simple example** - Single-file test first
6. **Ask for help with context:**
   - Error message (full traceback)
   - Output from `verify_setup.py`
   - Output from diagnostic notebook

---

## Further Help

Files in this repository:

- `GPU_INSTALLATION.md` - Fix GPU/storage issues
- `DOCUMENTATION.md` - Training Guide section for better results
- `src/diagnostics.py` - Diagnostic utilities
- `notebooks/inference_diagnostic.ipynb` - Complete diagnostics
- `verify_setup.py` - System verification script

Contact/Support:

- Run diagnostics: `notebooks/inference_diagnostic.ipynb`
- Check GPU: `python verify_setup.py`
- Try bicubic baseline before training

---

**Next action:** Choose your path above (GPU → Diagnostics → Train → Evaluate)
