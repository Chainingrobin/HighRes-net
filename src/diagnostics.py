"""
Diagnostic utilities for HighRes-Net inference troubleshooting
"""

import numpy as np
from pathlib import Path
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch


def _detect_naming_scheme(scene_path):
    """
    Detect which naming scheme is used in the dataset.
    Returns: 'standard' (LR_*.png, HR.png) or 'alternate' (lr_frame_*.png, hr_ground_truth.png)
    """
    scene_path = Path(scene_path)
    
    standard_lr = list(scene_path.glob("LR_*.png"))
    alternate_lr = list(scene_path.glob("lr_frame_*.png"))
    
    if len(standard_lr) > 0:
        return 'standard'
    elif len(alternate_lr) > 0:
        return 'alternate'
    else:
        return 'unknown'


def check_image_loading(scene_path):
    """
    Diagnose why images might appear black or fail to load properly.
    Supports both 'LR_*.png + HR.png' and 'lr_frame_*.png + hr_ground_truth.png' naming schemes.
    Also handles case-insensitive filenames.
    
    Args:
        scene_path: str or Path, path to scene directory
        
    Returns:
        dict with diagnostic info
    """
    scene_path = Path(scene_path)
    results = {
        'lr_images': [],
        'hr_image': None,
        'has_issues': False,
        'warnings': [],
        'naming_scheme': None
    }
    
    # Detect naming scheme
    scheme = _detect_naming_scheme(scene_path)
    results['naming_scheme'] = scheme
    
    print(f"\n{'='*70}")
    print(f"Image Loading Diagnostic for: {scene_path.name}")
    print(f"Detected naming scheme: {scheme}")
    print(f"{'='*70}")
    
    # Check LR images
    if scheme == 'standard':
        lr_files = sorted(scene_path.glob("LR_*.png"))
    elif scheme == 'alternate':
        lr_files = sorted(scene_path.glob("lr_frame_*.png"))
    else:
        results['warnings'].append("Unknown naming scheme")
        results['has_issues'] = True
        print(f"✗ Could not detect naming scheme")
        return results
    
    print(f"\nLR Images found: {len(lr_files)}")
    
    if len(lr_files) == 0:
        results['warnings'].append("No LR images found")
        results['has_issues'] = True
    
    for lr_file in lr_files[:3]:  # Check first 3
        try:
            img = io.imread(str(lr_file))
            info = {
                'name': lr_file.name,
                'shape': img.shape,
                'dtype': str(img.dtype),
                'min': int(img.min()),
                'max': int(img.max()),
                'mean': float(img.mean()),
                'valid': True
            }
            results['lr_images'].append(info)
            
            print(f"  ✓ {lr_file.name}")
            print(f"      Shape: {img.shape}, Dtype: {img.dtype}")
            print(f"      Range: [{img.min()}, {img.max()}], Mean: {img.mean():.1f}")
            
        except Exception as e:
            results['warnings'].append(f"{lr_file.name}: {str(e)}")
            results['has_issues'] = True
            print(f"  ✗ {lr_file.name}: {e}")
    
    # Check HR image with flexible naming
    print(f"\nHR Image (checking multiple naming conventions):")
    
    possible_hr_names = [
        'HR.png', 'hr.png', 'HR.PNG', 'hr.PNG',
        'hr_ground_truth.png', 'HR_ground_truth.png', 'hr_ground_truth.PNG',
        'hr_ground_truth.PNG', 'HR_ground_truth.PNG',
        'ground_truth.png', 'ground_truth.PNG',
        'GroundTruth.png', 'groundtruth.png'
    ]
    
    hr_file = None
    for hr_name in possible_hr_names:
        test_file = scene_path / hr_name
        if test_file.exists():
            hr_file = test_file
            break
    
    if hr_file is None:
        results['warnings'].append(f"HR image not found - tried: {', '.join(possible_hr_names)}")
        results['has_issues'] = True
        print(f"  ✗ HR image NOT FOUND")
        all_files = list(scene_path.glob("*"))
        print(f"  Files in directory: {[f.name for f in all_files]}")
    else:
        try:
            hr_raw = io.imread(str(hr_file))
            results['hr_image'] = {
                'name': hr_file.name,
                'shape': hr_raw.shape,
                'dtype': str(hr_raw.dtype),
                'min': int(hr_raw.min()),
                'max': int(hr_raw.max()),
                'mean': float(hr_raw.mean()),
                'unique_values': int(np.unique(hr_raw).size),
                'is_black': hr_raw.max() == 0
            }
            
            print(f"  ✓ {hr_file.name}")
            print(f"      Shape: {hr_raw.shape}, Dtype: {hr_raw.dtype}")
            print(f"      Range: [{hr_raw.min()}, {hr_raw.max()}], Mean: {hr_raw.mean():.1f}")
            print(f"      Unique values: {np.unique(hr_raw).size}")
            
            if hr_raw.max() == 0:
                results['warnings'].append("HR image is completely BLACK (all zeros)")
                results['has_issues'] = True
                print(f"  🔴 WARNING: HR image is all black!")
            
            if hr_raw.max() < 10:
                results['warnings'].append(f"HR image has very low values (max={hr_raw.max()})")
                print(f"  ⚠️ WARNING: HR has very low values (check loading)")
            
            
            
        except Exception as e:
            results['warnings'].append(f"Failed to load HR.png: {str(e)}")
            results['has_issues'] = True
            print(f"  ✗ Error loading HR.png: {e}")
    
    # Check scale relationship
    if len(results['lr_images']) > 0 and results['hr_image'] is not None:
        lr_h, lr_w = results['lr_images'][0]['shape'][:2]
        hr_h, hr_w = results['hr_image']['shape'][:2]
        
        scale_h = hr_h / lr_h if lr_h > 0 else 0
        scale_w = hr_w / lr_w if lr_w > 0 else 0
        
        print(f"\nScale Relationship:")
        print(f"  LR: {lr_h}×{lr_w} → HR: {hr_h}×{hr_w}")
        print(f"  Computed scale: {scale_h:.1f}x (H) × {scale_w:.1f}x (W)")
        
        if abs(scale_h - 3.0) > 0.1 or abs(scale_w - 3.0) > 0.1:
            results['warnings'].append(
                f"Scale is {scale_h:.1f}x, not 3x. Model expects 3x."
            )
            print(f"  ⚠️ WARNING: Scale is not 3x (model expects 3x)")
        
        results['scale'] = {'height': float(scale_h), 'width': float(scale_w)}
    
    print(f"\n{'='*70}")
    if results['has_issues']:
        print("⚠️ Issues found - see warnings above")
    else:
        print("✓ No issues detected")
    print(f"{'='*70}\n")
    
    return results


def compute_psnr(img_true, img_pred, data_range=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) using scikit-image
    
    Args:
        img_true: numpy array, reference image (float 0-1)
        img_pred: numpy array, predicted image (float 0-1)
        data_range: float, dynamic range of image (1.0 for normalized [0,1])
        
    Returns:
        float, PSNR in dB (higher is better, typical >25 dB is good)
        
    Formula: PSNR = 10 * log10(MAX^2 / MSE)
    where MAX is the maximum possible pixel value (data_range)
    and MSE is the mean squared error between images
    
    Example:
        - 20 dB: Poor quality, images quite different
        - 25 dB: Acceptable
        - 30 dB: Good quality
        - 35+ dB: Very good/excellent
    """
    if img_true is None or img_pred is None:
        return None
    
    img_true = np.asarray(img_true, dtype=np.float32)
    img_pred = np.asarray(img_pred, dtype=np.float32)
    
    img_true = np.clip(img_true, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)
    
    # Use scikit-image's built-in PSNR function
    try:
        psnr = peak_signal_noise_ratio(img_true, img_pred, data_range=data_range)
        return psnr
    except Exception as e:
        print(f"Error computing PSNR: {e}")
        return None


def compute_ssim(img_true, img_pred, data_range=1.0):
    """
    Compute Structural Similarity Index Measure (SSIM) using scikit-image
    
    Args:
        img_true: numpy array, reference image (float 0-1)
        img_pred: numpy array, predicted image (float 0-1)
        data_range: float, dynamic range of image (1.0 for normalized [0,1])
        
    Returns:
        float, SSIM value (range typically [0, 1], >0.9 is good)
        
    Formula: Compares luminance, contrast, and structure
    SSIM = (2*μ_x*μ_y + c1) * (2*σ_xy + c2) / 
           ((μ_x^2 + μ_y^2 + c1) * (σ_x^2 + σ_y^2 + c2))
    
    Where:
        μ = local mean
        σ = local variance
        σ_xy = local covariance
        c1, c2 = stability constants
    
    Advantages over PSNR:
    - Better matches human visual perception
    - Accounts for local structure, not just pixel differences
    
    Example:
        - 0.9 < SSIM: Good perceived quality
        - 0.95 < SSIM: Excellent, nearly identical
        - SSIM < 0.8: Visible differences
    """
    if img_true is None or img_pred is None:
        return None
    
    img_true = np.asarray(img_true, dtype=np.float32)
    img_pred = np.asarray(img_pred, dtype=np.float32)
    
    img_true = np.clip(img_true, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)
    
    # Handle different image shapes/dimensions
    if img_true.shape != img_pred.shape:
        return None
    
    try:
        # Multi-scale SSIM for better perceptual quality assessment
        ssim = structural_similarity(img_true, img_pred, data_range=data_range)
        return ssim
    except Exception as e:
        print(f"Error computing SSIM: {e}")
        return None


def compute_metrics(hr_true, sr_pred):
    """
    Compute comprehensive image quality metrics
    
    Args:
        hr_true: numpy array or None, reference HR image (ground truth)
        sr_pred: numpy array, predicted SR image (model output)
        
    Returns:
        dict with comprehensive metrics:
            'valid': bool - Whether metrics could be computed
            'psnr': float - Peak Signal-to-Noise Ratio (higher is better)
            'psnr_label': str - Quality assessment label
            'ssim': float - Structural Similarity (0-1, higher is better)
            'ssim_label': str - Quality assessment label
            'mse': float - Mean Squared Error (lower is better)
            'mae': float - Mean Absolute Error (lower is better)
    
    Interpretation Guide:
    ─────────────────────────────────────────────────────────────
    Metric    | Excellent  | Good      | Acceptable | Poor
    ─────────────────────────────────────────────────────────────
    PSNR      | >40 dB     | 30-40 dB  | 25-30 dB   | <25 dB
    SSIM      | >0.95      | 0.9-0.95  | 0.8-0.9    | <0.8
    ─────────────────────────────────────────────────────────────
    
    NOTE on PSNR/SSIM interpretation:
    - PSNR doesn't always match human perception (can be high but look blurry)
    - SSIM better matches visual perception
    - For SR tasks, improvement of 1-2 dB is meaningful
    - SSIM > 0.9 + PSNR > 28 dB = good SR quality
    """
    if hr_true is None:
        return {
            'valid': False,
            'reason': 'No HR reference image available'
        }
    
    psnr = compute_psnr(hr_true, sr_pred)
    ssim = compute_ssim(hr_true, sr_pred)
    
    return {
        'valid': True,
        'psnr': psnr,
        'psnr_label': _psnr_quality_label(psnr),
        'ssim': ssim,
        'ssim_label': _ssim_quality_label(ssim),
        'mse': np.mean((hr_true - sr_pred) ** 2),
        'mae': np.mean(np.abs(hr_true - sr_pred))
    }


def _psnr_quality_label(psnr):
    """Get quality assessment from PSNR value"""
    if psnr is None or np.isinf(psnr):
        return "Invalid"
    if psnr >= 40:
        return "Excellent (≥40 dB)"
    elif psnr >= 35:
        return "Very Good (35-40 dB)"
    elif psnr >= 30:
        return "Good (30-35 dB)"
    elif psnr >= 25:
        return "Acceptable (25-30 dB)"
    elif psnr >= 20:
        return "Poor (20-25 dB)"
    else:
        return "Very Poor (<20 dB)"


def _ssim_quality_label(ssim):
    """Get quality assessment from SSIM value"""
    if ssim is None:
        return "Invalid"
    if ssim >= 0.95:
        return "Excellent (≥0.95)"
    elif ssim >= 0.90:
        return "Very Good (0.90-0.95)"
    elif ssim >= 0.80:
        return "Good (0.80-0.90)"
    elif ssim >= 0.70:
        return "Acceptable (0.70-0.80)"
    elif ssim >= 0.50:
        return "Poor (0.50-0.70)"
    else:
        return "Very Poor (<0.50)"


def check_model_weights(weights_path=None):
    """
    Check if model has trained weights or using random initialization
    
    Args:
        weights_path: str or Path, path to weights file
        
    Returns:
        dict with weight status info
    """
    if weights_path is None:
        weights_path = Path("models/weights/HRNet.pth")
    else:
        weights_path = Path(weights_path)
    
    print(f"\n{'='*70}")
    print(f"Model Weight Status Check")
    print(f"{'='*70}")
    print(f"Looking for: {weights_path}")
    
    if weights_path.exists():
        try:
            state_dict = torch.load(str(weights_path), map_location='cpu')
            size_mb = weights_path.stat().st_size / (1024 ** 2)
            
            print(f"\n✓ Weights file found!")
            print(f"  File size: {size_mb:.1f} MB")
            print(f"  Parameters in checkpoint: {len(state_dict)} layers")
            
            return {
                'has_weights': True,
                'path': str(weights_path),
                'size_mb': size_mb,
                'num_layers': len(state_dict)
            }
        except Exception as e:
            print(f"\n✗ Error loading weights: {e}")
            return {
                'has_weights': False,
                'error': str(e)
            }
    else:
        print(f"\n✗ NO WEIGHTS FOUND")
        print(f"\n🔴 THIS EXPLAINS POOR OUTPUT QUALITY:")
        print(f"   - Model is using RANDOM initialized weights")
        print(f"   - Random weights = random noise output")
        print(f"   - Output will be grainy, darker, full of artifacts")
        print(f"\n✅ SOLUTIONS:")
        print(f"   1. Train the model on your dataset")
        print(f"   2. Use pre-trained weights if available")
        print(f"   3. For comparison: use bicubic upsampling baseline")
        
        return {
            'has_weights': False,
            'reason': 'Weights file not found - using random initialization'
        }
    
    print(f"{'='*70}\n")


def compare_with_bicubic(lr_image, hr_true=None):
    """
    Create a bicubic upsampling baseline for comparison
    
    Args:
        lr_image: numpy array, low-res image
        hr_true: numpy array or None, true HR image for metrics
        
    Returns:
        dict with bicubic output and metrics
    """
    from scipy import ndimage
    
    # Upsample 3x using bicubic
    bicubic_sr = ndimage.zoom(lr_image, 3, order=3)
    
    result = {
        'bicubic_output': bicubic_sr,
        'bicubic_shape': bicubic_sr.shape,
    }
    
    if hr_true is not None:
        result['metrics'] = compute_metrics(hr_true, bicubic_sr)
    
    return result
