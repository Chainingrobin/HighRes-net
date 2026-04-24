"""
Weight Quality Checker - Evaluates if trained weights are better than random
"""

import torch
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import io, img_as_float

def check_weight_statistics(model, label="Model"):
    """Analyze weight distribution to detect meaningful learning."""
    stats = {
        'label': label,
        'total_params': sum(p.numel() for p in model.parameters()),
        'layers': []
    }
    
    for name, param in model.named_parameters():
        w = param.data.detach().cpu().numpy()
        stats['layers'].append({
            'name': name,
            'shape': w.shape,
            'mean': float(np.mean(w)),
            'std': float(np.std(w)),
            'min': float(np.min(w)),
            'max': float(np.max(w)),
            'is_initialization': np.std(w) < 0.01 and np.mean(w) < 0.01  # All zeros/near-zeros
        })
    
    return stats


def compare_weight_distributions(stats_trained, stats_random):
    """Compare two weight distributions to detect learning."""
    print("\n" + "="*70)
    print("WEIGHT QUALITY ANALYSIS")
    print("="*70)
    
    print(f"\n{stats_trained['label']}:")
    print(f"  Total parameters: {stats_trained['total_params']:,}")
    
    trained_std = np.mean([l['std'] for l in stats_trained['layers']])
    random_std = np.mean([l['std'] for l in stats_random['layers']])
    
    trained_mean = np.abs(np.mean([l['mean'] for l in stats_trained['layers']]))
    random_mean = np.abs(np.mean([l['mean'] for l in stats_random['layers']]))
    
    print(f"  Average weight std: {trained_std:.6f}")
    print(f"  Average weight mean: {trained_mean:.6f}")
    
    # Check if weights have been modified from random initialization
    is_trained = (trained_std > random_std * 0.8) and (trained_std < random_std * 1.2)
    looks_random = np.all([l.get('is_initialization', False) for l in stats_trained['layers'][:3]])
    
    print(f"\n{stats_random['label']}:")
    print(f"  Average weight std: {random_std:.6f}")
    print(f"  Average weight mean: {random_mean:.6f}")
    
    print(f"\n📊 ANALYSIS:")
    if looks_random:
        print(f"  🔴 Weights appear to be RANDOM/UNINITIALIZED")
        print(f"     - All weights near zero (not trained)")
        print(f"     - Model output will be garbage")
        return 'random'
    elif abs(trained_std - random_std) < 0.001:
        print(f"  🟡 Weights look similar to random initialization")
        print(f"     - Possible underfitting or too few epochs")
        print(f"     - May not have trained properly")
        return 'questionable'
    else:
        print(f"  ✓ Weights appear TRAINED (different from random)")
        print(f"     - Weight distribution has changed")
        print(f"     - Model likely learned something useful")
        return 'trained'


def infer_with_model(model, lr_batch, alphas_batch, device):
    """Run inference and return output."""
    with torch.no_grad():
        sr = model(lr_batch, alphas_batch)
    return sr.detach().cpu().numpy()


def quick_inference_test(weights_path, num_test_scenes=5):
    """Quick test: compare trained model vs random model."""
    import sys
    import json
    from DataLoader import SimpleMicroscopyDataset, collateFunction
    from DeepNetworks.HRNet import HRNet
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = Path(__file__).parent.parent / 'config' / 'config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load dataset
    dataset_root = Path("D:/GUC/Datasets/HighRes input test")
    scene_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])[:num_test_scenes]
    scene_dirs_str = [str(d) for d in scene_dirs]
    
    if len(scene_dirs) == 0:
        print("❌ No test scenes found!")
        return
    
    dataset = SimpleMicroscopyDataset(
        imset_dirs=scene_dirs_str,
        config=config['training'],
        max_views=config['training']['n_views']
    )
    
    min_L = config['training']['min_L']
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collateFunction(min_L=min_L),
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize models
    model_trained = HRNet(config['network']).to(device)
    model_random = HRNet(config['network']).to(device)
    
    # Load trained weights
    if weights_path.exists():
        print(f"✓ Loading trained weights from {weights_path}")
        state_dict = torch.load(str(weights_path), map_location=device)
        model_trained.load_state_dict(state_dict)
    else:
        print(f"✗ Weights file not found at {weights_path}")
        print(f"  Using random initialization for 'trained' model")
    
    model_trained.eval()
    model_random.eval()
    
    # Compare weight statistics
    print("\n" + "="*70)
    stats_trained = check_weight_statistics(model_trained, "Loaded Model")
    stats_random = check_weight_statistics(model_random, "Random Initialization")
    
    weight_status = compare_weight_distributions(stats_trained, stats_random)
    
    # Run inference comparison on test batch
    print("\n" + "="*70)
    print("INFERENCE QUALITY TEST")
    print("="*70)
    
    psnr_trained_list = []
    psnr_random_list = []
    
    from skimage.metrics import peak_signal_noise_ratio
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_test_scenes:
            break
        
        lrs, alphas, hrs, hr_maps, names = batch
        lrs = lrs.float().to(device)
        alphas = alphas.float().to(device)
        hrs = hrs.float().numpy()
        
        # Trained model inference
        with torch.no_grad():
            sr_trained = model_trained(lrs, alphas)[0, 0].cpu().numpy()
        
        # Random model inference
        with torch.no_grad():
            sr_random = model_random(lrs, alphas)[0, 0].cpu().numpy()
        
        sr_trained = np.clip(sr_trained, 0, 1)
        sr_random = np.clip(sr_random, 0, 1)
        
        # Get HR and resample if needed
        hr_true = hrs[0]
        hr_true = np.clip(hr_true, 0, 1)
        
        # Match sizes
        if hr_true.shape != sr_trained.shape:
            scale = sr_trained.shape[0] / hr_true.shape[0]
            hr_true = ndimage.zoom(hr_true, scale, order=3)
        
        # Compute metrics
        if hr_true.max() > 0:  # Only compute if HR is not all black
            psnr_t = peak_signal_noise_ratio(hr_true, sr_trained, data_range=1.0)
            psnr_r = peak_signal_noise_ratio(hr_true, sr_random, data_range=1.0)
            
            psnr_trained_list.append(psnr_t)
            psnr_random_list.append(psnr_r)
            
            print(f"\nScene {batch_idx + 1}: {names[0]}")
            print(f"  Trained PSNR: {psnr_t:.2f} dB")
            print(f"  Random PSNR:  {psnr_r:.2f} dB")
            print(f"  Difference:   {psnr_t - psnr_r:+.2f} dB {'✓ Better' if psnr_t > psnr_r else '✗ Worse'}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if len(psnr_trained_list) > 0:
        mean_psnr_trained = np.mean(psnr_trained_list)
        mean_psnr_random = np.mean(psnr_random_list)
        avg_improvement = mean_psnr_trained - mean_psnr_random
        
        print(f"\nAverage PSNR (trained):  {mean_psnr_trained:.2f} dB")
        print(f"Average PSNR (random):   {mean_psnr_random:.2f} dB")
        print(f"Average improvement:     {avg_improvement:+.2f} dB")
        
        if avg_improvement > 1.0:
            print(f"\n✓ TRAINED MODEL IS BETTER than random baseline!")
            print(f"  Improvement: {avg_improvement:.2f} dB indicates learning occurred")
            print(f"  → Weights are usable and should be improved with more training")
        elif avg_improvement > 0:
            print(f"\n🟡 MARGINAL improvement over random")
            print(f"  Improvement: {avg_improvement:.2f} dB is small")
            print(f"  → Model may be underfitted - try more epochs or data")
        else:
            print(f"\n🔴 NO IMPROVEMENT or WORSE than random")
            print(f"  Difference: {avg_improvement:.2f} dB")
            print(f"  → Training may have failed - check training loop")
    else:
        print("\nCannot compute metrics - check if HR images are loading (not all black)")
    
    print("\n" + "="*70)
    
    return {
        'weight_status': weight_status,
        'mean_psnr_trained': np.mean(psnr_trained_list) if psnr_trained_list else None,
        'mean_psnr_random': np.mean(psnr_random_list) if psnr_random_list else None,
        'improvement': np.mean(psnr_trained_list) - np.mean(psnr_random_list) if psnr_trained_list else None
    }


if __name__ == "__main__":
    weights_path = Path("../models/weights/HRNet.pth")
    
    print("HighRes-Net Weight Quality Checker")
    print("="*70)
    
    result = quick_inference_test(weights_path, num_test_scenes=3)
