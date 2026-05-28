""" Python script to load and preprocess TIFF patch datasets for 2x super resolution """

from collections import OrderedDict
import numpy as np
from os.path import join, basename, exists

import glob
import skimage
from skimage import io

import torch
from torch.utils.data import Dataset


def get_patch(img, x, y, size=32):
    """
    Slices out a square patch from `img` starting from the (x,y) top-left corner.
    If `img` is a 3D array of shape (l, n, m), the same (x,y) is broadcasted across the first dimension.
    Args:
        img: numpy.ndarray (n, m) or (l, n, m), input image(s)
        x, y: int, top-left corner of the patch
        size: int, patch size
    Returns:
        patch: numpy.ndarray (size, size) or (l, size, size)
    """
    patch = img[..., x:(x + size), y:(y + size)]
    return patch


class ImageSet(OrderedDict):
    """
    An OrderedDict subclass to group image assets with pretty-print functionality.
    """

    def __init__(self, *args, **kwargs):
        super(ImageSet, self).__init__(*args, **kwargs)

    def __repr__(self):
        dict_info = f"{'name':>10} : {self['name']}"
        for name, v in self.items():
            if hasattr(v, 'shape'):
                dict_info += f"\n{name:>10} : {v.shape} {v.__class__.__name__} ({v.dtype})"
            else:
                dict_info += f"\n{name:>10} : {v.__class__.__name__} ({v})"
        return dict_info


def read_tiff_patch(patch_dir, max_lr_frames=7):
    """
    Reads a single pre-augmented TIFF patch directory.
    
    Expects structure:
        patch_dir/
            LR_0.tiff (or .tif), LR_1.tiff, ..., LR_6.tiff  (7x 128×128, 8-bit grayscale)
            HR.tiff (or .tif)                                (256×256, 8-bit grayscale)
    
    Args:
        patch_dir: str, path to patch folder
        max_lr_frames: int, maximum LR frames to load (default 7)
    
    Returns:
        ImageSet dict with keys: name, lr, hr, hr_map, clearances
    """
    
    # Load LR frames (support both .tiff and .tif extensions)
    lr_paths = sorted(glob.glob(join(patch_dir, 'LR_*.tiff')))[:max_lr_frames]
    if len(lr_paths) == 0:
        lr_paths = sorted(glob.glob(join(patch_dir, 'LR_*.tif')))[:max_lr_frames]
    if len(lr_paths) == 0:
        raise FileNotFoundError(f"No LR_*.tiff or LR_*.tif files found in {patch_dir}")
    
    lr_images = np.array([io.imread(path) for path in lr_paths], dtype=np.uint8)
    
    # Load HR image (support both .tiff and .tif extensions)
    hr_path = join(patch_dir, 'HR.tiff')
    if not exists(hr_path):
        hr_path = join(patch_dir, 'HR.tif')
    if not exists(hr_path):
        raise FileNotFoundError(f"HR.tiff or HR.tif not found in {patch_dir}")
    hr = np.array(io.imread(hr_path), dtype=np.uint8)
    
    # Create validity mask (all pixels valid - pre-cleaned dataset)
    hr_map = np.ones(hr.shape[:2], dtype=bool)
    
    # Create uniform clearance scores (all frames equally valid - no clouds/artifacts)
    clearances = np.ones(len(lr_images), dtype=np.float32)
    
    # Organize into ImageSet
    imageset = ImageSet(
        name=basename(patch_dir),
        lr=lr_images,
        hr=hr,
        hr_map=hr_map,
        clearances=clearances,
    )
    
    return imageset


class TiffPatchDataset(Dataset):
    """
    PyTorch Dataset for pre-augmented TIFF patches (2x super resolution).
    
    Expects patch directories containing:
        LR_0.tiff, LR_1.tiff, ..., LR_6.tiff (7x 128×128 frames)
        HR.tiff (256×256 ground truth)
    """
    
    def __init__(self, patch_dirs, config=None, max_views=7):
        """
        Args:
            patch_dirs: list of str, paths to patch directories
            config: dict, training config (unused, kept for compatibility)
            max_views: int, number of LR views per patch (default 7)
        """
        super().__init__()
        self.patch_dirs = patch_dirs if isinstance(patch_dirs, list) else [patch_dirs]
        self.max_views = max_views
    
    def __len__(self):
        return len(self.patch_dirs)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            patch_dir = self.patch_dirs[index]
        else:
            raise KeyError('index must be int')

        imset = read_tiff_patch(patch_dir, max_lr_frames=self.max_views)

        imset['lr']     = torch.from_numpy(skimage.img_as_float(imset['lr']).astype(np.float32))
        imset['hr']     = torch.from_numpy(skimage.img_as_float(imset['hr']).astype(np.float32))
        imset['hr_map'] = torch.from_numpy(imset['hr_map'].astype(np.float32))
        return imset

class collateFunction():
    """Util class to create batches with padding for variable LR frame counts."""

    def __init__(self, min_L=7):
        """
        Args:
            min_L: int, minimum number of LR frames per batch (pads with zeros if fewer)
        """
        self.min_L = min_L

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        Custom collate function to pad batches to uniform frame count.
        
        Args:
            batch: list of ImageSet dicts
        
        Returns:
            padded_lr_batch: tensor (B, min_L, H, W), low-resolution images
            alpha_batch: tensor (B, min_L), frame validity mask (1=real, 0=padded)
            hr_batch: tensor (B, 1, H, W), high-resolution images
            hr_map_batch: tensor (B, H, W), validity maps
            names_batch: list of patch names
        """
        
        lr_batch = []
        alpha_batch = []
        hr_batch = []
        hr_map_batch = []
        names_batch = []

        for imageset in batch:
            lrs = imageset['lr']
            L, H, W = lrs.shape  # (num_frames, height, width)

            # Pad LR frames to min_L if needed
            if L >= self.min_L:
                lr_batch.append(lrs[:self.min_L])
                alpha_batch.append(torch.ones(self.min_L))
            else:
                pad = torch.zeros(self.min_L - L, H, W)
                lr_batch.append(torch.cat([lrs, pad], dim=0))
                alpha_batch.append(torch.cat([torch.ones(L), torch.zeros(self.min_L - L)], dim=0))

            # Add HR (must exist for training)
            hr = imageset['hr']
            if hr is None:
                raise ValueError(f"HR image missing for patch '{imageset['name']}'. Check HR.tiff exists.")
            hr_batch.append(hr)

            hr_map_batch.append(imageset['hr_map'])
            names_batch.append(imageset['name'])

        # Stack into batches
        padded_lr_batch = torch.stack(lr_batch, dim=0)      # (B, min_L, H, W)
        alpha_batch = torch.stack(alpha_batch, dim=0)        # (B, min_L)
        hr_batch = torch.stack(hr_batch, dim=0)              # (B, H, W)
        hr_batch = hr_batch.unsqueeze(1)                     # (B, 1, H, W) - add channel dim
        hr_map_batch = torch.stack(hr_map_batch, dim=0)      # (B, H, W)

        return padded_lr_batch, alpha_batch, hr_batch, hr_map_batch, names_batch
