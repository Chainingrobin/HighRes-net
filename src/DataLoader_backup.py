""" Python script to load, augment and preprocess batches of data """

from collections import OrderedDict
import numpy as np
from os.path import join, exists, basename, isfile

import glob
import skimage
from skimage import io

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def get_patch(img, x, y, size=32):
    """
    Slices out a square patch from `img` starting from the (x,y) top-left corner.
    If `im` is a 3D array of shape (l, n, m), then the same (x,y) is broadcasted across the first dimension,
    and the output has shape (l, size, size).
    Args:
        img: numpy.ndarray (n, m), input image
        x, y: int, top-left corner of the patch
        size: int, patch size
    Returns:
        patch: numpy.ndarray (size, size)
    """
    
    patch = img[..., x:(x + size), y:(y + size)]   # using ellipsis to slice arbitrary ndarrays
    return patch


class ImageSet(OrderedDict):
    """
    An OrderedDict derived class to group the assets of an imageset, with a pretty-print functionality.
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


def sample_clearest(clearances, n=None, beta=50, seed=None):
    """
    Given a set of clearances, samples `n` indices with probability proportional to their clearance.
    Args:
        clearances: numpy.ndarray, clearance scores
        n: int, number of low-res views to read
        beta: float, inverse temperature. beta 0 = uniform sampling. beta +infinity = argmax.
        seed: int, random seed
    Returns:
        i_sample: numpy.ndarray (n), sampled indices
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    e_c = np.exp(beta * clearances / clearances.max()) ##### FIXME: This is numerically unstable. 
    p = e_c / e_c.sum()
    idx = range(len(p))
    i_sample = np.random.choice(idx, size=n, p=p, replace=False)
    return i_sample


def read_imageset(imset_dir, create_patches=False, patch_size=64, seed=None, top_k=None, beta=0.):
    """
    Retrieves all assets from the given directory.
    Args:
        imset_dir: str, imageset directory.
        create_patches: bool, samples a random patch or returns full image (default).
        patch_size: int, size of low-res patch.
        top_k: int, number of low-res views to read.
            If top_k = None (default), low-views are loaded in the order of clearance.
            Otherwise, top_k views are sampled with probability proportional to their clearance.
        beta: float, parameter for random sampling of a reference proportional to its clearance.
        load_lr_maps: bool, reads the status maps for the LR views (default=True).
    Returns:
        dict, collection of the following assets:
          - name: str, imageset name.
          - lr: numpy.ndarray, low-res images.
          - hr: high-res image.
          - hr_map: high-res status map.
          - clearances: precalculated average clearance (see save_clearance.py)
    """

    # Read asset names
    idx_names = np.array([basename(path)[2:-4] for path in glob.glob(join(imset_dir, 'QM*.png'))])
    idx_names = np.sort(idx_names)
    
    clearances = np.zeros(len(idx_names))
    if isfile(join(imset_dir, 'clearance.npy')):
        try:
            clearances = np.load(join(imset_dir, 'clearance.npy'))  # load clearance scores
        except Exception as e:
            print("please call the save_clearance.py before call DataLoader")
            print(e)
    else:
        raise Exception("please call the save_clearance.py before call DataLoader")

    if top_k is not None and top_k > 0:
        top_k = min(top_k, len(idx_names))
        i_samples = sample_clearest(clearances, n=top_k, beta=beta, seed=seed)
        idx_names = idx_names[i_samples]
        clearances = clearances[i_samples]
    else:
        i_clear_sorted = np.argsort(clearances)[::-1]  # max to min
        clearances = clearances[i_clear_sorted]
        idx_names = idx_names[i_clear_sorted]

    lr_images = np.array([io.imread(join(imset_dir, f'LR{i}.png')) for i in idx_names], dtype=np.uint16)

    hr_map = np.array(io.imread(join(imset_dir, 'SM.png')), dtype=np.bool)
    if exists(join(imset_dir, 'HR.png')):
        hr = np.array(io.imread(join(imset_dir, 'HR.png')), dtype=np.uint16)
    else:
        hr = None  # no high-res image in test data

    if create_patches:
        if seed is not None:
            np.random.seed(seed)

        max_x = lr_images[0].shape[0] - patch_size
        max_y = lr_images[0].shape[1] - patch_size
        x = np.random.randint(low=0, high=max_x)
        y = np.random.randint(low=0, high=max_y)
        lr_images = get_patch(lr_images, x, y, patch_size)  # broadcasting slicing coordinates across all images
        hr_map = get_patch(hr_map, x * 3, y * 3, patch_size * 3)

        if hr is not None:
            hr = get_patch(hr, x * 3, y * 3, patch_size * 3)

    # Organise all assets into an ImageSet (OrderedDict)
    imageset = ImageSet(name=basename(imset_dir),
                        lr=np.array(lr_images),
                        hr=hr,
                        hr_map=hr_map,
                        clearances=clearances,
                        )

    return imageset


class ImagesetDataset(Dataset):
    """ Derived Dataset class for loading many imagesets from a list of directories."""

    def __init__(self, imset_dir, config, seed=None, top_k=-1, beta=0.):

        super().__init__()
        self.imset_dir = imset_dir
        self.name_to_dir = {basename(im_dir): im_dir for im_dir in imset_dir}
        self.create_patches = config["create_patches"]
        self.patch_size = config["patch_size"]
        self.seed = seed  # seed for random patches
        self.top_k = top_k
        self.beta = beta
        
    def __len__(self):
        return len(self.imset_dir)        

    def __getitem__(self, index):
        """ Returns an ImageSet dict of all assets in the directory of the given index."""    

        if isinstance(index, int):
            imset_dir = [self.imset_dir[index]]
        elif isinstance(index, str):
            imset_dir = [self.name_to_dir[index]]
        elif isinstance(index, slice):
            imset_dir = self.imset_dir[index]
        else:
            raise KeyError('index must be int, string, or slice')

        imset = [read_imageset(imset_dir=dir_,
                                  create_patches=self.create_patches,
                                  patch_size=self.patch_size,
                                  seed=self.seed,
                                  top_k=self.top_k,
                                  beta=self.beta,)
                    for dir_ in tqdm(imset_dir, disable=(len(imset_dir) < 11))]

        if len(imset) == 1:
            imset = imset[0]

        imset_list = imset if isinstance(imset, list) else [imset]
        for i, imset_ in enumerate(imset_list):
            imset_['lr'] = torch.from_numpy(skimage.img_as_float(imset_['lr']).astype(np.float32))
            if imset_['hr'] is not None:
                imset_['hr'] = torch.from_numpy(skimage.img_as_float(imset_['hr']).astype(np.float32))
                imset_['hr_map'] = torch.from_numpy(imset_['hr_map'].astype(np.float32))
            imset_list[i] = imset_

        return imset if not isinstance(imset_list, list) else (imset_list[0] if len(imset_list) == 1 else imset_list)


def _detect_naming_scheme(imset_dir):
    """
    Detects the naming scheme used in the dataset.
    Returns: 'standard' (LR_*.png, HR.png) or 'alternate' (lr_frame_*.png, hr_ground_truth.png)
    """
    standard_lr = glob.glob(join(imset_dir, 'LR_*.png'))
    alternate_lr = glob.glob(join(imset_dir, 'lr_frame_*.png'))
    
    if len(standard_lr) > 0:
        return 'standard'
    elif len(alternate_lr) > 0:
        return 'alternate'
    else:
        raise FileNotFoundError(f"No LR images found in {imset_dir}. Expected LR_*.png or lr_frame_*.png")


def read_microscopy_imageset(imset_dir, create_patches=False, patch_size=64, seed=None, max_views=7):
    """
    Reads a simple microscopy imageset with flexible naming.
    Supports both formats:
      - Standard: LR_1.png, LR_2.png, ..., HR.png
      - Alternate: lr_frame_01.png, lr_frame_02.png, ..., hr_ground_truth.png
    
    Args:
        imset_dir: str, imageset directory path
        create_patches: bool, whether to extract random patches
        patch_size: int, size of patches to extract
        seed: int, random seed for reproducibility
        max_views: int, maximum number of LR views to load
        
    Returns:
        ImageSet dict with keys: name, lr, hr, hr_map, clearances
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Detect naming scheme
    scheme = _detect_naming_scheme(imset_dir)
    
    # Find all LR images (supports both LR_*.png and lr_frame_*.png)
    if scheme == 'standard':
        lr_paths = sorted(glob.glob(join(imset_dir, 'LR_*.png')))
    else:  # alternate
        lr_paths = sorted(glob.glob(join(imset_dir, 'lr_frame_*.png')))
    
    lr_paths = lr_paths[:max_views] if max_views > 0 else lr_paths
    
    if len(lr_paths) == 0:
        raise FileNotFoundError(f"No LR images found in {imset_dir}")
    
    # Load LR images (DON'T cast to float32 here - let imread determine dtype)
    lr_images = np.array([io.imread(path) for path in lr_paths])
    
    # Load HR image with flexible naming detection
    hr = None
    hr_path = None
    
    # Try multiple HR file name variations (case-insensitive search)
    possible_hr_names = [
        'HR.png', 'hr.png', 'HR.PNG', 'hr.PNG',
        'hr_ground_truth.png', 'HR_ground_truth.png', 'hr_ground_truth.PNG',
        'hr_ground_truth.PNG', 'HR_ground_truth.PNG',
        'ground_truth.png', 'ground_truth.PNG',
        'GroundTruth.png', 'groundtruth.png'
    ]
    
    for hr_name in possible_hr_names:
        test_path = join(imset_dir, hr_name)
        if exists(test_path):
            hr_path = test_path
            hr = np.array(io.imread(test_path))  # DON'T force float32 - let skimage handle normalization
            break
    
    if hr is None:
        # List all files for debugging
        all_files = glob.glob(join(imset_dir, '*'))
        file_names = [basename(f) for f in all_files]
        raise FileNotFoundError(
            f"HR image not found in {imset_dir}\n"
            f"Tried: {', '.join(possible_hr_names)}\n"
            f"Files in directory: {file_names}"
        )
    
    # Create dummy status map (all valid pixels)
    if hr is not None:
        # Resample HR from 4x to 3x scale to match model output
        # LR is 128×128, so SR will be 384×384 (128*3)
        # HR is currently 512×512 (128*4)
        # Downsample HR to 384×384 so loss can be computed
        lr_h, lr_w = lr_images[0].shape[:2]
        target_shape = (lr_h * 3, lr_w * 3)  # 384×384 for 128×128 LR
        
        if hr.shape[:2] != target_shape:
            from scipy import ndimage
            scale_factor = target_shape[0] / hr.shape[0]
            hr = ndimage.zoom(hr, scale_factor, order=3)
            print(f"  HR resampled from {hr.shape} to {target_shape} to match 3x SR output")
        
        hr_map = np.ones(target_shape, dtype=bool)
    else:
        # If no HR, use LR size as reference (upsampled by 3x)
        h, w = lr_images[0].shape
        hr_map = np.ones((h * 3, w * 3), dtype=bool)
    
    # Generate dummy clearance scores (uniform = all views equally valid)
    clearances = np.ones(len(lr_images), dtype=np.float32)
    
    # Extract patches if requested
    if create_patches:
        if hr is not None:
            max_x = lr_images[0].shape[0] - patch_size
            max_y = lr_images[0].shape[1] - patch_size
        else:
            max_x = lr_images[0].shape[0] - patch_size
            max_y = lr_images[0].shape[1] - patch_size
            
        if max_x > 0 and max_y > 0:
            x = np.random.randint(low=0, high=max_x)
            y = np.random.randint(low=0, high=max_y)
            
            lr_images = get_patch(lr_images, x, y, patch_size)
            
            if hr is not None:
                hr = get_patch(hr, x * 3, y * 3, patch_size * 3)
            hr_map = get_patch(hr_map, x * 3, y * 3, patch_size * 3)
    
    # Organize into ImageSet
    imageset = ImageSet(
        name=basename(imset_dir),
        lr=np.array(lr_images),
        hr=hr,
        hr_map=hr_map,
        clearances=clearances,
    )
    
    return imageset


class SimpleMicroscopyDataset(Dataset):
    """
    PyTorch Dataset for simple microscopy imagesets (no ESA format required).
    Expects folder structure: scene_folder/LR_1.png, LR_2.png, ..., HR.png
    """
    
    def __init__(self, imset_dirs, config, seed=None, max_views=7):
        """
        Args:
            imset_dirs: list of str, paths to scene directories
            config: dict, training config with create_patches, patch_size
            seed: int, random seed
            max_views: int, maximum number of LR views to read per scene
        """
        super().__init__()
        self.imset_dirs = imset_dirs if isinstance(imset_dirs, list) else [imset_dirs]
        self.create_patches = config.get("create_patches", False)
        self.patch_size = config.get("patch_size", 64)
        self.seed = seed
        self.max_views = max_views
        
    def __len__(self):
        return len(self.imset_dirs)
    
    def __getitem__(self, index):
        """Returns an ImageSet dict for the scene at given index."""
        
        if isinstance(index, int):
            imset_dir = self.imset_dirs[index]
        else:
            raise KeyError('index must be int')
        
        imset = read_microscopy_imageset(
            imset_dir=imset_dir,
            create_patches=self.create_patches,
            patch_size=self.patch_size,
            seed=self.seed,
            max_views=self.max_views,
        )
        
        # Convert to torch tensors with float32
        imset['lr'] = torch.from_numpy(skimage.img_as_float(imset['lr']).astype(np.float32))
        
        if imset['hr'] is not None:
            imset['hr'] = torch.from_numpy(skimage.img_as_float(imset['hr']).astype(np.float32))
            imset['hr_map'] = torch.from_numpy(imset['hr_map'].astype(np.float32))
        
        return imset


class collateFunction():
    """ Util class to create padded batches of data. """

    def __init__(self, min_L=32):
        """
        Args:
            min_L: int, pad length
        """
        
        self.min_L = min_L

    def __call__(self, batch):
        return self.collateFunction(batch)

    def collateFunction(self, batch):
        """
        Custom collate function to adjust a variable number of low-res images.
        Args:
            batch: list of imageset
        Returns:
            padded_lr_batch: tensor (B, min_L, W, H), low resolution images
            alpha_batch: tensor (B, min_L), low resolution indicator (0 if padded view, 1 otherwise)
            hr_batch: tensor (B, W, H), high resolution images
            hm_batch: tensor (B, W, H), high resolution status maps
            isn_batch: list of imageset names
        """
        
        lr_batch = []  # batch of low-resolution views
        alpha_batch = []  # batch of indicators (0 if padded view, 1 if genuine view)
        hr_batch = []  # batch of high-resolution views
        hm_batch = []  # batch of high-resolution status maps
        isn_batch = []  # batch of site names

        train_batch = True

        for imageset in batch:

            lrs = imageset['lr']
            L, H, W = lrs.shape

            if L >= self.min_L:  # pad input to top_k
                lr_batch.append(lrs[:self.min_L])
                alpha_batch.append(torch.ones(self.min_L))
            else:
                pad = torch.zeros(self.min_L - L, H, W)
                lr_batch.append(torch.cat([lrs, pad], dim=0))
                alpha_batch.append(torch.cat([torch.ones(L), torch.zeros(self.min_L - L)], dim=0))

            hr = imageset['hr']
            if hr is None:
                raise ValueError(f"HR image missing for scene '{imageset['name']}'. Check if HR.png or hr_ground_truth.png exists.")
            hr_batch.append(hr)

            hm_batch.append(imageset['hr_map'])
            isn_batch.append(imageset['name'])

        padded_lr_batch = torch.stack(lr_batch, dim=0)
        alpha_batch = torch.stack(alpha_batch, dim=0)
        hr_batch = torch.stack(hr_batch, dim=0)
        hr_batch = hr_batch.unsqueeze(1)  # Add channel dimension: [B, H, W] → [B, 1, H, W] to match SR output
        hm_batch = torch.stack(hm_batch, dim=0)

        return padded_lr_batch, alpha_batch, hr_batch, hm_batch, isn_batch
