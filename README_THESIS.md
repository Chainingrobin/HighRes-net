# Thesis Readme: HighRes-Net

This repository contains a PyTorch implementation of HighRes-Net for multi-frame, 2x super-resolution on grayscale TIFF data. The model fuses multiple low-resolution (LR) frames into a single super-resolved (SR) image and supports optional attention (CBAM and FusionAttention).

## Requirements

Install all dependencies from the pinned list:

```bash
python -m venv hr_env
hr_env\Scripts\activate
pip install -r requirements.txt
```

Key packages (from requirements.txt):

- Core: torch, torchvision, numpy, scipy, scikit-image
- Visualization: matplotlib, seaborn, plotly
- Notebooks: jupyter, ipython
- Training utilities: tensorboardX, tqdm
- Optional/dev tools: pytest, ipdb, line_profiler, jupyter_contrib_nbextensions

## Hardware and CUDA compatibility

- Training is GPU-recommended. CPU training is possible but slow.
- Inference can run on CPU, but GPU is strongly recommended for speed.
- If you plan to use CUDA, install a PyTorch build that matches your NVIDIA driver and CUDA version.
- If CUDA is unavailable, the code falls back to CPU.

## Data format (TIFF, square images)

### Training data (LR + HR)

Training uses patch directories under the path in config `paths.prefix` (default `data/`). The training script expects:

```
data/
  train/
    patch_000/
      LR_0.tif
      LR_1.tif
      ...
      LR_6.tif
      HR.tif
    patch_001/
      ...
```

Requirements:

- File format: `.tif` or `.tiff`
- Grayscale TIFFs
- LR frames are square (default 128x128)
- HR is 2x the LR size (default 256x256)
- The number of LR frames used is controlled by `training.n_views`

### Inference data (LR only)

For LR-only inference (no ground truth), each sample folder contains only LR frames:

```
<dataset_root>/
  image_000/
    LR_0.tif
    LR_1.tif
    ...
```

Notes:

- The inference notebook filters sample folder names with a prefix (default `image_`).
- If your folders are named `scene_XX` or similar, update the filter in the notebook.
- Inputs are expected to be square; non-square images are not validated.

## Configuration (config/config.json)

Edit `config/config.json` to control training, architecture, and paths.

### Paths

- `paths.prefix`: base data folder (training expects `<prefix>/train`)
- `paths.checkpoint_dir`: where training saves checkpoints
- `paths.tb_log_file_dir`: TensorBoard logs

### Network

- `network.use_cbam`: enable CBAM modules
- `network.use_checkpointing`: enable checkpointing to reduce VRAM
- `network.cbam_reduction`: channel reduction ratio for CBAM
- `network.cbam_spatial_kernel`: spatial kernel size for CBAM
- `network.encoder.*`: encoder depth and channels
- `network.recursive.*`: recursive fusion depth and channels
- `network.decoder.*`: upsample mode and final output conv

### Training (core)

- `training.num_epochs`: number of epochs
- `training.batch_size`: batch size
- `training.grad_accum_steps`: gradient accumulation steps
- `training.n_views`: number of LR frames used
- `training.min_L`: pad to this many frames if fewer are available
- `training.n_workers`: DataLoader workers
- `training.patch_size`: LR patch size (default 128)
- `training.crop`: crop border size for loss
- `training.val_proportion`: train/val split fraction
- `training.create_patches`: reserved for patch creation pipelines

### Training (AMP)

- `training.use_amp`: enable mixed precision
- `training.amp_dtype`: float16 or bfloat16

### Training (optimizer and schedule)

- `training.lr`: base learning rate
- `training.lr_step`: scheduler patience
- `training.lr_decay`: scheduler decay factor

### Training (attention and warmup)

- `training.fusion_lr`: learning rate for fusion components
- `training.fusion_blend_lr`: learning rate for fusion blend gate
- `training.use_cbam_warmup`: enable CBAM warmup schedule
- `training.cbam_warmup_epochs`: warmup length
- `training.cbam_lr_warmup`: warmup LR
- `training.base_lr_joint`: base LR after warmup
- `training.cbam_lr_joint`: CBAM LR after warmup

### Training (regularization and losses)

Some knobs are used in notebooks or experimental scripts. Keep them consistent if you use those paths.

- `training.lambda`: legacy shift regularization term
- `training.lambda_shift_reg`: explicit shift regularization term
- `training.lambda_range`: output range regularization
- `training.lambda_tv`: total variation regularization
- `training.lambda_edge_tv`: edge-aware TV regularization
- `training.edge_aware_k`: edge weighting factor
- `training.lambda_freq`: frequency-domain regularization
- `training.lambda_dir`: directional regularization
- `training.dir_emphasis_axis`: preferred axis for directional emphasis
- `training.dir_emphasis`: emphasis strength
- `training.lambda_axis_freq`: axis-specific frequency loss
- `training.axis_sigma_ratio`: axis frequency weighting
- `training.freq_low_ratio`: low-frequency band ratio
- `training.freq_high_ratio`: high-frequency band ratio
- `training.freq_use_log_magnitude`: use log magnitude in frequency loss
- `training.smoothness_mode`: smoothing mode selector

## Weight manager

Use `notebooks/weights_manager.ipynb` to:

- List saved weights in `models/weights/Base` and `models/weights/CBAM`
- Review temporary run candidates in `models/weights/_runs_tmp`
- Select the active weight used for inference

The active selection is stored at:

- `models/weights/active_weight_selection.json`

Inference also falls back to:

- `models/weights/HRNet.pth`

If you move the repository or data, update these paths in the notebook.

## Training: step-by-step

1. Create a Python environment and install requirements.
2. Prepare training patches under `data/train/` (see data format above).
3. Edit `config/config.json` (paths, views, batch size, epochs).
4. Train using either:
   - Notebook: `notebooks/training_run.ipynb`
   - Script: `python src/train.py --config config/config.json`
5. Checkpoints are saved under `models/weights/<run_folder>/`.

Optional CLI overrides:

- `--num_epochs`, `--batch_size`, `--lr`, `--lambda_reg`

## Inference: which notebook to use

### inference_diagnostic.ipynb

Use this when you have LR inputs and HR ground truth to compare against.

- Computes metrics and diagnostics
- Best for evaluation and validation

### inference_no_hr_upscale.ipynb

Use this when you have LR inputs only and no HR ground truth.

- Produces 2x SR output and a bicubic baseline
- Best for visual inspection

## Paths you may need to change

- `dataset_root` in `notebooks/inference_no_hr_upscale.ipynb`
- `export_dir` in `notebooks/inference_no_hr_upscale.ipynb`
- `paths.prefix` in `config/config.json`
- Any absolute paths in your local environment

## Tips for new users

- Always verify the active weights before running inference.
- If outputs look noisy or dark, confirm that trained weights are loaded.
- Keep LR inputs square to avoid layout issues.
- Reduce `training.n_workers` to 0 on Windows if DataLoader workers crash.

## Troubleshooting

See `TROUBLESHOOTING.md` for common setup and runtime issues.
