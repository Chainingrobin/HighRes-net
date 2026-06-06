# HighRes-Net: Multi-Frame Super-Resolution

Multi-frame super-resolution (MFSR) using PyTorch. Trained on synthetic microscopy data with HRNet architecture + ShiftNet registration + range regularization.

## Quick Start (5 minutes)

```bash
# Setup
python -m venv hr_env
hr_env\Scripts\activate
pip install -r requirements.txt

# Test
python verify_setup.py
jupyter notebook notebooks/inference_microscopy.ipynb
```

## Documentation

👉 **START HERE:** [README_THESIS.md](README_THESIS.md) — Complete guide to training, tuning, and troubleshooting

- **Setup & Installation** — System requirements, CUDA setup
- **Training Guide** — Step-by-step training on your data
- **Range Regularization** — Technical details of the loss function
- **Parameter Tuning** — Lambda, learning rate, batch size
- **Troubleshooting** — Common issues & solutions

## Notebooks

- `training_run.ipynb` — Train the model (50-300 epochs)
- `inference_microscopy.ipynb` — Run super-resolution on your images
- `inference_diagnostic.ipynb` — Analyze results with metrics

## Project Structure

```
src/                       # Core training/inference code
├── train.py              # Training script
├── predict.py            # Inference wrapper
├── DataLoader.py         # Dataset handling
├── DeepNetworks/
│   ├── HRNet.py          # Super-resolution network
│   └── ShiftNet.py       # Registration network
└── ...                   # Utilities

notebooks/                 # Jupyter workflows
├── training_run.ipynb    # Main training
├── inference_microscopy.ipynb
└── inference_diagnostic.ipynb

config/config.json        # Training configuration
models/weights/HRNet.pth  # Trained weights
```

## Key Features

✅ **Multi-frame fusion** — Aligns and combines 7+ LR images  
✅ **Range regularization** — Forces network to use full output range  
✅ **Built-in diagnostics** — Check alignment, metrics, weights  
✅ **Configurable training** — Adjust for your data size & GPU  
✅ **Fast inference** — ~50ms for 7-frame fusion on RTX 4060

## Citation

Based on HighRes-Net (ESA Kelvin Competition 2019)

```bibtex
@article{highresnet,
  title={HighRes-Net: Recursive Fusion for Multi-Frame Super-Resolution},
  author={Deudon, Michel and others},
  year={2019}
}
```

## License

Apache 2.0 License. See [LICENSE](LICENSE)

---
