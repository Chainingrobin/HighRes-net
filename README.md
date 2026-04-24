# HighRes-Net: Multi-Frame Super-Resolution

# this was used for ESA satelitte stuff and the fluff has been removed too so check and compare with the original repo to see changes in scripts etc

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

👉 **START HERE:** [DOCUMENTATION.md](DOCUMENTATION.md) — Complete guide to training, tuning, and troubleshooting

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

## Performance

After training on 12-15 scenes (RTX 4060):

| Metric     | Random | Trained  |
| ---------- | ------ | -------- |
| PSNR       | 9.5 dB | 26-28 dB |
| vs Bicubic | -5 dB  | +2-5 dB  |
| SSIM       | 0.42   | 0.85+    |

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

**Need help?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or run `python verify_setup.py`
