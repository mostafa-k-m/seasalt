# MPDenoiseNet

Official PyTorch implementation for the paper:

> **MPDenoiseNet: Resource-Efficient Deep Learning Approach for Image Denoising**  
> Mostafa Kamal, Walid Al-Atabany  
> *Neural Processing Letters*, Springer, 2026  
> DOI: [10.1007/s11063-025-11803-4](https://link.springer.com/article/10.1007/s11063-025-11803-4)

---

## Overview

MPDenoiseNet is a **multi-path, resource-efficient** deep learning network for blind image denoising. It handles four noise types — **Gaussian, Poisson, Bernoulli, and Salt-and-Pepper** — while being significantly faster and less memory-intensive than transformer-only baselines like Restormer.

The architecture combines three parallel processing paths:

- **SeConv Branch** — 7 Selective Convolutional blocks targeting Salt-and-Pepper and Bernoulli noise
- **Anisotropic Diffusion Branch** — 5 novel learnable AD blocks for edge-preserving smoothing of Gaussian/Poisson noise
- **Transformer Blocks** — 10 Restormer-style MDTA blocks for global context
- **Asymmetric U-Net AutoEncoder** — final reconstruction and perceptual refinement stage

The full transformation is:

$$\hat{I} = f_{AE}(f_{TF}(f_{Emb}(\text{Concat}(I,\ f_{SeConv}(I),\ f_{AD}(I)))))$$

---

## Key Results

MPDenoiseNet matches or exceeds Restormer's denoising quality while using significantly fewer resources:

| Image Size | Restormer Memory | **MPDenoiseNet Memory** | Restormer Throughput | **MPDenoiseNet Throughput** |
|-----------|-----------------|------------------------|---------------------|----------------------------|
| 256×256   | 520 MB          | **307 MB**             | 21.4 img/s          | **41.6 img/s**             |
| 512×512   | 1687 MB         | **989 MB**             | 6.4 img/s           | **31.6 img/s**             |
| 1024×1024 | 6424 MB         | **3718 MB**            | 1.2 img/s           | **23.6 img/s**             |
| 2048×2048 | OOM             | **14638 MB** ✓         | OOM                 | **2.5 img/s** ✓            |

Evaluated on BSD68, Kodak24, Set12, and Urban100.

---

## Repository Structure
```
seasalt/
├── seasalt/salt_net/                     # MPDenoiseNet model definition
├── restormer_implementation/             # Restormer baseline
├── paper/sections/experiments/           # Experiment configs and results
├── train.py                              # Training script
├── benchmark.py                          # Throughput & memory benchmarking
├── evaluation_script.py                  # Quantitative evaluation (PSNR/SSIM)
├── evaluation_script_restormer_local.py  # Restormer evaluation
├── train_noise_to_salt.ipynb             # Stage 1 training notebook
├── train_noise_to_salt_2nd_stage.ipynb   # Stage 2 training notebook
└── eval.ipynb                            # Results exploration notebook
```

---

## Installation
```bash
git clone https://github.com/mostafa-k-m/seasalt.git
cd seasalt
poetry install
```

---

## Training
```bash
python train.py
```

The model was trained for 250 epochs on 432 grayscale images using 64×64 patches (stride 10) with random flip/rotation augmentation. Optimization used Adam with a custom `MixL1SSIMLoss` (α=0.84) combining L1 and MS-SSIM.

---

## Evaluation
```bash
python evaluation_script.py
```

Evaluates on BSD68, Kodak24, Urban100, and Set12 across all four noise types.

---

## Benchmarking
```bash
python benchmark.py
```

---

## Citation
```bibtex
@article{kamal2026mpdenoisenet,
  title   = {MPDenoiseNet: Resource-Efficient Deep Learning Approach for Image Denoising},
  author  = {Kamal, Mostafa and Al-Atabany, Walid},
  journal = {Neural Processing Letters},
  volume  = {58},
  pages   = {7},
  year    = {2026},
  doi     = {10.1007/s11063-025-11803-4},
  url     = {https://link.springer.com/article/10.1007/s11063-025-11803-4}
}
```
