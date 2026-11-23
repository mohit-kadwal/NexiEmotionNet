# NexiEmotionNet

**A Lightweight Channel-Spatial Attention Convolutional Neural Network for Real-Time Facial Emotion Recognition**

Mohit Kadwal (Prestige Institute of Engineering Management and Research, Indore, India) and
Maya Ingle (SCSIT, DAVV Indore, India)

This repository contains the code and result tables for the paper above. NexiEmotionNet stacks four custom
convolution blocks (two 3×3 conv + BatchNorm + ReLU + 2×2 max-pooling each), followed by a simplified CBAM
(average-pool channel attention + 7×7 spatial attention), global average pooling, dropout and a fully connected
classifier. It is trained **from scratch** on the Real-World Affective Faces Database (RAF-DB) with focal loss,
AdamW and a one-cycle learning-rate schedule.

## Main results (RAF-DB test set, 3,068 images, mean ± SD over 5 seeds)

| Metric | Value |
|---|---|
| Accuracy | 78.81 ± 0.77 % |
| Macro-F1 | 69.30 ± 1.07 % |
| Weighted F1 | 78.44 ± 0.81 % |
| Parameters | 1,184,538 (1.185 M) |
| FLOPs (224×224, 2 × MACs) | 5.173 G |
| GPU latency (RTX 5060, batch 1, FP32) | 2.23 ms |
| CPU latency (4 threads, batch 1, FP32) | 25.5 ± 0.9 ms |

Comparison under the identical training pipeline (trained from scratch, same data, augmentation, loss,
optimizer, scheduler, epochs and seeds):

| Model | Accuracy (%) | Params (M) | GFLOPs | GPU latency (ms) | CPU latency (ms) |
|---|---|---|---|---|---|
| NexiEmotionNet | 78.81 ± 0.77 | 1.185 | 5.173 | 2.23 | 25.5 ± 0.9 |
| ResNet-18 | 83.42 ± 0.58 | 11.18 | 3.627 | 2.99 | 21.5 ± 1.0 |
| MobileNetV2 | 80.95 ± 0.36 | 2.233 | 0.599 | 2.98 | 10.3 ± 0.2 |
| EfficientNet-B0 | 78.40 ± 0.49 | 4.017 | 0.769 | 5.01 | 14.7 ± 0.2 |

NexiEmotionNet has the fewest parameters and the lowest GPU latency, but it needs more FLOPs and has a higher
CPU latency than the baselines, and ResNet-18 and MobileNetV2 are significantly more accurate. See the paper for
the ablation study, per-class results and statistical tests.

## Repository contents

```
NexiEmotionNet/
├── NexiEmotionNet_experiments.ipynb   # complete experiment notebook (run top to bottom)
├── requirements.txt                   # Python packages
├── data/README.md                     # how to obtain RAF-DB and the expected folder layout
├── results/
│   ├── tables/                        # CSV result tables produced by the notebook
│   └── figures/                       # confusion matrices (300 dpi PNG)
├── CITATION.cff
├── LICENSE
└── .gitignore
```

## Environment

The reported results were produced with:

| Item | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 (8 GB) |
| OS | Windows 10 (build 19045) |
| Python | 3.13.7 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA / cuDNN | 12.8 / 9.19.0 |

Install a CUDA build of PyTorch first (see <https://pytorch.org> for the command that matches your driver), for example:

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

All computations use FP32 (TF32 disabled) with `cudnn.deterministic = True` and `cudnn.benchmark = False`.
`torch.use_deterministic_algorithms` is not enabled, so bit-identical results across machines are not guaranteed;
expect small differences in accuracy on different hardware.

## How to reproduce

1. Obtain RAF-DB from its authors (it cannot be redistributed here) and place `RAF DB Dataset.zip` next to the
   notebook. See [`data/README.md`](data/README.md) for the expected layout.
2. Open `NexiEmotionNet_experiments.ipynb` and run the cells from top to bottom. The notebook trains
   NexiEmotionNet, the three ablation variants and the three baselines for seeds 0–4 (30 epochs, batch size 64),
   measures parameters, FLOPs and latency, and runs the statistical tests.
3. Every finished run is saved as a JSON file in `NexiEmotionNet_results/runs/`. If the kernel stops you can
   re-run the notebook and finished runs are skipped. Tables and figures are written to
   `NexiEmotionNet_results/tables/` and `NexiEmotionNet_results/figures/`.

Training uses an NVIDIA GPU and takes several hours for all 35 runs (5 seeds × 7 configurations).
Use the same GPU for all runs so that the latency comparison is fair.

## Training configuration

| Setting | Value |
|---|---|
| Input | 224×224 RGB, normalised with mean = std = 0.5 |
| Augmentation | horizontal flip (p = 0.5), rotation ±10°, colour jitter (0.2 for brightness, contrast, saturation and hue) |
| Optimizer | AdamW (weight decay 0.01, default betas) |
| Scheduler | OneCycleLR, max_lr 1e-3, pct_start 0.3, cosine annealing, div_factor 25, final_div_factor 1e4, stepped every batch |
| Loss | focal loss, γ = 2.0, α = 0.25 (cross-entropy in ablation variants a and b) |
| Epochs / batch size | 30 / 64 |
| Dropout | 0.4 after global average pooling |
| Evaluation | final-epoch model on the RAF-DB test split, no validation set, no checkpoint selection |
| Seeds | 0, 1, 2, 3, 4 |

## Citation

If you use this code, please cite the paper (details to be completed after publication):

```bibtex
@article{kadwal_nexiemotionnet,
  title   = {NexiEmotionNet: A Lightweight Channel-Spatial Attention Convolutional Neural Network for Real-Time Facial Emotion Recognition},
  author  = {Kadwal, Mohit and Ingle, Maya},
  journal = {TO BE ADDED},
  year    = {TO BE ADDED}
}
```

GitHub also offers a "Cite this repository" button based on [`CITATION.cff`](CITATION.cff).

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). RAF-DB is **not** part of this repository
and is subject to its own license from the dataset authors.

## Contact

Mohit Kadwal, Prestige Institute of Engineering Management and Research, Indore, India.
