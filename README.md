# Learning ImageNet Representations with Variational Autoencoders and GANs

CSE473 Computational Intelligence — Team T24

## VAE Architecture

We implemented a convolutional Variational Autoencoder in PyTorch for 64 × 64 RGB images. The encoder consists of four convolutional layers followed by separate mean and log-variance outputs. The reparameterization trick is used to sample the latent vector, and the decoder reconstructs the image using transposed convolutions.

The objective combines squared-error reconstruction loss and KL divergence, weighted by β.

[Architecture and layer dimensions](docs/vae_architecture.md) · [Architecture diagram](docs/vae_architecture.svg) · [Model implementation](vae.py)

## Baseline Training

The baseline model was trained for 20 epochs with latent dimension 64, β = 1, Adam optimizer, learning rate 0.001, and batch size 128. The original baseline is a separate run from E1 in the six-experiment study.

[Training script](train_vae.py) · [Baseline results](results/baseline/) · [Baseline checkpoint](https://drive.google.com/file/d/1US8y1jP3kFUNhqHgGHwFSbYWOc4yca17/view)

## VAE Experiments

Six experiments were completed using the same 11,529 training and 1,281 validation images and a fixed 20-epoch budget. Changes were made to the latent dimension, β, and KL warm-up, one factor at a time relative to E1.

| Experiment | Latent dimension | Training β | KL warm-up | Validation MSE ↓ | Negative ELBO (β=1) ↓ |
|---|---:|---:|---|---:|---:|
| E1 — Baseline | 64 | 1.0 | Off | 0.01856 | 305.57 |
| E2 — Latent 32 | 32 | 1.0 | Off | 0.02007 | 307.25 |
| E3 — Latent 128 | 128 | 1.0 | Off | 0.01741 | **295.68** |
| E4 — Beta 0.5 | 64 | 0.5 | Off | **0.01723** | 303.89 |
| E5 — Beta 2 | 64 | 2.0 | Off | 0.02031 | 312.94 |
| E6 — KL warm-up | 64 | 1.0 | First 10 epochs | 0.01854 | 305.86 |

The negative ELBO was evaluated using β = 1 for all configurations. **E3 (latent dimension 128)** achieved the lowest negative ELBO and was selected for further evaluation. **E4** achieved the lowest reconstruction MSE but had higher KL divergence.

[Experiments and analysis](docs/vae_experiments.md) · [Detailed results](results/experiment_comparison.csv) · [Original Colab notebook](results/T24_VAE_Experiments.ipynb)

## Training and Validation Curves

![Reconstruction error](results/plots/train_vs_validation_reconstruction.svg)

![KL divergence](results/plots/train_vs_validation_kl.svg)

[Negative ELBO curves](results/plots/validation_negative_elbo.svg) · [Reconstruction comparison](https://drive.google.com/file/d/1TsbPtCPc1NcFqxHNxqn0OKpaFdexHd-r/view)

## Model Checkpoints

The dataset and model weights are stored on Google Drive rather than GitHub.

[All six experiment checkpoints](https://drive.google.com/drive/folders/11uIykXqDXNIukItKjxJA81N2B253q0rB) · [Selected E3 checkpoint](https://drive.google.com/file/d/1RvYug7dPSXC1YuVNoEC67lg2w_yTJDc7/view)
