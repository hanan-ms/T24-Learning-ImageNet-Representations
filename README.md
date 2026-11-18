# Learning ImageNet Representations with Variational Autoencoders and GANs

**Team T24 — CSE473 Computational Intelligence**

This private repository is for the **full team project**. Its current content covers the **convolutional VAE baseline and the start of the six-run VAE ablation study**; GAN, VAE-GAN, latent-space analysis, classification and final reporting are future team work.

## Current files

- `vae.py`: from-scratch PyTorch convolutional encoder, decoder, VAE with reparameterization, and Gaussian squared-error reconstruction + beta-weighted KL loss.
- `train_vae.py`: original 20-epoch standalone **baseline reproduction** script with checkpoint resume. This script is **not** the six-run notebook/function.
- `results/baseline/vae_baseline_evaluation.json`: validation metrics from the **original** 20-epoch baseline run.
- `results/baseline/vae_baseline_run_record.json`: original run settings, time, memory and checkpoint SHA-256.
- `results/E1_Baseline/history.json` and `summary.json`: 20 epochs of E1 training/validation and its final metrics.
- `results/E2_Latent32/history.json` and `summary.json`: 20 epochs of E2 (latent dimension 32) and its final metrics.
- `results/E3_Latent128/history.json` and `summary.json`: 20 epochs of E3 (latent dimension 128) and its final metrics.
- `results/predictions.md`: hypotheses for E2–E6, committed **after E1** and before the remaining planned runs.

### Architecture

Input: 64×64×3 RGB (normalized to [0,1]). Four stride-2 convolution layers with output channels 32, 64, 128, 256. Encoder outputs a mean and log-variance vector. Sample `z = mean + exp(0.5 * logvar) * epsilon` where `epsilon ~ N(0,I)`. The decoder uses a linear layer and four transposed convolutions to reconstruct 64×64×3 RGB.

Loss per image averaged over a batch: **sum of squared pixel errors + beta × KL divergence**. For the stated fixed-variance Gaussian reconstruction model (variance 0.5), the squared-error term is the negative Gaussian log-likelihood **up to an additive constant**. Reported negative ELBO values omit this constant.

### Baseline reference results

Original 20-epoch baseline (not identical to E1): validation pixel MSE **0.018248**, reconstruction loss **241.9633**, KL **56.3608**, negative ELBO **298.3241**, active units **63/64**. Recorded training time **62.10 s**; peak allocated GPU memory **179.13 MiB**; Tesla T4. See JSON files for exact values.

New six-run study **E1 (epoch 20)**: pixel MSE **0.018563**, validation reconstruction **249.324**, KL **56.249**, negative ELBO at beta=1 **305.572**, active units **61/64**. The best **reported epoch-level** E1 negative ELBO was **300.984 at epoch 19**; the saved checkpoint on Drive is epoch 20. Stochastic decoder reconstructions can cause evaluation variation.

Completed **E2, latent dimension 32 (epoch 20)**: pixel MSE **0.020070**, validation reconstruction **259.918**, KL **47.332**, negative ELBO at beta=1 **307.249**, active units **32/32**. Run time over the logged training+validation epochs **60.59 s**, peak allocated GPU memory **149.05 MiB**. E1 corresponding values: **66.51 s**, **153.66 MiB**.

Completed **E3, latent dimension 128 (epoch 20)**: pixel MSE **0.017414**, validation reconstruction **236.923**, KL **58.754**, negative ELBO at beta=1 **295.677**, active units **124/128**. Logged epoch-loop training+validation time **58.42 s**, peak allocated GPU memory **162.87 MiB**. At epoch 20, E3 has the lowest validation negative ELBO of E1/E2/E3. The different run times may reflect GPU runtime variation, not just model capacity. Pixel MSE is measured on `decoder(mean)`, while ELBO reconstruction is sampled from the posterior.

### Study design (20 epochs per run)

| Run | Latent size | Training beta | KL warm-up |
| --- | ---: | ---: | --- |
| E1 baseline | 64 | 1 | Off |
| E2 small latent | 32 | 1 | Off |
| E3 large latent | 128 | 1 | Off |
| E4 low beta | 64 | 0.5 | Off |
| E5 high beta | 64 | 2 | Off |
| E6 warm-up | 64 | 1 | On, first 10 epochs |

Compare all runs by **validation negative ELBO evaluated with beta=1**, validation reconstruction and KL, pixel MSE, active latent units (variance of posterior mean across validation examples >0.01), runtime, peak GPU memory and train/validation curves. **E1, E2 and E3 are complete; E4–E6 are pending.**

## Data and model checkpoints (Google Drive)

Binary dataset, checkpoints, and images are kept on Drive rather than Git:

- [T24 VAE Results folder](https://drive.google.com/drive/folders/13Dd6eMQw5WgDvtS75a2h4Qf71xs8T7A5)
- [Original baseline checkpoint (epoch 20)](https://drive.google.com/file/d/1US8y1jP3kFUNhqHgGHwFSbYWOc4yca17/view)
- [Six experiments folder](https://drive.google.com/drive/folders/11uIykXqDXNIukItKjxJA81N2B253q0rB)
- [E1 epoch-20 checkpoint](https://drive.google.com/file/d/1k1mF9-VPvdCLs8cr3O2N7gspdpdIgBGY/view)
- [E2 epoch-20 checkpoint](https://drive.google.com/file/d/1BDwHuDl8Yqd53TBiQzAsbLrgDMdpSVNg/view)
- [E3 epoch-20 checkpoint](https://drive.google.com/file/d/1RvYug7dPSXC1YuVNoEC67lg2w_yTJDc7/view)
- [Original baseline training curves](https://drive.google.com/file/d/15V4r2NNd_BnKg1EJQUScUX7EYFJBsdUo/view)
- [Original baseline reconstruction examples](https://drive.google.com/file/d/149Mz3JF7Cu2_PbdFMrjzBeL95ZPmXhiS/view)

Drive access is separately controlled; team members may need sharing permission.

## Running the original baseline

1. In Google Colab, select a GPU and mount Drive:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
2. Ensure `images.npy`, `train_indices.npy`, `val_indices.npy` are accessible under `/content/drive/MyDrive/T24_Dataset` (or change `DATA_DIR` in `train_vae.py`).
3. Place `vae.py` and `train_vae.py` in the same working directory, and change `OUTPUT_DIR` in `train_vae.py` to a directory you can write.
4. Run `python train_vae.py` with PyTorch and NumPy installed. **Be aware:** if a `checkpoint.pth` already exists at `OUTPUT_DIR`, the script resumes from it. Set a fresh output directory for a new run.

The separate E1–E6 experiment loop was executed in a Colab notebook and is **not yet committed as a runnable script**. Its recorded E1 history and checkpoint are available above. To reproduce the study completely, the team should commit that notebook or a standalone experiment runner next.

## Status and integrity

- `results/predictions.md` explicitly notes that E1 was run before predictions were written; do **not** claim E1 was pre-registered.
- The original baseline checkpoint SHA-256 is recorded in `vae_baseline_run_record.json`; the team should verify it after downloading.
- Keep large `.npy`, `.pth`, image datasets and cache files out of Git. 
- No test-set model selection. Use validation for comparing models; preserve the test split for final evaluation.
