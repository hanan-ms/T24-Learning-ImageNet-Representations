# VAE study predictions — Team T24

**Date prepared:** 2026-11-18  
**Study:** Learning ImageNet Representations with Variational Autoencoders and GANs

## Pre-registration status

**E1 (baseline) was already trained before this document was written.** Its results are observed findings, not a pre-registered prediction. At the time this file was prepared, the shared experiment folder contained E1 only. This commit records hypotheses for **E2–E6 before their planned training**; check Git commit dates and run logs for the final chronology.

## Experimental design and predictions

Each run changes **one factor relative to E1**, retaining the same 20-epoch budget and otherwise matching settings.

| Run | Latent dim | Target beta | KL warm-up | Prediction (hypothesis, not a finding) |
| --- | ---: | ---: | --- | --- |
| E1 — Baseline | 64 | 1.0 | Off | Reference run; already completed |
| E2 — Latent32 | 32 | 1.0 | Off | Reconstruction error may increase because the bottleneck contains fewer dimensions |
| E3 — Latent128 | 128 | 1.0 | Off | Reconstruction may improve with more capacity; parameter count and possibly memory use may increase |
| E4 — Beta0_5 | 64 | 0.5 | Off | Weaker KL regularization may reduce reconstruction error while increasing KL |
| E5 — Beta2 | 64 | 2.0 | Off | Stronger KL regularization may reduce KL while increasing reconstruction error |
| E6 — Warmup | 64 | 1.0 | On | Ramping KL weight from 0.1 to 1 over the first 10 epochs may improve latent utilization and reduce collapse risk |

## Controlled settings

- **Dataset:** 10-class ImageNet subset, 64 × 64 RGB.
- **Split:** 11,529 training images; 1,281 validation images.
- **Architecture:** same convolutional encoder and decoder; only latent size changes in E2/E3.
- **Training:** 20 epochs, batch size 128, Adam learning rate 0.001, seed 42.
- **Warm-up schedule:** weight = target beta × min(1, epoch / 10), for epochs numbered from 1.
- **Comparison:** compute validation reconstruction, KL, and **negative ELBO with beta=1** for every run, even when training beta differs.
- **Additional measurements:** per-pixel validation MSE; active units (variance of posterior means across validation data > 0.01); cumulative training time; peak allocated GPU memory; reconstruction and KL curves.
- **Model selection:** prioritize lowest comparable validation negative ELBO, and inspect reconstructions and latent utilization. Do not assume higher beta or a larger latent vector must win.

## Posterior collapse watch

A near-zero KL together with very few active latent units may suggest the encoder is ignoring the input and posteriors approach the standard normal prior. Check these signals alongside reconstructions; neither a single KL value nor a single active-unit count is conclusive on its own.

## E1 observed results (recorded after training, not predictions)

- Epoch 20: validation MSE 0.01856, KL 56.25, negative ELBO (beta=1, without likelihood constant) 305.57, active units 61/64.
- Lowest **reported** epoch negative ELBO was 300.98 at epoch 19.
- These values were reported from the E1 Colab training output. See `results/E1_Baseline/history.json` for the saved numeric history.

## Study sequencing

1. E1 was run before this prediction document existed.
2. Record and commit predictions for E2–E6.
3. Run E2–E6 without changing their hypotheses in retrospect.
4. Compare observed results with these predictions in the final report.
