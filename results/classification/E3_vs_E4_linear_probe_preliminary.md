# E3 versus E4 — frozen encoder classification probe

**Status:** preliminary validation comparison; the source-code provenance check for exact row alignment of `metadata.csv` and `images.npy` has **not yet been completed**.

## Setup

Both encoder checkpoints come from the six-run VAE study after 20 epochs.

- E3: convolutional VAE with **latent dimension 128**, training beta=1.
- E4: convolutional VAE with **latent dimension 64**, training beta=0.5.
- Encoder weights are **frozen**; deterministic posterior mean `mu(x)` provides the feature vectors.
- Same 11,529 training samples and 1,281 validation samples, indexed by `train_indices.npy` and `val_indices.npy`.
- Labels taken from `metadata.csv` with integer class IDs 0 through 9, subject to verifying image/metadata row correspondence.
- Each probe: `StandardScaler()` + `LogisticRegression(max_iter=2000)`, with a **new classifier trained only on the training split**.
- No test data used.

## User-reported Colab validation results

| Metric | E3 (latent 128) | E4 (latent 64) | Difference |
|---|---:|---:|---:|
| Accuracy | **47.54%** | 42.86% | E3 +4.68 percentage points |
| Balanced accuracy | **47.32%** | 42.67% | E3 +4.65 percentage points |
| Number of training feature vectors | 11,529 | 11,529 | Same |
| Features per vector | 128 | 64 | Architecture varies |
| Validation samples | 1,281 | 1,281 | Same |

The E4 training feature array shape was reported as `(11529,64)`, while E3 yielded `(11529,128)`.

### Interpretation

The observed validation accuracy supports **E3 as the better of these two tested frozen representations** under the same linear-probe protocol. This agrees with E3's better beta=1 validation negative ELBO (295.68 versus E4's 303.89), although E4 has slightly lower reconstruction pixel MSE (0.01723 versus 0.01741). Better pixel reconstruction **does not imply** better class separability in a linear probe.

**Important caveat:** E3 has twice the latent feature dimensions of E4 *and* a different training beta. Therefore the probe difference does **not** isolate the causal effect of beta or a single architectural factor. It compares two selected candidate representations. Since the same validation set has been used repeatedly, take account of possible model-selection bias before reporting a final generalization estimate.

### Integrity and next evaluation

1. Confirm exact row alignment of `images.npy` with `metadata.csv` using the dataset construction code and/or an independent visual class audit; matching row counts is not proof.
2. Freeze selection and evaluate the **chosen encoder only once on held-out test labels**, when final evaluation is authorized by the team. Do not use the test set to tune the classifier or model.
3. Save version details, exact probing code, and optionally class-wise precision/recall and confusion matrices for E4 (currently only aggregate E4 metrics were reported).
4. Compare against a random-initialized encoder or another simple baseline to better isolate the benefit of learned features.

These two accuracy values are **user-reported Colab output**, not independently rerun inside the connector.
