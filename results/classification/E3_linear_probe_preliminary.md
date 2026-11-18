# E3 frozen-encoder linear probe — preliminary validation result

**Status:** preliminary, user-reported Colab execution; label-to-image row alignment is **not yet independently verified** against the original dataset creation process. Do not treat as final validated score until that provenance check is complete.

## Protocol
- Data: 12,810 RGB 64×64 images; 11,529 training / 1,281 validation indices; reserved test set not touched.
- Image features: `VAE(latent_dim=128)` from the saved E3 epoch-20 checkpoint, frozen encoder; deterministic posterior mean `mu(x)` used as 128-dimensional features.
- Classifier: `StandardScaler` followed by `LogisticRegression(max_iter=2000)` (scikit-learn), fit on training data only and evaluated on validation images.
- Class labels: 0–9 from `metadata.csv`, indexed by `train_indices` / `val_indices` under the assumption that row order matches `images.npy`.
- Reconstructed feature array shapes: train `(11529,128)`, validation `(1281,128)`.

## Results
- **Validation accuracy: 47.54%**
- **Balanced accuracy: 47.32%**

| Class | Precision | Recall | F1 | Validation support |
|---:|---:|---:|---:|---:|
| 0 | 0.57 | 0.63 | 0.60 | 130 |
| 1 | 0.39 | 0.49 | 0.44 | 130 |
| 2 | 0.32 | 0.25 | 0.28 | 130 |
| 3 | 0.46 | 0.52 | 0.49 | 127 |
| 4 | 0.47 | 0.39 | 0.43 | 130 |
| 5 | 0.41 | 0.44 | 0.43 | 130 |
| 6 | 0.50 | 0.45 | 0.47 | 130 |
| 7 | 0.38 | 0.29 | 0.33 | 114 |
| 8 | 0.49 | 0.54 | 0.51 | 130 |
| 9 | 0.71 | 0.73 | 0.72 | 130 |

- Macro average precision/recall/F1 approximately 0.47 each.
- Weighted average precision 0.47, recall 0.48, F1 0.47.
- A balanced random-chance benchmark for 10 classes is 10% accuracy, but this is not a controlled random-encoder baseline.

## Limitations and next steps
1. **Required integrity check:** verify that the image array and metadata rows were written in the same order. Matching row counts and disjoint splits **do not prove** this correspondence.
2. Evaluate an **E4 frozen encoder** with an independent, identically configured logistic regression trained on the **same training/validation indices**, and compare accuracy and balanced accuracy.
3. For stronger evidence of representation learning, compare with a randomly initialized encoder and/or a simple raw-pixel baseline under the same linear-probe method.
4. Save the original Colab classifier cell and environment versions. Do not tune on or otherwise inspect the test set prior to final model selection.

Source: classification metrics and per-class report shared from Colab output. The validation score is not a claimed test-set result.
