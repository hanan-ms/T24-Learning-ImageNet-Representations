"""Regenerate VAE experiment curves from results/*/history.json.

Run from repository root:
    python plot_vae_study.py

Requires matplotlib. One chart per image; writes PNG and SVG plots in
results/plots/. Inputs are committed experiment histories, not new training.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
RUNS = [
    "E1_Baseline",
    "E2_Latent32",
    "E3_Latent128",
    "E4_Beta0_5",
    "E5_Beta2",
    "E6_Warmup",
]

PLOT_CONFIG = [
    ("val_recon", "Validation Reconstruction Error", "SSE per image",
     "validation_reconstruction"),
    ("val_kl", "Validation KL Divergence", "KL per image",
     "validation_kl"),
    ("negative_elbo_beta1", "Validation Negative ELBO (beta=1)",
     "Reconstruction + KL per image", "validation_negative_elbo"),
    ("val_mse", "Validation Pixel MSE from decoder(mean)", "MSE",
     "validation_mse"),
]


def load_histories():
    histories = {}
    for name in RUNS:
        path = RESULTS / name / "history.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        records = json.loads(path.read_text())
        if not records or any("epoch" not in r for r in records):
            raise ValueError(f"Invalid history data in {path}")
        histories[name] = records
    return histories


def save_plot(histories, key, title, ylabel, filename, start_epoch=1):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for name in RUNS:
        history = [r for r in histories[name] if r["epoch"] >= start_epoch]
        epochs = [r["epoch"] for r in history]
        values = [r[key] for r in history]
        ax.plot(epochs, values, label=name, linewidth=2)

    ax.set(
        title=title,
        xlabel="Epoch",
        ylabel=ylabel,
        xlim=(start_epoch, 20)
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"{filename}.{ext}", dpi=160)
    plt.close(fig)


def main():
    histories = load_histories()
    PLOTS.mkdir(parents=True, exist_ok=True)
    for key, title, ylabel, filename in PLOT_CONFIG:
        save_plot(histories, key, title, ylabel, filename)
    save_plot(
        histories,
        "negative_elbo_beta1",
        "Validation Negative ELBO (beta=1), Epochs 10–20",
        "Reconstruction + KL per image",
        "validation_negative_elbo_epochs10to20",
        start_epoch=10,
    )
    print(f"Saved VAE curves in {PLOTS}")


if __name__ == "__main__":
    main()
