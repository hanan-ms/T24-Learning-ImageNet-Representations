"""
Train the six VAE ablation configurations from scratch, or resume any run.

This runner reconstructs the Colab study procedure using the archived
configuration, metrics and checkpoint formats. The original notebook should
also be committed for exact provenance. Do not expect bitwise identical
numbers across PyTorch/CUDA versions and hardware.

Example (Google Colab after mounting Drive):
    python run_vae_study.py --run all \
        --data-dir /content/drive/MyDrive/T24_Dataset \
        --output-dir /content/drive/MyDrive/T24_VAE_Results/VAE_6_Experiments

Use --run E3_Latent128 to run just one experiment.
Use --output-dir with an empty new directory to retrain from scratch.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vae import VAE, vae_loss


CONFIGS = [
    {"name": "E1_Baseline", "latent_dim": 64, "beta": 1.0, "warmup": False},
    {"name": "E2_Latent32", "latent_dim": 32, "beta": 1.0, "warmup": False},
    {"name": "E3_Latent128", "latent_dim": 128, "beta": 1.0, "warmup": False},
    {"name": "E4_Beta0_5", "latent_dim": 64, "beta": 0.5, "warmup": False},
    {"name": "E5_Beta2", "latent_dim": 64, "beta": 2.0, "warmup": False},
    {"name": "E6_Warmup", "latent_dim": 64, "beta": 1.0, "warmup": True},
]


class ImageSubset(Dataset):
    def __init__(self, images, indices):
        self.images = images
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image = self.images[self.indices[index]]
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1).copy())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir):
    images = np.load(data_dir / "images.npy")
    train_indices = np.load(data_dir / "train_indices.npy")
    val_indices = np.load(data_dir / "val_indices.npy")

    if images.ndim != 4 or images.shape[1:] != (64, 64, 3):
        raise ValueError("Expected images.npy with shape (N, 64, 64, 3)")
    if not (0 <= images.min() and images.max() > 1 and images.max() <= 255):
        raise ValueError(
            "This study expects raw 0–255 images. "
            "Do not apply a second /255 normalization to already normalized data."
        )
    if len(np.intersect1d(train_indices, val_indices)) > 0:
        raise ValueError("Training/validation indices overlap")

    return images, train_indices, val_indices


def validate(model, loader, device, latent_dim):
    model.eval()
    recon_sum = 0.0
    kl_sum = 0.0
    deterministic_sq_error = 0.0
    num_images = 0
    means = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstructed, mean, logvar = model(batch)
            _, recon, kl = vae_loss(
                reconstructed, batch, mean, logvar, beta=1.0
            )
            n = batch.size(0)
            recon_sum += recon.item() * n
            kl_sum += kl.item() * n
            deterministic_output = model.decoder(mean)
            deterministic_sq_error += (
                (deterministic_output - batch).pow(2)
            ).sum().item()
            means.append(mean.cpu())
            num_images += n

    if num_images == 0:
        raise ValueError("Validation set is empty")
    posterior_means = torch.cat(means, dim=0)
    active_units = int(
        (posterior_means.var(dim=0, unbiased=False) > 0.01).sum().item()
    )
    recon = recon_sum / num_images
    kl = kl_sum / num_images

    return {
        "val_recon": recon,
        "val_kl": kl,
        "val_mse": deterministic_sq_error / (num_images * 3 * 64 * 64),
        "negative_elbo_beta1": recon + kl,
        "active_units": active_units,
    }


def run_experiment(config, data, args, device):
    images, train_indices, val_indices = data
    name = config["name"]
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pth"
    history_path = run_dir / "history.json"

    seed_all(args.seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_data = ImageSubset(images, train_indices)
    val_data = ImageSubset(images, val_indices)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True, generator=loader_generator, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = VAE(latent_dim=config["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    global_step = 0
    history = []
    total_time = 0.0
    peak_memory = 0.0

    if checkpoint_path.exists():
        ckpt = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if ckpt["config"] != config:
            raise ValueError("Existing checkpoint does not match this run config")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        history = ckpt["history"]
        total_time = ckpt["training_time"]
        peak_memory = ckpt["peak_memory_mb"]
        torch.set_rng_state(ckpt["torch_rng_state"])
        loader_generator.set_state(ckpt["loader_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_states"])
        print(f"Resuming {name} from epoch {start_epoch}")

    print(f"\nSTARTING: {name}")
    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()

        effective_beta = (
            config["beta"] * min(1.0, (epoch + 1) / args.warmup_epochs)
            if config["warmup"] else config["beta"]
        )

        model.train()
        total_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output, mean, logvar = model(batch)
            loss, recon, kl = vae_loss(
                output, batch, mean, logvar, beta=effective_beta
            )
            if not torch.isfinite(loss).item():
                raise ValueError(f"Non-finite loss in {name} epoch {epoch+1}")
            loss.backward()
            optimizer.step()
            n = batch.size(0)
            total_loss_sum += loss.item() * n
            train_recon_sum += recon.item() * n
            train_kl_sum += kl.item() * n
            global_step += 1

        val = validate(model, val_loader, device, config["latent_dim"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        total_time += epoch_time
        memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if device.type == "cuda" else 0.0
        )
        peak_memory = max(peak_memory, memory)

        entry = {
            "epoch": epoch + 1,
            "beta_used": effective_beta,
            "train_loss": total_loss_sum / len(train_data),
            "train_recon": train_recon_sum / len(train_data),
            "train_kl": train_kl_sum / len(train_data),
            **val,
            "epoch_time": epoch_time,
            "peak_memory_mb": memory,
        }
        history.append(entry)

        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "config": config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "training_time": total_time,
                "peak_memory_mb": peak_memory,
                "torch_rng_state": torch.get_rng_state(),
                "loader_rng_state": loader_generator.get_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all()
                    if device.type == "cuda" else []
                ),
            },
            checkpoint_path,
        )
        history_path.write_text(json.dumps(history, indent=4) + "\n")

        print(
            f"{name} | Epoch {epoch+1}/{args.epochs} | "
            f"VAL MSE: {val['val_mse']:.5f} | "
            f"KL: {val['val_kl']:.2f} | "
            f"ELBO: {val['negative_elbo_beta1']:.2f} | "
            f"Active: {val['active_units']}/{config['latent_dim']}"
        )

    if not history:
        raise ValueError(f"No history for {name}")
    print(f"COMPLETED: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", choices=["all"] + [c["name"] for c in CONFIGS],
        default="all"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("/content/drive/MyDrive/T24_Dataset")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/content/drive/MyDrive/T24_VAE_Results/VAE_6_Experiments")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.warmup_epochs <= 0:
        parser.error("Epochs, batch size and warm-up epochs must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(args.data_dir)
    print(f"Device: {device}; images: {len(data[0])}; "
          f"train: {len(data[1])}; val: {len(data[2])}")

    selected = CONFIGS if args.run == "all" else [
        c for c in CONFIGS if c["name"] == args.run
    ]
    for config in selected:
        run_experiment(config, data, args, device)


if __name__ == "__main__":
    main()
