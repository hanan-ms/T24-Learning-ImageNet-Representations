import os
import time
import random
import json
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from vae import VAE, vae_loss

# Settings
DATA_DIR = "/content/drive/MyDrive/T24_Dataset"
OUTPUT_DIR = "/content/drive/MyDrive/T24_VAE_Results/repro_run"

LATENT_DIM = 64
BETA = 1.0
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Dataset
class VAEDataset(Dataset):
    def __init__(self, images, indices):
        self.images = images
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.images[self.indices[idx]]
        image = image.astype(np.float32)

        if image.max() > 1:
            image /= 255.0

        return torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

images = np.load(DATA_DIR + "/images.npy")
train_indices = np.load(DATA_DIR + "/train_indices.npy")
val_indices = np.load(DATA_DIR + "/val_indices.npy")

train_loader = DataLoader(
    VAEDataset(images, train_indices),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    VAEDataset(images, val_indices),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Model and optimizer
model = VAE(latent_dim=LATENT_DIM).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

checkpoint_path = OUTPUT_DIR + "/checkpoint.pth"

start_epoch = 0
global_step = 0
total_training_time = 0.0
history = []

# Resume if checkpoint exists
if os.path.exists(checkpoint_path):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch = ckpt["epoch"]
    global_step = ckpt["global_step"]
    history = ckpt["history"]
    total_training_time = ckpt["training_time"]

    torch.set_rng_state(ckpt["torch_rng_state"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(
            ckpt["cuda_rng_state"]
        )

    print("Resuming from epoch:", start_epoch)

# Training
for epoch in range(start_epoch, EPOCHS):
    start_time = time.perf_counter()
    model.train()

    train_loss_sum = 0.0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        reconstructed, mean, logvar = model(batch)

        loss, recon, kl = vae_loss(
            reconstructed, batch, mean, logvar, BETA
        )

        if not torch.isfinite(loss).item():
            raise ValueError("Non-finite loss")

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * batch.size(0)
        global_step += 1

    train_loss = train_loss_sum / len(train_loader.dataset)

    # Validation
    model.eval()

    val_loss_sum = 0.0
    val_recon_sum = 0.0
    val_kl_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            reconstructed, mean, logvar = model(batch)

            loss, recon, kl = vae_loss(
                reconstructed, batch, mean, logvar, BETA
            )

            n = batch.size(0)
            val_loss_sum += loss.item() * n
            val_recon_sum += recon.item() * n
            val_kl_sum += kl.item() * n

    val_loss = val_loss_sum / len(val_loader.dataset)
    val_recon = val_recon_sum / len(val_loader.dataset)
    val_kl = val_kl_sum / len(val_loader.dataset)

    epoch_time = time.perf_counter() - start_time
    total_training_time += epoch_time

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_recon": val_recon,
        "val_kl": val_kl,
        "epoch_time": epoch_time
    })

    # Save checkpoint
    torch.save({
        "epoch": epoch + 1,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "training_time": total_training_time,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state()
            if device.type == "cuda" else None
        )
    }, checkpoint_path)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train: {train_loss:.4f} | "
        f"VAL: {val_loss:.4f}"
    )

with open(OUTPUT_DIR + "/history.json", "w") as f:
    json.dump(history, f, indent=4)

print("Training completed!")
