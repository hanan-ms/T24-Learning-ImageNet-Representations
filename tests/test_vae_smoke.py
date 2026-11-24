"""Fast CPU-only architecture and training-runner smoke tests.

Uses synthetic images, never the project dataset or held-out test set.
Run with: python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vae import VAE, vae_loss
from run_vae_study import CONFIGS, load_dataset, run_experiment


class TestVAESmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_architecture_shapes_and_backward(self):
        for latent_dim in (32, 64, 128):
            with self.subTest(latent_dim=latent_dim):
                model = VAE(latent_dim=latent_dim)
                x = torch.rand(2, 3, 64, 64)
                mu, logvar = model.encoder(x)
                self.assertEqual(tuple(mu.shape), (2, latent_dim))
                self.assertEqual(tuple(logvar.shape), (2, latent_dim))
                z = model.reparameterize(mu, logvar)
                self.assertEqual(tuple(z.shape), (2, latent_dim))
                y = model.decoder(z)
                self.assertEqual(tuple(y.shape), (2, 3, 64, 64))
                self.assertGreaterEqual(float(y.min()), 0.0)
                self.assertLessEqual(float(y.max()), 1.0)
                loss, recon, kl = vae_loss(y, x, mu, logvar, beta=1)
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(torch.isfinite(recon).item())
                self.assertTrue(torch.isfinite(kl).item())
                self.assertGreaterEqual(float(kl.detach()), 0.0)
                loss.backward()
                self.assertIsNotNone(model.encoder.conv_layers[0].weight.grad)

    def test_loss_definition(self):
        original = torch.zeros((2, 3, 64, 64))
        reconstructed = torch.ones_like(original)
        mu = torch.zeros((2, 64))
        logvar = torch.zeros_like(mu)
        loss, recon, kl = vae_loss(
            reconstructed, original, mu, logvar, beta=2.0
        )
        self.assertAlmostEqual(float(recon), 3 * 64 * 64)
        self.assertAlmostEqual(float(kl), 0.0, places=5)
        self.assertAlmostEqual(float(loss), 3 * 64 * 64)
        mu[:, 0] = 1.0
        loss, recon, kl = vae_loss(
            reconstructed, original, mu, logvar, beta=2.0
        )
        self.assertAlmostEqual(float(kl), 0.5, places=5)
        self.assertAlmostEqual(
            float(loss), float(recon) + 2.0 * float(kl), places=3
        )

    def test_synthetic_data_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            images = np.random.default_rng(42).integers(
                0, 256, size=(6, 64, 64, 3), dtype=np.uint8
            )
            np.save(p / "images.npy", images)
            np.save(p / "train_indices.npy", np.array([0, 1, 2, 3]))
            np.save(p / "val_indices.npy", np.array([4, 5]))
            result = load_dataset(p)
            self.assertEqual(len(result[0]), 6)
            self.assertEqual(len(result[1]), 4)
            self.assertEqual(len(result[2]), 2)

    def test_six_configs_one_epoch_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            rng = np.random.default_rng(42)
            images = rng.integers(
                0, 256, size=(6, 64, 64, 3), dtype=np.uint8
            )
            data = (
                images,
                np.array([0, 1, 2, 3]),
                np.array([4, 5])
            )
            args = SimpleNamespace(
                output_dir=Path(folder),
                batch_size=2,
                learning_rate=0.001,
                seed=42,
                epochs=1,
                warmup_epochs=10,
            )
            for config in CONFIGS:
                with self.subTest(name=config["name"]):
                    run_experiment(config, data, args, torch.device("cpu"))
                    cp = args.output_dir / config["name"] / "checkpoint.pth"
                    history_path = cp.parent / "history.json"
                    self.assertTrue(cp.exists())
                    self.assertTrue(history_path.exists())
                    ckpt = torch.load(
                        cp, map_location="cpu", weights_only=True
                    )
                    self.assertEqual(ckpt["epoch"], 1)
                    self.assertEqual(len(ckpt["history"]), 1)
                    record = ckpt["history"][0]
                    self.assertAlmostEqual(
                        record["negative_elbo_beta1"],
                        record["val_recon"] + record["val_kl"],
                        places=3,
                    )
                    self.assertLessEqual(
                        record["active_units"], config["latent_dim"]
                    )
                    self.assertGreaterEqual(
                        record["active_units"], 0
                    )
                    expected_beta = (
                        config["beta"] / 10 if config["warmup"]
                        else config["beta"]
                    )
                    self.assertAlmostEqual(
                        record["beta_used"], expected_beta
                    )
                    # Already completed one epoch: resume should not retrain.
                    run_experiment(config, data, args, torch.device("cpu"))
                    resume = torch.load(
                        cp, map_location="cpu", weights_only=True
                    )
                    self.assertEqual(len(resume["history"]), 1)


if __name__ == "__main__":
    unittest.main()
