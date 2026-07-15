# -*- coding: utf-8 -*-
"""
1D-SVD denoising autoencoder.

The 1D-SVD input is a flat concatenation
    [U[:, :r].flatten(order='C'), S[:r], V[:r, :].flatten(order='C')]
so a plain Conv1d across the whole vector would slide across the U|S|V
boundaries, which is meaningless. Instead we run three independent MLP
branches over the three segments, concatenate their bottlenecks, and
decode back through three heads to the original segments.

Output activations:
    * U, V heads : linear (values are in roughly [-1, 1] but not bounded)
    * S head     : softplus (singular values are strictly positive)

Loss is plain MSE over the full 4232-vector (see
train_svd1d_autoencoder.py). U/V dominate by count; S will get a small
signal but that is fine, since downstream cares about the reconstructed
image which is mostly driven by U and V.
"""

import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


DEFAULT_H = 16
DEFAULT_W = 512
DEFAULT_R = 8


def svd1d_split_sizes(H=DEFAULT_H, W=DEFAULT_W, r=DEFAULT_R):
    return H * r, r, r * W  # U_flat, S, V_flat


def split_svd1d(x, H=DEFAULT_H, W=DEFAULT_W, r=DEFAULT_R):
    """Split a (..., L) tensor into (U_flat, S, V_flat) segments."""
    nU, nS, nV = svd1d_split_sizes(H, W, r)
    return (
        x[..., :nU],
        x[..., nU:nU + nS],
        x[..., nU + nS:nU + nS + nV],
    )


def reconstruct_2d(x, H=DEFAULT_H, W=DEFAULT_W, r=DEFAULT_R):
    """(..., L) -> (..., H, W) via U @ diag(S) @ V.

    Works on a batch of 1D SVDs and returns the batched 2D image.
    """
    u_flat, s, v_flat = split_svd1d(x, H, W, r)
    U = u_flat.reshape(*u_flat.shape[:-1], H, r)
    V = v_flat.reshape(*v_flat.shape[:-1], r, W)
    return U @ (s.unsqueeze(-1) * V)


class SegmentNormalizedMSE(nn.Module):
    """MSE normalized per-segment by the target's own per-sample mean-square.

    Splits both output and target into (U, S, V) via split_svd1d, then for each
    segment divides the squared error by the target segment's mean-square
    (detached, so no gradient flows through the denominator). The three
    normalized segment MSEs are summed to give one scalar per sample, so
    U/S/V each contribute full-strength — unlike plain MSE over the flat
    vector, where V's small natural magnitude lets the model collapse to
    ~zero on U/V while S carries the loss.
    """

    def __init__(self, H=DEFAULT_H, W=DEFAULT_W, r=DEFAULT_R, eps=1e-8):
        super().__init__()
        self.H, self.W, self.r = H, W, r
        self.eps = eps

    def forward(self, outputs, targets):
        u_out, s_out, v_out = split_svd1d(outputs, self.H, self.W, self.r)
        u_tgt, s_tgt, v_tgt = split_svd1d(targets, self.H, self.W, self.r)

        with torch.no_grad():
            u_denom = u_tgt.pow(2).mean(dim=-1, keepdim=True) + self.eps
            s_denom = s_tgt.pow(2).mean(dim=-1, keepdim=True) + self.eps
            v_denom = v_tgt.pow(2).mean(dim=-1, keepdim=True) + self.eps

        u_loss = ((u_out - u_tgt).pow(2) / u_denom).mean(dim=-1)
        s_loss = ((s_out - s_tgt).pow(2) / s_denom).mean(dim=-1)
        v_loss = ((v_out - v_tgt).pow(2) / v_denom).mean(dim=-1)

        return u_loss + s_loss + v_loss


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, out_activation=None):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if out_activation is not None:
            layers.append(out_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SVD1D_Autoencoder(nn.Module):
    """Segmented-MLP autoencoder over the (U, S, V) blocks of a 1D SVD."""

    def __init__(
        self,
        H=DEFAULT_H,
        W=DEFAULT_W,
        r=DEFAULT_R,
        u_encoder_hidden=(256, 128),
        s_encoder_hidden=(32, 32),
        v_encoder_hidden=(2048, 512),
        bottleneck=256,
        u_decoder_hidden=(128, 256),
        s_decoder_hidden=(32, 32),
        v_decoder_hidden=(512, 2048),
        dtype=torch.float32,
    ):
        super().__init__()
        self.H, self.W, self.r = H, W, r
        self.nU, self.nS, self.nV = svd1d_split_sizes(H, W, r)
        self.dtype = dtype

        u_enc_out = u_encoder_hidden[-1] if u_encoder_hidden else self.nU
        s_enc_out = s_encoder_hidden[-1] if s_encoder_hidden else self.nS
        v_enc_out = v_encoder_hidden[-1] if v_encoder_hidden else self.nV

        self.u_encoder = _MLP(self.nU, u_encoder_hidden[:-1], u_enc_out)
        self.s_encoder = _MLP(self.nS, s_encoder_hidden[:-1], s_enc_out)
        self.v_encoder = _MLP(self.nV, v_encoder_hidden[:-1], v_enc_out)

        concat_dim = u_enc_out + s_enc_out + v_enc_out
        self.fuse = nn.Sequential(
            nn.Linear(concat_dim, bottleneck),
            nn.ReLU(),
        )

        self.u_decoder = _MLP(bottleneck, u_decoder_hidden, self.nU)
        self.s_decoder = _MLP(bottleneck, s_decoder_hidden, self.nS,
                              out_activation=nn.Softplus())
        self.v_decoder = _MLP(bottleneck, v_decoder_hidden, self.nV)

        self.to(dtype)

    def forward(self, x):
        u_in, s_in, v_in = split_svd1d(x, self.H, self.W, self.r)
        z = torch.cat([
            self.u_encoder(u_in),
            self.s_encoder(s_in),
            self.v_encoder(v_in),
        ], dim=-1)
        z = self.fuse(z)
        u_out = self.u_decoder(z)
        s_out = self.s_decoder(z)
        v_out = self.v_decoder(z)
        return torch.cat([u_out, s_out, v_out], dim=-1)

    def train_model(self, train_dataloader, val_dataloader, criterion, optimizer,
                    scheduler, model_save_dir, identifier, device,
                    checkpoints_enabled=True, resume_from_checkpoint=False,
                    max_epochs=10, figures_dir=None):
        self.to(device)
        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_epoch = 0
        start_epoch = 0

        if figures_dir is None:
            figures_dir = model_save_dir
        os.makedirs(figures_dir, exist_ok=True)

        checkpoint_path = os.path.join(model_save_dir, f"{identifier}_checkpoint.pth")

        if checkpoints_enabled and resume_from_checkpoint and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            train_losses = checkpoint["train_losses"]
            val_losses = checkpoint["val_losses"]
            best_val_loss = checkpoint["best_val_loss"]
            best_epoch = checkpoint["best_epoch"]

        log_path = os.path.join(model_save_dir, f"{identifier}_run_time_info.txt")
        with open(log_path, "a") as f:
            f.write(
                f"Training resumed at {datetime.datetime.now()} from epoch {start_epoch}\n"
                if start_epoch > 0
                else f"Training started at {datetime.datetime.now()}\n"
            )

            best_model = self.state_dict().copy()

            for epoch in range(start_epoch, max_epochs):
                self.train()
                running_train_loss = 0.0
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    inputs, labels = batch
                    inputs = inputs.to(device, torch.float32)
                    labels = labels.squeeze()
                    labels = labels.to(device, torch.float32)

                    outputs = self(inputs)
                    losses = criterion(outputs, labels)
                    loss = torch.mean(losses)
                    loss.backward()
                    optimizer.step()
                    running_train_loss += loss.item()

                train_loss = running_train_loss / len(train_dataloader)
                train_losses.append(train_loss)

                self.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for batch in val_dataloader:
                        inputs, labels = batch
                        inputs = inputs.to(device, torch.float32)
                        labels = labels.squeeze()
                        labels = labels.to(device, torch.float32)

                        outputs = self(inputs)
                        loss = torch.mean(criterion(outputs, labels))
                        running_val_loss += loss.item()

                val_loss = running_val_loss / len(val_dataloader)
                val_losses.append(val_loss)

                f.write(
                    f"Epoch [{epoch+1}/{max_epochs}] - "
                    f"Train Loss: {train_loss:.10f}, "
                    f"Validation Loss: {val_loss:.10f}\n\n"
                )
                print(
                    f"Epoch [{epoch+1}/{max_epochs}] - "
                    f"Train Loss: {train_loss:.10f}, "
                    f"Validation Loss: {val_loss:.10f}"
                )

                should_stop = scheduler.step(val_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model = self.state_dict().copy()
                    torch.save(
                        self.state_dict(),
                        os.path.join(model_save_dir, f"{identifier}_best_model.pth"),
                    )

                if checkpoints_enabled:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                            "best_val_loss": best_val_loss,
                            "best_epoch": best_epoch,
                        },
                        checkpoint_path,
                    )

                if should_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    f.write(f"Early stopping at epoch {epoch+1}\n")
                    break
                f.flush()

        run_summary_path = os.path.join(model_save_dir, f"{identifier}_run_summary.txt")
        with open(run_summary_path, "w") as file:
            file.write(f"Number of Epochs for Best Model: {best_epoch + 1}\n")
            file.write(f"Final Training Loss: {train_losses[-1]:.10f}\n")
            file.write(f"Final Validation Loss: {val_losses[-1]:.10f}\n")

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.scatter(best_epoch, val_losses[best_epoch], marker="*",
                    color="red", label="Best Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("1D-SVD Autoencoder: Train/Val Loss")
        plt.legend()
        plt.savefig(os.path.join(figures_dir, f"{identifier}_losses.pdf"))
        plt.close()

        return best_model, best_epoch, train_losses[-1], val_losses[-1], best_val_loss
