import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.traj_verification import read_colvar_fields
from src.models.jepa_model import JEPAModel

CONTACT_MAP_PATH = Path("data/jepa_contact_maps.pt")
EQ_CONTACT_MAP_PATH = Path("data/eq_jepa_contact_maps.pt")
WEIGHTS_PATH = Path("models/best_jepa.pth")
COLVAR_PATH = Path("data/metad_S1_BAR_COLVAR")
OUTPUT_PLOT_PATH = Path("figures/metadynamics_anomaly.png")

LATENT_DIM = 128
PREDICTOR_HIDDEN_DIM = 256
EXPECTED_FRAMES = 10000
BATCH_SIZE_METADYNAMICS = 128
POCKET_DISTANCE_COL = "cv1"
COLVAR_DISTANCE_UNIT = "nm"


def get_device():
    if not torch.backends.mps.is_available():
        raise SystemExit("use MPS backend")
    return torch.device("mps")


def load_contact_maps(path):
    contact_maps = torch.load(path, map_location="cpu")
    if contact_maps.dim() == 3:
        contact_maps = contact_maps.unsqueeze(1)
    if contact_maps.dim() != 4:
        raise ValueError("contact_maps must have shape (frames, 1, atoms, atoms)")
    return contact_maps.float()


def load_colvar_distance(path, column):
    fields = read_colvar_fields(path)
    colvar_df = pd.read_csv(path, comment="#", header=None, delimiter="\s+")
    if fields and len(fields) == colvar_df.shape[1]:
        colvar_df.columns = fields
    else:
        colvar_df.columns = [f"col{i}" for i in range(colvar_df.shape[1])]

    if column not in colvar_df.columns:
        raise ValueError(f"column '{column}' not found in COLVAR file")

    return colvar_df[column].to_numpy(dtype=np.float32)


def align_series(energies, distances):
    max_len = min(len(energies), len(distances))
    energies = energies[:max_len]
    distances = distances[:max_len]
    frames = np.arange(max_len)
    return frames, energies, distances


def compute_latent_distances(contact_maps, model, centroid, device, batch_size):
    if not batch_size or batch_size <= 0 or batch_size >= contact_maps.shape[0]:
        meta_latents = model.target_encoder(contact_maps.to(device))
        diffs = meta_latents - centroid
        return torch.linalg.norm(diffs, dim=1)

    distances = []
    for start in range(0, contact_maps.shape[0], batch_size):
        batch = contact_maps[start : start + batch_size].to(device)
        batch_latents = model.target_encoder(batch)
        batch_distances = torch.linalg.norm(batch_latents - centroid, dim=1)
        distances.append(batch_distances.cpu())

    return torch.cat(distances, dim=0)


def main():
    device = get_device()

    if not CONTACT_MAP_PATH.exists():
        raise SystemExit(f"missing contact map tensor: {CONTACT_MAP_PATH}")
    if not EQ_CONTACT_MAP_PATH.exists():
        raise SystemExit(f"missing contact map tensor: {EQ_CONTACT_MAP_PATH}")
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"missing model weights: {WEIGHTS_PATH}")
    if not COLVAR_PATH.exists():
        raise SystemExit(f"missing COLVAR file: {COLVAR_PATH}")

    model = JEPAModel(
        in_channels=1,
        latent_dim=LATENT_DIM,
        predictor_hidden_dim=PREDICTOR_HIDDEN_DIM,
    )
    model.eval()
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    eq_contact_maps = load_contact_maps(EQ_CONTACT_MAP_PATH)
    eq_contact_maps = eq_contact_maps.to(device)

    contact_maps = load_contact_maps(CONTACT_MAP_PATH)
    n_frames = contact_maps.shape[0]
    if n_frames != EXPECTED_FRAMES:
        print(f"warning: expected {EXPECTED_FRAMES} frames, found {n_frames}")

    with torch.no_grad():
        eq_latents = model.target_encoder(eq_contact_maps)
        eq_centroid = eq_latents.mean(dim=0)
        centroid_norm = torch.linalg.norm(eq_centroid).item()

        energies = compute_latent_distances(
            contact_maps,
            model,
            eq_centroid,
            device,
            BATCH_SIZE_METADYNAMICS,
        )

    energies = energies.detach().cpu().numpy().astype(np.float32)
    print(f"\nequilibrium centroid L2 norm: {centroid_norm:.6f}")
    if energies.size:
        print(
            "energy stats (min/mean/max): "
            f"{energies.min():.6f} / {energies.mean():.6f} / {energies.max():.6f}"
        )

    distance_series = load_colvar_distance(COLVAR_PATH, POCKET_DISTANCE_COL)
    if COLVAR_DISTANCE_UNIT == "nm":
        distance_series = distance_series * 10.0

    frames, energies, distances = align_series(energies, distance_series)

    os.makedirs(OUTPUT_PLOT_PATH.parent, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Simulation frame")
    ax1.set_ylabel("EB-JEPA latent displacement (L2)", color="tab:red", weight="bold")
    ax1.plot(frames, energies, color="tab:red", label="Latent displacement")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Pocket distance (A)", color="tab:blue", weight="bold")
    ax2.plot(frames, distances, color="tab:blue", label="Pocket distance")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT_PATH, dpi=300)
    plt.close(fig)

    print(f"\nsaved anomaly plot to {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
