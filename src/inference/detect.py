import os
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
import torch

from src.data.traj_verification import read_colvar_fields
from src.models.jepa_model import JEPAModel
from src.training.loss import jepa_loss

CONTACT_MAP_PATH = Path("data/jepa_contact_maps.pt")
META_XTC_PATH = Path("data/patch_traj.xtc")
META_PDB_PATH = Path("data/patch_topology.pdb")
WEIGHTS_PATH = Path("models/best_jepa.pth")
COLVAR_PATH = Path("data/metad_S1_BAR_COLVAR")
OUTPUT_PLOT_PATH = Path("figures/metadynamics_anomaly.png")

LATENT_DIM = 128
PREDICTOR_HIDDEN_DIM = 256
TEMPORAL_GAP = 10
EXPECTED_FRAMES = 2000
SMOOTH_WINDOW = 20
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


def load_colvar_distance(path, column, traj_time):
    fields = read_colvar_fields(path)
    colvar_df = pd.read_csv(path, comment="#", header=None, delimiter="\s+")
    if fields and len(fields) == colvar_df.shape[1]:
        colvar_df.columns = fields
    else:
        colvar_df.columns = [f"col{i}" for i in range(colvar_df.shape[1])]

    time_col = "time" if "time" in colvar_df.columns else colvar_df.columns[0]

    if column not in colvar_df.columns:
        raise ValueError(f"column '{column}' not found in COLVAR file")

    colvar_time = colvar_df[time_col].to_numpy(dtype=np.float32)
    distance_series = colvar_df[column].to_numpy(dtype=np.float32)

    interpolated = np.interp(traj_time, colvar_time, distance_series)
    return interpolated.astype(np.float32)


def align_series(energies, distances, gap):
    if len(distances) <= gap:
        raise ValueError("COLVAR series is too short for the requested temporal gap")

    max_len = min(len(energies), len(distances) - gap)
    energies = energies[:max_len]
    distances = distances[gap : gap + max_len]
    frames = np.arange(gap, gap + max_len)
    return frames, energies, distances


def main():
    device = get_device()

    if not CONTACT_MAP_PATH.exists():
        raise SystemExit(f"missing contact map tensor: {CONTACT_MAP_PATH}")
    if not META_XTC_PATH.exists():
        raise SystemExit(f"missing trajectory: {META_XTC_PATH}")
    if not META_PDB_PATH.exists():
        raise SystemExit(f"missing topology: {META_PDB_PATH}")
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"missing model weights: {WEIGHTS_PATH}")
    if not COLVAR_PATH.exists():
        raise SystemExit(f"missing COLVAR file: {COLVAR_PATH}")

    contact_maps = load_contact_maps(CONTACT_MAP_PATH)
    n_frames = contact_maps.shape[0]
    if n_frames < TEMPORAL_GAP + 1:
        raise SystemExit("contact map tensor does not contain enough frames")
    if n_frames != EXPECTED_FRAMES:
        print(f"warning: expected {EXPECTED_FRAMES} frames, found {n_frames}")

    model = JEPAModel(
        in_channels=1,
        latent_dim=LATENT_DIM,
        predictor_hidden_dim=PREDICTOR_HIDDEN_DIM,
    )
    model.eval()
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    energies = []
    with torch.no_grad():
        for t in range(n_frames - TEMPORAL_GAP):
            context = contact_maps[t].unsqueeze(0).to(device)
            target = contact_maps[t + TEMPORAL_GAP].unsqueeze(0).to(device)

            z_context, z_target, z_pred = model(context, target)
            energy, _ = jepa_loss(
                z_context,
                z_target,
                z_pred,
                apply_reg=False,
            )
            energies.append(energy.item())

    raw_energies = np.array(energies, dtype=np.float32)

    # low-pass filter to smooth the thermal noise with a short rolling window
    energies = pd.Series(raw_energies).rolling(
        window=SMOOTH_WINDOW, min_periods=1, center=True
    ).mean().to_numpy()
    
    # # establish baseline stats from stable equilibrium phase
    # eq_mean = smoothed_energies[:1000].mean()
    # eq_std = smoothed_energies[:1000].std()
    
    # # calculate anomaly z-scores relative to equilibrium baseline
    # z_scores = (smoothed_energies - eq_mean) / eq_std

    # energies = z_scores

    # downsample colvar distance series to match energy series length
    meta_traj = md.load(str(META_XTC_PATH), top=str(META_PDB_PATH))
    traj_time = meta_traj.time
    if len(traj_time) != n_frames:
        if len(traj_time) % n_frames == 0:
            stride = len(traj_time) // n_frames
            traj_time = traj_time[::stride]
        else:
            step = (traj_time[-1] - traj_time[0]) / max(n_frames - 1, 1)
            traj_time = np.arange(n_frames, dtype=np.float32) * step + traj_time[0]

    distance_series = load_colvar_distance(COLVAR_PATH, POCKET_DISTANCE_COL, traj_time)
    if COLVAR_DISTANCE_UNIT == "nm":
        distance_series = distance_series * 10.0

    # align energy and distance series based on temporal gap used for prediction
    frames, energies, distances = align_series(energies, distance_series, TEMPORAL_GAP)

    os.makedirs(OUTPUT_PLOT_PATH.parent, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Simulation frame")
    ax1.set_ylabel("EB-JEPA energy (MSE)", color="tab:red", weight="bold")
    ax1.plot(frames, energies, color="tab:red", label="EB-JEPA energy")
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

    print(f"saved anomaly plot to {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()