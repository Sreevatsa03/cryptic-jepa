import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import torch

from src.models.jepa_model import JEPAModel

ALPHA = 15.0
D0 = 0.8


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trajectory(xtc_path, pdb_path):
    xtc_path = Path(xtc_path)
    pdb_path = Path(pdb_path)
    if not xtc_path.exists():
        raise SystemExit(f"missing trajectory: {xtc_path}")
    if not pdb_path.exists():
        raise SystemExit(f"missing topology: {pdb_path}")
    return md.load(str(xtc_path), top=str(pdb_path))


def contact_maps_from_coords(coords):
    distances = torch.cdist(coords, coords)
    contact_maps = 1.0 / (1.0 + torch.exp(ALPHA * (distances - D0)))
    return contact_maps.unsqueeze(1)


@torch.no_grad()
def compute_mse_series(coords, model, device, temporal_gap, batch_size):
    """
    Compute the JEPA prediction MSE series for a trajectory
    """
    n_frames = coords.shape[0]
    if n_frames <= temporal_gap:
        raise ValueError("trajectory is too short for the temporal gap")

    energies = []
    max_start = n_frames - temporal_gap
    for start in range(0, max_start, batch_size):
        end = min(start + batch_size, max_start)
        context_coords = coords[start:end]
        target_coords = coords[start + temporal_gap : end + temporal_gap]

        context_maps = contact_maps_from_coords(context_coords)
        target_maps = contact_maps_from_coords(target_coords)

        context_maps = context_maps.to(device)
        target_maps = target_maps.to(device)

        _, z_target, z_pred = model(context_maps, target_maps)
        mse = (z_pred - z_target).pow(2).mean(dim=1)
        energies.append(mse.cpu())

    return torch.cat(energies, dim=0).numpy()


def smooth_series(values, window):
    if window <= 1:
        return values
    window = int(window)
    half = window // 2
    pad_left = half
    pad_right = window - 1 - half
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def save_baseline_stats(path, mean, std, temporal_gap, n_frames, smooth_window):
    payload = {
        "mean": float(mean),
        "std": float(std),
        "temporal_gap": int(temporal_gap),
        "n_frames": int(n_frames),
        "smooth_window": int(smooth_window),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_baseline_stats(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = float(payload.get("mean"))
    std = float(payload.get("std"))
    smooth_window = payload.get("smooth_window")
    return mean, std, smooth_window


def save_anomaly_indices(path, indices, z_scores=None, mse=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, indices.astype(np.int64))
        return

    if z_scores is None or mse is None:
        np.savetxt(path, indices.astype(np.int64), fmt="%d", delimiter=",")
        return

    rows = np.column_stack([indices.astype(np.int64), z_scores, mse])
    header = "index,z_score,mse"
    np.savetxt(path, rows, fmt=["%d", "%.6f", "%.6f"], delimiter=",", header=header)


def main():
    parser = argparse.ArgumentParser(
        description="Score metadynamics frames using baseline JEPA prediction error",
    )
    parser.add_argument(
        "--eq-xtc",
        default="data/eq_patch_traj.xtc",
        help="equilibrium patch trajectory",
    )
    parser.add_argument(
        "--eq-pdb",
        default="data/eq_patch_topology.pdb",
        help="equilibrium patch topology",
    )
    parser.add_argument(
        "--meta-xtc",
        default="data/patch_traj.xtc",
        help="metadynamics patch trajectory",
    )
    parser.add_argument(
        "--meta-pdb",
        default="data/patch_topology.pdb",
        help="metadynamics patch topology",
    )
    parser.add_argument(
        "--weights",
        default="models/best_jepa.pth",
        help="trained JEPA weights",
    )
    parser.add_argument(
        "--baseline-stats-path",
        default="data/baseline_stats.json",
        help="path to write/read baseline stats",
    )
    parser.add_argument(
        "--recompute-baseline",
        action="store_true",
        help="force recomputation of baseline stats",
    )
    parser.add_argument("--temporal-gap", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--predictor-hidden-dim", type=int, default=256)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="rolling mean window size for MSE smoothing (1 disables)",
    )
    parser.add_argument(
        "--output-anomalies",
        default="data/anomaly_frames.npy",
        help="output path for anomaly frame indices",
    )
    args = parser.parse_args()

    device = get_device()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"missing model weights: {weights_path}")
    
    model = JEPAModel(
        in_channels=1,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # compute baseline stats from equilibrium trajectory or load from file if available
    stats_path = Path(args.baseline_stats_path)
    if stats_path.exists() and not args.recompute_baseline:
        baseline_mean, baseline_std, smooth_window = load_baseline_stats(stats_path)
        if smooth_window is not None and int(smooth_window) != int(args.smooth_window):
            raise SystemExit(
                "baseline stats were computed with a different smooth window; "
                "recompute or use matching --smooth-window"
            )
    else:
        eq_traj = load_trajectory(args.eq_xtc, args.eq_pdb)
        eq_coords = torch.tensor(eq_traj.xyz, dtype=torch.float32)
        eq_mse = compute_mse_series(
            eq_coords,
            model,
            device,
            args.temporal_gap,
            args.batch_size,
        )
        eq_mse = smooth_series(eq_mse, args.smooth_window)
        baseline_mean = float(eq_mse.mean())
        baseline_std = float(eq_mse.std())
        save_baseline_stats(
            stats_path,
            baseline_mean,
            baseline_std,
            args.temporal_gap,
            len(eq_mse),
            args.smooth_window,
        )

    if baseline_std <= 0:
        raise SystemExit("baseline std is non-positive; cannot compute z-scores")

    # compute MSE series for metadynamics trajectory and calculate z-scores relative to baseline
    meta_traj = load_trajectory(args.meta_xtc, args.meta_pdb)
    meta_coords = torch.tensor(meta_traj.xyz, dtype=torch.float32)
    meta_mse = compute_mse_series(
        meta_coords,
        model,
        device,
        args.temporal_gap,
        args.batch_size,
    )
    meta_mse = smooth_series(meta_mse, args.smooth_window)

    # calculate anomaly z-scores relative to equilibrium baseline
    z_scores = (meta_mse - baseline_mean) / baseline_std
    target_indices = np.arange(len(meta_mse), dtype=np.int64) + args.temporal_gap
    anomaly_mask = z_scores > args.z_threshold
    anomaly_indices = target_indices[anomaly_mask]

    save_anomaly_indices(
        Path(args.output_anomalies),
        anomaly_indices,
        z_scores[anomaly_mask],
        meta_mse[anomaly_mask],
    )

    print(
        "baseline mean={:.6f} std={:.6f}; detected {} anomalies".format(
            baseline_mean,
            baseline_std,
            anomaly_indices.size,
        )
    )


if __name__ == "__main__":
    main()
