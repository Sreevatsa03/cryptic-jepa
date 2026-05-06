import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.jepa_model import JEPAModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_contact_maps(path):
    contact_maps = torch.load(path, map_location="cpu")
    if contact_maps.dim() == 3:
        contact_maps = contact_maps.unsqueeze(1)
    return contact_maps.float()


def load_baseline_stats(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = float(payload.get("mean"))
    std = float(payload.get("std"))
    temporal_gap = int(payload.get("temporal_gap", 0))
    smooth_window = int(payload.get("smooth_window", 1))
    return mean, std, temporal_gap, smooth_window


def smooth_series(values, window):
    """
    Smooth a series of values using a moving average with edge padding
    """
    if window <= 1:
        return values
    window = int(window)
    half = window // 2
    pad_left = half
    pad_right = window - 1 - half
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


@torch.no_grad()
def compute_mse_series(contact_maps, model, device, temporal_gap, batch_size):
    """
    Compute the JEPA prediction MSE series for a trajectory of contact maps
    """
    n_frames = contact_maps.shape[0]
    if n_frames <= temporal_gap:
        raise ValueError("contact maps are too short for the temporal gap")

    energies = []
    max_start = n_frames - temporal_gap
    for start in range(0, max_start, batch_size):
        end = min(start + batch_size, max_start)
        context = contact_maps[start:end].to(device)
        target = contact_maps[start + temporal_gap : end + temporal_gap].to(device)

        _, z_target, z_pred = model(context, target)
        mse = (z_pred - z_target).pow(2).mean(dim=1)
        energies.append(mse.cpu())

    return torch.cat(energies, dim=0).numpy()


def standardize_embeddings(embeddings):
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (embeddings - mean) / std


def two_nn_id(embeddings):
    """
    Compute the Two-NN intrinsic dimensionality estimate for a set of embeddings
    """
    if len(embeddings) < 3:
        return np.nan

    data = embeddings.astype(np.float64)
    sq = np.sum(data * data, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (data @ data.T)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)

    nearest = np.partition(dist, 1, axis=1)
    r1 = nearest[:, 0]
    r2 = nearest[:, 1]

    valid = r1 > 0
    if not np.any(valid):
        return np.nan
    mu = r2[valid] / r1[valid]
    return len(mu) / np.sum(np.log(mu))


def bootstrap_id(embeddings, rng, n_boot, sample_size):
    """
    Perform bootstrap subsampling (without replacement) to estimate variability.
    """
    if n_boot <= 0:
        return np.array([])
    n = len(embeddings)
    if n < 3:
        return np.array([])
    
    # default to 80% subsampling if no explicit sample_size is passed
    size = int(n * 0.8) if sample_size <= 0 else min(sample_size, n - 1)

    # ensure at least 3 samples for Two-NN math
    size = max(3, size)
    
    estimates = []
    for _ in range(n_boot):
        # replace=False to prevent distance=0 errors
        idx = rng.choice(n, size=size, replace=False)
        estimates.append(two_nn_id(embeddings[idx]))
    return np.array(estimates, dtype=np.float64)


def summarize_bootstrap(estimates):
    """
    Summarize bootstrap estimates by computing mean and 95% confidence interval percentiles
    """
    if estimates.size == 0:
        return np.nan, np.nan, np.nan
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(estimates))
    low = float(np.percentile(estimates, 2.5))
    high = float(np.percentile(estimates, 97.5))
    return mean, low, high


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Two-NN intrinsic dimensionality for baseline vs transition",
    )
    parser.add_argument("--contact-maps", default="data/jepa_contact_maps.pt")
    parser.add_argument("--weights", default="models/best_jepa.pth")
    parser.add_argument("--baseline-stats", default="data/baseline_stats.json")
    parser.add_argument(
        "--anomaly-indices",
        default="data/anomaly_frames.npy",
        help="anomaly frame indices used to define the transition subset",
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--predictor-hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-gap", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--baseline-z", type=float, default=1.0)
    parser.add_argument("--transition-z", type=float, default=2.0)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="minimum samples required per subset before falling back to quantiles",
    )
    parser.add_argument(
        "--baseline-quantile",
        type=float,
        default=0.25,
        help="fallback quantile for baseline subset",
    )
    parser.add_argument(
        "--transition-quantile",
        type=float,
        default=0.9,
        help="fallback quantile for transition subset",
    )
    parser.add_argument(
        "--use-quantiles",
        action="store_true",
        help="use quantile thresholds instead of z-score cutoffs",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--bootstrap-sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="z-score embeddings before Two-NN (recommended)",
    )
    parser.add_argument(
        "--output",
        default="figures/two_nn_baseline_transition.png",
        help="output plot path",
    )
    parser.add_argument(
        "--output-stats",
        default="data/two_nn_stats.csv",
        help="output CSV path for Two-NN stats",
    )
    parser.add_argument(
        "--use-anomaly-frames",
        action="store_true",
        help="define transition subset from anomaly frames instead of z-cutoffs",
    )
    args = parser.parse_args()

    contact_map_path = Path(args.contact_maps)
    if not contact_map_path.exists():
        raise SystemExit(f"missing contact map tensor: {contact_map_path}")

    stats_path = Path(args.baseline_stats)
    if not stats_path.exists():
        raise SystemExit(
            f"missing baseline stats: {stats_path} (run anomaly_scorer first)"
        )

    baseline_mean, baseline_std, stats_gap, stats_window = load_baseline_stats(stats_path)
    if stats_gap and stats_gap != args.temporal_gap:
        raise SystemExit("temporal gap does not match baseline stats")
    if stats_window and stats_window != args.smooth_window:
        raise SystemExit("smooth window does not match baseline stats")

    device = get_device()
    contact_maps = load_contact_maps(contact_map_path).to(device)

    model = JEPAModel(
        in_channels=1,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # extract latent embeddings for all frames to use for Two-NN
    print("extracting latent embeddings")
    z_embeddings = []
    for start in range(0, len(contact_maps), args.batch_size):
        batch = contact_maps[start : start + args.batch_size]
        z = model.context_encoder(batch)
        z_embeddings.append(z.cpu().numpy())
    embeddings = np.concatenate(z_embeddings, axis=0)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    if args.standardize:
        embeddings = standardize_embeddings(embeddings)

    # compute MSE series for the trajectory and convert to z-scores using baseline stats
    print("computing MSE series for z-score partition")
    mse = compute_mse_series(
        contact_maps,
        model,
        device,
        args.temporal_gap,
        args.batch_size,
    )
    mse = smooth_series(mse, args.smooth_window)
    if baseline_std <= 0:
        raise SystemExit("baseline std is non-positive; cannot compute z-scores")
    z_scores = (mse - baseline_mean) / baseline_std

    # partition frames into baseline vs transition based on z-score thresholds
    n_frames = embeddings.shape[0]
    z_by_frame = np.full(n_frames, np.nan, dtype=np.float32)
    z_by_frame[args.temporal_gap : args.temporal_gap + len(z_scores)] = z_scores

    print(f"maximum z-score in trajectory: {np.nanmax(z_by_frame):.3f}")

    # select baseline and transition subsets based on anomaly frames or z-score thresholds
    finite = np.isfinite(z_by_frame)
    z_vals = z_by_frame[finite]
    if z_vals.size == 0:
        raise SystemExit("no finite z-scores available for partitioning")

    baseline_idx = None
    transition_idx = None

    # define transition subset from anomaly frames
    if args.use_anomaly_frames:
        anomaly_path = Path(args.anomaly_indices)
        if not anomaly_path.exists():
            raise SystemExit(f"missing anomaly indices: {anomaly_path}")
        anomaly_frames = np.load(anomaly_path).astype(np.int64)
        anomaly_frames = anomaly_frames[(anomaly_frames >= 0) & (anomaly_frames < n_frames)]
        transition_idx = np.unique(anomaly_frames)

        baseline_idx = np.where(z_by_frame < args.baseline_z)[0]
        if baseline_idx.size < args.min_samples:
            baseline_cut = np.quantile(z_vals, args.baseline_quantile)
            baseline_idx = np.where(z_by_frame <= baseline_cut)[0]
            print(
                "insufficient baseline samples for z-cutoffs; falling back to quantiles: "
                "baseline <= {:.3f}".format(baseline_cut)
            )

        print("using anomaly frames for transition subset")
    else:
        # use z-score cutoffs by default but fall back to quantiles if there are too few samples
        if args.use_quantiles:
            baseline_cut = np.quantile(z_vals, args.baseline_quantile)
            transition_cut = np.quantile(z_vals, args.transition_quantile)
            baseline_idx = np.where(z_by_frame <= baseline_cut)[0]
            transition_idx = np.where(z_by_frame >= transition_cut)[0]
            print(
                "using quantile thresholds: baseline <= {:.3f}, transition >= {:.3f}".format(
                    baseline_cut, transition_cut
                )
            )
        else:
            baseline_idx = np.where(z_by_frame < args.baseline_z)[0]
            transition_idx = np.where(z_by_frame > args.transition_z)[0]

            if baseline_idx.size < args.min_samples or transition_idx.size < args.min_samples:
                baseline_cut = np.quantile(z_vals, args.baseline_quantile)
                transition_cut = np.quantile(z_vals, args.transition_quantile)
                baseline_idx = np.where(z_by_frame <= baseline_cut)[0]
                transition_idx = np.where(z_by_frame >= transition_cut)[0]
                print(
                    "insufficient samples for z-cutoffs; falling back to quantiles: "
                    "baseline <= {:.3f}, transition >= {:.3f}".format(
                        baseline_cut, transition_cut
                    )
                )
    
    if baseline_idx.size < 3 or transition_idx.size < 3:
        raise SystemExit(
            "insufficient baseline or transition samples for Two-NN after partitioning"
        )

    print(
        "subset sizes: baseline={} transition={}".format(
            baseline_idx.size,
            transition_idx.size,
        )
    )

    rng = np.random.default_rng(args.seed)
    if args.max_samples > 0:
        if baseline_idx.size > args.max_samples:
            baseline_idx = rng.choice(baseline_idx, size=args.max_samples, replace=False)
        if transition_idx.size > args.max_samples:
            transition_idx = rng.choice(transition_idx, size=args.max_samples, replace=False)

    baseline_embeddings = embeddings[baseline_idx]
    transition_embeddings = embeddings[transition_idx]

    baseline_id = two_nn_id(baseline_embeddings)
    transition_id = two_nn_id(transition_embeddings)

    # perform bootstrap resampling to estimate variability of the Two-NN estimates and compute confidence intervals
    boot_size = args.bootstrap_sample if args.bootstrap_sample > 0 else 0
    baseline_boot = bootstrap_id(baseline_embeddings, rng, args.bootstrap, boot_size)
    transition_boot = bootstrap_id(transition_embeddings, rng, args.bootstrap, boot_size)

    baseline_mean, baseline_low, baseline_high = summarize_bootstrap(baseline_boot)
    transition_mean, transition_low, transition_high = summarize_bootstrap(transition_boot)

    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        "subset,n,id,boot_mean,boot_low,boot_high\n"
        f"baseline,{len(baseline_embeddings)},{baseline_id:.6f},{baseline_mean:.6f},{baseline_low:.6f},{baseline_high:.6f}\n"
        f"transition,{len(transition_embeddings)},{transition_id:.6f},{transition_mean:.6f},{transition_low:.6f},{transition_high:.6f}\n",
        encoding="utf-8",
    )

    # create bar plot comparing baseline vs transition Two-NN estimates with error bars for bootstrap confidence intervals
    labels = ["Baseline", "Transition"]
    ids = [baseline_id, transition_id]
    if np.isfinite(baseline_low) and np.isfinite(baseline_high):
        base_err_low = max(0.0, baseline_id - baseline_low)
        base_err_high = max(0.0, baseline_high - baseline_id)
    else:
        base_err_low = 0.0
        base_err_high = 0.0

    if np.isfinite(transition_low) and np.isfinite(transition_high):
        trans_err_low = max(0.0, transition_id - transition_low)
        trans_err_high = max(0.0, transition_high - transition_id)
    else:
        trans_err_low = 0.0
        trans_err_high = 0.0

    errs = [
        [base_err_low, base_err_high],
        [trans_err_low, trans_err_high],
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(labels, ids, color=["#4C78A8", "#E45756"], alpha=0.9)
    plt.errorbar(labels, ids, yerr=np.array(errs).T, fmt="none", ecolor="black", capsize=4)
    plt.ylabel("Intrinsic Dimensionality (Two-NN)", weight="bold")
    plt.title("Baseline vs Transition Intrinsic Dimensionality")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"saved two-nn plot to {output_path}")
    print(f"saved two-nn stats to {stats_path}")


if __name__ == "__main__":
    main()