import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "i",
    "j",
    "res_i",
    "resname_i",
    "res_j",
    "resname_j",
    "in_pocket_i",
    "in_pocket_j",
}


def load_mapped_contacts(path):
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    return df


def attach_saliency(df, saliency):
    """
    Attach saliency values to contact pairs based on their atom indices
    """
    max_index = saliency.shape[0]
    valid = (df["i"] >= 0) & (df["j"] >= 0) & (df["i"] < max_index) & (df["j"] < max_index)
    if not valid.all():
        df = df[valid].copy()
    df["saliency"] = saliency[df["i"].to_numpy(), df["j"].to_numpy()]
    return df


def ordered_columns(df):
    res_i = df["res_i"].to_numpy()
    res_j = df["res_j"].to_numpy()
    swap = res_i > res_j

    res_a = np.where(swap, res_j, res_i)
    res_b = np.where(swap, res_i, res_j)
    name_a = np.where(swap, df["resname_j"], df["resname_i"])
    name_b = np.where(swap, df["resname_i"], df["resname_j"])
    pocket_a = np.where(swap, df["in_pocket_j"], df["in_pocket_i"])
    pocket_b = np.where(swap, df["in_pocket_i"], df["in_pocket_j"])

    df = df.copy()
    df["res_a"] = res_a
    df["res_b"] = res_b
    df["resname_a"] = name_a
    df["resname_b"] = name_b
    df["in_pocket_a"] = pocket_a
    df["in_pocket_b"] = pocket_b
    return df


def aggregate_pairs(df):
    """
    Aggregate contact pairs by residue pairs and compute mean/max saliency and counts
    """
    grouped = df.groupby(["res_a", "res_b"], as_index=False)
    agg = grouped.agg(
        resname_a=("resname_a", "first"),
        resname_b=("resname_b", "first"),
        in_pocket_a=("in_pocket_a", "first"),
        in_pocket_b=("in_pocket_b", "first"),
        mean_saliency=("saliency", "mean"),
        max_saliency=("saliency", "max"),
        count=("saliency", "size"),
    )
    agg = agg.sort_values("mean_saliency", ascending=False)
    return agg


def gate_residues(df, min_count):
    """
    Count how many times each residue appears in the contact pairs and filter by minimum count
    """
    counts = pd.concat(
        [
            df[["res_i", "resname_i"]].rename(
                columns={"res_i": "res", "resname_i": "resname"}
            ),
            df[["res_j", "resname_j"]].rename(
                columns={"res_j": "res", "resname_j": "resname"}
            ),
        ],
        ignore_index=True,
    )
    counts = counts.groupby(["res", "resname"], as_index=False).size()
    counts = counts.rename(columns={"size": "count"})
    counts = counts[counts["count"] >= min_count]
    counts = counts.sort_values(["res"])
    return counts


def gate_segments(residue_df, min_length):
    """
    Identify contiguous segments of residues that meet the minimum count threshold for gate inclusion, and aggregate their total counts
    """
    residues = residue_df["res"].to_numpy()
    counts = residue_df.set_index("res")["count"].to_dict()
    if residues.size == 0:
        return pd.DataFrame(columns=["start", "end", "length", "total_count"])

    residues = np.sort(residues)
    segments = []
    start = residues[0]
    prev = residues[0]
    total = counts[start]

    # iterate through sorted residues and identify contiguous segments, summing counts for each segment
    for res in residues[1:]:
        if res == prev + 1:
            total += counts[res]
        else:
            length = prev - start + 1
            if length >= min_length:
                segments.append((start, prev, length, total))
            start = res
            total = counts[res]
        prev = res

    length = prev - start + 1
    if length >= min_length:
        segments.append((start, prev, length, total))

    return pd.DataFrame(segments, columns=["start", "end", "length", "total_count"])


def main():
    parser = argparse.ArgumentParser(
        description="Filter saliency contacts and rank inter-residue latches",
    )
    parser.add_argument(
        "--mapped-contacts",
        default="data/saliency_contacts_mapped.csv",
        help="mapped saliency contacts CSV",
    )
    parser.add_argument(
        "--saliency-map",
        default="data/saliency_map.npy",
        help="saliency map numpy file",
    )
    parser.add_argument(
        "--output-filtered",
        default="data/contacts_filtered.csv",
        help="output CSV for filtered contacts with saliency",
    )
    parser.add_argument(
        "--output-latches",
        default="data/latches.csv",
        help="output CSV for aggregated inter-residue pairs",
    )
    parser.add_argument(
        "--output-gate-residues",
        default="data/gate_residues.csv",
        help="output CSV for gate residue counts",
    )
    parser.add_argument(
        "--output-gate-segments",
        default="data/gate_segments.csv",
        help="output CSV for contiguous gate segments",
    )
    parser.add_argument(
        "--gate-min-count",
        type=int,
        default=2,
        help="minimum count to include a residue in the gate",
    )
    parser.add_argument(
        "--gate-min-length",
        type=int,
        default=3,
        help="minimum contiguous length for a gate segment",
    )
    args = parser.parse_args()

    mapped_path = Path(args.mapped_contacts)
    if not mapped_path.exists():
        raise SystemExit(f"missing mapped contacts file: {mapped_path}")

    saliency_path = Path(args.saliency_map)
    if not saliency_path.exists():
        raise SystemExit(f"missing saliency map file: {saliency_path}")

    df = load_mapped_contacts(mapped_path)
    saliency = np.load(saliency_path)
    df = attach_saliency(df, saliency)

    df = df[df["res_i"] != df["res_j"]].copy()
    df = ordered_columns(df)

    filtered_path = Path(args.output_filtered)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filtered_path, index=False)

    # aggregate contacts by residue pairs to identify top latches, and count/filter residues for gate definition
    latches = aggregate_pairs(df)
    latches_path = Path(args.output_latches)
    latches_path.parent.mkdir(parents=True, exist_ok=True)
    latches.to_csv(latches_path, index=False)

    gate_res = gate_residues(df, args.gate_min_count)
    gate_res_path = Path(args.output_gate_residues)
    gate_res_path.parent.mkdir(parents=True, exist_ok=True)
    gate_res.to_csv(gate_res_path, index=False)

    segments = gate_segments(gate_res, args.gate_min_length)
    segments_path = Path(args.output_gate_segments)
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(segments_path, index=False)

    print(f"saved filtered contacts to {filtered_path}")
    print(f"saved latch ranking to {latches_path}")
    print(f"saved gate residues to {gate_res_path}")
    print(f"saved gate segments to {segments_path}")


if __name__ == "__main__":
    main()
