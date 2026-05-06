import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_latches(path):
    df = pd.read_csv(path)
    required = {"res_a", "res_b", "mean_saliency"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    return df


def load_gate_residues(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "res" not in df.columns:
        return set()
    return set(int(x) for x in df["res"].to_numpy())


def load_gate_segments(path):
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if not {"start", "end"}.issubset(df.columns):
        return []
    return [(int(row["start"]), int(row["end"])) for _, row in df.iterrows()]


def gate_set_from_segments(segments):
    gate = set()
    for start, end in segments:
        gate.update(range(start, end + 1))
    return gate


def parse_latch_list(value):
    if not value:
        return set()
    pairs = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
        elif ":" in item:
            left, right = item.split(":", 1)
        else:
            raise ValueError(f"invalid latch pair '{item}'; use 'A-B' format")
        res_a = int(left.strip())
        res_b = int(right.strip())
        pairs.add((min(res_a, res_b), max(res_a, res_b)))
    return pairs


def build_heatmap_matrix(df):
    """
    Build a heatmap matrix from latch data
    """
    residues = np.unique(
        np.concatenate([df["res_a"].to_numpy(), df["res_b"].to_numpy()])
    )
    residues = np.sort(residues)
    index = {res: idx for idx, res in enumerate(residues)}
    mat = np.full((len(residues), len(residues)), np.nan, dtype=np.float64)

    for _, row in df.iterrows():
        i = index[int(row["res_a"])]
        j = index[int(row["res_b"])]
        val = float(row["mean_saliency"])
        mat[i, j] = val
        mat[j, i] = val

    return residues, mat, index


def plot_heatmap(residues, mat, index, output_path, label_every, highlight_pairs):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="magma", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean saliency", weight="bold")

    ticks = np.arange(len(residues))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    labels = [str(r) if (idx % label_every == 0) else "" for idx, r in enumerate(residues)]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_xlabel("Residue", weight="bold")
    ax.set_ylabel("Residue", weight="bold")
    ax.set_title("Residue Pair Saliency Heatmap")

    if highlight_pairs:
        points_x = []
        points_y = []
        for res_a, res_b in highlight_pairs:
            if res_a not in index or res_b not in index:
                continue
            i = index[res_a]
            j = index[res_b]
            points_x.extend([j, i])
            points_y.extend([i, j])
        if points_x:
            ax.scatter(
                points_x,
                points_y,
                s=80,
                facecolors="none",
                edgecolors="white",
                linewidths=1.5,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_ranked_table(df, output_csv, output_png, top_n, highlight_pairs):
    """
    Plot a ranked table of top latches with saliency values
    """
    df = df.sort_values("mean_saliency", ascending=False).head(top_n).copy()
    if "resname_a" in df.columns and "resname_b" in df.columns:
        df["pair"] = (
            df["resname_a"].astype(str)
            + df["res_a"].astype(str)
            + "-"
            + df["resname_b"].astype(str)
            + df["res_b"].astype(str)
        )
    else:
        df["pair"] = df["res_a"].astype(str) + "-" + df["res_b"].astype(str)

    df["highlight"] = [
        "*" if (min(a, b), max(a, b)) in highlight_pairs else ""
        for a, b in zip(df["res_a"], df["res_b"])
    ]

    # output CSV with pair and saliency columns for reference
    out_cols = ["pair", "mean_saliency", "max_saliency", "count", "highlight"]
    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan

    table_df = df[out_cols].copy()
    table_df.to_csv(output_csv, index=False)

    fig, ax = plt.subplots(figsize=(7, 0.35 * (len(table_df) + 2)))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def classify_pairs(df, gate_residues):
    """
    Classify residue pairs into categories based on gate membership
    """
    categories = []
    for _, row in df.iterrows():
        res_a = int(row["res_a"])
        res_b = int(row["res_b"])
        in_a = res_a in gate_residues
        in_b = res_b in gate_residues
        if in_a and in_b:
            categories.append("intra_gate")
        elif in_a or in_b:
            categories.append("gate_scaffold")
        else:
            categories.append("scaffold")
    return categories


def plot_schematic(df, gate_residues, output_png, top_n, highlight_pairs):
    """
    Plot a schematic bar plot of top latches colored by category
    """
    df = df.sort_values("mean_saliency", ascending=False).head(top_n).copy()
    df["category"] = classify_pairs(df, gate_residues)
    df["label"] = df["res_a"].astype(str) + "-" + df["res_b"].astype(str)

    # define colors for categories
    colors = {
        "gate_scaffold": "#E45756",
        "intra_gate": "#F2B447",
        "scaffold": "#72B7B2",
    }
    bar_colors = [colors.get(cat, "#999999") for cat in df["category"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    edge_colors = [
        "black" if (min(a, b), max(a, b)) in highlight_pairs else "none"
        for a, b in zip(df["res_a"], df["res_b"])
    ]
    ax.bar(
        df["label"],
        df["mean_saliency"],
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=1.2,
    )
    ax.set_ylabel("Mean saliency", weight="bold")
    ax.set_xlabel("Residue pair", weight="bold")
    ax.set_title("Top Latches by Saliency")
    ax.tick_params(axis="x", rotation=45)

    legend_labels = [
        ("gate_scaffold", "Gate-Scaffold"),
        ("intra_gate", "Intra-Gate"),
        ("scaffold", "Scaffold"),
    ]
    handles = []
    for key, label in legend_labels:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[key], label=label))
    ax.legend(handles=handles, frameon=False)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate saliency report figures (heatmap, table, schematic)",
    )
    parser.add_argument(
        "--latches",
        default="data/latches.csv",
        help="latch ranking CSV",
    )
    parser.add_argument(
        "--gate-segments",
        default="data/gate_segments.csv",
        help="gate segments CSV (preferred for gate definition)",
    )
    parser.add_argument(
        "--heatmap-out",
        default="figures/saliency_heatmap.png",
        help="output heatmap path",
    )
    parser.add_argument(
        "--table-csv",
        default="data/saliency_latch_table.csv",
        help="output ranked table CSV",
    )
    parser.add_argument(
        "--table-fig",
        default="figures/saliency_latch_table.png",
        help="output ranked table figure",
    )
    parser.add_argument(
        "--schematic-out",
        default="figures/saliency_latch_schematic.png",
        help="output schematic bar plot",
    )
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--label-every", type=int, default=1)
    parser.add_argument(
        "--highlight-latches",
        default="51-229,207-213,60-196",
        help="comma-separated residue pairs to highlight",
    )
    args = parser.parse_args()

    latch_path = Path(args.latches)
    if not latch_path.exists():
        raise SystemExit(f"missing latch file: {latch_path}")

    # latch data and gate definition from segments
    df = load_latches(latch_path)
    segments = load_gate_segments(Path(args.gate_segments))
    gate_residues = gate_set_from_segments(segments)
    highlight_pairs = parse_latch_list(args.highlight_latches)

    heatmap_out = Path(args.heatmap_out)
    heatmap_out.parent.mkdir(parents=True, exist_ok=True)

    residues, mat, index = build_heatmap_matrix(df)
    plot_heatmap(residues, mat, index, heatmap_out, args.label_every, highlight_pairs)

    table_csv = Path(args.table_csv)
    table_csv.parent.mkdir(parents=True, exist_ok=True)
    table_fig = Path(args.table_fig)
    table_fig.parent.mkdir(parents=True, exist_ok=True)
    plot_ranked_table(df, table_csv, table_fig, args.top_n, highlight_pairs)

    schematic_out = Path(args.schematic_out)
    schematic_out.parent.mkdir(parents=True, exist_ok=True)
    plot_schematic(df, gate_residues, schematic_out, args.top_n, highlight_pairs)

    print(f"saved heatmap to {heatmap_out}")
    print(f"saved table CSV to {table_csv}")
    print(f"saved table figure to {table_fig}")
    print(f"saved schematic to {schematic_out}")


if __name__ == "__main__":
    main()
