import argparse
from pathlib import Path

import pandas as pd


def build_resi_selection(residues):
    """
    Build a PyMOL selection string for a list of residue numbers, grouping contiguous residues into ranges
    For example, [10, 11, 12, 15, 16] would become "10-12+15-16"
    """
    if not residues:
        return ""
    residues = sorted(set(int(x) for x in residues))
    segments = []
    start = residues[0]
    prev = residues[0]
    for res in residues[1:]:
        if res == prev + 1:
            prev = res
            continue
        segments.append((start, prev))
        start = res
        prev = res
    segments.append((start, prev))

    parts = []
    for start, end in segments:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    return "+".join(parts)


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
        pair = (min(res_a, res_b), max(res_a, res_b))
        pairs.add(pair)
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate a PyMOL script highlighting gate and latch residues",
    )
    parser.add_argument(
        "--patch-topology",
        default="data/shared_atoms_topology.pdb",
        help="topology used for visualization; use shared atoms topology if available to ensure residue numbers match between gate/latch CSVs and structure",
    )
    parser.add_argument(
        "--trajectory",
        default="data/patch_traj_shared.xtc",
        help="metadynamics trajectory",
    )
    parser.add_argument(
        "--gate-residues",
        default="data/gate_residues.csv",
        help="gate residues CSV",
    )
    parser.add_argument(
        "--latches",
        default="data/latches.csv",
        help="latch ranking CSV",
    )
    parser.add_argument(
        "--keep-latches",
        default="",
        help="comma-separated residue pairs to keep (e.g., '51-229,207-213')",
    )
    parser.add_argument(
        "--drop-latches",
        default="",
        help="comma-separated residue pairs to drop (e.g., '203-207,200-201')",
    )
    parser.add_argument("--top-latches", type=int, default=5)
    parser.add_argument("--frame", type=int, default=1580)
    parser.add_argument("--frame-offset", type=int, default=100)
    parser.add_argument(
        "--frame-is-one-based",
        action="store_true",
        help="treat --frame as one-based index (PyMOL default)",
    )
    parser.add_argument(
        "--output-pml",
        default="figures/pymol_gate_latch.pml",
        help="output PyMOL script path",
    )
    parser.add_argument(
        "--image-prefix",
        default="figures/latch",
        help="prefix for rendered images",
    )
    args = parser.parse_args()

    gate_path = Path(args.gate_residues)
    latch_path = Path(args.latches)

    # read gate residues and latch pairs from CSVs, if they exist, and build PyMOL selections
    gate_residues = []
    if gate_path.exists():
        gate_df = pd.read_csv(gate_path)
        if "res" in gate_df.columns:
            gate_residues = gate_df["res"].tolist()
    latch_pairs = []
    if latch_path.exists():
        latch_df = pd.read_csv(latch_path)
        if {"res_a", "res_b", "mean_saliency"}.issubset(latch_df.columns):
            latch_df = latch_df.sort_values("mean_saliency", ascending=False)
        latch_pairs = list(zip(latch_df["res_a"], latch_df["res_b"]))

    keep_pairs = parse_latch_list(args.keep_latches)
    drop_pairs = parse_latch_list(args.drop_latches)
    if keep_pairs:
        latch_pairs = [
            pair
            for pair in latch_pairs
            if (min(pair), max(pair)) in keep_pairs
        ]
    if drop_pairs:
        latch_pairs = [
            pair
            for pair in latch_pairs
            if (min(pair), max(pair)) not in drop_pairs
        ]
    if not keep_pairs and args.top_latches > 0:
        latch_pairs = latch_pairs[: args.top_latches]

    # build PyMOL selection strings
    gate_sel = build_resi_selection(gate_residues)
    if args.frame_is_one_based:
        frame_main = args.frame
    else:
        frame_main = args.frame + 1
    frame_later = frame_main + args.frame_offset

    output_path = Path(args.output_pml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # PyMOL commands to load the structure and trajectory, show the protein, and highlight gate and latch residues
    lines = [
        f"load {args.patch_topology}, protein",
        f"load_traj {args.trajectory}, protein",
        "bg_color white",
        "hide everything, all",
        "show cartoon, protein",
        "color gray80, protein",
        "set cartoon_transparency, 0.3",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "set stick_radius, 0.4",
        "set stick_transparency, 0.0",
    ]

    # highlight gate residues in red and latch pairs in orange, showing latch pairs as sticks
    if gate_sel:
        lines.append(f"select gate, resi {gate_sel}")
        lines.append("color red, gate")
        lines.append("show cartoon, gate")

    focus_parts = []
    if gate_sel:
        focus_parts.append("gate")

    for idx, (res_a, res_b) in enumerate(latch_pairs, start=1):
        lines.append(f"select latch{idx}, resi {int(res_a)}+{int(res_b)}")
        lines.append(f"show sticks, latch{idx}")
        lines.append(f"color orange, latch{idx}")
        focus_parts.append(f"latch{idx}")

    if focus_parts:
        focus_sel = " or ".join(focus_parts)
        lines.append(f"select focus, {focus_sel}")

    # set the view to focus on the gate region, and save images for the main frame and a later frame to show conformational changes
    lines.extend(
        [
            f"frame {frame_main}",
            "center focus",
            "zoom focus, 8.5",
            "orient focus",
            "scene gate_view, store",
            "pseudoatom panel_label, pos=[0.0, 0.0, 0.0], label='Apex'",
            "show labels, panel_label",
            "color black, panel_label",
            f"png {args.image_prefix}_frame{frame_main}.png, dpi=300",
            "delete panel_label",
            f"frame {frame_later}",
            "scene gate_view, recall",
            "pseudoatom panel_label, pos=[0.0, 0.0, 0.0], label='Post'",
            "show labels, panel_label",
            "color black, panel_label",
            f"png {args.image_prefix}_frame{frame_later}.png, dpi=300",
            "delete panel_label",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved PyMOL script to {output_path}")


if __name__ == "__main__":
    main()
