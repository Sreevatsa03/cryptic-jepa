import argparse
from pathlib import Path

import mdtraj as md
import numpy as np
import torch

from src.data.traj_verification import (
    get_pocket_residue_indices,
    ligand_resname,
    pocket_source_pdb,
    pocket_threshold,
)
from src.models.jepa_model import JEPAModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_contact_maps(path):
    contact_maps = torch.load(path, map_location="cpu")
    if contact_maps.dim() == 3:
        contact_maps = contact_maps.unsqueeze(1)
    if contact_maps.dim() != 4:
        raise ValueError("contact_maps must have shape (frames, 1, atoms, atoms)")
    return contact_maps.float()


def select_anomaly_indices(indices, top_k, mode, seed):
    if indices.size == 0:
        return indices
    if top_k is None or top_k <= 0 or top_k >= indices.size:
        return indices
    if mode == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(indices, size=top_k, replace=False))
    return indices[:top_k]


@torch.no_grad()
def compute_frame_mse(model, context, target):
    _, z_target, z_pred = model(context, target)
    return (z_pred - z_target).pow(2).mean()


def main():
    parser = argparse.ArgumentParser(
        description="Backpropagate JEPA prediction error to contact maps",
    )
    parser.add_argument(
        "--contact-maps",
        default="data/jepa_contact_maps.pt",
        help="contact map tensor path",
    )
    parser.add_argument(
        "--weights",
        default="models/best_jepa.pth",
        help="trained JEPA weights",
    )
    parser.add_argument(
        "--anomaly-indices",
        default="data/anomaly_frames.npy",
        help="anomaly frame indices",
    )
    parser.add_argument("--temporal-gap", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--sample",
        choices=["first", "random"],
        default="first",
        help="how to select top-k anomaly frames",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--predictor-hidden-dim", type=int, default=256)
    parser.add_argument("--top-contacts", type=int, default=50)
    parser.add_argument(
        "--output-saliency",
        default="data/saliency_map.npy",
        help="output path for the aggregated saliency map",
    )
    parser.add_argument(
        "--output-contacts",
        default="data/saliency_contacts.csv",
        help="output CSV path for top contact pairs",
    )
    parser.add_argument(
        "--patch-topology",
        default="data/shared_atoms_topology.pdb",
        help="patch topology used to map atom indices to residue numbers; use shared atoms topology if available",
    )
    parser.add_argument(
        "--pocket-source-pdb",
        default=pocket_source_pdb,
        help="complex pdb used to define the pocket residues",
    )
    parser.add_argument(
        "--ligand-resname",
        default=ligand_resname,
        help="ligand residue name used to define pocket residues",
    )
    parser.add_argument(
        "--pocket-threshold",
        type=float,
        default=pocket_threshold,
        help="distance threshold (nm) for pocket residues",
    )
    parser.add_argument(
        "--skip-residue-mapping",
        action="store_true",
        help="skip mapping atom indices to residue numbers",
    )
    args = parser.parse_args()

    contact_map_path = Path(args.contact_maps)
    if not contact_map_path.exists():
        raise SystemExit(f"missing contact map tensor: {contact_map_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"missing model weights: {weights_path}")

    anomaly_path = Path(args.anomaly_indices)
    if not anomaly_path.exists():
        raise SystemExit(f"missing anomaly indices: {anomaly_path}")

    device = get_device()
    contact_maps = load_contact_maps(contact_map_path)

    model = JEPAModel(
        in_channels=1,
        latent_dim=args.latent_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # load anomaly frame indices and select top-k
    anomaly_indices = np.load(anomaly_path).astype(np.int64)
    anomaly_indices = select_anomaly_indices(
        anomaly_indices,
        args.top_k,
        args.sample,
        args.seed,
    )

    if anomaly_indices.size == 0:
        raise SystemExit("no anomaly indices found")

    # compute saliency maps for selected anomaly frames and aggregate by averaging
    gradients = []
    for target_idx in anomaly_indices:
        context_idx = target_idx - args.temporal_gap
        if context_idx < 0 or target_idx >= contact_maps.shape[0]:
            continue

        context = contact_maps[context_idx].unsqueeze(0).to(device)
        target = contact_maps[target_idx].unsqueeze(0).to(device)
        context.requires_grad_(True)

        z_context, z_target, z_pred = model(context, target)
        loss = (z_pred - z_target).pow(2).mean()
        loss.backward()

        grad = context.grad.detach().abs().squeeze(0).squeeze(0).cpu().numpy()
        gradients.append(grad)

        model.zero_grad(set_to_none=True)

    if not gradients:
        raise SystemExit("no valid anomaly frames after temporal gap filtering")

    saliency = np.mean(np.stack(gradients, axis=0), axis=0)
    saliency = (saliency + saliency.T) / 2.0

    output_path = Path(args.output_saliency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, saliency.astype(np.float32))

    masked = np.triu(saliency, k=1)
    flat = masked.reshape(-1)
    if args.top_contacts <= 0:
        top_contacts = flat.size
    else:
        top_contacts = min(args.top_contacts, flat.size)

    top_indices = np.argsort(flat)[-top_contacts:][::-1]
    row_idx, col_idx = np.unravel_index(top_indices, saliency.shape)

    print(f"saved saliency map to {output_path}")
    contact_rows = [(int(i), int(j)) for i, j in zip(row_idx, col_idx)]
    contact_path = Path(args.output_contacts)
    contact_path.parent.mkdir(parents=True, exist_ok=True)

    with contact_path.open("w", encoding="utf-8") as handle:
        handle.write("i,j\n")
        for i, j in contact_rows:
            handle.write(f"{i},{j}\n")

    print(f"saved contact pairs to {contact_path}")

    if args.skip_residue_mapping:
        return

    patch_topology = Path(args.patch_topology)
    if not patch_topology.exists():
        print(f"warning: missing patch topology at {patch_topology}; skipping mapping")
        return

    patch = md.load(str(patch_topology))
    if patch.topology.n_atoms != saliency.shape[0]:
        print(
            "warning: patch topology atom count does not match saliency size; "
            "skipping residue mapping"
        )
        return

    atom_resseq = [atom.residue.resSeq for atom in patch.topology.atoms]
    atom_resname = [atom.residue.name for atom in patch.topology.atoms]

    pocket_residue_set = None
    pocket_source = Path(args.pocket_source_pdb)
    if pocket_source.exists():
        pocket_residue_indices, _ = get_pocket_residue_indices(
            args.pocket_source_pdb,
            args.ligand_resname,
            args.pocket_threshold,
        )
        pocket_residue_set = set(int(x) for x in pocket_residue_indices)

    mapped_path = contact_path.with_name(f"{contact_path.stem}_mapped.csv")
    with mapped_path.open("w", encoding="utf-8") as handle:
        header = "i,j,res_i,resname_i,res_j,resname_j,in_pocket_i,in_pocket_j\n"
        handle.write(header)
        for i, j in contact_rows:
            res_i = atom_resseq[i]
            res_j = atom_resseq[j]
            name_i = atom_resname[i]
            name_j = atom_resname[j]
            if pocket_residue_set is None:
                in_i = ""
                in_j = ""
            else:
                in_i = "1" if res_i in pocket_residue_set else "0"
                in_j = "1" if res_j in pocket_residue_set else "0"
            handle.write(
                f"{i},{j},{res_i},{name_i},{res_j},{name_j},{in_i},{in_j}\n"
            )

    print(f"saved mapped contacts to {mapped_path}")

    print("top contact pairs mapped to residues:")
    for i, j in contact_rows:
        res_i = atom_resseq[i]
        res_j = atom_resseq[j]
        name_i = atom_resname[i]
        name_j = atom_resname[j]
        if pocket_residue_set is not None:
            flag_i = "" if res_i in pocket_residue_set else "!"
            flag_j = "" if res_j in pocket_residue_set else "!"
            print(f"{i},{j} -> {res_i}{flag_i}({name_i}), {res_j}{flag_j}({name_j})")
        else:
            print(f"{i},{j} -> {res_i}({name_i}), {res_j}({name_j})")


if __name__ == "__main__":
    main()
