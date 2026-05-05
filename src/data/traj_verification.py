import mdtraj as md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pocket definition
pocket_source_pdb = 'data/complex_S1_site680_CHEMBL4105630_rep1_ref_nw.pdb'
ligand_resname = 'MOL'
pocket_threshold = 1.0  # in nm

# metadynamics inputs
meta_colvar_file = 'data/metad_S1_BAR_COLVAR'
meta_pdb_file = 'data/metad_S1_BAR_meta_ref.pdb'
meta_xtc = 'data/metad_S1_BAR_meta_protein.xtc'

# equilibrium inputs
eq_xtc = 'data/complex_S1_site680_CHEMBL4105630_rep1_complexmd_fit_nw.xtc'
eq_pdb = pocket_source_pdb
eq_start_time = 40000


def read_colvar_fields(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#! FIELDS"):
                return line.strip().split()[2:]
    return None


def get_pocket_residue_indices(complex_pdb, ligand_resname, threshold):
    """
    Identify protein residues that are within a specified distance of the ligand in the complex structure
    """
    t = md.load(complex_pdb)

    ligand_names = set(res.name for res in t.topology.residues if not res.is_protein and not res.is_water)
    print(f"\npotential ligand residue names: {ligand_names}")

    # select ligand atoms based on residue name
    print(f"\nselecting ligand atoms with resname {ligand_resname}")
    ligand_indices = t.topology.select(f"resname {ligand_resname}")
    if ligand_indices.size == 0:
        raise SystemExit(f"no ligand atoms found for resname {ligand_resname} in {complex_pdb}")

    # compute neighboring protein atoms within distance threshold
    threshold_angstrom = threshold * 10
    print(f"\nidentifying protein atoms within {threshold_angstrom} angstroms of the ligand")
    protein_indices = t.topology.select("protein")
    neighbor_atoms = md.compute_neighbors(
        t,
        threshold,
        query_indices=ligand_indices,
        haystack_indices=protein_indices,
    )[0]

    neighbor_residues = set()
    for atom_idx in neighbor_atoms:
        atom = t.topology.atom(atom_idx)
        neighbor_residues.add(atom.residue.resSeq)

    # define pocket residues
    pocket_residue_indices = np.array(sorted(neighbor_residues), dtype=int)
    print("\nexact PDB residue numbers for the pocket:")
    print(f"pocket_residue_indices = np.array({repr(list(pocket_residue_indices))})")
    print(f"\nnumber of residues in the pocket: {len(pocket_residue_indices)}")

    return pocket_residue_indices, t.topology


def verify_residue_mapping(source_topology, target_topology, pocket_residue_indices, label):
    """
    Verify that the pocket residues defined by their residue numbers in the source topology can be unambiguously mapped to residues in the target topology
    """
    print(f"\nverifying residue mapping between pocket source and {label} topology")

    # check that all pocket residue numbers exist in target topology and collect candidate matches
    pocket_residue_set = set(int(x) for x in pocket_residue_indices)
    pocket_residues = [
        res
        for res in source_topology.residues
        if res.is_protein and res.resSeq in pocket_residue_set
    ]

    # build a mapping of residue number to residues in target topology for efficient lookup
    target_res_by_resseq = {}
    for res in target_topology.residues:
        if not res.is_protein:
            continue
        target_res_by_resseq.setdefault(res.resSeq, []).append(res)

    def get_chain_label(residue):
        chain = residue.chain
        chain_id = getattr(chain, "id", None)
        if chain_id not in (None, "", " "):
            return str(chain_id)
        chain_id = getattr(chain, "chain_id", None)
        if chain_id not in (None, "", " "):
            return str(chain_id)
        chain_index = getattr(chain, "index", None)
        if chain_index is not None:
            return str(chain_index)
        return None

    # check for mismatches in residue number, chain identifier, and residue name
    mismatches = []
    for res in pocket_residues:
        candidates = target_res_by_resseq.get(res.resSeq, [])
        if not candidates:
            mismatches.append(f"resSeq {res.resSeq} ({res.name}) missing in {label} topology")
            continue
        chain_id = get_chain_label(res)
        if chain_id not in (None, "", " "):
            candidates = [c for c in candidates if get_chain_label(c) == chain_id]
            if not candidates:
                mismatches.append(
                    f"resSeq {res.resSeq} chain {chain_id} missing in {label} topology"
                )
                continue
        name_matches = [c for c in candidates if c.name == res.name]
        if len(name_matches) == 1:
            continue
        if len(name_matches) == 0:
            mismatches.append(f"resSeq {res.resSeq} name mismatch (complex {res.name})")
        else:
            mismatches.append(f"resSeq {res.resSeq} has multiple matches; ambiguous mapping")

    # report mismatches and exit if any are found
    if mismatches:
        print(f"{label} residue mapping check failed; do not reuse pocket resSeq directly")
        for msg in mismatches[:10]:
            print(f" - {msg}")
        if len(mismatches) > 10:
            print(f" - ... {len(mismatches) - 10} more")
        print("align residue numbering between structures or derive pocket residues in the target topology")
        raise SystemExit(1)

    print(f"{label} residue mapping check passed for pocket residues")


def prepare_traj(traj):
    """
    Apply standard preprocessing steps to an MDTraj trajectory
    """
    print("\nfixing periodic boundary conditions")
    traj.image_molecules(inplace=True)

    print("\naligning trajectory to reference backbone")
    backbone_indices = traj.topology.select("backbone")
    traj.superpose(traj, frame=0, atom_indices=backbone_indices)


def extract_patch(traj, pocket_residue_indices):
    """
    Extract a trajectory slice containing only the atoms of the pocket residues
    """
    res_list = [str(i) for i in pocket_residue_indices]
    res_selection_string = "(resSeq " + " or resSeq ".join(res_list) + ") and element != H"
    print(f"\nselection string: {res_selection_string}")

    target_atom_indices = traj.topology.select(res_selection_string)
    if target_atom_indices.size == 0:
        raise SystemExit("no atoms matched the pocket selection in the target topology")

    print(f"\nextracted {len(target_atom_indices)} heavy atoms for the patch")
    return traj.atom_slice(target_atom_indices)


def run_metadynamics(
    meta_xtc,
    meta_pdb_file,
    colvar_file,
    pocket_residue_indices,
    pocket_topology,
    output_patch_xtc,
    output_patch_topology,
    output_plot,
):
    """
    Load the metadynamics trajectory, verify that the pocket residues can be mapped to the trajectory topology, extract the patch trajectory, and plot the patch RMSD against the metadynamics collective variables for verification
    """
    print("testing metadynamics topology alignment")
    try:
        test_meta = md.load_frame(meta_xtc, 0, top=meta_pdb_file)
        print("metadynamics trajectory matches the pdb topology")
        print(f"metadynamics trajectory shape: {test_meta.xyz.shape}")
    except Exception as e:
        print(f"topology mismatch: {e}")
        print(
            "you must extract a reference structure directly from the 'metad_S1_BAR_meta.tpr' file using GROMACS (gmx trjconv) to use as your topology for the metadynamics trajectory"
        )
        print(
            "gmx trjconv -s data/metad_S1_BAR_meta.tpr -f data/metad_S1_BAR_meta.xtc -dump 0 -o data/metad_S1_BAR_meta_ref.pdb"
        )
        print(
            "gmx trjconv -s data/metad_S1_BAR_meta.tpr -f data/metad_S1_BAR_meta.xtc -o data/metad_S1_BAR_meta_protein.xtc"
        )
        raise SystemExit(1)
    
    # load and prepare the metadynamics trajectory
    traj = md.load(meta_xtc, top=meta_pdb_file)
    prepare_traj(traj)

    verify_residue_mapping(pocket_topology, traj.topology, pocket_residue_indices, "metadynamics")
    patch_traj = extract_patch(traj, pocket_residue_indices)

    patch_traj.save_xtc(output_patch_xtc)
    patch_traj[0].save_pdb(output_patch_topology)
    print(
        f"\nsaved patch trajectory with shape {patch_traj.xyz.shape} to {output_patch_xtc} and topology to {output_patch_topology}"
    )

    # load COLVAR data and align to trajectory timestamps for verification plotting
    print("\nCOLVAR verification")
    fields = read_colvar_fields(colvar_file)
    colvar_df = pd.read_csv(colvar_file, comment="#", header=None, delimiter="\s+")
    if fields and len(fields) == colvar_df.shape[1]:
        colvar_df.columns = fields
    else:
        colvar_df.columns = [f"col{i}" for i in range(colvar_df.shape[1])]

    time_col = "time" if "time" in colvar_df.columns else colvar_df.columns[0]
    cv1_col = "cv1" if "cv1" in colvar_df.columns else colvar_df.columns[1]
    cv2_col = "cv2" if "cv2" in colvar_df.columns else colvar_df.columns[2]

    colvar_time = np.array(colvar_df[time_col].values, dtype=np.float32)
    cv1 = np.array(colvar_df[cv1_col].values, dtype=np.float32)
    cv2 = np.array(colvar_df[cv2_col].values, dtype=np.float32)
    print(f"\nloaded {len(cv1)} frames from COLVAR")

    traj_time = patch_traj.time
    print(f"\npatch trajectory spans from {traj_time[0]} to {traj_time[-1]} time units")

    print("\naligning COLVAR data to trajectory timestamps")
    cv1 = np.interp(traj_time, colvar_time, cv1)
    cv2 = np.interp(traj_time, colvar_time, cv2)

    patch_rmsd = md.rmsd(patch_traj, patch_traj, frame=0) * 10

    # create a dual-axis plot of patch RMSD and COLVAR CVs over time for verification
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_rmsd = "tab:red"
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Patch RMSD ($\\AA$)", color=color_rmsd, weight="bold")
    ax1.plot(traj_time, patch_rmsd, color=color_rmsd, alpha=0.9, label="Patch RMSD")
    ax1.tick_params(axis="y", labelcolor=color_rmsd)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Metadynamics CVs", color="black", weight="bold")
    ax2.plot(traj_time, cv1, color="tab:blue", alpha=0.6, label="CV1")
    ax2.plot(traj_time, cv2, color="tab:green", alpha=0.6, label="CV2")
    ax2.tick_params(axis="y", labelcolor="black")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Patch RMSD vs. Metadynamics CVs")
    fig.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"plot saved to {output_plot}")


def run_equilibrium(
    eq_xtc,
    eq_pdb,
    start_time,
    pocket_residue_indices,
    pocket_topology,
    output_patch_xtc,
    output_patch_topology,
    output_plot,
):
    """
    Load the equilibrium trajectory, verify that the pocket residues can be mapped to the trajectory topology, extract the patch trajectory, and plot the patch RMSD over time for verification
    """
    print("\nloading rep1 equilibrium trajectory")
    traj = md.load(eq_xtc, top=eq_pdb)

    print("\nfixing periodic boundary conditions")
    traj.image_molecules(inplace=True)

    # slice trajectory to isolate stable window and align to first frame for RMSD calculation
    print("\nslicing trajectory to isolate the 40ns-100ns stable window")
    valid_indices = np.where(traj.time >= start_time)[0]
    if valid_indices.size == 0:
        raise SystemExit(f"no frames found at or after {start_time}")
    clean_traj = traj[valid_indices]

    print("\naligning cleaned trajectory to its own first frame")
    backbone_indices = clean_traj.topology.select("backbone")
    clean_traj.superpose(clean_traj, frame=0, atom_indices=backbone_indices)

    verify_residue_mapping(pocket_topology, clean_traj.topology, pocket_residue_indices, "equilibrium")
    patch_traj = extract_patch(clean_traj, pocket_residue_indices)

    patch_rmsd = md.rmsd(patch_traj, patch_traj, frame=0) * 10

    plt.figure(figsize=(10, 5))
    plt.plot(patch_traj.time, patch_rmsd, color="tab:green", alpha=0.9)
    plt.xlabel("Time (ps)")
    plt.ylabel("Patch RMSD ($\\AA$)", weight="bold")
    plt.title(f"Sliced Equilibrium Baseline")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"plot saved to {output_plot}")

    patch_traj.save_xtc(output_patch_xtc)
    patch_traj[0].save_pdb(output_patch_topology)


def main():
    pocket_residue_indices, pocket_topology = get_pocket_residue_indices(
        pocket_source_pdb,
        ligand_resname,
        pocket_threshold,
    )

    run_metadynamics(
        meta_xtc=meta_xtc,
        meta_pdb_file=meta_pdb_file,
        colvar_file=meta_colvar_file,
        pocket_residue_indices=pocket_residue_indices,
        pocket_topology=pocket_topology,
        output_patch_xtc="data/patch_traj.xtc",
        output_patch_topology="data/patch_topology.pdb",
        output_plot="figures/colvar_verification_plot.png",
    )

    run_equilibrium(
        eq_xtc=eq_xtc,
        eq_pdb=eq_pdb,
        start_time=eq_start_time,
        pocket_residue_indices=pocket_residue_indices,
        pocket_topology=pocket_topology,
        output_patch_xtc="data/eq_patch_traj.xtc",
        output_patch_topology="data/eq_patch_topology.pdb",
        output_plot="figures/final_training_baseline_plot.png",
    )


if __name__ == "__main__":
    main()