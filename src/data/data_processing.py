import mdtraj as md
import torch


def atom_key(atom):
    """
    Generate a unique key for an atom based on its chain, residue, and name
    """
    res = atom.residue
    chain = res.chain
    chain_id = getattr(chain, "id", None)
    if chain_id in (None, "", " "):
        chain_id = getattr(chain, "chain_id", None)
    if chain_id in (None, "", " "):
        chain_id = getattr(chain, "index", None)
    return (str(chain_id), int(res.resSeq), res.name, atom.name)


def aligned_atom_indices(eq_topology, meta_topology):
    """
    Find indices of atoms that are common between two topologies based on their unique keys
    """
    eq_keys = [atom_key(atom) for atom in eq_topology.atoms]
    meta_keys = [atom_key(atom) for atom in meta_topology.atoms]

    eq_map = {}
    for idx, key in enumerate(eq_keys):
        if key in eq_map:
            raise SystemExit(f"duplicate atom key in equilibrium topology: {key}")
        eq_map[key] = idx

    meta_map = {}
    for idx, key in enumerate(meta_keys):
        if key in meta_map:
            raise SystemExit(f"duplicate atom key in metadynamics topology: {key}")
        meta_map[key] = idx

    common_keys = [key for key in eq_keys if key in meta_map]
    if not common_keys:
        raise SystemExit("no shared atoms found between equilibrium and metadynamics patches")

    eq_indices = [eq_map[key] for key in common_keys]
    meta_indices = [meta_map[key] for key in common_keys]
    return eq_indices, meta_indices


def process_trajectory_to_tensors(traj, output_path):
    """
    Convert an MDTraj trajectory to a tensor of continuous contact maps and save to disk
    """
    coords = torch.tensor(traj.xyz, dtype=torch.float32)
    print(f"coordinate tensor shape: {coords.shape}")

    print("\ncomputing pairwise distance matrices")
    distances = torch.cdist(coords, coords)

    print("\napplying continuous contact normalization")
    alpha = 15.0
    d_0 = 0.8

    contact_maps = 1.0 / (1.0 + torch.exp(alpha * (distances - d_0)))
    contact_maps = contact_maps.unsqueeze(1)

    torch.save(contact_maps, output_path)
    print(f"saved final tensor of shape {contact_maps.shape} to {output_path}")


def main(
    eq_xtc,
    eq_pdb,
    meta_xtc,
    meta_pdb,
    eq_out,
    meta_out,
    shared_topology_out,
    eq_shared_xtc_out,
    meta_shared_xtc_out,
):
    """
    Load equilibrium and metadynamics patch trajectories, align their topologies to find shared atoms,
    slice the trajectories to those shared atoms, and process each trajectory into tensors of continuous contact maps
    """
    print(f"loading equilibrium patch trajectory from {eq_xtc}")
    eq_traj = md.load(eq_xtc, top=eq_pdb)

    print(f"loading metadynamics patch trajectory from {meta_xtc}")
    meta_traj = md.load(meta_xtc, top=meta_pdb)

    eq_step = eq_traj.timestep
    meta_step = meta_traj.timestep
    print(f"\ntimesteps: eq={eq_step} ps, meta={meta_step} ps")

    target_step = max(eq_step, meta_step)
    if eq_step != meta_step:
        if eq_step < target_step:
            stride = int(target_step / eq_step)
            print(f"downsampling equilibrium trajectory by a factor of {stride} to match {target_step} ps")
            eq_traj = eq_traj[::stride]
            
        elif meta_step < target_step:
            stride = int(target_step / meta_step)
            print(f"downsampling metadynamics trajectory by a factor of {stride} to match {target_step} ps")
            meta_traj = meta_traj[::stride]

    if len(eq_traj.time) > 1:
        resolution = float(eq_traj.time[1] - eq_traj.time[0])
    elif len(meta_traj.time) > 1:
        resolution = float(meta_traj.time[1] - meta_traj.time[0])
    else:
        resolution = float(target_step)
    print(f"effective resolution after downsampling: {resolution:.3f} ps/frame")

    eq_indices, meta_indices = aligned_atom_indices(eq_traj.topology, meta_traj.topology)
    print(
        f"\nusing {len(eq_indices)} shared atoms to enforce matching tensor shapes "
        f"(eq={eq_traj.n_atoms}, meta={meta_traj.n_atoms})"
    )

    eq_traj = eq_traj.atom_slice(eq_indices)
    meta_traj = meta_traj.atom_slice(meta_indices)

    if eq_traj.n_atoms != meta_traj.n_atoms:
        raise SystemExit("atom count mismatch after intersection; check topology alignment")

    shared_topology_out = str(shared_topology_out)
    eq_traj[0].save_pdb(shared_topology_out)
    print(f"saved shared-atoms topology to {shared_topology_out}")

    eq_shared_xtc_out = str(eq_shared_xtc_out)
    meta_shared_xtc_out = str(meta_shared_xtc_out)
    eq_traj.save_xtc(eq_shared_xtc_out)
    meta_traj.save_xtc(meta_shared_xtc_out)
    print(
        "saved shared-atoms trajectories to {} and {}".format(
            eq_shared_xtc_out,
            meta_shared_xtc_out,
        )
    )

    print("\nprocessing equilibrium baseline tensors")
    process_trajectory_to_tensors(eq_traj, eq_out)

    print("\nprocessing metadynamics tensors")
    process_trajectory_to_tensors(meta_traj, meta_out)


if __name__ == "__main__":
    EQ_XTC_IN = "data/eq_patch_traj.xtc"
    EQ_PDB_IN = "data/eq_patch_topology.pdb"
    META_XTC_IN = "data/patch_traj.xtc"
    META_PDB_IN = "data/patch_topology.pdb"

    EQ_TENSOR_OUT = "data/eq_jepa_contact_maps.pt"
    META_TENSOR_OUT = "data/jepa_contact_maps.pt"
    SHARED_TOPOLOGY_OUT = "data/shared_atoms_topology.pdb"
    EQ_SHARED_XTC_OUT = "data/eq_patch_traj_shared.xtc"
    META_SHARED_XTC_OUT = "data/patch_traj_shared.xtc"

    main(
        EQ_XTC_IN,
        EQ_PDB_IN,
        META_XTC_IN,
        META_PDB_IN,
        EQ_TENSOR_OUT,
        META_TENSOR_OUT,
        SHARED_TOPOLOGY_OUT,
        EQ_SHARED_XTC_OUT,
        META_SHARED_XTC_OUT,
    )