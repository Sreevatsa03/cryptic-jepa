import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

# paths
eq_xtc = 'data/complex_S1_site680_CHEMBL4105630_rep1_complexmd_fit_nw.xtc'
colvar_file = 'data/metad_S1_BAR_COLVAR'
pdb_file = 'data/metad_S1_BAR_meta_ref.pdb'
meta_xtc = 'data/metad_S1_BAR_meta_protein.xtc'

# test topology alignment
print("testing metadynamics topology alignment")
try:
    # load only the first frame to test quickly
    test_meta = md.load_frame(meta_xtc, 0, top=pdb_file)
    print("metadynamics trajectory matches the pdb topology")
    print(f"metadynamics trajectory shape: {test_meta.xyz.shape}")
except Exception as e:
    print(f"topology mismatch: {e}")
    print("you must extract a reference structure directly from the 'metad_S1_BAR_meta.tpr' file using GROMACS (gmx trjconv) to use as your topology for the metadynamics trajectory")
    print("gmx trjconv -s data/metad_S1_BAR_meta.tpr -f data/metad_S1_BAR_meta.xtc -dump 0 -o data/metad_S1_BAR_meta_ref.pdb")
    print("gmx trjconv -s data/metad_S1_BAR_meta.tpr -f data/metad_S1_BAR_meta.xtc -o data/metad_S1_BAR_meta_protein.xtc")
    exit()

# load the full metadynamics trajectory
traj = md.load(meta_xtc, top=pdb_file)

# superpose the trajectory to the reference structure using backbone atoms to remove global translations and rotations
print("\naligning trajectory to reference backbone...")
backbone_indices = traj.topology.select("backbone")
traj.superpose(traj, frame=0, atom_indices=backbone_indices)

# load the complex structure to identify the pocket residues
complex_pdb = 'data/complex_S1_site680_CHEMBL4105630_rep1_ref_nw.pdb'
t = md.load(complex_pdb)

# print possible residue names
ligand_names = set(res.name for res in t.topology.residues if not res.is_protein and not res.is_water)
print(f"\npotential ligand residue names: {ligand_names}")

# using MOL by default
print("\nselecting ligand atoms with resname MOL")
ligand_indices = t.topology.select("resname MOL") 

# find protein atoms within 10 angstroms of any ligand atom
threshold = 1.0  # in nm
print(f"\nidentifying protein atoms within {threshold*10} angstroms of the ligand")
protein_indices = t.topology.select("protein")
neighbor_atoms = md.compute_neighbors(t, threshold, query_indices=ligand_indices, haystack_indices=protein_indices)[0]

# map atom indices back to their PDB residue sequence numbers (1-indexed)
neighbor_residues = set()
for atom_idx in neighbor_atoms:
    atom = t.topology.atom(atom_idx)
    neighbor_residues.add(atom.residue.resSeq)

pocket_residue_indices = np.sort(list(neighbor_residues))
print(f"\nexact PDB residue numbers for the pocket:")
print(f"pocket_residue_indices = np.array({repr(list(pocket_residue_indices))})")

# define spatial patch around the cryptic pocket in BIN1 using the identified residue numbers
pocket_residue_indices

# print number of residues in the pocket
print(f"\nnumber of residues in the pocket: {len(pocket_residue_indices)}")

# create an MDTraj atom selection string for heavy atoms of these residues
res_list = [str(i) for i in pocket_residue_indices]
res_selection_string = "(resSeq " + " or resSeq ".join(res_list) + ") and element != H"
print(f"\nselection string: {res_selection_string}")

# extract the indices of the heavy atoms in the selected residues
target_atom_indices = traj.topology.select(res_selection_string)
print(f"\nextracted {len(target_atom_indices)} heavy atoms for the patch")

# slice the trajectory to keep only these atoms
patch_traj = traj.atom_slice(target_atom_indices)
print(f"\npatch trajectory shape: {patch_traj.xyz.shape}")