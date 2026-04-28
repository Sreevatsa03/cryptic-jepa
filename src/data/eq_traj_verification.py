from src.data.traj_verification import (
       get_pocket_residue_indices,
       run_equilibrium,
       ligand_resname,
       pocket_source_pdb,
       pocket_threshold,
)

eq_xtc = "data/complex_S1_site680_CHEMBL4105630_rep1_complexmd_fit_nw.xtc"
eq_pdb = "data/complex_S1_site680_CHEMBL4105630_rep1_ref_nw.pdb"
eq_start_time = 40000


if __name__ == "__main__":
       pocket_residue_indices, pocket_topology = get_pocket_residue_indices(
              pocket_source_pdb,
              ligand_resname,
              pocket_threshold,
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