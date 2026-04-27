# cryptic-jepa

we will assume for this experiment that you are running this code in a conda environment with the necessary dependencies installed. if you have not set up your environment yet, please download the packages in `requirements.txt` and install them using pip or conda.

we will also assume you are running this on an ARM Mac, so if you are using a different operating system, you may need to adjust some of the installation instructions for GROMACS and other dependencies, as well as the torch code to ensure it runs on cuda if you have an NVIDIA GPU.

you will also have to have GROMACS installed to extract the reference structure for the metadynamics trajectory, which is necessary for the topology alignment step in `traj_verification.py`. Instructions for this are included in the script.

if you have a Mac, use the following command to install GROMACS using Homebrew:

```bash
brew install gromacs
```

## data

[1]Ç. Özkurt, “MD trajectories for cryptic pocket discovery in Alzheimer's disease risk proteins BIN1, PICALM, and CD2AP (CRYPTAD) Part 1”. Zenodo, Apr. 18, 2026. doi: 10.5281/zenodo.19643896.

this experiment focuses on the BIN1 protein, and the data files for this experiment are available at the above link.

download the necessary data files from the above link and place them in a folder named `data` in this repository.

- `complex_S1_site680_CHEMBL4105630_rep1_ref_nw.pdb`
- `complex_S1_site680_CHEMBL4105630_rep1_complexmd_fit_nw.xtc`
- `metad_S1_BAR_meta.xtc`
- `metad_S1_BAR_meta.tpr`
- `metad_S1_BAR_COLVAR`

run GROMACS to extract the reference structure for the metadynamics trajectory:

- enter `1` when prompted to select the group for the output structure, as this is the group containing the protein atoms. this will ensure that the reference structure contains only the protein atoms, which is necessary for the topology alignment step in `traj_verification.py`.

```bash
gmx trjconv -s data/metad_S1_BAR_meta.tpr -f data/metad_S1_BAR_meta.xtc -dump 0 -o data/metad_S1_BAR_meta_ref.pdb
```

and to extract the protein-only trajectory for the metadynamics simulation:

```bash
gmx trjconv -s data/metad_S1_BAR_meta.tpr -f data/metad_S1_BAR_meta.xtc -o data/metad_S1_BAR_meta_protein.xtc
```
