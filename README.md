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

verify the trajectories and the residue mapping between the pocket residues and the trajectory topologies using `src/data/traj_verification.py`:

```bash
python -m src.data.traj_verification
```

preprocess the trajectories and generate contact maps in a tensor using `src/data/data_processing.py`:

```bash
python -m src.data.data_processing
```

## dataset and dataloader

after generating the contact maps with `src/data/data_processing.py`, use the dataset and dataloader utilities to build JEPA context-target pairs with a max gap and optional gaussian jitter.

smoke test the dataloader on the equilibrium tensors:

```bash
python -m src.data.dataloader_smoketest
```

use the dataloader in your training script:

```python
from src.data.dataloader import create_dataloaders

train_loader, val_loader = create_dataloaders(
 contact_map_path="data/eq_jepa_contact_maps.pt",
 batch_size=32,
 max_gap=50,
 jitter_std=0.01,
 val_split=0.1,
 seed=42,
)
```

## training

Minor Recommendations for the Training Loop
• Loss Monitoring: Because 601 frames is a small set, watch for the training loss falling significantly lower than the validation loss. If this happens, increase jitter_std to 0.02 or 0.03.
• Batch Size: For 597×597 matrices, a batch_size of 32 (default in dataloader.py) is a good starting point for most modern GPUs. If you hit Out-Of-Memory (OOM) errors, drop to 16.
