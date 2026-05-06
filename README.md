# cryptic-jepa

we will assume for this experiment that you are running this code in a conda environment with the necessary dependencies installed. if you have not set up your environment yet, please download the packages in `requirements.txt` and install them using pip or conda.

we will also assume you are running this on an ARM Mac, so if you are using a different operating system, you may need to adjust some of the installation instructions for GROMACS and other dependencies, as well as the torch code to ensure it runs on cuda if you have an NVIDIA GPU.

you will also have to have GROMACS installed to extract the reference structure for the metadynamics trajectory, which is necessary for the topology alignment step in `traj_verification.py`. instructions for this are included in the script.

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

this will also write shared-atom topology and trajectories for downstream visualization:

- `data/shared_atoms_topology.pdb`
- `data/patch_traj_shared.xtc`
- `data/eq_patch_traj_shared.xtc`

## dataset and dataloader

after generating the contact maps with `src/data/data_processing.py`, use the dataset and dataloader utilities to build JEPA context-target pairs with a max gap and optional gaussian jitter.

the goal here is to teach the model local temporal consistency in the equilibrium regime. the `max_gap` defines how far apart context/target frames can be, and `jitter_std` adds small noise to encourage robustness to thermal fluctuations.

smoke test the dataloader on the equilibrium tensors:

```bash
python -m tests.dataloader_smoketest
```

use the dataloader in your training script if not using built-in training utilities

```python
from src.data.dataloader import create_dataloaders

train_loader, val_loader = create_dataloaders(
    contact_map_path="data/eq_jepa_contact_maps.pt",
    batch_size=32,       # optimal for variance stability and VRAM
    max_gap=50,
    jitter_std=0.01,
    val_split=0.1,
    pin_memory=True,     # ensure this is True for GPU training
    drop_last=True       # drops the final uneven batch to prevent variance spikes
)
```

## training

train the EB-JEPA model using `src/training/train.py`:

```bash
python -m src.training.train
```

the training objective is a JEPA prediction error with variance/covariance regularizers, using equilibrium-only data to establish a stable baseline manifold.

## inference

run the aligned anomaly plot (JEPA energy vs. pocket distance) using `src/inference/detect.py`:

```bash
python -m src.inference.detect
```

if you change the temporal gap or smoothing window in the script, keep those values consistent with the evaluation steps below.

this plot is used to test whether the JEPA prediction error rises ahead of the pocket distance increase (a leading indicator of opening), after time alignment between the model output and the COLVAR series.

the output plot is saved to:

- `figures/metadynamics_anomaly.png`

## evaluation

### anomaly scoring

compute baseline statistics from the equilibrium trajectory and z-score the metadynamics trajectory using `src/evaluation/anomaly_scorer.py`:

```bash
python -m src.evaluation.anomaly_scorer
```

important args to keep consistent across the pipeline:

- `--temporal-gap` and `--smooth-window` define the baseline used for z-scores
- the same values should be used for `two_nn` and any downstream analysis that consumes `baseline_stats.json`

use `--help` to see additional options.

this step establishes a quantitative anomaly signal relative to equilibrium noise and produces a list of candidate transition frames.

outputs:

- `data/baseline_stats.json`
- `data/anomaly_frames.npy`

### saliency mapping

backpropagate the prediction error into the contact map to generate a saliency map:

```bash
python -m src.evaluation.saliency_mapping
```

if you changed `--temporal-gap` for scoring, pass the same value here.

this step identifies which contact-map entries drive the anomaly score, providing atom-level attribution for the transition signal.

outputs:

- `data/saliency_map.npy`
- `data/saliency_contacts.csv`
- `data/saliency_contacts_mapped.csv`

### contact filtering (phase 1)

filter intra-residue pairs, aggregate residue-level contacts, and identify gate segments:

```bash
python -m src.evaluation.contact_filtering
```

outputs:

- `data/contacts_filtered.csv`
- `data/latches.csv`
- `data/gate_residues.csv`
- `data/gate_segments.csv`

this step converts the atom-level saliency into residue-level latches and a contiguous gate region, which is easier to interpret biologically.

### report plots

generate report-ready figures from the saliency results:

```bash
python -m src.evaluation.saliency_plots
```

outputs:

- `figures/saliency_heatmap.png`
- `figures/saliency_latch_table.png`
- `figures/saliency_latch_schematic.png`
- `data/saliency_latch_table.csv`

the schematic highlights gate-scaffold latches; the heatmap gives a dense pairwise view; the table is a ranked summary for narrative clarity.

### two-nn intrinsic dimensionality

compare baseline vs. transition intrinsic dimensionality using anomaly frames as the transition subset:

```bash
python -m src.evaluation.two_nn --standardize --use-anomaly-frames
```

if you changed `--temporal-gap` or `--smooth-window` in anomaly scoring, pass the same values here.
use `--help` for more options.

this is a quantitative sanity check on whether the transition subset occupies a different latent complexity than baseline.

outputs:

- `figures/two_nn_baseline_transition.png`
- `data/two_nn_stats.csv`

### pymol visualization (optional)

generate a PyMOL script to visualize the gate and selected latches:

```bash
python -m src.evaluation.pymol_gate_latch --keep-latches "51-229,207-213,60-196"
```

use `--help` to see optional arguments for frames and latch selection.

then run PyMOL:

```bash
pymol -cq figures/pymol_gate_latch.pml
```

this is a supporting visualization to anchor the residue-level latches back onto the 3d structure, showing how they relate to the gate region and to each other.
