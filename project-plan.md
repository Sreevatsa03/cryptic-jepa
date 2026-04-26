# Project Plan: EB-JEPA for Cryptic Pocket Detection in MD Trajectories

### Phase 1: Local Data Extraction & Verification
* **Target Selection:** Download the equilibrium complex trajectory (`.xtc`), the metadynamics trajectory (`.xtc`), the reference topology (`.pdb`), and the `COLVAR` log for a single target (e.g., BIN1) from the CRYPTAD dataset.
* **Temporal & Spatial Down-sampling:** Load trajectories using `MDTraj` with a strict temporal stride (e.g., `stride=10`). Isolate a ~50-residue spatial patch directly surrounding the cryptic pocket to reduce the input grid and fit within local unified memory limits.
* **Signal Verification:** Before tensor conversion, plot the distance between two heavy atoms on opposite lips of the patched pocket across the metadynamics trajectory. Cross-reference this with the `COLVAR` log to prove the pocket-opening signal survives the striding.
* **Matrix Transformation:** Extract the heavy-atom coordinates for the patched residues. Compute the 50x50 minimum pairwise Euclidean distance matrices. Apply z-scoring independently to each residue pair across the equilibrium data to preserve sub-Ångström structural shifts. Save as batched PyTorch tensors.

### Phase 2: Architecture Construction & Equilibrium Training
* **Hardware Routing:** Hardcode PyTorch to utilize the Apple Metal Performance Shaders (`mps`) backend to ensure the local GPU handles the matrix operations.
* **Model Assembly:** Construct the EB-JEPA utilizing Vision Transformer (ViT) backbones. 
    * *Context Encoder:* Standard ViT updated via gradient descent to map unmasked historical trajectory matrices to latent space ($z_x$).
    * *Target Encoder:* Identical ViT updated strictly via Exponential Moving Average (EMA) to map masked future matrices to target latent space ($z_y$).
    * *Predictor:* Narrow ViT to forecast the target state ($\hat{z}_y$).
* **Baseline Training:** Train the architecture exclusively on the batched equilibrium tensors to internalize standard conformational breathing. 
* **Stability Enforcement:** Integrate Weak-SigReg into the loss function and log the stable rank (sRank) across temporal batches to prevent representation collapse in the latent space.

### Phase 3: Inference Run
* **Environment Lock:** Execute `caffeinate -i python inference_script.py` in the macOS terminal to block sleep state and ensure continuous execution over the 72-hour window.
* **Execution:** Stream the batched metadynamics trajectory tensors through the trained architecture.
* **Logging:** Compute the L2 distance between the predictor output ($\hat{z}_y$) and the target encoder output ($z_y$) for each temporal window. Write these energy-based surprise arrays to disk continuously to prevent data loss in the event of a crash.

### Phase 4: Validation & Finalization
* **Metric Cross-Referencing:** Plot the generated L2 surprise arrays against time. Overlay the physical transition states from the `COLVAR` log. A successful model will show sharp L2 anomaly spikes aligning precisely with the physical pocket opening.
* **Visual Proof:** Map the highest L2-scoring frames back to the 3D structure. Render the open state in PyMOL or VMD to visually correlate the mathematical anomaly with the biophysical event.
* **Final Compilation:** Assemble the methodology, sRank stability plots, L2 vs. COLVAR plots, and visual renders into the final deliverable for the May 6 deadline. Ensure the independent z-scoring and heavy-atom distance rationale are explicitly defended.

### References
1. [PI-JEPA](https://arxiv.org/abs/2604.01349)
2. [CRYPTAD Dataset](https://zenodo.org/records/19643896)
3. [Koopman Invariants](https://arxiv.org/abs/2511.09783)