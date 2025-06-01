# EPG: Extended Phase Graph Simulations for MRI

This repository provides Python-based tools for simulating MRI signal evolution using the Extended Phase Graph (EPG) formalism. It includes both original non-vectorized scripts and newer, optimized versions that support batched (vectorized) inputs for efficient simulation of various scenarios, including Magnetization Transfer (MT) and Multiple Quantum Coherences (MQC). The simulations can leverage GPU acceleration if a CUDA-enabled GPU and PyTorch are correctly configured.

The primary goal of this project is to offer tools for research, education, and development in quantitative MRI, pulse sequence design, and understanding complex MR phenomena.

---

## Scientific Context

The EPG formalism is a powerful and efficient tool to model the evolution of magnetization under the action of arbitrary RF pulse sequences, relaxation, exchange, gradients, and other physical effects in MR imaging. It enables simulation of complex sequences (spin echo, gradient echo, bSSFP, MT, CEST, etc.) and is foundational for sequence optimization, quantitative mapping, and understanding the effects of non-idealities in MRI.

Key references:
- Hennig J. Echoes—how to generate, recognize, use or avoid them in MR imaging sequences. Part I: Fundamental and not so fundamental properties of spin echoes. Concepts Magn Reson. 1991;3:125-143.
- Weigel M. Extended phase graphs: dephasing, RF pulses, and echoes—pure and simple. J Magn Reson Imaging. 2015;41(2):266-295.
- Zur Y, Stokar S, Bendel P. An analysis of fast imaging sequences with steady-state transverse magnetization refocusing. Magn Reson Med. 1988;6(2):175-193.
- Gloor M, Scheffler K, Bieri O. Quantitative magnetization transfer imaging using balanced SSFP. Magn Reson Med. 2008;60(3):691-700.

---

## Repository Structure

The repository is organized as follows:

*   **`non-parallelized-sims/`**: Contains the original EPG simulation scripts. These are generally simpler to understand for basic EPG concepts but are not optimized for batch processing or speed.
*   **`vectorized-sims/`**: Contains newer, optimized EPG simulation scripts. These versions support batched inputs (allowing multiple parameter sets to be simulated simultaneously) and can run efficiently on CPUs or GPUs (via PyTorch). This is the recommended location for most current simulation tasks.
*   **`wiki/`**: Contains detailed Markdown-based documentation on the different EPG models implemented:
    *   `basic_epg_model.md`: Explanation of the fundamental EPG model.
    *   `mt_epg_model.md`: Details on the EPG model for Magnetization Transfer.
    *   `mqc_epg_model.md`: Information on the EPG model for Multiple Quantum Coherences.
    *   `extended_epg_model.md`: Overview of various extensions to the EPG formalism.
*   `background.md`: A general introduction to the EPG formalism, its importance, and core concepts.
*   `ideas.md`: A list of potential future improvements and ideas for this project.

---

## Getting Started

1.  **Understand EPG:**
    *   For a general introduction to EPG, start with [`background.md`](./background.md).
    *   For detailed explanations of specific models, refer to the documents in the [`wiki/`](./wiki/) folder.

2.  **Using the Simulations:**
    *   The latest, optimized, and batched simulations are located in the [`vectorized-sims/`](./vectorized-sims/) folder. Check the `if __name__ == "__main__":` block within each script for example usage.
    *   The original, non-vectorized scripts are in [`non-parallelized-sims/`](./non-parallelized-sims/) and can be useful for understanding the basic EPG algorithm steps in a simpler context.

3.  **Dependencies:**
    *   Python >= 3.8
    *   PyTorch (required for all simulations, enables CPU/GPU execution)
    *   NumPy (often used by PyTorch or for utility)
    *   (Optional, if used by specific scripts) Matplotlib for plotting.

4.  **Running Simulations:**
    *   Clone this repository:
        ```bash
        git clone <repository_url>
        cd <repository_name>
        ```
    *   Ensure you have PyTorch installed (e.g., `pip install torch torchvision torchaudio`).
    *   You can then run the simulation scripts directly, e.g.:
        ```bash
        python vectorized-sims/epg_mri_vectorized.py
        ```
    *   The device for computation (CPU or GPU) is typically determined automatically within the scripts or can be set as a parameter.

---

## Simulation Scripts Overview

The primary simulation scripts are found in the `vectorized-sims/` directory:

*   **`epg_mri_vectorized.py`**: Standard EPG for single-pool systems, supporting batched inputs.
    *   `forward(flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0)`
*   **`epg_extended_vectorized.py`**: Extended EPG model incorporating MT, diffusion, flow, etc., with batched inputs.
    *   Key parameters include those for relaxation, exchange, diffusion (`D`, `bval`), flow (`v`), etc.
*   **`epg_mqc_vectorized.py`**: EPG simulation with Multiple Quantum Coherences, batched.
    *   `forward(flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0)`
    *   Key parameter: `max_mqc_order`.
*   **`epg_mr_mt_gpu_vectorized.py`**: Two-pool Magnetization Transfer simulation, batched and optimized for GPU.
    *   `forward(flip_angles, phases, T1f, T2f, T1b, T2b, kf, kb, TR, TE, B0=0.0, B1=1.0, wf=1.0, wb=0.1)`
*   **`epg_mri_time_varying_gradients.py`**: EPG simulation extended to support time-varying gradients (placeholder for full implementation), batched.
    *   Key parameters: `gradient_waveform`.

Refer to the example usage within each script and the documentation in the `wiki/` for more details on parameters.

*(Note: Plotting utilities like `epg_plotting_tools` might exist or be developed separately.)*

---

## Plotting Utilities

The `epg_plotting_tool.py` script (located at the root of the repository) provides functions to visualize EPG simulation results and pulse sequences:

*   `plot_pulse_sequence(flip_angles, phases, TR)`: Displays the RF pulse sequence.
*   `plot_epg_evolution(epg_states, max_display_order, batch_idx, pool_idx)`: Shows the time evolution of different k-orders for Fp, Fm, and Z states (2D plot).
*   `plot_epg_snapshot(epg_states, time_step_idx, batch_idx, pool_idx, max_k_order)`: Displays a snapshot of Fp(k), Fm(k), and Z(k) magnitudes vs. k-order at a specific time point (2D plot).
*   `plot_epg_F_states_3D(epg_states, batch_idx, pool_idx, component, kind)`: Provides a 3D visualization of F-state magnitudes (Fp, Fm, or both) over time and k-order (e.g., as a surface or wireframe plot).

These utilities are demonstrated in the Jupyter notebooks located in the `examples/` directory.

---

## Examples

The `examples/` directory contains Jupyter notebooks (derived from Python scripts) that demonstrate how to use the EPG simulation scripts and plotting utilities:

*   **`example_basic_epg.ipynb`**: Shows how to run a basic EPG simulation using `vectorized-sims/epg_mri_vectorized.py` and visualize the pulse sequence, 2D EPG state evolution, 2D state snapshots, and 3D F-state plots.
*   **`example_extended_epg.ipynb`**: Demonstrates an EPG simulation with Magnetization Transfer (MT) using `vectorized-sims/epg_extended_vectorized.py`. It includes visualizations for different pools and highlights MT effects.

To run these examples, ensure you have Jupyter Notebook or JupyterLab installed (e.g., `pip install notebook jupyterlab`). The Python scripts (`.py`) in the `examples/` directory can be converted to notebooks (e.g., using `jupytext` or by opening them in VS Code with the Jupyter extension) or run directly, though notebooks offer a more interactive experience for visualization.

---

## Future Ideas

For potential enhancements, new features, and future directions for this project, please see [`ideas.md`](./ideas.md).
