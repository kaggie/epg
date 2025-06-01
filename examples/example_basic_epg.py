# %% [markdown]
# # Basic EPG Simulation and Visualization
#
# This notebook demonstrates a simple Extended Phase Graph (EPG) simulation using `epg_mri_vectorized.py` from the `vectorized-sims` directory and visualizes its states with `epg_plotting_tool.py`.

# %% [markdown]
# ## Setup
#
# First, we import the necessary libraries and add the parent directory to the system path to allow importing our custom EPG simulation and plotting modules.

# %% [code]
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import custom modules
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the simulation class and plotting functions
# Assuming vectorizedsims is in the parent directory relative to the examples folder when run
try:
    from vectorizedsims.epg_mri_vectorized import EPGSimulation
    from epg_plotting_tool import plot_pulse_sequence, plot_epg_evolution, plot_epg_snapshot, plot_epg_F_states_3D
    MODULES_FOUND = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run from the 'examples/' directory or that the main project directory is in PYTHONPATH.")
    MODULES_FOUND = False

# %% [markdown]
# ## Define Simulation Parameters
#
# Here, we set the tissue parameters (T1, T2), sequence parameters (number of pulses, TR, TE, flip angles, phases), and simulation settings.

# %% [code]
if MODULES_FOUND:
    # Tissue parameters
    T1 = 1000.0  # ms
    T2 = 100.0   # ms

    # Sequence parameters
    n_pulses = 20
    TR = 20.0    # ms (short TR to see evolution)
    TE = 2.0     # ms (relevant for signal readout, but EPG tracks states throughout TR)

    # Example: A series of 30-degree pulses
    flip_angles_deg = torch.ones(n_pulses) * 30.0
    # Uncomment below for a spin-echo like train (approximate, simple phases)
    # flip_angles_deg = torch.ones(n_pulses) * 90.0
    # flip_angles_deg[1::2] = 180.0 # Make every second pulse 180 degrees

    flip_angles_rad = torch.deg2rad(flip_angles_deg)
    phases_rad = torch.zeros(n_pulses) # Simple zero phase for all pulses

    # Simulation settings
    n_states = 15 # Number of EPG states to track
    device = 'cpu'

# %% [markdown]
# ## Run the Simulation
#
# We instantiate the `EPGSimulation` class and run the forward pass with the defined parameters.
# The `epg_mri_vectorized` module expects inputs to be PyTorch tensors. For simplicity, this example uses a single set of tissue parameters (batch size of 1).

# %% [code]
if MODULES_FOUND:
    epg_sim = EPGSimulation(n_states=n_states, device=device)

    # Wrap parameters in tensors for the vectorized simulation
    T1_tensor = torch.tensor([T1], device=device)
    T2_tensor = torch.tensor([T2], device=device)
    # B0 and B1 can be defaulted in the simulation if not critical for this example
    B0_tensor = torch.tensor([0.0], device=device)
    B1_tensor = torch.tensor([1.0], device=device)

    epg_states = epg_sim(flip_angles_rad, phases_rad, T1_tensor, T2_tensor, TR, TE, B0=B0_tensor, B1=B1_tensor)
    print(f"Simulation complete. Number of time steps (states stored): {len(epg_states)}")
    if len(epg_states) > 0:
        print(f"Shape of Fp tensor at first time step: {epg_states[0][0].shape}")

# %% [markdown]
# ## Visualize Pulse Sequence
#
# Let's plot the RF pulse sequence that was applied.

# %% [code]
if MODULES_FOUND and len(epg_states) > 0:
    plot_pulse_sequence(flip_angles_rad, phases_rad, TR=TR)
    plt.show()
else:
    print("Skipping pulse sequence plot as modules were not found or simulation failed.")

# %% [markdown]
# ## Visualize EPG State Evolution (2D)
#
# The `plot_epg_evolution` function displays the magnitudes of Fp, Fm, and Z states for the first few k-orders over time (pulse number). This helps visualize how different coherence pathways evolve.

# %% [code]
if MODULES_FOUND and len(epg_states) > 0:
    plot_epg_evolution(epg_states, max_display_order=5, batch_idx=0) # batch_idx=0 as we have a single batch
    plt.show()
else:
    print("Skipping 2D EPG evolution plot.")

# %% [markdown]
# ## Visualize EPG State Snapshot (2D)
#
# The `plot_epg_snapshot` function shows the distribution of EPG states (Fp, Fm, Z) across k-orders at a single point in time (a specific pulse/time step).

# %% [code]
if MODULES_FOUND and len(epg_states) > 0:
    snapshot_time_step = n_pulses // 2
    plot_epg_snapshot(epg_states, time_step_idx=snapshot_time_step, batch_idx=0, max_k_order=n_states)
    plt.show()
else:
    print("Skipping 2D EPG snapshot plot.")

# %% [markdown]
# ## Visualize F-State Evolution (3D)
#
# The `plot_epg_F_states_3D` function provides a 3D view of either |Fp(k,t)| or |Fm(k,t)|, showing magnitude as a surface or wireframe plot against k-order and time step. This can give a more holistic view of the transverse coherence evolution.

# %% [code]
if MODULES_FOUND and len(epg_states) > 0:
    plot_epg_F_states_3D(epg_states,
                         batch_idx=0,
                         component='Fp',
                         kind='surface',
                         max_k_order=10, # Reduced for clearer 3D plot
                         max_time_steps=n_pulses,
                         title_suffix=" (Fp states - Surface)")
    plt.show()

    plot_epg_F_states_3D(epg_states,
                         batch_idx=0,
                         component='Fm',
                         kind='wireframe',
                         max_k_order=10, # Reduced for clearer 3D plot
                         max_time_steps=n_pulses,
                         title_suffix=" (Fm states - Wireframe)")
    plt.show()
else:
    print("Skipping 3D F-state evolution plot.")

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrated setting up and running a basic EPG simulation and visualizing the results using the provided plotting tools. You can modify the tissue and sequence parameters in the "Define Simulation Parameters" section to explore different scenarios. For more complex simulations involving multiple pools, diffusion, or other effects, refer to `epg_extended_vectorized.py`.

# %% [code]
# This is the final cell.
if MODULES_FOUND:
    print("Example notebook execution finished.")
else:
    print("Example notebook execution could not fully complete due to import errors.")
