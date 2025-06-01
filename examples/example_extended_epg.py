# %% [markdown]
# # Extended EPG Simulation: Magnetization Transfer (MT)
#
# This notebook demonstrates an Extended Phase Graph (EPG) simulation featuring Magnetization Transfer (MT) using `epg_extended_vectorized.py` from the `vectorizedsims` directory. We will visualize the states of different pools using `epg_plotting_tool.py`.

# %% [markdown]
# ## Setup
#
# Import necessary libraries and configure paths for custom module imports.

# %% [code]
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import custom modules
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from vectorizedsims.epg_extended_vectorized import EPGSimulation as EPGSimulationExtended
    from epg_plotting_tool import plot_pulse_sequence, plot_epg_evolution, plot_epg_snapshot, plot_epg_F_states_3D
    MODULES_FOUND = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run from the 'examples/' directory or that the main project directory is in PYTHONPATH.")
    MODULES_FOUND = False

# %% [markdown]
# ## Define Simulation Parameters for Magnetization Transfer (MT)
#
# We will set up parameters for a two-pool MT model: a free water pool and a macromolecular (bound) pool. We'll use a series of on-resonance, low flip angle pulses to observe the MT effect.

# %% [code]
if MODULES_FOUND:
    # Device configuration
    device = 'cpu'

    # --- Two-Pool Magnetization Transfer Parameters ---
    # Pool A (Free water)
    T1f = 1200.0  # ms
    T2f = 80.0    # ms
    # Pool B (Macromolecular protons / Bound pool)
    T1b = 250.0   # ms
    T2b = 10.0    # ms (very short T2 for bound pool)

    # Exchange rates (Hz)
    # k_fb: rate from free pool to bound pool
    # k_bf: rate from bound pool to free pool
    # Using values similar to examples in epg_extended_vectorized.py for consistency
    k_exch_param = torch.tensor([[20.0, 60.0]], device=device) # [k_free->bound, k_bound->free] in Hz

    # pool_params for EPGSimulationExtended: (batch_size, n_pools, 2 [T1,T2])
    # Batch size of 1 for this example.
    pool_params_mt = torch.tensor([[[T1f, T2f], [T1b, T2b]]], device=device)

    # --- Sequence Parameters ---
    n_pulses = 30
    TR = 50.0    # ms
    TE = 2.0     # ms (typically time to echo, EPG tracks states throughout TR)

    # RF pulses: series of low flip angle pulses to drive the system towards steady-state
    flip_angles_deg = torch.ones(n_pulses) * 15.0 # 15-degree pulses
    flip_angles_rad = torch.deg2rad(flip_angles_deg)
    phases_rad = torch.zeros(n_pulses) # Zero phase for all pulses

    # B0 offset and B1 scale factor
    B0_offset = torch.tensor([0.0], device=device) # Hz, on-resonance
    B1_scale = torch.tensor([1.0], device=device)  # No B1 inhomogeneity

    # --- Simulation Settings for EPGSimulationExtended ---
    n_states = 10 # Number of EPG coherence states (k-orders)
    n_pools = 2   # Number of magnetization pools (for MT)

# %% [markdown]
# ## Run the Extended EPG Simulation (MT)
#
# Instantiate `EPGSimulationExtended` with `n_pools=2` and `mt=True`, then run the simulation.
# The main T1/T2 arguments to the `forward` method usually correspond to the primary pool (free water).

# %% [code]
if MODULES_FOUND:
    epg_sim_mt = EPGSimulationExtended(
        n_states=n_states,
        n_pools=n_pools,
        device=device,
        mt=True # Enable Magnetization Transfer calculations
    )

    # T1/T2 for the primary (free water) pool for the simulation call
    # The model uses these as defaults if pool_params is not comprehensive
    # or for aspects not covered by pool_params specific logic.
    T1_water_for_sim = torch.tensor([T1f], device=device)
    T2_water_for_sim = torch.tensor([T2f], device=device)

    print("Running MT simulation...")
    epg_states_mt = epg_sim_mt(
        flip_angles_rad,
        phases_rad,
        T1_water_for_sim, # T1 for free water pool
        T2_water_for_sim, # T2 for free water pool
        TR,
        TE,
        B0=B0_offset,
        B1=B1_scale,
        pool_params=pool_params_mt,    # T1/T2 for all pools
        k_exch_rates=k_exch_param  # Exchange rates between pools
    )
    print(f"MT Simulation complete. Number of time steps: {len(epg_states_mt)}")
    if len(epg_states_mt) > 0:
        Fp0, Fm0, Z0 = epg_states_mt[0]
        print(f"Shape of Fp tensor from first time step (Batch, Pool, States): {Fp0.shape}")


# %% [markdown]
# ## Visualize Results (MT)
#
# We'll plot the pulse sequence and then the EPG state evolution and snapshots for each pool.

# %% [markdown]
# ### RF Pulse Sequence

# %% [code]
if MODULES_FOUND and 'epg_states_mt' in locals() and len(epg_states_mt) > 0:
    plot_pulse_sequence(flip_angles_rad, phases_rad, TR=TR)
    plt.show()
else:
    print("Skipping pulse sequence plot.")

# %% [markdown]
# ### EPG State Evolution - Pool 0 (Free Water)
#
# This plot shows the evolution of |Fp|, |Fm|, and Z states for the free water pool.

# %% [code]
if MODULES_FOUND and 'epg_states_mt' in locals() and len(epg_states_mt) > 0:
    plot_epg_evolution(epg_states_mt, max_display_order=3, batch_idx=0, pool_idx=0, title_suffix=" (Pool 0: Free Water)")
    plt.show()
else:
    print("Skipping EPG evolution plot for Pool 0.")

# %% [markdown]
# ### EPG State Evolution - Pool 1 (Bound Pool)
#
# This plot shows the evolution of |Fp|, |Fm|, and Z states for the bound macromolecular pool. Notice the much faster T2 decay reflected in Fp/Fm states if they are generated.

# %% [code]
if MODULES_FOUND and 'epg_states_mt' in locals() and len(epg_states_mt) > 0:
    plot_epg_evolution(epg_states_mt, max_display_order=3, batch_idx=0, pool_idx=1, title_suffix=" (Pool 1: Bound Pool)")
    plt.show()
else:
    print("Skipping EPG evolution plot for Pool 1.")

# %% [markdown]
# ### EPG State Snapshot at mid-sequence - Pool 0 (Free Water)

# %% [code]
if MODULES_FOUND and 'epg_states_mt' in locals() and len(epg_states_mt) > 0:
    snapshot_idx = n_pulses // 2
    plot_epg_snapshot(epg_states_mt, time_step_idx=snapshot_idx, batch_idx=0, pool_idx=0, max_k_order=n_states, title_suffix=" (Pool 0: Free Water)")
    plt.show()
else:
    print("Skipping EPG snapshot for Pool 0.")

# %% [markdown]
# ### EPG State Snapshot at mid-sequence - Pool 1 (Bound Pool)

# %% [code]
if MODULES_FOUND and 'epg_states_mt' in locals() and len(epg_states_mt) > 0:
    snapshot_idx = n_pulses // 2
    plot_epg_snapshot(epg_states_mt, time_step_idx=snapshot_idx, batch_idx=0, pool_idx=1, max_k_order=n_states, title_suffix=" (Pool 1: Bound Pool)")
    plt.show()
else:
    print("Skipping EPG snapshot for Pool 1.")

# %% [markdown]
# ### 3D Fp State Evolution - Pool 0 (Free Water)
#
# This provides a 3D visualization of the |Fp(k,t)| states for the free water pool.

# %% [code]
if MODULES_FOUND and 'epg_states_mt' in locals() and len(epg_states_mt) > 0:
    plot_epg_F_states_3D(epg_states_mt,
                         batch_idx=0,
                         pool_idx=0,
                         component='Fp',
                         kind='surface',
                         max_k_order=5,
                         max_time_steps=n_pulses,
                         title_suffix=" (Pool 0: Free Water)")
    plt.show()
else:
    print("Skipping 3D Fp state evolution plot for Pool 0.")

# %% [markdown]
# ### 3D Fm State Evolution - Pool 1 (Bound Pool)
#
# This provides a 3D visualization of the |Fm(k,t)| states for the bound pool. Due to very short T2, these states are expected to be small.

# %% [code]
if MODULES_FOUND and 'epg_states_mt' in locals() and len(epg_states_mt) > 0:
    plot_epg_F_states_3D(epg_states_mt,
                         batch_idx=0,
                         pool_idx=1, # Selecting Pool 1
                         component='Fm',
                         kind='wireframe',
                         max_k_order=5,
                         max_time_steps=n_pulses,
                         title_suffix=" (Pool 1: Bound Pool - Fm states)")
    plt.show()
else:
    print("Skipping 3D Fm state evolution plot for Pool 1.")

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrated an EPG simulation with Magnetization Transfer, showing how to set up a two-pool model and visualize the distinct behavior of each pool. The `epg_extended_vectorized.py` script can also model other effects like diffusion and flow, which can be explored by enabling the corresponding flags and providing relevant parameters.

# %% [code]
if MODULES_FOUND:
    print("Extended EPG example notebook execution finished.")
else:
    print("Extended EPG example notebook execution could not fully complete due to import errors.")
