import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os
from mpl_toolkits.mplot3d import Axes3D

def plot_pulse_sequence(flip_angles, phases=None, TR=1.0, ax=None, title="RF Pulse Sequence"):
    """
    Plot the RF pulse sequence (flip angle and phase vs. time).
    Args:
        flip_angles (torch.Tensor or np.ndarray): (N,) array or tensor, flip angles in radians.
        phases (torch.Tensor, np.ndarray, optional): (N,) array or tensor, phases in radians. Defaults to zeros.
        TR (float, optional): Time between pulses (ms or arbitrary units). Defaults to 1.0.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on. If None, creates a new figure and axes.
        title (str, optional): Title for the plot.
    """
    if not isinstance(flip_angles, (torch.Tensor, np.ndarray)) or len(flip_angles) == 0:
        print("Warning: flip_angles must be a non-empty PyTorch Tensor or NumPy array.")
        return
    N = len(flip_angles)
    t = np.arange(N) * TR

    fa_np = flip_angles.cpu().numpy() if isinstance(flip_angles, torch.Tensor) else np.asarray(flip_angles)
    flip_deg = np.rad2deg(fa_np)

    if phases is not None:
        ph_np = phases.cpu().numpy() if isinstance(phases, torch.Tensor) else np.asarray(phases)
        phase_deg = np.rad2deg(ph_np)
    else:
        phase_deg = np.zeros(N)

    create_fig = ax is None
    if create_fig:
        fig, ax_arr = plt.subplots(2, 1, sharex=True, figsize=(10, 5)) # Wider figure
    else:
        ax_arr = ax # Expect ax to be an array of two axes if provided
        fig = ax_arr[0].get_figure() if isinstance(ax_arr, (list, np.ndarray)) else ax_arr.get_figure()


    ax_arr[0].stem(t, flip_deg)
    ax_arr[0].set_ylabel("Flip angle (deg)")
    ax_arr[0].set_title(title)
    ax_arr[0].grid(True, linestyle=':', alpha=0.7)

    ax_arr[1].stem(t, phase_deg, linefmt='orangered', markerfmt='o', basefmt=' ') # Changed color
    ax_arr[1].set_ylabel("Phase (deg)")
    ax_arr[1].set_xlabel("Time (a.u. or ms)")
    ax_arr[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    if create_fig: # Only call plt.show() if this function created the figure
        plt.show()

def plot_epg_evolution(epg_states, max_display_order=5, mqc=False, batch_idx=0, pool_idx=0, title_suffix=""):
    """
    Plot the evolution of EPG states over time (magnitudes of Fp, Fm, Z states vs. pulse number).
    Args:
        epg_states (list): List of tuples, where each tuple contains (Fp, Fm, Z) state tensors for a time point.
                           Tensors can be 1D (states), 2D (batch, states), or 3D (batch, pool, states).
        max_display_order (int, optional): Max EPG order (k-state index) to display. Defaults to 5.
        mqc (bool, optional): If True, assumes MQC EPG states (structure might differ, current MQC plotting is basic). Defaults to False.
        batch_idx (int, optional): Batch index to plot if data is batched. Defaults to 0.
        pool_idx (int, optional): Pool index to plot if data has multiple pools (e.g., for MT). Defaults to 0.
        title_suffix (str, optional): Suffix to append to the plot title. Defaults to "".
    """
    if not epg_states:
        print("EPG states list is empty. Cannot plot EPG evolution.")
        return
    N = len(epg_states)

    all_Fp_selected, all_Fm_selected, all_Z_selected = [], [], []
    actual_max_k_for_plot = 0

    for s_idx, s_tuple in enumerate(epg_states):
        if not (isinstance(s_tuple, tuple) and len(s_tuple) >= 3):
            print(f"Warning: State tuple at index {s_idx} is not as expected (Fp, Fm, Z). Skipping.")
            continue
        s_Fp, s_Fm, s_Z = s_tuple[0], s_tuple[1], s_tuple[2]

        # Determine current max_k available for this time step after selection
        current_max_k = 0
        # Handle Fp
        if s_Fp.ndim == 3: s_Fp_sel = s_Fp[batch_idx, pool_idx, :]
        elif s_Fp.ndim == 2: s_Fp_sel = s_Fp[batch_idx, :] if not mqc else s_Fp[:, s_Fp.shape[1]//2+1] # Basic MQC F+
        elif s_Fp.ndim == 1: s_Fp_sel = s_Fp[:]
        else: raise ValueError(f"Unsupported Fp tensor ndim: {s_Fp.ndim} at state {s_idx}")
        current_max_k = max(current_max_k, s_Fp_sel.shape[0])
        all_Fp_selected.append(np.abs(s_Fp_sel.cpu().numpy()))

        if not mqc:
            if s_Fm.ndim == 3: s_Fm_sel = s_Fm[batch_idx, pool_idx, :]
            elif s_Fm.ndim == 2: s_Fm_sel = s_Fm[batch_idx, :]
            elif s_Fm.ndim == 1: s_Fm_sel = s_Fm[:]
            else: raise ValueError(f"Unsupported Fm tensor ndim: {s_Fm.ndim} at state {s_idx}")
            current_max_k = max(current_max_k, s_Fm_sel.shape[0])
            all_Fm_selected.append(np.abs(s_Fm_sel.cpu().numpy()))

        # Handle Z
        if s_Z.ndim == 3: s_Z_sel = s_Z[batch_idx, pool_idx, :]
        elif s_Z.ndim == 2: s_Z_sel = s_Z[batch_idx, :] if not mqc else s_Z[:, s_Z.shape[1]//2] # Basic MQC Z0
        elif s_Z.ndim == 1: s_Z_sel = s_Z[:]
        else: raise ValueError(f"Unsupported Z tensor ndim: {s_Z.ndim} at state {s_idx}")
        current_max_k = max(current_max_k, s_Z_sel.shape[0])
        all_Z_selected.append(s_Z_sel.cpu().numpy())

        if s_idx == 0: actual_max_k_for_plot = current_max_k # Use k-states from first step
        actual_max_k_for_plot = min(actual_max_k_for_plot, current_max_k) # Ensure consistency if k changes

    # Trim states to consistent actual_max_k_for_plot before padding for max_display_order
    all_Fp_selected = [s[:actual_max_k_for_plot] for s in all_Fp_selected]
    if not mqc: all_Fm_selected = [s[:actual_max_k_for_plot] for s in all_Fm_selected]
    all_Z_selected = [s[:actual_max_k_for_plot] for s in all_Z_selected]

    # Pad with NaNs if max_display_order > actual_max_k_for_plot
    k_to_plot_count = min(max_display_order, actual_max_k_for_plot)

    Fp_np = np.array([np.pad(s, (0, k_to_plot_count - len(s)), 'constant', constant_values=np.nan) if len(s) < k_to_plot_count else s[:k_to_plot_count] for s in all_Fp_selected])
    if not mqc:
        Fm_np = np.array([np.pad(s, (0, k_to_plot_count - len(s)), 'constant', constant_values=np.nan) if len(s) < k_to_plot_count else s[:k_to_plot_count] for s in all_Fm_selected])
    Z_np = np.array([np.pad(s, (0, k_to_plot_count - len(s)), 'constant', constant_values=np.nan) if len(s) < k_to_plot_count else s[:k_to_plot_count] for s in all_Z_selected])

    t = np.arange(N)
    plt.figure(figsize=(12, 7)) # Wider figure

    for k in range(k_to_plot_count):
        if mqc: plt.plot(t, Fp_np[:, k], label=f'|Fp_sel| k={k}')
        else:
            plt.plot(t, Fp_np[:, k], label=f'|Fp| k={k}')
            plt.plot(t, Fm_np[:, k], '--', label=f'|Fm| k={k}')
        plt.plot(t, Z_np[:, k], ':', label=f'Z k={k}')

    plt.xlabel('Pulse number')
    plt.ylabel('Magnitude')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Legend outside

    title = "EPG State Evolution"
    if mqc: title += " (MQC)"
    ref_tensor_for_dim_check = epg_states[0][0]
    if ref_tensor_for_dim_check.ndim > 1 and not mqc : # MQC batching might be different
        title += f" (Batch {batch_idx}"
        if ref_tensor_for_dim_check.ndim == 3:
            title += f", Pool {pool_idx}"
        title += ")"
    title += title_suffix
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for external legend
    plt.show()

def plot_epg_snapshot(epg_states, time_step_idx, batch_idx=0, pool_idx=0, max_k_order=None, title_suffix=""):
    """
    Plot a snapshot of EPG states (|Fp(k)|, |Fm(k)|, Z(k) vs. k-order) at a specific time step.
    Args:
        epg_states (list): List of (Fp, Fm, Z) tuples.
        time_step_idx (int): Index of the time step in epg_states to plot.
        batch_idx (int, optional): Batch index if data is batched. Defaults to 0.
        pool_idx (int, optional): Pool index if data is pooled (for 3D tensors). Defaults to 0.
        max_k_order (int, optional): Maximum k-order (index) to display. If None, displays all available states.
        title_suffix (str, optional): Suffix for the plot title. Defaults to "".
    """
    if not epg_states:
        print("EPG states list is empty. Cannot plot snapshot.")
        return
    if not (0 <= time_step_idx < len(epg_states)):
        raise ValueError(f"time_step_idx {time_step_idx} is out of bounds for epg_states (length {len(epg_states)}).")

    s_tuple = epg_states[time_step_idx]
    if not (isinstance(s_tuple, tuple) and len(s_tuple) >= 3):
        print(f"Warning: State tuple at time_step_idx {time_step_idx} is not as expected (Fp, Fm, Z). Cannot plot.")
        return
    Fp_t, Fm_t, Z_t = s_tuple

    # Handle batch/pool slicing
    if Fp_t.ndim == 3: Fp_sel = Fp_t[batch_idx, pool_idx, :]
    elif Fp_t.ndim == 2: Fp_sel = Fp_t[batch_idx, :]
    elif Fp_t.ndim == 1: Fp_sel = Fp_t[:]
    else: raise ValueError(f"Unsupported Fp tensor ndim: {Fp_t.ndim}")

    if Fm_t.ndim == 3: Fm_sel = Fm_t[batch_idx, pool_idx, :]
    elif Fm_t.ndim == 2: Fm_sel = Fm_t[batch_idx, :]
    elif Fm_t.ndim == 1: Fm_sel = Fm_t[:]
    else: raise ValueError(f"Unsupported Fm tensor ndim: {Fm_t.ndim}")

    if Z_t.ndim == 3: Z_sel = Z_t[batch_idx, pool_idx, :]
    elif Z_t.ndim == 2: Z_sel = Z_t[batch_idx, :]
    elif Z_t.ndim == 1: Z_sel = Z_t[:]
    else: raise ValueError(f"Unsupported Z tensor ndim: {Z_t.ndim}")

    Fp_np = torch.abs(Fp_sel).cpu().numpy()
    Fm_np = torch.abs(Fm_sel).cpu().numpy()
    Z_np = Z_sel.cpu().numpy()

    num_k_states_available = Fp_np.shape[-1]
    k_indices = np.arange(num_k_states_available)

    current_max_k = min(max_k_order, num_k_states_available) if max_k_order is not None else num_k_states_available

    Fp_np = Fp_np[:current_max_k]
    Fm_np = Fm_np[:current_max_k]
    Z_np = Z_np[:current_max_k]
    k_indices = k_indices[:current_max_k]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 7)) # Wider figure

    axs[0].stem(k_indices, Fp_np, label='|Fp(k)|')
    axs[0].set_ylabel('Magnitude')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)

    axs[1].stem(k_indices, Fm_np, label='|Fm(k)|', linefmt='orangered', markerfmt='o')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    axs[2].stem(k_indices, Z_np, label='Z(k)', linefmt='forestgreen', markerfmt='s')
    axs[2].set_ylabel('Magnitude')
    axs[2].set_xlabel('k-order index')
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.7)

    title = f"EPG State Snapshot at Time Step {time_step_idx}"
    if Fp_t.ndim > 1: title += f" (Batch {batch_idx}"
    if Fp_t.ndim == 3: title += f", Pool {pool_idx}"
    if Fp_t.ndim > 1: title += ")"
    title += title_suffix

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_epg_F_states_3D(epg_states, batch_idx=0, pool_idx=0, max_k_order=None, max_time_steps=None, component='Fp', kind='surface', title_suffix=""):
    """
    Plot 3D evolution of F+ or F- states (|Fp(k,t)| or |Fm(k,t)|).
    Args:
        epg_states (list): List of (Fp, Fm, Z) tuples.
        batch_idx (int, optional): Index of the batch to plot. Defaults to 0.
        pool_idx (int, optional): Index of the pool to plot (if applicable for 3D data). Defaults to 0.
        max_k_order (int, optional): Maximum k-order (index) to display. If None, displays all available.
        max_time_steps (int, optional): Maximum number of time steps (pulses) to display. If None, displays all.
        component (str, optional): Which component to plot: 'Fp', 'Fm', or 'both'. Defaults to 'Fp'.
        kind (str, optional): Type of 3D plot: 'surface' or 'wireframe'. Defaults to 'surface'.
        title_suffix (str, optional): Optional suffix for the plot title. Defaults to "".
    """
    if not epg_states:
        print("EPG states list is empty. Cannot plot 3D F-states.")
        return

    num_total_time_steps = len(epg_states)
    time_indices_to_plot = np.arange(min(num_total_time_steps, max_time_steps) if max_time_steps is not None else num_total_time_steps)
    if len(time_indices_to_plot) == 0:
        print("No time steps to plot.")
        return

    first_Fp_state_tensor = epg_states[0][0]
    if first_Fp_state_tensor.ndim == 3: num_k_states_available = first_Fp_state_tensor.shape[2]
    elif first_Fp_state_tensor.ndim == 2: num_k_states_available = first_Fp_state_tensor.shape[1]
    else: num_k_states_available = first_Fp_state_tensor.shape[0]

    k_to_plot_count = min(max_k_order, num_k_states_available) if max_k_order is not None else num_k_states_available
    k_orders_to_plot_indices = np.arange(k_to_plot_count)
    if k_to_plot_count == 0:
        print("No k-orders to plot.")
        return

    magnitudes_Fp_list = []
    magnitudes_Fm_list = []

    for t_idx in time_indices_to_plot:
        s_tuple = epg_states[t_idx]
        if not (isinstance(s_tuple, tuple) and len(s_tuple) >=3): continue # Basic validation
        Fp_t, Fm_t, _ = s_tuple

        if Fp_t.ndim == 3: sel_Fp_t = Fp_t[batch_idx, pool_idx, :k_to_plot_count]
        elif Fp_t.ndim == 2: sel_Fp_t = Fp_t[batch_idx, :k_to_plot_count]
        else: sel_Fp_t = Fp_t[:k_to_plot_count]
        magnitudes_Fp_list.append(torch.abs(sel_Fp_t).cpu().numpy())

        if Fm_t.ndim == 3: sel_Fm_t = Fm_t[batch_idx, pool_idx, :k_to_plot_count]
        elif Fm_t.ndim == 2: sel_Fm_t = Fm_t[batch_idx, :k_to_plot_count]
        else: sel_Fm_t = Fm_t[:k_to_plot_count]
        magnitudes_Fm_list.append(torch.abs(sel_Fm_t).cpu().numpy())

    Fp_array = np.array(magnitudes_Fp_list).T
    Fm_array = np.array(magnitudes_Fm_list).T

    if Fp_array.size == 0 and Fm_array.size == 0 :
        print("No data available for selected component(s) to plot.")
        return

    fig = plt.figure(figsize=(12, 8)) # Slightly larger figure
    ax = fig.add_subplot(111, projection='3d')

    T, K = np.meshgrid(time_indices_to_plot, k_orders_to_plot_indices)

    plot_title = f"3D EPG {component if component != 'both' else 'Fp and Fm'} Magnitude Evolution"
    ref_tensor_for_dim_check = epg_states[0][0]
    if ref_tensor_for_dim_check.ndim > 1: plot_title += f" (Batch {batch_idx}"
    if ref_tensor_for_dim_check.ndim == 3: plot_title += f", Pool {pool_idx}"
    if ref_tensor_for_dim_check.ndim > 1: plot_title += ")"
    plot_title += title_suffix

    alpha_val = 0.75 # Default alpha
    r_stride, c_stride = 1, max(1, len(time_indices_to_plot)//20)


    if component == 'Fp' or component == 'both':
        if Fp_array.size > 0:
            if kind == 'surface':
                ax.plot_surface(K, T, Fp_array, cmap='viridis', alpha=alpha_val, rstride=r_stride, cstride=c_stride, label='|Fp|')
            elif kind == 'wireframe':
                ax.plot_wireframe(K, T, Fp_array, color='deepskyblue', rstride=r_stride, cstride=c_stride, label='|Fp|')
            # TODO: Add scatter, bar3d for Fp
        else: print("No Fp data to plot.")


    if component == 'Fm' or component == 'both':
        if Fm_array.size > 0:
            offset_factor = 0.01 * ax.get_zlim()[1] if component == 'both' and kind=='surface' else 0 # Slight offset for 'both' surface
            if kind == 'surface':
                ax.plot_surface(K, T, Fm_array + offset_factor, cmap='magma', alpha=alpha_val, rstride=r_stride, cstride=c_stride, label='|Fm|')
            elif kind == 'wireframe':
                ax.plot_wireframe(K, T, Fm_array, color='orangered', rstride=r_stride, cstride=c_stride, label='|Fm|')
            # TODO: Add scatter, bar3d for Fm
        else: print("No Fm data to plot.")

    ax.set_xlabel('k-order index')
    ax.set_ylabel('Time Step (Pulse No.)')
    ax.set_zlabel('Magnitude')
    ax.set_title(plot_title)

    # Creating proxy artists for legend if 'both' components are plotted, as surface plots don't directly support legend entries well.
    if component == 'both':
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=plt.cm.viridis(0.5), label='|Fp|'),
                           Patch(facecolor=plt.cm.magma(0.5), label='|Fm|')]
        ax.legend(handles=legend_elements, loc='upper left')

    ax.view_init(elev=25., azim=-120) # Adjusted view
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust path to access modules from the parent directory of 'epg-simulation'
    # This assumes 'epg-simulation' is the root, and examples are in 'epg-simulation/examples/'
    # So, to import from 'vectorizedsims' or 'non_parallelized_sims' located at root:
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    NON_VEC_MODULE_FOUND = False
    VECTORIZED_MODULE_FOUND = False
    MQC_MODULE_FOUND = False

    try:
        from nonparallelizedsims.epg_mri import EPGSimulation as EPGSimulationNonVectorized
        from nonparallelizedsims.epg_mri import EPGSimulationMQC
        NON_VEC_MODULE_FOUND = True
        MQC_MODULE_FOUND = True # Assuming MQC is part of non-vectorized for now
    except ImportError as e:
        print(f"Could not import non-vectorized modules (epg_mri): {e}")

    try:
        from vectorizedsims.epg_mri_vectorized import EPGSimulation as EPGSimulationVec
        VECTORIZED_MODULE_FOUND = True
    except ImportError as e:
        print(f"Could not import vectorized EPGSimulation (epg_mri_vectorized): {e}")

    # Example 1: Plot a simple pulse sequence
    n_pulses_ex1 = 8
    flip_angles_ex1 = torch.deg2rad(torch.tensor([90, 180, 90, 180, 90, 180, 90, 180], dtype=torch.float))
    phases_ex1 = torch.zeros(n_pulses_ex1)
    # plot_pulse_sequence(flip_angles_ex1, phases_ex1, TR=5.0)

    if NON_VEC_MODULE_FOUND:
        T1_ex2, T2_ex2 = 1000.0, 100.0
        TR_ex2, TE_ex2 = 500.0, 20.0
        epg_non_vec = EPGSimulationNonVectorized(n_states=10)
        states_non_vec = epg_non_vec(flip_angles_ex1, phases_ex1, float(T1_ex2), float(T2_ex2), TR_ex2, TE_ex2)
        # plot_epg_evolution(states_non_vec, max_display_order=4, title_suffix=" (Non-Vectorized)")

    vectorized_states_ex3 = None
    if VECTORIZED_MODULE_FOUND:
        n_pulses_ex3 = 30
        flip_angles_ex3 = torch.deg2rad(torch.ones(n_pulses_ex3, dtype=torch.float) * 35)
        phases_ex3 = torch.linspace(0, 2*np.pi, n_pulses_ex3) # Varying phases

        T1_vec_ex3 = torch.tensor([1000.0])
        T2_vec_ex3 = torch.tensor([70.0])
        TR_ex3 = 50.0
        TE_ex3 = 5.0
        B0_vec_ex3 = torch.tensor([10.0]) # Add some off-resonance
        B1_vec_ex3 = torch.tensor([1.0])

        epg_vec_ex3 = EPGSimulationVec(n_states=20, device='cpu')
        vectorized_states_ex3 = epg_vec_ex3(flip_angles_ex3, phases_ex3,
                                          T1_vec_ex3, T2_vec_ex3, TR_ex3, TE_ex3,
                                          B0=B0_vec_ex3, B1=B1_vec_ex3)

        if len(vectorized_states_ex3) > 0:
            print(f"Shape of Fp from vectorized_states_ex3[0]: {vectorized_states_ex3[0][0].shape}")
            plot_epg_evolution(vectorized_states_ex3, max_display_order=6, batch_idx=0, title_suffix=" (Vectorized Example)")
            snapshot_idx = n_pulses_ex3 // 3
            plot_epg_snapshot(vectorized_states_ex3, time_step_idx=snapshot_idx, batch_idx=0, max_k_order=15, title_suffix=" (Vectorized Example)")

            print("Plotting 3D EPG Fp states (Surface)...")
            plot_epg_F_states_3D(vectorized_states_ex3, batch_idx=0, component='Fp', kind='surface', max_k_order=10, max_time_steps=n_pulses_ex3)

            print("Plotting 3D EPG Fm states (Wireframe)...")
            plot_epg_F_states_3D(vectorized_states_ex3, batch_idx=0, component='Fm', kind='wireframe', max_k_order=10, max_time_steps=n_pulses_ex3, title_suffix=" (Fm)")

            print("Plotting 3D EPG Fp and Fm states (Surface)...")
            plot_epg_F_states_3D(vectorized_states_ex3, batch_idx=0, component='both', kind='surface', max_k_order=8, max_time_steps=n_pulses_ex3)
    else:
        print("Skipping vectorized examples as epg_mri_vectorized module not found.")

    if MQC_MODULE_FOUND and NON_VEC_MODULE_FOUND:
        epg_mqc_ex = EPGSimulationMQC(n_states=6, max_mqc_order=2)
        states_mqc_ex = epg_mqc_ex(flip_angles_ex1, phases_ex1, float(T1_ex2), float(T2_ex2), TR_ex2, TE_ex2)
        # plot_epg_evolution(states_mqc_ex, max_display_order=3, mqc=True, title_suffix=" (MQC Example)")

    print("Plotting examples finished. If plots are not showing, ensure you are not in a headless environment or try running script directly.")
