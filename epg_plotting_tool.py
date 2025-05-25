import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_pulse_sequence(flip_angles, phases=None, TR=1.0, ax=None):
    """
    Plot the RF pulse sequence (flip angle and phase vs. time).
    Args:
        flip_angles: (N,) array or tensor, flip angles in radians
        phases: (N,) array or tensor, phases in radians (optional)
        TR: float, time between pulses (ms or arbitrary units)
        ax: matplotlib axis (optional)
    """
    N = len(flip_angles)
    t = np.arange(N) * TR
    flip_deg = np.rad2deg(flip_angles.cpu().numpy()) if isinstance(flip_angles, torch.Tensor) else np.rad2deg(flip_angles)
    if phases is not None:
        phase_deg = np.rad2deg(phases.cpu().numpy()) if isinstance(phases, torch.Tensor) else np.rad2deg(phases)
    else:
        phase_deg = np.zeros(N)
    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 4))
    else:
        fig = None
    ax[0].stem(t, flip_deg, use_line_collection=True)
    ax[0].set_ylabel("Flip angle (deg)")
    ax[0].set_title("RF Pulse Sequence")
    ax[1].stem(t, phase_deg, linefmt='tab:orange', markerfmt='o', basefmt=' ', use_line_collection=True)
    ax[1].set_ylabel("Phase (deg)")
    ax[1].set_xlabel("Time (a.u. or ms)")
    plt.tight_layout()
    if fig is not None:
        plt.show()

def plot_epg_evolution(epg_states, max_display_order=5, mqc=False):
    """
    Plot the evolution of EPG states over time.
    Args:
        epg_states: list of (F, Z) or (Fp, Fm, Z) for each time point
        max_display_order: int, max EPG order (spatial coherence) to display
        mqc: if True, plot MQC orders for F (expects F[k, q])
    """
    N = len(epg_states)
    if mqc:
        # For MQC: plot |F+| for q=+1, |DQ| for q=+2, Z
        Fplus = np.array([np.abs(s[0][:max_display_order, s[0].shape[1]//2+1].cpu().numpy()) for s in epg_states])  # q=+1
        DQ = np.array([np.abs(s[0][:max_display_order, s[0].shape[1]//2+2].cpu().numpy()) for s in epg_states])    # q=+2
        Z = np.array([s[1][:max_display_order].cpu().numpy() for s in epg_states])
        t = np.arange(N)
        plt.figure(figsize=(10, 6))
        for k in range(max_display_order):
            plt.plot(t, Fplus[:, k], label=f'|F+|, k={k}')
            plt.plot(t, DQ[:, k], '--', label=f'|DQ|, k={k}')
            plt.plot(t, Z[:, k], ':', label=f'Z, k={k}')
        plt.xlabel('Pulse number')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.title("EPG Evolution with MQC")
        plt.tight_layout()
        plt.show()
    else:
        # Standard EPG: plot |Fp|, |Fm|, Z for k=0..max_display_order-1
        Fp = np.array([np.abs(s[0][:max_display_order].cpu().numpy()) for s in epg_states])
        Fm = np.array([np.abs(s[1][:max_display_order].cpu().numpy()) for s in epg_states])
        Z = np.array([s[2][:max_display_order].cpu().numpy() for s in epg_states])
        t = np.arange(N)
        plt.figure(figsize=(10, 6))
        for k in range(max_display_order):
            plt.plot(t, Fp[:, k], label=f'|Fp|, k={k}')
            plt.plot(t, Fm[:, k], '--', label=f'|Fm|, k={k}')
            plt.plot(t, Z[:, k], ':', label=f'Z, k={k}')
        plt.xlabel('Pulse number')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.title("EPG State Evolution")
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    import torch
    from epg_mri import EPGSimulation, EPGSimulationMQC

    # Example 1: Plot a simple pulse sequence
    n_pulses = 8
    flip_angles = torch.deg2rad(torch.tensor([90, 180, 90, 180, 90, 180, 90, 180]))
    phases = torch.zeros(n_pulses)
    plot_pulse_sequence(flip_angles, phases, TR=5.0)

    # Example 2: Plot EPG evolution (standard)
    T1, T2 = 1000.0, 100.0
    TR, TE = 500.0, 20.0
    epg = EPGSimulation(n_states=10)
    states = epg(flip_angles, phases, T1, T2, TR, TE)
    plot_epg_evolution(states, max_display_order=4, mqc=False)

    # Example 3: Plot EPG evolution with MQC
    epg_mqc = EPGSimulationMQC(n_states=6, max_mqc_order=2)
    states_mqc = epg_mqc(flip_angles, phases, T1, T2, TR, TE)
    plot_epg_evolution(states_mqc, max_display_order=3, mqc=True)
