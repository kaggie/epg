import torch
import torch.nn as nn
import math

SCRIPT_VERSION_INFO = "epg_mri_v2_RF_first" # Version string

class EPGSimulation(nn.Module):
    """
    Extended Phase Graph (EPG) simulation for MRI.
    Simulates evolution of magnetization under T1, T2, B0, and B1 variations.
    This version supports vectorized inputs for T1, T2, B0, and B1.
    Order of operations in TR: RF -> Relax -> B0 -> Shift.
    """

    def __init__(self, n_states=21, device='cpu'):
        super().__init__()
        self.n_states = n_states
        self.device = device

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0):
        N_pulses = len(flip_angles)
        batch_size = 1
        params_to_check = [T1, T2, B0, B1]
        for p in params_to_check:
            if isinstance(p, torch.Tensor) and p.ndim > 0 and p.shape[0] > 1:
                if batch_size == 1: batch_size = p.shape[0]
                elif p.shape[0] != batch_size: raise ValueError("Inconsistent batch sizes.")

        T1 = torch.as_tensor(T1, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)
        T2 = torch.as_tensor(T2, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)
        B0_val = torch.as_tensor(B0, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)
        B1_val = torch.as_tensor(B1, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)

        Fp = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Fm = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(batch_size, self.n_states, dtype=torch.float, device=self.device)
        Z[..., 0] = 1.0

        epg_states = []
        E1 = torch.exp(-TR / T1)
        E2 = torch.exp(-TR / T2)
        phi_b0_val = (2 * math.pi * B0_val * TR / 1000.0)

        for i_pulse in range(N_pulses):
            # New Order: RF -> Relax -> B0 -> Shift
            current_flip_angle = flip_angles[i_pulse]
            current_phase = phases[i_pulse]
            alpha = current_flip_angle * B1_val
            beta = current_phase
            Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, alpha, beta)

            Fp, Fm, Z = self.relax(Fp, Fm, Z, E1, E2)
            Fp, Fm = self.apply_b0(Fp, Fm, phi_b0_val)

            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z)
            epg_states.append((Fp.clone().detach(), Fm.clone().detach(), Z.clone().detach()))
        return epg_states

    def relax(self, Fp, Fm, Z, E1, E2):
        Fp = Fp * E2
        Fm = Fm * E2
        Z_relaxed = Z * E1
        Z_relaxed[..., 0] = Z[..., 0] * E1[...,0] + (1 - E1[...,0])
        return Fp, Fm, Z_relaxed

    def apply_b0(self, Fp, Fm, phi):
        phase_factor = torch.exp(1j * phi)
        Fp = Fp * phase_factor
        Fm = Fm * torch.conj(phase_factor)
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        cos_a2 = torch.cos(alpha / 2)
        sin_a2 = torch.sin(alpha / 2)
        exp_ib = torch.exp(1j * beta)
        exp_mib = torch.exp(-1j * beta)
        Z_complex = Z.to(torch.cfloat)
        Fp_new = cos_a2**2 * Fp + sin_a2**2 * torch.conj(Fm) * exp_ib**2 + 1j * cos_a2 * sin_a2 * (Z_complex * exp_ib)
        Fm_new = sin_a2**2 * torch.conj(Fp) * exp_mib**2 + cos_a2**2 * Fm - 1j * cos_a2 * sin_a2 * (Z_complex * exp_mib)
        Z_new_complex = -1j * sin_a2 * cos_a2 * (Fp * exp_mib - Fm * exp_ib) + (cos_a2**2 - sin_a2**2) * Z_complex
        Z_new = Z_new_complex.real
        return Fp_new, Fm_new, Z_new

    def epg_shift(self, Fp, Fm, Z):
        Fp_shifted = torch.roll(Fp, shifts=1, dims=1)
        Fm_shifted = torch.roll(Fm, shifts=-1, dims=1)
        Z_shifted = torch.roll(Z, shifts=1, dims=1)
        Fp_shifted[..., 0] = 0
        Fm_shifted[..., -1] = 0
        Z_shifted[..., 0] = 0
        return Fp_shifted, Fm_shifted, Z_shifted

if __name__ == "__main__":
    # Basic example remains for direct script execution
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running EPG MRI Vectorized on device: {device}")
    print(f"SCRIPT VERSION: {SCRIPT_VERSION_INFO}")
    n_states_ex = 5
    n_pulses_ex = 3
    flip_angles_ex = torch.ones(n_pulses_ex, device=device) * torch.deg2rad(torch.tensor(90.0, device=device))
    phases_ex = torch.zeros(n_pulses_ex, device=device)
    TR_ex, TE_ex = 100.0, 10.0
    print("\\n--- Scalar Test ---")
    T1_s, T2_s = 1000.0, 80.0
    B0_s, B1_s = 0.0, 1.0
    epg_scalar = EPGSimulation(n_states=n_states_ex, device=device)
    states_scalar = epg_scalar(flip_angles_ex, phases_ex, T1_s, T2_s, TR_ex, TE_ex, B0=B0_s, B1=B1_s)
    print("Scalar run output (state k=0, state k=1 for batch 0):")
    for i, (Fp_s, Fm_s, Z_s) in enumerate(states_scalar):
        print(f"Pulse {i+1}: Fp0={Fp_s[0,0].item():.3f}, Fp1={Fp_s[0,1].item():.3f}, Z0={Z_s[0,0]:.3f}, Z1={Z_s[0,1]:.3f}")
    print("\\n--- Vectorized Test ---")
    batch_size_ex = 2
    T1_v = torch.tensor([1000.0, 800.0], device=device)
    T2_v = torch.tensor([80.0, 60.0], device=device)
    B0_v = torch.tensor([0.0, 10.0], device=device)
    B1_v = torch.tensor([1.0, 0.9], device=device)
    epg_vectorized = EPGSimulation(n_states=n_states_ex, device=device)
    states_vectorized = epg_vectorized(flip_angles_ex, phases_ex, T1_v, T2_v, TR_ex, TE_ex, B0=B0_v, B1=B1_v)
    print(f"Vectorized run output (Batch Size {batch_size_ex}, state k=0, k=1):")
    for i_pulse, (Fp_v, Fm_v, Z_v) in enumerate(states_vectorized):
        print(f"Pulse {i_pulse+1}:")
        for b_idx in range(batch_size_ex):
            print(f"  B{b_idx}: Fp0={Fp_v[b_idx,0].item():.3f}, Fp1={Fp_v[b_idx,1].item():.3f}, Z0={Z_v[b_idx,0]:.3f}, Z1={Z_v[b_idx,1]:.3f}")
