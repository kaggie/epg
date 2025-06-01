import torch
import torch.nn as nn
import math

SCRIPT_VERSION_INFO = "epg_mri_slice_profile_v1"

class EPGSimulationSliceProfile(nn.Module):
    """
    EPG simulation incorporating slice profile effects.
    Each point in the slice profile is treated as a batch item.
    Assumes uniform T1, T2, B0, B1 across the slice for this version.
    Order of operations in TR: RF -> Relax -> B0 -> Shift.
    """

    def __init__(self, n_states=21, device='cpu'):
        super().__init__()
        self.n_states = n_states
        self.device = device

    def forward(self, nominal_flip_angles_rad_seq, phases_rad_seq,
                slice_profile_factors, # 1D tensor, e.g., (num_sub_slices,)
                T1_ms, T2_ms, TR_ms, TE_ms,
                B0_hz=0.0, global_B1_scale=1.0):
        """
        Simulate EPG evolution with slice profile effects.

        Args:
            nominal_flip_angles_rad_seq (torch.Tensor): 1D tensor of nominal flip angles for the sequence (num_pulses,).
            phases_rad_seq (torch.Tensor): 1D tensor of RF phases for the sequence (num_pulses,).
            slice_profile_factors (torch.Tensor): 1D tensor of scaling factors for flip angles,
                                                 representing the slice profile (num_sub_slices,).
                                                 If None, assumes ideal profile (factor of 1.0).
            T1_ms, T2_ms (float): Uniform T1, T2 for the slice.
            TR_ms, TE_ms (float): Repetition and echo times.
            B0_hz (float, optional): Uniform B0 offset for the slice.
            global_B1_scale (float, optional): Uniform B1 scaling factor for the slice.

        Returns:
            list: List of (Fp, Fm, Z) state tuples, each tensor of shape (num_sub_slices, n_states).
        """
        num_pulses = nominal_flip_angles_rad_seq.shape[0]

        if slice_profile_factors is None:
            slice_profile_factors = torch.tensor([1.0], device=self.device)
        slice_profile_factors = slice_profile_factors.to(self.device, dtype=torch.float)

        num_sub_slices = slice_profile_factors.shape[0] # This is our "batch_size"

        T1 = torch.full((num_sub_slices, 1), T1_ms, dtype=torch.float, device=self.device)
        T2 = torch.full((num_sub_slices, 1), T2_ms, dtype=torch.float, device=self.device)
        B0_val = torch.full((num_sub_slices, 1), B0_hz, dtype=torch.float, device=self.device)

        effective_flip_angles = slice_profile_factors.unsqueeze(1) * \
                                nominal_flip_angles_rad_seq.unsqueeze(0).to(self.device) * \
                                global_B1_scale

        Fp = torch.zeros(num_sub_slices, self.n_states, dtype=torch.cfloat, device=self.device)
        Fm = torch.zeros(num_sub_slices, self.n_states, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(num_sub_slices, self.n_states, dtype=torch.float, device=self.device)
        Z[..., 0] = 1.0

        epg_states_over_time = []

        E1 = torch.exp(-TR_ms / T1)
        E2 = torch.exp(-TR_ms / T2)
        phi_b0_val = (2 * math.pi * B0_val * TR_ms / 1000.0)

        phases_rad_seq = phases_rad_seq.to(self.device)

        for i_pulse in range(num_pulses):
            current_effective_alphas = effective_flip_angles[:, i_pulse].unsqueeze(-1)
            current_phase = phases_rad_seq[i_pulse]

            Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, current_effective_alphas, current_phase)
            Fp, Fm, Z = self.relax(Fp, Fm, Z, E1, E2)
            Fp, Fm = self.apply_b0(Fp, Fm, phi_b0_val)
            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z)

            epg_states_over_time.append((Fp.clone().detach(), Fm.clone().detach(), Z.clone().detach()))

        return epg_states_over_time

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

    def get_averaged_signal(self, states_list):
        if not states_list:
            return torch.empty(0, device=self.device if hasattr(self, 'device') else 'cpu')

        signal_state_idx = 1 if self.n_states > 1 else 0
        if self.n_states == 1 and states_list:
             print("Warning: n_states=1, averaged signal taken from Fp[:,0] which is zero post-shift.")

        avg_signals = []
        for Fp, Fm, Z in states_list:
            if Fp.numel() == 0:
                avg_signals.append(torch.tensor(0.0, device=Fp.device))
                continue

            current_n_states_in_fp = Fp.shape[-1]
            actual_signal_idx = signal_state_idx
            if current_n_states_in_fp == 1:
                actual_signal_idx = 0
            elif signal_state_idx >= current_n_states_in_fp:
                print(f"Warning: signal_state_idx {signal_state_idx} out of bounds for Fp with {current_n_states_in_fp} states. Using index 0.")
                actual_signal_idx = 0

            signal_per_sub_slice = torch.abs(Fp[..., actual_signal_idx])
            avg_signal_for_tr = torch.mean(signal_per_sub_slice)
            avg_signals.append(avg_signal_for_tr)

        if not avg_signals:
             return torch.empty(0, device=self.device if hasattr(self, 'device') else 'cpu')
        return torch.stack(avg_signals)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"EPG Slice Profile Simulation. Device: {device}")
    print(f"SCRIPT VERSION: {SCRIPT_VERSION_INFO}")

    n_s = 11
    n_pulses_ex = 50

    nominal_flips_deg_ex = torch.ones(n_pulses_ex) * 30
    # Corrected variable name here:
    nominal_flips_rad_ex = torch.deg2rad(nominal_flips_deg_ex).to(device)
    phases_rad_ex = torch.zeros(n_pulses_ex, device=device)

    profile_factors_ex = torch.tensor([1.0], device=device)

    T1_ex = 1000.0; T2_ex = 80.0; TR_ex = 10.0; TE_ex = TR_ex / 2.0
    B0_ex = 0.0; B1_global_ex = 1.0

    print(f"Simulating with {len(profile_factors_ex)} sub-slices, {n_pulses_ex} pulses.")
    epg_sp_model = EPGSimulationSliceProfile(n_states=n_s, device=device)

    # Corrected variable name in the call:
    all_states_over_time = epg_sp_model(
        nominal_flips_rad_ex, phases_rad_ex, profile_factors_ex,
        T1_ex, T2_ex, TR_ex, TE_ex, B0_ex, B1_global_ex
    )
    print(f"Number of TRs simulated: {len(all_states_over_time)}")
    if all_states_over_time:
        Fp_last, Fm_last, Z_last = all_states_over_time[-1]
        print(f"Shape of Fp at last TR (num_sub_slices, n_states): {Fp_last.shape}")

    averaged_signal_series = epg_sp_model.get_averaged_signal(all_states_over_time)
    print(f"\nAveraged signal series shape: {averaged_signal_series.shape}")
    print(f"Averaged signal values (first 10): {averaged_signal_series[:10]}")

    num_sub_slices_ex = 7
    profile_factors_ex_tri = torch.cat((torch.linspace(0.2,1,4)[:-1],torch.linspace(1,0.2,4))).to(device)
    print(f"\nSimulating with profile: {profile_factors_ex_tri}")
    # Corrected variable name in the call:
    all_states_tri = epg_sp_model(
        nominal_flips_rad_ex, phases_rad_ex, profile_factors_ex_tri,
        T1_ex, T2_ex, TR_ex, TE_ex, B0_ex, B1_global_ex)
    avg_signal_tri = epg_sp_model.get_averaged_signal(all_states_tri)
    print(f"Averaged signal for triangular profile (first 10): {avg_signal_tri[:10]}")

    print("\nScript execution finished.")
