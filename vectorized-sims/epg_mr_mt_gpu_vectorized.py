import torch
import torch.nn as nn

class EPGSimulationMT_GPU_Vectorized(nn.Module):
    """
    Vectorized Extended Phase Graph (EPG) simulation for Magnetization Transfer (MT) with GPU support.
    Two-pool model: free (water) and bound (macromolecular) pools with exchange.
    Simulates T1, T2, B0, B1, and exchange between pools.
    Supports batched inputs for vectorized computation on the specified device.
    """

    def __init__(self, n_states=21, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_states = n_states
        self.device = device
        # Simple print to confirm device, can be removed later
        # print(f"EPGSimulationMT_GPU_Vectorized initialized on device: {self.device}")

    def _to_batched_tensor(self, value, batch_size):
        """Helper to convert value to a tensor of shape (batch_size, 1) on self.device."""
        if not isinstance(value, torch.Tensor):
            # Python scalar or list/tuple of scalars for each batch item
            if isinstance(value, (list, tuple)): # e.g. [v1, v2] for batch_size=2
                if len(value) != batch_size:
                    raise ValueError(f"List/tuple parameter length {len(value)} does not match batch_size {batch_size}")
                value = torch.tensor(value, dtype=torch.float, device=self.device)
            else: # Python scalar, expand to batch_size
                value = torch.tensor(value, dtype=torch.float, device=self.device).expand(batch_size)
        else:
            # Input is already a tensor
            value = value.to(self.device, dtype=torch.float)
            if value.ndim == 0: # Scalar tensor
                value = value.expand(batch_size)
            elif value.shape == (batch_size,): # Already (B,)
                pass
            elif value.numel() == 1: # Tensor with 1 element, e.g. tensor([1000.])
                value = value.squeeze().expand(batch_size)
            elif value.shape[0] != batch_size :
                 # This case could be ambiguous e.g. if a 1D tensor of other size is passed for batch_size > 1
                 raise ValueError(f"Tensor parameter shape {value.shape} not compatible with batch_size {batch_size}")

        return value.view(batch_size, 1) # Ensure final shape is (B,1)

    def forward(
        self,
        flip_angles, # (N_pulses,)
        phases,      # (N_pulses,)
        T1f, T2f,
        T1b, T2b,
        kf, kb,
        TR, TE,     # Scalars
        B0=0.0, B1=1.0,
        wf=1.0, wb=0.1
    ):
        N_pulses = flip_angles.shape[0]
        # These are per-pulse, not batched across simulations, so just move to device
        flip_angles = flip_angles.to(self.device)
        phases = phases.to(self.device)

        # Determine batch_size from the parameters that can be batched
        batch_size = 1
        batched_params_candidates = [T1f, T2f, T1b, T2b, kf, kb, B0, B1, wf, wb]
        for p_val in batched_params_candidates:
            if isinstance(p_val, torch.Tensor):
                if p_val.ndim > 0 and p_val.shape[0] > 1 : # If it's a tensor like (N,) or (N, M)
                    current_bs_candidate = p_val.shape[0]
                    if batch_size == 1: batch_size = current_bs_candidate
                    elif batch_size != current_bs_candidate:
                        raise ValueError(f"Inconsistent batch sizes in input parameters: found {current_bs_candidate} and {batch_size}")
            elif isinstance(p_val, (list,tuple)): # If python list/tuple
                 current_bs_candidate = len(p_val)
                 if current_bs_candidate > 1:
                    if batch_size == 1: batch_size = current_bs_candidate
                    elif batch_size != current_bs_candidate:
                        raise ValueError(f"Inconsistent batch sizes for list parameters: found {current_bs_candidate} and {batch_size}")

        # Convert all parameters to batched tensors of shape (batch_size, 1)
        T1f = self._to_batched_tensor(T1f, batch_size)
        T2f = self._to_batched_tensor(T2f, batch_size)
        T1b = self._to_batched_tensor(T1b, batch_size)
        T2b = self._to_batched_tensor(T2b, batch_size)
        kf = self._to_batched_tensor(kf, batch_size)
        kb = self._to_batched_tensor(kb, batch_size)
        B0_val = self._to_batched_tensor(B0, batch_size)
        B1_val = self._to_batched_tensor(B1, batch_size) # B1 is used to scale flip angle
        wf = self._to_batched_tensor(wf, batch_size)
        wb = self._to_batched_tensor(wb, batch_size)

        # Initialize EPG states: (batch_size, n_states)
        Fp_f = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Fm_f = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Z_f = torch.zeros(batch_size, self.n_states, dtype=torch.float, device=self.device)
        Fp_b = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Fm_b = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Z_b = torch.zeros(batch_size, self.n_states, dtype=torch.float, device=self.device)

        Z_f[..., 0] = wf.squeeze()
        Z_b[..., 0] = wb.squeeze()

        epg_states = []

        E1f = torch.exp(-TR / T1f)
        E2f = torch.exp(-TR / T2f)
        E1b = torch.exp(-TR / T1b)
        E2b = torch.exp(-TR / T2b)
        phi_b0_val = (2 * torch.pi * B0_val * TR / 1000.0)

        for i_pulse in range(N_pulses):
            Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b = self.relax_exchange(
                Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b,
                E1f, E2f, E1b, E2b, kf, kb, TR, wf, wb
            )

            Fp_f, Fm_f = self.apply_b0(Fp_f, Fm_f, phi_b0_val)
            Fp_b, Fm_b = self.apply_b0(Fp_b, Fm_b, phi_b0_val)

            current_flip_angle = flip_angles[i_pulse]
            current_phase = phases[i_pulse]
            alpha_rf = current_flip_angle * B1_val    # (B,1)
            beta_rf = current_phase                   # scalar
            Fp_f, Fm_f, Z_f = self.apply_rf(Fp_f, Fm_f, Z_f, alpha_rf, beta_rf)

            Fp_f, Fm_f, Z_f = self.epg_shift(Fp_f, Fm_f, Z_f)
            Fp_b, Fm_b, Z_b = self.epg_shift(Fp_b, Fm_b, Z_b)

            epg_states.append((
                Fp_f.clone().detach(), Fm_f.clone().detach(), Z_f.clone().detach(),
                Fp_b.clone().detach(), Fm_b.clone().detach(), Z_b.clone().detach()
            ))
        return epg_states

    def relax_exchange(
        self, Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b,
        E1f, E2f, E1b, E2b, kf, kb, TR, wf, wb
    ):
        Fp_f = Fp_f * E2f
        Fm_f = Fm_f * E2f
        Fp_b = Fp_b * E2b
        Fm_b = Fm_b * E2b

        dZf_exchange = (-kf * Z_f + kb * Z_b) * (TR / 1000.0)
        dZb_exchange = (-kb * Z_b + kf * Z_f) * (TR / 1000.0)

        Z_f_t1 = Z_f * E1f
        Z_b_t1 = Z_b * E1b

        # M0 recovery for Z0 state. wf/wb are (B,1), E1f/E1b are (B,1)
        # Z_f_t1[...,0] is (B,). (1-E1f) is (B,1). wf is (B,1)
        Z_f_t1[..., 0] += (1 - E1f.squeeze(-1)) * wf.squeeze(-1)
        Z_b_t1[..., 0] += (1 - E1b.squeeze(-1)) * wb.squeeze(-1)

        Z_f_new = Z_f_t1 + dZf_exchange
        Z_b_new = Z_b_t1 + dZb_exchange

        return Fp_f, Fm_f, Z_f_new, Fp_b, Fm_b, Z_b_new

    def apply_b0(self, Fp, Fm, phi):
        phase_effect = torch.exp(1j * phi)
        Fp = Fp * phase_effect
        Fm = Fm * torch.exp(-1j * phi) # More robust than conj for phi potentially > pi or < -pi
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        cos_a2 = torch.cos(alpha / 2)
        sin_a2 = torch.sin(alpha / 2)
        exp_ib = torch.exp(1j * beta)
        exp_mib = torch.exp(-1j * beta)

        Fp_new = cos_a2**2 * Fp + sin_a2**2 * torch.conj(Fm) * exp_ib**2 + 1j * cos_a2 * sin_a2 * (Z * exp_ib)
        Fm_new = sin_a2**2 * torch.conj(Fp) * exp_mib**2 + cos_a2**2 * Fm - 1j * cos_a2 * sin_a2 * (Z * exp_mib)
        Z_new = (
            -1j * sin_a2 * cos_a2 * (Fp * exp_mib - Fm * exp_ib) +
            (cos_a2**2 - sin_a2**2) * Z
        ).real
        return Fp_new, Fm_new, Z_new

    def epg_shift(self, Fp, Fm, Z):
        Fp_shifted = torch.roll(Fp, shifts=1, dims=1)
        Fm_shifted = torch.roll(Fm, shifts=-1, dims=1)
        Z_shifted = torch.roll(Z, shifts=1, dims=1)

        Fp_shifted[:, 0] = 0
        Fm_shifted[:, -1] = 0
        Z_shifted[:, 0] = 0
        return Fp_shifted, Fm_shifted, Z_shifted

# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running EPG MT GPU Vectorized on device: {device}")

    n_states_ex = 5
    n_pulses_ex = 3

    flip_angles_ex = torch.ones(n_pulses_ex, device=device) * torch.deg2rad(torch.tensor(45.0, device=device))
    phases_ex = torch.zeros(n_pulses_ex, device=device)
    TR_ex, TE_ex = 100.0, 10.0

    print("\n--- Scalar MT Test ---")
    T1f_s, T2f_s = 1200.0, 40.0
    T1b_s, T2b_s = 400.0, 10.0
    kf_s, kb_s = 3.0, 20.0
    wf_s, wb_s = 0.85, 0.15
    B0_s, B1_s = 0.0, 1.0

    epg_scalar = EPGSimulationMT_GPU_Vectorized(n_states=n_states_ex, device=device)
    states_scalar = epg_scalar(
        flip_angles_ex, phases_ex,
        T1f_s, T2f_s, T1b_s, T2b_s, kf_s, kb_s,
        TR_ex, TE_ex, B0_s, B1_s, wf_s, wb_s
    )

    print("Scalar run output (state_k=0 for Z_f, Z_b):")
    for i, (fpf, fmf, zf, fpb, fmb, zb) in enumerate(states_scalar):
        print(f"Pulse {i+1}: Zf0={zf[0,0]:.3f}, Zb0={zb[0,0]:.3f}")

    print("\n--- Vectorized MT Test ---")
    batch_size_ex = 2
    T1f_v = torch.tensor([1200.0, 1100.0], device=device)
    T2f_v = torch.tensor([40.0, 50.0], device=device)
    T1b_v = torch.tensor([400.0, 350.0], device=device)
    T2b_v = torch.tensor([10.0, 12.0], device=device)
    kf_v = torch.tensor([3.0, 4.0], device=device)
    kb_v = torch.tensor([20.0, 25.0], device=device)
    wf_v = torch.tensor([0.85, 0.8], device=device)
    wb_v = torch.tensor([0.15, 0.2], device=device)
    B0_v = torch.tensor([0.0, 10.0], device=device)
    B1_v = torch.tensor([1.0, 0.95], device=device)

    epg_vectorized = EPGSimulationMT_GPU_Vectorized(n_states=n_states_ex, device=device)
    states_vectorized = epg_vectorized(
        flip_angles_ex, phases_ex,
        T1f_v, T2f_v, T1b_v, T2b_v, kf_v, kb_v,
        TR_ex, TE_ex, B0_v, B1_v, wf_v, wb_v
    )

    print(f"Vectorized run output (Batch Size {batch_size_ex}, state_k=0 for Z_f, Z_b):")
    for i_pulse, (fpf_v, fmf_v, zf_v, fpb_v, fmb_v, zb_v) in enumerate(states_vectorized):
        print(f"Pulse {i_pulse+1}:")
        for b_idx in range(batch_size_ex):
            print(f"  B{b_idx}: Zf0={zf_v[b_idx,0]:.3f}, Zb0={zb_v[b_idx,0]:.3f}")
