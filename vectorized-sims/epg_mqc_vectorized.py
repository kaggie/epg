import torch
import torch.nn as nn

class EPGSimulationMQC(nn.Module):
    """
    Extended Phase Graph (EPG) simulation for MRI with Multiple Quantum Coherences (MQC).
    Vectorized version.
    Simulates evolution of magnetization under T1, T2, B0, and B1 variations,
    and explicitly tracks multiple quantum orders.
    """

    def __init__(self, n_states=21, max_mqc_order=2, device='cpu'):
        super().__init__()
        self.n_states = n_states  # Number of EPG spatial orders (coherence orders k)
        self.max_mqc_order = max_mqc_order  # Maximum MQC order to simulate
        self.device = device
        self.n_mqc = 2 * self.max_mqc_order + 1 # Total MQC states q = -max .. 0 .. +max

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0):
        """
        Simulate EPG evolution with MQC (vectorized).
        Args:
            flip_angles: (N_pulses,) tensor, RF pulse flip angles in radians.
            phases: (N_pulses,) tensor, RF pulse phases in radians.
            T1: float or (batch_size,) tensor, longitudinal relaxation time (ms).
            T2: float or (batch_size,) tensor, transverse relaxation time (ms).
            TR: float, repetition time (ms).
            TE: float, echo time (ms). (Currently not used in MQC core logic but common for EPG)
            B0: float or (batch_size,) tensor, B0 inhomogeneity (Hz).
            B1: float or (batch_size,) tensor, B1 scale (unitless).
        Returns:
            epg_states: list of (F, Z) tuples.
                       F: (batch_size, n_states, n_mqc) tensor
                       Z: (batch_size, n_states) tensor
        """
        N_pulses = len(flip_angles)

        # Determine batch_size
        batch_size = 1
        params_to_check = [T1, T2, B0, B1]
        for p in params_to_check:
            if isinstance(p, torch.Tensor) and p.ndim > 0 and p.shape[0] > 1:
                if batch_size == 1:
                    batch_size = p.shape[0]
                elif p.shape[0] != batch_size:
                    raise ValueError("Inconsistent batch sizes in input parameters.")

        # Prepare parameters for broadcasting (shape (batch_size, 1))
        T1 = torch.as_tensor(T1, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)
        T2 = torch.as_tensor(T2, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)
        B0_val = torch.as_tensor(B0, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)
        B1_val = torch.as_tensor(B1, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)

        # Initialize states with batch dimension
        # F: (batch_size, n_states, n_mqc)
        # Z: (batch_size, n_states)
        F = torch.zeros(batch_size, self.n_states, self.n_mqc, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(batch_size, self.n_states, dtype=torch.float, device=self.device)
        Z[..., 0] = 1.0  # Initial Z0 magnetization for all batches Z[b, state_0] = 1.0

        epg_states = [] # List to store (F, Z) state tuples at each pulse

        # Precompute relaxation and B0 phase terms (batched)
        E1 = torch.exp(-TR / T1)  # Shape: (batch_size, 1)
        E2 = torch.exp(-TR / T2)  # Shape: (batch_size, 1)
        phi_b0_val = (2 * torch.pi * B0_val * TR / 1000.0)  # Shape: (batch_size, 1)

        for i_pulse in range(N_pulses):
            # 1. Relaxation
            F, Z = self.relax(F, Z, E1, E2)

            # 2. B0 Off-resonance
            F = self.apply_b0(F, phi_b0_val)

            # 3. RF pulse
            current_flip_angle = flip_angles[i_pulse]  # Scalar for this pulse
            current_phase = phases[i_pulse]            # Scalar for this pulse
            # Alpha and Beta after B1 scaling, alpha is batched if B1 is batched
            alpha_rf = current_flip_angle * B1_val  # Shape: (batch_size, 1)
            beta_rf = current_phase                 # Scalar
            F, Z = self.apply_rf(F, Z, alpha_rf, beta_rf)

            # 4. EPG shift (gradient dephasing)
            F, Z = self.epg_shift(F, Z) # Note: Z is not shifted in this MQC model's epg_shift

            # Store cloned and detached states
            epg_states.append((F.clone().detach(), Z.clone().detach()))

        return epg_states

    def relax(self, F, Z, E1, E2):
        # F: (batch_size, n_states, n_mqc), Z: (batch_size, n_states)
        # E1, E2: (batch_size, 1)

        # MQC states decay with T2
        F = F * E2.unsqueeze(-1)  # E2 (B,1,1) broadcasts with F (B,S,Q)

        # Longitudinal states Z relax with T1 and recover towards M0 (assumed 1 for Z0)
        Z_relaxed = Z * E1 # E1 (B,1) broadcasts with Z (B,S)
        Z_relaxed[..., 0] = Z[..., 0] * E1[...,0] + (1 - E1[...,0]) # Z0 state recovery for each batch
        return F, Z_relaxed

    def apply_b0(self, F, phi):
        # F: (batch_size, n_states, n_mqc)
        # phi: (batch_size, 1) - B0 off-resonance phase accrual per TR

        # MQC order vector q = [-max_mqc_order, ..., +max_mqc_order]
        q_vec = torch.arange(-self.max_mqc_order, self.max_mqc_order + 1,
                             dtype=torch.float, device=F.device)  # Shape: (n_mqc,)

        # Phase for each MQC order q is q*phi
        # phi_per_q shape: (batch_size, n_mqc) by (B,1) * (1,Q)
        phi_per_q = phi * q_vec.view(1, -1)

        # Apply phase: exp(i * q * phi)
        phase_factors = torch.exp(1j * phi_per_q)  # Shape: (batch_size, n_mqc)

        # Broadcast phase_factors (B,1,Q) to F (B,S,Q)
        F = F * phase_factors.unsqueeze(1)
        return F

    def apply_rf(self, F, Z, alpha, beta):
        # F: (batch_size, n_states, n_mqc)
        # Z: (batch_size, n_states)
        # alpha: (batch_size, 1) - flip angle
        # beta: scalar - phase

        # Trigonometric terms for RF pulse, need shape (B,1,1) for broadcasting with F parts
        cos_a2 = torch.cos(alpha / 2).unsqueeze(-1)  # (B,1,1)
        sin_a2 = torch.sin(alpha / 2).unsqueeze(-1)  # (B,1,1)

        # Phase terms (beta is scalar)
        exp_ib = torch.exp(1j * beta)   # scalar complex
        exp_mib = torch.exp(-1j * beta) # scalar complex

        F_new = torch.zeros_like(F)  # (B,S,Q)
        Z_expanded = Z.unsqueeze(-1) # (B,S,1) for broadcasting in F+/- updates

        # Indices for MQC orders
        idx_p1 = self.mqc_idx(+1)
        idx_m1 = self.mqc_idx(-1)

        # Update F_new for q = +1 (F+ states)
        term1_p1 = cos_a2**2 * F[..., idx_p1].unsqueeze(-1)
        term2_p1 = sin_a2**2 * torch.conj(F[..., idx_m1]).unsqueeze(-1) * exp_ib**2
        term3_p1 = 1j * cos_a2 * sin_a2 * (Z_expanded * exp_ib)
        F_new[..., idx_p1] = (term1_p1 + term2_p1 + term3_p1).squeeze(-1)

        # Update F_new for q = -1 (F- states)
        term1_m1 = sin_a2**2 * torch.conj(F[..., idx_p1]).unsqueeze(-1) * exp_mib**2
        term2_m1 = cos_a2**2 * F[..., idx_m1].unsqueeze(-1)
        term3_m1 = -1j * cos_a2 * sin_a2 * (Z_expanded * exp_mib)
        F_new[..., idx_m1] = (term1_m1 + term2_m1 + term3_m1).squeeze(-1)

        # Update Z_new (longitudinal states, q=0 implicit for Z)
        s_cos_a2 = cos_a2.squeeze(-1) # (B,1)
        s_sin_a2 = sin_a2.squeeze(-1) # (B,1)
        Z_new = (
            -1j * s_sin_a2 * s_cos_a2 * (F[..., idx_m1] * exp_ib - F[..., idx_p1] * exp_mib) +
            (s_cos_a2**2 - s_sin_a2**2) * Z
        ).real # Z states must be real

        if self.max_mqc_order >= 2:
            idx_p2 = self.mqc_idx(+2)
            idx_m2 = self.mqc_idx(-2)
            F_new[..., idx_p2] = (
                cos_a2**2 * F[..., idx_p2].unsqueeze(-1) +
                sin_a2**2 * torch.conj(F[..., idx_m2]).unsqueeze(-1) * exp_ib**4
            ).squeeze(-1)
            F_new[..., idx_m2] = (
                sin_a2**2 * torch.conj(F[..., idx_p2]).unsqueeze(-1) * exp_mib**4 +
                cos_a2**2 * F[..., idx_m2].unsqueeze(-1)
            ).squeeze(-1)

        q_range_loop = torch.arange(-self.max_mqc_order, self.max_mqc_order + 1, device=F.device)
        for q_val_loop in q_range_loop:
            q_val_int = q_val_loop.item()
            if abs(q_val_int) > 2:
                 idx_q = self.mqc_idx(q_val_int)
                 if 0 <= idx_q < self.n_mqc:
                    F_new[..., idx_q] = F[..., idx_q]
        return F_new, Z_new

    def epg_shift(self, F, Z):
        # F: (batch_size, n_states, n_mqc)
        # Z: (batch_size, n_states)

        F_shifted = torch.roll(F, shifts=1, dims=1)
        F_shifted[:, 0, :] = 0

        return F_shifted, Z

    def mqc_idx(self, q_order):
        """Return index into MQC axis for quantum order q_order."""
        return q_order + self.max_mqc_order

# Example usage:
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running EPG MQC on device: {device}")

    n_states_ex = 7
    max_mqc_order_ex = 2
    n_pulses_ex = 3

    flip_angles_ex = torch.ones(n_pulses_ex, device=device) * torch.deg2rad(torch.tensor(90.0, device=device))
    phases_ex = torch.zeros(n_pulses_ex, device=device)
    TR_ex, TE_ex = 100.0, 10.0

    print("\n--- Scalar MQC Test ---")
    T1_s, T2_s = 1000.0, 80.0
    B0_s, B1_s = 0.0, 1.0

    epg_scalar = EPGSimulationMQC(n_states=n_states_ex, max_mqc_order=max_mqc_order_ex, device=device)
    states_scalar = epg_scalar(flip_angles_ex, phases_ex, T1_s, T2_s, TR_ex, TE_ex, B0=B0_s, B1=B1_s)

    print("Scalar run output (F[batch=0, state_k=0, MQC_order_idx], Z[batch=0, state_k=0]):")
    for i, (F_s, Z_s) in enumerate(states_scalar):
        print(f"Pulse {i+1}: F0_p1={F_s[0,0,epg_scalar.mqc_idx(+1)].real:.3f}, "
              f"F0_m1={F_s[0,0,epg_scalar.mqc_idx(-1)].real:.3f}, "
              f"F0_p2={F_s[0,0,epg_scalar.mqc_idx(+2)].real:.3f}, Z0={Z_s[0,0]:.3f}")

    print("\n--- Vectorized MQC Test ---")
    batch_size_ex = 2
    T1_v = torch.tensor([1000.0, 800.0], device=device)
    T2_v = torch.tensor([80.0, 60.0], device=device)
    B0_v = torch.tensor([0.0, 5.0], device=device)
    B1_v = torch.tensor([1.0, 0.9], device=device)

    epg_vectorized = EPGSimulationMQC(n_states=n_states_ex, max_mqc_order=max_mqc_order_ex, device=device)
    states_vectorized = epg_vectorized(flip_angles_ex, phases_ex, T1_v, T2_v, TR_ex, TE_ex, B0=B0_v, B1=B1_v)

    print(f"Vectorized run output (Batch Size {batch_size_ex}):")
    for i_pulse, (F_v, Z_v) in enumerate(states_vectorized):
        print(f"Pulse {i_pulse+1}:")
        for b_idx in range(batch_size_ex):
            print(f"  B{b_idx}: F0_p1={F_v[b_idx,0,epg_vectorized.mqc_idx(+1)].real:.3f}, "
                  f"F0_m1={F_v[b_idx,0,epg_vectorized.mqc_idx(-1)].real:.3f}, "
                  f"F0_p2={F_v[b_idx,0,epg_vectorized.mqc_idx(+2)].real:.3f}, Z0={Z_v[b_idx,0]:.3f}")
