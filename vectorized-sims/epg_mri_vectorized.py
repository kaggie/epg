import torch
import torch.nn as nn

SCRIPT_VERSION_INFO = "epg_mri_v2_RF_first"

class EPGSimulation(nn.Module):
    """
    Extended Phase Graph (EPG) simulation for MRI.
    Simulates evolution of magnetization under T1, T2, B0, and B1 variations.
    This version supports vectorized inputs for T1, T2, B0, and B1.
    """

    def __init__(self, n_states=21, device='cpu'):
        super().__init__()
        self.n_states = n_states  # Number of EPG states (F+, F-, Z)
        self.device = device

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0):
        """
        Simulate EPG evolution with vectorized parameters.
        Args:
            flip_angles: (N,) tensor, RF pulse flip angles in radians.
            phases: (N,) tensor, RF pulse phases in radians.
            T1: float or (batch_size,) tensor, longitudinal relaxation time (ms).
            T2: float or (batch_size,) tensor, transverse relaxation time (ms).
            TR: float, repetition time (ms).
            TE: float, echo time (ms).
            B0: float or (batch_size,) tensor, B0 inhomogeneity (Hz).
            B1: float or (batch_size,) tensor, B1 scale (unitless).
        Returns:
            epg_states: list of ( (batch_size, n_states), (batch_size, n_states), (batch_size, n_states) ) tuples,
                        representing Fp, Fm, Z states at each pulse step.
        """
        N = len(flip_angles)

        # Determine batch_size based on tensor inputs
        params = [T1, T2, B0, B1]
        batch_size = 1
        for p in params:
            if isinstance(p, torch.Tensor) and p.ndim > 0:
                if batch_size == 1:
                    batch_size = p.shape[0]
                elif p.shape[0] != batch_size:
                    raise ValueError("Inconsistent batch sizes for T1, T2, B0, B1.")

        # Ensure parameters are tensors and expand if necessary for broadcasting
        T1 = torch.as_tensor(T1, dtype=torch.float, device=self.device).view(batch_size, 1)
        T2 = torch.as_tensor(T2, dtype=torch.float, device=self.device).view(batch_size, 1)
        B0_val = torch.as_tensor(B0, dtype=torch.float, device=self.device).view(batch_size, 1) # Renamed to avoid conflict
        B1_val = torch.as_tensor(B1, dtype=torch.float, device=self.device).view(batch_size, 1) # Renamed to avoid conflict

        # Initialize EPG states with batch dimension
        # Fp, Fm are complex; Z is real
        Fp = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Fm = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(batch_size, self.n_states, dtype=torch.float, device=self.device)
        Z[:, 0] = 1.0  # Initial longitudinal magnetization for all items in batch

        # Collect state evolution for output
        epg_states = []

        # Precompute relaxation and B0 phase terms (now potentially batched)
        E1 = torch.exp(-TR / T1)  # (batch_size, 1)
        E2 = torch.exp(-TR / T2)  # (batch_size, 1)
        # B0 in Hz, TR in ms. phi_b0 is (batch_size, 1)
        phi_b0_val = 2 * torch.pi * B0_val * TR / 1000.0 # Renamed

        for i in range(N):
            # Order for v2: RF -> Relax -> B0 -> Shift

            # 1. Apply RF pulse
            alpha = flip_angles[i] * B1_val
            beta = phases[i]
            Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, alpha, beta)

            # 2. Relaxation
            Fp, Fm, Z = self.relax(Fp, Fm, Z, E1, E2)

            # 3. B0 dephasing
            Fp, Fm = self.apply_b0(Fp, Fm, phi_b0_val)

            # 4. EPG shift (gradient dephasing)
            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z)

            # Store current state
            epg_states.append((Fp.clone(), Fm.clone(), Z.clone()))

            # 4. (Optional) Readout at TE
            # Can add readout signal here if desired

        return epg_states

    def relax(self, Fp, Fm, Z, E1, E2):
        # E1, E2 are (batch_size, 1), Fp, Fm, Z are (batch_size, n_states)
        # Operations are automatically broadcasted correctly.
        Fp = E2 * Fp
        Fm = E2 * Fm
        # Ensure Z update happens correctly for batched Z and E1
        # Z is (batch_size, n_states), E1 is (batch_size, 1)
        # (1-E1) is also (batch_size, 1). Z[0] is M0, assumed 1.
        # The update rule Z = E1*Z + (1-E1)*M0 where M0 is Z[:,0] before relaxation.
        # However, the standard EPG formulation assumes M0=1 and applies to Z states.
        # Z has states Z0, Z1, Z2... Zn. Only Z0 recovers towards M0.
        # For batched states, Z[:,0] is the Z0 state for each batch item.
        Z_relaxed = E1 * Z
        Z_relaxed[:, 0] = E1[:,0] * Z[:, 0] + (1 - E1[:,0]) # M0 is 1
        return Fp, Fm, Z_relaxed

    def apply_b0(self, Fp, Fm, phi):
        # phi is (batch_size, 1)
        # Fp, Fm are (batch_size, n_states)
        # Apply phase accrual due to B0 off-resonance
        # Ensure phi is complex for exp
        phi_complex = 1j * phi
        Fp = Fp * torch.exp(phi_complex)
        Fm = Fm * torch.exp(-phi_complex)
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        """
        Apply an RF rotation (alpha = flip angle (batch_size,1), beta = phase (scalar))
        """
        # alpha is (batch_size, 1), beta is scalar
        # Fp, Fm, Z are (batch_size, n_states)
        cos_a2 = torch.cos(alpha / 2)  # (batch_size, 1)
        sin_a2 = torch.sin(alpha / 2)  # (batch_size, 1)

        # exp_ib and exp_mib need to be broadcastable with Fp, Fm, Z if beta were a tensor.
        # Since beta is scalar, exp_ib and exp_mib are scalar complex numbers.
        exp_ib = torch.exp(1j * beta)
        exp_mib = torch.exp(-1j * beta)

        # Need to ensure Z (real) is correctly promoted to complex for additions/subtractions involving it.
        # And then ensure the result for Z_new is real.
        Z_complex = Z.to(torch.cfloat)

        # All terms like cos_a2**2 are (batch_size,1) and will broadcast with (batch_size, n_states)
        Fp_new = cos_a2**2 * Fp + \
                 sin_a2**2 * torch.conj(Fm) * exp_ib**2 + \
                 1j * cos_a2 * sin_a2 * (Z_complex * exp_ib)

        Fm_new = sin_a2**2 * torch.conj(Fp) * exp_mib**2 + \
                 cos_a2**2 * Fm - \
                 1j * cos_a2 * sin_a2 * (Z_complex * exp_mib)

        Z_new_complex = -1j * sin_a2 * cos_a2 * (Fp * exp_mib - Fm * exp_ib) + \
                        (cos_a2**2 - sin_a2**2) * Z_complex

        # Z_new should be real. Taking .real should be safe if calculations are correct.
        # Small imaginary parts might remain due to precision; consider warning or torch.real_if_close.
        Z_new = Z_new_complex.real

        return Fp_new, Fm_new, Z_new

    def epg_shift(self, Fp, Fm, Z):
        # Shift states for the effect of gradients (dephasing)
        # This operation is per batch item, along the n_states dimension (dim=1)
        Fp = torch.roll(Fp, 1, dims=1)
        Fm = torch.roll(Fm, -1, dims=1)

        # Set Fp[0] and Fm[-1] (i.e. Fp_batch_0_state_0, Fm_batch_0_state_n-1) to 0 for all batch items
        Fp[:, 0] = 0
        Fm[:, -1] = 0 # Corrected to -1 for the last state index
        return Fp, Fm, Z

# Example usage:
if __name__ == "__main__":
    n_pulses = 10
    flip_angles = torch.ones(n_pulses) * torch.deg2rad(torch.tensor(90.0))  # 90 degree pulses
    phases = torch.zeros(n_pulses) # Scalar phase for all pulses, or (n_pulses,) if varying

    # --- Scalar example (backward compatibility) ---
    print("--- Scalar T1, T2, B0, B1 ---")
    T1_scalar, T2_scalar = 1000.0, 80.0  # ms
    TR_scalar, TE_scalar = 500.0, 20.0   # ms (TR is not batched in this design)
    B0_scalar, B1_scalar = 0.0, 1.0      # Hz and unitless scale

    epg_scalar = EPGSimulation(n_states=21, device='cpu')
    states_scalar = epg_scalar(flip_angles, phases, T1_scalar, T2_scalar, TR_scalar, TE_scalar, B0_scalar, B1_scalar)

    # Output for the first state (or aggregate if needed)
    for i, (Fp, Fm, Z) in enumerate(states_scalar):
        # Fp, Fm, Z are (batch_size=1, n_states)
        print(f"Pulse {i+1}: Fp0={Fp[0,0].real:.4f}, Fm0={Fm[0,0].real:.4f}, Z0={Z[0,0]:.4f}")

    # --- Vectorized example ---
    print("\n--- Vectorized T1, T2, B0, B1 ---")
    batch_s = 3
    T1_vec = torch.tensor([800.0, 1000.0, 1200.0]) # ms
    T2_vec = torch.tensor([60.0, 80.0, 100.0])    # ms
    TR_vec = 500.0 # TR remains scalar for now
    TE_vec = 20.0  # TE remains scalar
    B0_vec = torch.tensor([-5.0, 0.0, 5.0])       # Hz
    B1_vec = torch.tensor([0.9, 1.0, 1.1])        # unitless scale

    epg_vectorized = EPGSimulation(n_states=21, device='cpu')
    states_vectorized = epg_vectorized(flip_angles, phases, T1_vec, T2_vec, TR_vec, TE_vec, B0_vec, B1_vec)

    # Output for each item in the batch
    for i, (Fp_b, Fm_b, Z_b) in enumerate(states_vectorized):
        # Fp_b, Fm_b, Z_b are (batch_size, n_states)
        print(f"Pulse {i+1}:")
        for batch_idx in range(batch_s):
            print(f"  Batch {batch_idx}: Fp0={Fp_b[batch_idx,0].real:.4f}, Fm0={Fm_b[batch_idx,0].real:.4f}, Z0={Z_b[batch_idx,0]:.4f}")
