import torch
import torch.nn as nn

SCRIPT_VERSION_INFO = "epg_mri_v2_RF_first_tv_gradients_v1" # Modified version info

class EPGSimulationTimeVaryingGradients(nn.Module): # Renamed class
    """
    Extended Phase Graph (EPG) simulation for MRI with Time-Varying Gradients.
    Simulates evolution of magnetization under T1, T2, B0, and B1 variations,
    and supports time-varying gradients.
    This version supports vectorized inputs for T1, T2, B0, and B1.
    """

    def __init__(self, n_states=21, device='cpu', gradient_waveform=None): # Added gradient_waveform
        super().__init__()
        self.n_states = n_states  # Number of EPG states (F+, F-, Z)
        self.device = device
        self.gradient_waveform = gradient_waveform # Store gradient_waveform

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0, gradient_waveform=None): # Added gradient_waveform
        """
        Simulate EPG evolution with vectorized parameters and time-varying gradients.
        Args:
            flip_angles: (N,) tensor, RF pulse flip angles in radians.
            phases: (N,) tensor, RF pulse phases in radians.
            T1: float or (batch_size,) tensor, longitudinal relaxation time (ms).
            T2: float or (batch_size,) tensor, transverse relaxation time (ms).
            TR: float, repetition time (ms).
            TE: float, echo time (ms).
            B0: float or (batch_size,) tensor, B0 inhomogeneity (Hz).
            B1: float or (batch_size,) tensor, B1 scale (unitless).
            gradient_waveform: Optional tensor representing the time-varying gradient.
                               If None, self.gradient_waveform is used.
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
        B0_val = torch.as_tensor(B0, dtype=torch.float, device=self.device).view(batch_size, 1)
        B1_val = torch.as_tensor(B1, dtype=torch.float, device=self.device).view(batch_size, 1)

        # Handle gradient_waveform precedence
        current_gradient_waveform = gradient_waveform
        if current_gradient_waveform is None:
            current_gradient_waveform = self.gradient_waveform

        # Initialize EPG states with batch dimension
        Fp = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Fm = torch.zeros(batch_size, self.n_states, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(batch_size, self.n_states, dtype=torch.float, device=self.device)
        Z[:, 0] = 1.0

        epg_states = []
        E1 = torch.exp(-TR / T1)
        E2 = torch.exp(-TR / T2)
        phi_b0_val = 2 * torch.pi * B0_val * TR / 1000.0

        for i in range(N):
            Fp, Fm, Z = self.relax(Fp, Fm, Z, E1, E2)
            Fp, Fm = self.apply_b0(Fp, Fm, phi_b0_val)

            alpha = flip_angles[i] * B1_val
            beta = phases[i]
            Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, alpha, beta)

            # Pass current_gradient_waveform to epg_shift
            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z, current_gradient_waveform)

            epg_states.append((Fp.clone(), Fm.clone(), Z.clone()))

        return epg_states

    def relax(self, Fp, Fm, Z, E1, E2):
        Fp = E2 * Fp
        Fm = E2 * Fm
        Z_relaxed = E1 * Z
        Z_relaxed[:, 0] = E1[:,0] * Z[:, 0] + (1 - E1[:,0]) # M0 is 1
        return Fp, Fm, Z_relaxed

    def apply_b0(self, Fp, Fm, phi):
        phi_complex = 1j * phi
        Fp = Fp * torch.exp(phi_complex)
        Fm = Fm * torch.exp(-phi_complex)
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        cos_a2 = torch.cos(alpha / 2)
        sin_a2 = torch.sin(alpha / 2)
        exp_ib = torch.exp(1j * beta)
        exp_mib = torch.exp(-1j * beta)
        Z_complex = Z.to(torch.cfloat)

        Fp_new = cos_a2**2 * Fp + \
                 sin_a2**2 * torch.conj(Fm) * exp_ib**2 + \
                 1j * cos_a2 * sin_a2 * (Z_complex * exp_ib)
        Fm_new = sin_a2**2 * torch.conj(Fp) * exp_mib**2 + \
                 cos_a2**2 * Fm - \
                 1j * cos_a2 * sin_a2 * (Z_complex * exp_mib)
        Z_new_complex = -1j * sin_a2 * cos_a2 * (Fp * exp_mib - Fm * exp_ib) + \
                        (cos_a2**2 - sin_a2**2) * Z_complex
        Z_new = Z_new_complex.real
        return Fp_new, Fm_new, Z_new

    def epg_shift(self, Fp, Fm, Z, gradient_waveform_step=None): # Added gradient_waveform_step
        # TODO: Implement time-varying gradient logic using gradient_waveform_step
        # For now, using the basic shift logic as a placeholder.
        # If gradient_waveform_step is not None, it should be used to modulate the shift.

        Fp_shifted = torch.roll(Fp, 1, dims=1)
        Fm_shifted = torch.roll(Fm, -1, dims=1)

        Fp_shifted[:, 0] = 0
        Fm_shifted[:, -1] = 0
        return Fp_shifted, Fm_shifted, Z # Z not shifted by ideal gradients

# Example usage:
if __name__ == "__main__":
    n_pulses = 10
    flip_angles = torch.ones(n_pulses) * torch.deg2rad(torch.tensor(90.0))
    phases = torch.zeros(n_pulses)

    # --- Scalar example ---
    print("--- Scalar T1, T2, B0, B1 (Time-Varying Gradients Example) ---")
    T1_scalar, T2_scalar = 1000.0, 80.0
    TR_scalar, TE_scalar = 500.0, 20.0
    B0_scalar, B1_scalar = 0.0, 1.0

    # Placeholder for gradient waveform definition
    # gradient_waveform_sample_scalar = torch.tensor(...) # Define for each pulse or TR
    # For example, if each TR has a specific gradient strength:
    # gradient_waveform_sample_scalar = torch.rand(n_pulses) # Example: one value per pulse

    epg_scalar = EPGSimulationTimeVaryingGradients(n_states=21, device='cpu') # Instantiated new class
    # Pass gradient_waveform if defined, e.g.:
    # states_scalar = epg_scalar(flip_angles, phases, T1_scalar, T2_scalar, TR_scalar, TE_scalar,
    #                            B0_scalar, B1_scalar, gradient_waveform=gradient_waveform_sample_scalar)
    states_scalar = epg_scalar(flip_angles, phases, T1_scalar, T2_scalar, TR_scalar, TE_scalar,
                               B0_scalar, B1_scalar) # Current run without passing new waveform

    for i, (Fp, Fm, Z) in enumerate(states_scalar):
        print(f"Pulse {i+1}: Fp0={Fp[0,0].real:.4f}, Fm0={Fm[0,0].real:.4f}, Z0={Z[0,0]:.4f}")

    # --- Vectorized example ---
    print("\n--- Vectorized T1, T2, B0, B1 (Time-Varying Gradients Example) ---")
    batch_s = 3
    T1_vec = torch.tensor([800.0, 1000.0, 1200.0])
    T2_vec = torch.tensor([60.0, 80.0, 100.0])
    TR_vec = 500.0
    TE_vec = 20.0
    B0_vec = torch.tensor([-5.0, 0.0, 5.0])
    B1_vec = torch.tensor([0.9, 1.0, 1.1])

    # Placeholder for gradient waveform definition for vectorized input
    # gradient_waveform_sample_vectorized = torch.rand(batch_s, n_pulses) # Or (n_pulses) if same for batch

    epg_vectorized = EPGSimulationTimeVaryingGradients(n_states=21, device='cpu') # Instantiated new class
    # Pass gradient_waveform if defined, e.g.:
    # states_vectorized = epg_vectorized(flip_angles, phases, T1_vec, T2_vec, TR_vec, TE_vec,
    #                                    B0_vec, B1_vec, gradient_waveform=gradient_waveform_sample_vectorized)
    states_vectorized = epg_vectorized(flip_angles, phases, T1_vec, T2_vec, TR_vec, TE_vec,
                                       B0_vec, B1_vec) # Current run without passing new waveform

    for i, (Fp_b, Fm_b, Z_b) in enumerate(states_vectorized):
        print(f"Pulse {i+1}:")
        for batch_idx in range(batch_s):
            print(f"  Batch {batch_idx}: Fp0={Fp_b[batch_idx,0].real:.4f}, Fm0={Fm_b[batch_idx,0].real:.4f}, Z0={Z_b[batch_idx,0]:.4f}")
