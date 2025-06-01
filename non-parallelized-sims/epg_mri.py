import torch
import torch.nn as nn

class EPGSimulation(nn.Module):
    """
    Extended Phase Graph (EPG) simulation for MRI.
    Simulates evolution of magnetization under T1, T2, B0, and B1 variations.
    """

    def __init__(self, n_states=21, device='cpu'):
        super().__init__()
        self.n_states = n_states  # Number of EPG states (F+, F-, Z)
        self.device = device

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0):
        """
        Simulate EPG evolution.
        Args:
            flip_angles: (N,) tensor, RF pulse flip angles in radians.
            phases: (N,) tensor, RF pulse phases in radians.
            T1: float or tensor, longitudinal relaxation time (ms).
            T2: float or tensor, transverse relaxation time (ms).
            TR: float, repetition time (ms).
            TE: float, echo time (ms).
            B0: float or tensor, B0 inhomogeneity (Hz).
            B1: float or tensor, B1 scale (unitless).
        Returns:
            epg_states: list of tensors, EPG state vectors at each step.
        """
        N = len(flip_angles)
        Fp = torch.zeros(self.n_states, dtype=torch.cfloat, device=self.device)
        Fm = torch.zeros(self.n_states, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(self.n_states, dtype=torch.float, device=self.device)
        Z[0] = 1.0  # Initial longitudinal magnetization

        # Collect state evolution for output
        epg_states = []

        E1 = torch.exp(-TR / T1)
        E2 = torch.exp(-TR / T2)
        phi_b0 = 2 * torch.pi * B0 * TR / 1000.0  # B0 in Hz, TR in ms

        for i in range(N):
            # 1. Relaxation and dephasing
            Fp, Fm, Z = self.relax(Fp, Fm, Z, E1, E2)
            Fp, Fm = self.apply_b0(Fp, Fm, phi_b0)

            # 2. Apply RF pulse (flip angle and phase, possibly B1 scaled)
            alpha = flip_angles[i] * B1
            beta = phases[i]
            Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, alpha, beta)

            # 3. EPG shift (gradient dephasing)
            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z)

            # Store current state
            epg_states.append((Fp.clone(), Fm.clone(), Z.clone()))

            # 4. (Optional) Readout at TE
            # Can add readout signal here if desired

        return epg_states

    def relax(self, Fp, Fm, Z, E1, E2):
        Fp = E2 * Fp
        Fm = E2 * Fm
        Z = E1 * Z + (1 - E1)
        return Fp, Fm, Z

    def apply_b0(self, Fp, Fm, phi):
        # Apply phase accrual due to B0 off-resonance
        Fp = Fp * torch.exp(1j * phi)
        Fm = Fm * torch.exp(-1j * phi)
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        """
        Apply an RF rotation (alpha = flip angle, beta = phase)
        """
        cos_a2 = torch.cos(alpha / 2)
        sin_a2 = torch.sin(alpha / 2)
        exp_ib = torch.exp(1j * beta)
        exp_mib = torch.exp(-1j * beta)

        Fp_new = cos_a2**2 * Fp + sin_a2**2 * torch.conj(Fm) * exp_ib**2 + 1j * cos_a2 * sin_a2 * (Z * exp_ib)
        Fm_new = sin_a2**2 * torch.conj(Fp) * exp_mib**2 + cos_a2**2 * Fm - 1j * cos_a2 * sin_a2 * (Z * exp_mib)
        Z_new = -1j * sin_a2 * cos_a2 * (Fp * exp_mib - Fm * exp_ib) + (cos_a2**2 - sin_a2**2) * Z

        return Fp_new, Fm_new, Z_new

    def epg_shift(self, Fp, Fm, Z):
        # Shift states for the effect of gradients (dephasing)
        Fp = torch.roll(Fp, 1, 0)
        Fm = torch.roll(Fm, -1, 0)
        Fp[0] = 0
        Fm[-1] = 0
        return Fp, Fm, Z

# Example usage:
if __name__ == "__main__":
    n_pulses = 10
    flip_angles = torch.ones(n_pulses) * torch.deg2rad(torch.tensor(90.0))  # 90 degree pulses
    phases = torch.zeros(n_pulses)
    T1, T2 = 1000.0, 80.0  # ms
    TR, TE = 500.0, 20.0   # ms
    B0, B1 = 0.0, 1.0

    epg = EPGSimulation(n_states=21, device='cpu')
    states = epg(flip_angles, phases, T1, T2, TR, TE, B0, B1)

    for i, (Fp, Fm, Z) in enumerate(states):
        print(f"Pulse {i+1}: Fp={Fp[0].real:.4f}, Fm={Fm[0].real:.4f}, Z={Z[0]:.4f}")
