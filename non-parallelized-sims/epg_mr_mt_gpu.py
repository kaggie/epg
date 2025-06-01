import torch
import torch.nn as nn

class EPGSimulationMT_GPU(nn.Module):
    """
    Extended Phase Graph (EPG) simulation for Magnetization Transfer (MT) with full GPU support.
    Two-pool model: free (water) and bound (macromolecular) pools with exchange.
    Simulates T1, T2, B0, B1, and exchange between pools.
    All computations run on the specified device.
    """

    def __init__(self, n_states=21, device=None):
        super().__init__()
        # If device is None, default to GPU if available
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_states = n_states
        self.device = device

    def forward(
        self,
        flip_angles,
        phases,
        T1f, T2f,           # Free pool relaxation times (ms)
        T1b, T2b,           # Bound pool relaxation times (ms)
        kf, kb,             # Exchange rates (Hz): kf = f->b, kb = b->f
        TR, TE,
        B0=0.0, B1=1.0,     # Off-resonance (Hz), B1 scaling
        wf=1.0, wb=0.1      # Equilibrium magnetization fractions (wf+wb=1)
    ):
        """
        Simulate EPG-MT evolution (two-pool).
        Args:
            flip_angles: (N,) tensor, RF pulse flip angles in radians (applied to free pool only).
            phases: (N,) tensor, RF pulse phases in radians.
            T1f, T2f: floats or tensors, free pool relaxation times (ms).
            T1b, T2b: floats or tensors, bound pool relaxation times (ms).
            kf, kb: floats or tensors, exchange rates f->b and b->f (Hz).
            TR, TE: float, repetition and echo time (ms).
            B0: float or tensor, B0 inhomogeneity (Hz).
            B1: float, B1 scaling factor.
            wf, wb: floats, equilibrium fractions of free and bound pools (wf+wb=1).
        Returns:
            epg_states: list of (Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b) at each step.
        """
        # Move all input tensors and scalars to device
        flip_angles = flip_angles.to(self.device)
        phases = phases.to(self.device)

        T1f = torch.tensor(T1f, dtype=torch.float, device=self.device)
        T2f = torch.tensor(T2f, dtype=torch.float, device=self.device)
        T1b = torch.tensor(T1b, dtype=torch.float, device=self.device)
        T2b = torch.tensor(T2b, dtype=torch.float, device=self.device)
        kf = torch.tensor(kf, dtype=torch.float, device=self.device)
        kb = torch.tensor(kb, dtype=torch.float, device=self.device)
        B0 = torch.tensor(B0, dtype=torch.float, device=self.device)
        B1 = torch.tensor(B1, dtype=torch.float, device=self.device)
        wf = torch.tensor(wf, dtype=torch.float, device=self.device)
        wb = torch.tensor(wb, dtype=torch.float, device=self.device)

        N = len(flip_angles)
        Fp_f = torch.zeros(self.n_states, dtype=torch.cfloat, device=self.device)
        Fm_f = torch.zeros(self.n_states, dtype=torch.cfloat, device=self.device)
        Z_f = torch.zeros(self.n_states, dtype=torch.float, device=self.device)
        Fp_b = torch.zeros(self.n_states, dtype=torch.cfloat, device=self.device)
        Fm_b = torch.zeros(self.n_states, dtype=torch.cfloat, device=self.device)
        Z_b = torch.zeros(self.n_states, dtype=torch.float, device=self.device)
        Z_f[0] = wf
        Z_b[0] = wb

        epg_states = []

        E1f = torch.exp(-TR / T1f)
        E2f = torch.exp(-TR / T2f)
        E1b = torch.exp(-TR / T1b)
        E2b = torch.exp(-TR / T2b)
        phi_b0 = 2 * torch.pi * B0 * TR / 1000.0  # B0 in Hz, TR in ms

        for i in range(N):
            # 1. Relaxation and exchange
            Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b = self.relax_exchange(
                Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b,
                E1f, E2f, E1b, E2b, kf, kb, TR, wf, wb
            )
            Fp_f, Fm_f = self.apply_b0(Fp_f, Fm_f, phi_b0)
            Fp_b, Fm_b = self.apply_b0(Fp_b, Fm_b, phi_b0)  # Bound pool may have different offset (edit if needed)

            # 2. Apply RF pulse (only to free pool)
            alpha = flip_angles[i] * B1
            beta = phases[i]
            Fp_f, Fm_f, Z_f = self.apply_rf(Fp_f, Fm_f, Z_f, alpha, beta)

            # 3. EPG shift (gradient dephasing, both pools)
            Fp_f, Fm_f, Z_f = self.epg_shift(Fp_f, Fm_f, Z_f)
            Fp_b, Fm_b, Z_b = self.epg_shift(Fp_b, Fm_b, Z_b)

            # Store state
            epg_states.append((
                Fp_f.clone(), Fm_f.clone(), Z_f.clone(),
                Fp_b.clone(), Fm_b.clone(), Z_b.clone()
            ))

        return epg_states

    def relax_exchange(
        self, Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b,
        E1f, E2f, E1b, E2b, kf, kb, TR, wf, wb
    ):
        # Relaxation
        Fp_f = E2f * Fp_f
        Fm_f = E2f * Fm_f
        Fp_b = E2b * Fp_b
        Fm_b = E2b * Fm_b

        # Longitudinal relaxation + exchange (Euler step for Bloch-McConnell)
        dZf = -kf * Z_f + kb * Z_b
        dZb = -kb * Z_b + kf * Z_f
        Z_f = E1f * Z_f + (1 - E1f) * wf + dZf * TR / 1000.0
        Z_b = E1b * Z_b + (1 - E1b) * wb + dZb * TR / 1000.0
        return Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b

    def apply_b0(self, Fp, Fm, phi):
        # Apply phase accrual due to B0 off-resonance
        Fp = Fp * torch.exp(1j * phi)
        Fm = Fm * torch.exp(-1j * phi)
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        # Standard EPG RF rotation (free pool only)
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

# Example usage
if __name__ == "__main__":
    n_pulses = 10
    flip_angles = torch.ones(n_pulses, device='cuda' if torch.cuda.is_available() else 'cpu') * torch.deg2rad(torch.tensor(90.0, device='cuda' if torch.cuda.is_available() else 'cpu'))
    phases = torch.zeros(n_pulses, device=flip_angles.device)
    # Pool parameters
    T1f, T2f = 1000.0, 80.0    # ms (free)
    T1b, T2b = 1000.0, 10.0    # ms (bound)
    TR, TE = 500.0, 20.0       # ms
    kf, kb = 3.0, 6.0          # Hz (kf = f->b, kb = b->f)
    wf, wb = 0.9, 0.1          # Equilibrium fractions
    B0, B1 = 0.0, 1.0

    epg = EPGSimulationMT_GPU(n_states=21)
    states = epg(
        flip_angles, phases,
        T1f, T2f, T1b, T2b, kf, kb,
        TR, TE, B0, B1, wf, wb
    )

    for i, (Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b) in enumerate(states):
        print(f"Pulse {i+1}: Fp_f={Fp_f[0].real:.4f}, Z_f={Z_f[0]:.4f}, Z_b={Z_b[0]:.4f}")
