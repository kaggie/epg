import torch
import torch.nn as nn

class EPGSimulationMQC(nn.Module):
    """
    Extended Phase Graph (EPG) simulation for MRI with Multiple Quantum Coherences (MQC).
    Simulates evolution of magnetization under T1, T2, B0, and B1 variations,
    and explicitly tracks multiple quantum orders.
    """

    def __init__(self, n_states=21, max_mqc_order=2, device='cpu'):
        super().__init__()
        self.n_states = n_states  # Number of EPG spatial orders (coherence orders k)
        self.max_mqc_order = max_mqc_order  # Maximum MQC order to simulate (e.g., 2 for single and double quantum)
        self.device = device

        # Fp/Fm for each MQC order, Z only for longitudinal (order 0)
        # F[k, q]: k=spatial order, q=MQC order (e.g. -2, -1, 0, +1, +2)
        # Map q from [-max_mqc_order, max_mqc_order], so index = q + max_mqc_order
        self.n_mqc = 2 * max_mqc_order + 1

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0):
        """
        Simulate EPG evolution with Multiple Quantum Coherences (MQC).
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
                       Each entry is (F, Z), where
                       F: [n_states, n_mqc] tensor of cfloat (MQC orders)
                       Z: [n_states] tensor of float (longitudinal)
        """
        N = len(flip_angles)
        F = torch.zeros(self.n_states, self.n_mqc, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(self.n_states, dtype=torch.float, device=self.device)
        # Initial state: Z0 = 1, all MQC = 0
        Z[0] = 1.0

        epg_states = []

        E1 = torch.exp(-TR / T1)
        E2 = torch.exp(-TR / T2)
        phi_b0 = 2 * torch.pi * B0 * TR / 1000.0  # B0 in Hz, TR in ms

        for i in range(N):
            # 1. Relaxation (single quantum only, but can generalize)
            F, Z = self.relax(F, Z, E1, E2)
            F = self.apply_b0(F, phi_b0)

            # 2. RF pulse (couples MQC orders)
            alpha = flip_angles[i] * B1
            beta = phases[i]
            F, Z = self.apply_rf(F, Z, alpha, beta)

            # 3. EPG shift (gradient dephasing: k-order shift)
            F, Z = self.epg_shift(F, Z)

            # Store current state
            epg_states.append((F.clone(), Z.clone()))

        return epg_states

    def relax(self, F, Z, E1, E2):
        # Relaxation: all MQC orders except q=0 decay with T2, only Z (longitudinal) recovers
        F = E2 * F
        Z = E1 * Z + (1 - E1)
        return F, Z

    def apply_b0(self, F, phi):
        # Off-resonance phase accrual: each quantum order q gets phase q*phi
        # F[k, q] -> exp(i*q*phi)
        n_mqc = F.shape[1]
        q_vec = torch.arange(-self.max_mqc_order, self.max_mqc_order+1, device=F.device)
        phase = torch.exp(1j * q_vec * phi)
        F = F * phase  # [n_states, n_mqc] * [n_mqc]
        return F

    def apply_rf(self, F, Z, alpha, beta):
        """
        RF pulse mixes MQC orders.
        Here, we implement the mixing for up to double quantum coherence (q=-2, -1, 0, +1, +2).
        See Weigel et al., JMRI 2015, Table 1 for general case.
        """
        n_states, n_mqc = F.shape
        F_new = torch.zeros_like(F)
        # Prepare indices for MQC orders: q = -max_mqc_order, ..., +max_mqc_order
        q_range = torch.arange(-self.max_mqc_order, self.max_mqc_order+1, device=F.device)

        cos_a2 = torch.cos(alpha / 2)
        sin_a2 = torch.sin(alpha / 2)
        exp_ib = torch.exp(1j * beta)
        exp_mib = torch.exp(-1j * beta)

        # Main MQC rotation
        # Only nonzero elements for -2 <= q <= 2 shown for simplicity

        # q = +1 (single quantum, F+): couples to q=0 (Z) and q=+2 (DQ)
        F_new[:, self.mqc_idx(+1)] = (
            cos_a2**2 * F[:, self.mqc_idx(+1)] +
            sin_a2**2 * torch.conj(F[:, self.mqc_idx(-1)]) * exp_ib**2 +
            1j*cos_a2*sin_a2 * (Z * exp_ib)
        )
        # q = -1 (single quantum, F-): couples to q=0 (Z) and q=-2 (DQ)
        F_new[:, self.mqc_idx(-1)] = (
            sin_a2**2 * torch.conj(F[:, self.mqc_idx(+1)]) * exp_mib**2 +
            cos_a2**2 * F[:, self.mqc_idx(-1)] -
            1j*cos_a2*sin_a2 * (Z * exp_mib)
        )
        # q = 0 (longitudinal): couples to F+ and F-
        Z_new = (
            -1j*sin_a2*cos_a2*(F[:, self.mqc_idx(-1)]*exp_ib - F[:, self.mqc_idx(+1)]*exp_mib)
            + (cos_a2**2 - sin_a2**2)*Z
        )

        # q = +2 (double quantum): couples to F- and itself
        if self.max_mqc_order >= 2:
            F_new[:, self.mqc_idx(+2)] = (
                cos_a2**2 * F[:, self.mqc_idx(+2)] +
                sin_a2**2 * torch.conj(F[:, self.mqc_idx(-2)]) * exp_ib**4
            )
            # q = -2
            F_new[:, self.mqc_idx(-2)] = (
                sin_a2**2 * torch.conj(F[:, self.mqc_idx(+2)]) * exp_mib**4 +
                cos_a2**2 * F[:, self.mqc_idx(-2)]
            )

        # Copy unchanged for other (higher) MQC orders if implemented
        for q in q_range:
            if abs(q) > 2:
                F_new[:, self.mqc_idx(q)] = F[:, self.mqc_idx(q)]

        return F_new, Z_new

    def epg_shift(self, F, Z):
        # Shift all MQC orders spatially
        F_shifted = torch.zeros_like(F)
        # For each MQC order, shift as in standard EPG
        for q_idx in range(F.shape[1]):
            F_shifted[:, q_idx] = torch.roll(F[:, q_idx], 1, 0)
            F_shifted[0, q_idx] = 0
        # Standard Z shift is not needed (longitudinal coherence does not shift)
        return F_shifted, Z

    def mqc_idx(self, q):
        """Return index into MQC axis for quantum order q."""
        return q + self.max_mqc_order

# Example usage:
if __name__ == "__main__":
    n_pulses = 10
    flip_angles = torch.ones(n_pulses) * torch.deg2rad(torch.tensor(90.0))  # 90 degree pulses
    phases = torch.zeros(n_pulses)
    T1, T2 = 1000.0, 80.0  # ms
    TR, TE = 500.0, 20.0   # ms
    B0, B1 = 0.0, 1.0

    epg = EPGSimulationMQC(n_states=21, max_mqc_order=2, device='cpu')
    states = epg(flip_angles, phases, T1, T2, TR, TE, B0, B1)

    for i, (F, Z) in enumerate(states):
        # Print only spatial order 0 for each MQC
        print(f"Pulse {i+1}: F+ {F[0,epg.mqc_idx(+1)].real:.4f}, F- {F[0,epg.mqc_idx(-1)].real:.4f}, DQ {F[0,epg.mqc_idx(+2)].real:.4f}, Z {Z[0]:.4f}")
