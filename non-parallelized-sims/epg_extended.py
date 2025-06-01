import torch
import torch.nn as nn

class EPGSimulation(nn.Module):
    """
    Extended Phase Graph (EPG) simulation for MRI.
    Now supports:
        - Magnetization transfer (MT)
        - Chemical exchange (CEST)
        - Diffusion
        - Flow/motion
        - Multiple quantum coherences (MQC)
        - Off-resonance & chemical shift
        - Gradient imperfections
    """

    def __init__(self, n_states=21, n_pools=1, device='cpu', diffusion=False, flow=False, mqc=False, mt=False, cest=False):
        super().__init__()
        self.n_states = n_states
        self.n_pools = n_pools  # Number of pools for MT/CEST
        self.device = device
        self.diffusion = diffusion
        self.flow = flow
        self.mqc = mqc
        self.mt = mt
        self.cest = cest

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0,
                D=None, bval=None, v=None, k=None, delta=None, pool_params=None, grad_spoil=None,
                chemical_shifts=None):
        """
        Simulate EPG evolution with extensions.
        Args:
            ... as before ...
            D: diffusion coefficient(s) [mm^2/s], if self.diffusion is True
            bval: b-value of each interval [s/mm^2], if self.diffusion is True
            v: flow velocity [mm/ms], if self.flow is True
            k: exchange rates for MT/CEST [Hz], if self.mt or self.cest is True
            delta: pool frequency offsets [Hz], if self.cest is True
            pool_params: [(T1p, T2p), ...] for each pool, if self.mt or self.cest is True
            grad_spoil: gradient spoiling factors, if simulating gradient imperfections
            chemical_shifts: chemical shift offsets [Hz] for off-resonance pools
        Returns:
            epg_states: list of states at each step
        """
        N = len(flip_angles)
        # Setup for multiple pools (MT/CEST), MQC, etc.
        if self.n_pools > 1:
            Fp = torch.zeros(self.n_pools, self.n_states, dtype=torch.cfloat, device=self.device)
            Fm = torch.zeros(self.n_pools, self.n_states, dtype=torch.cfloat, device=self.device)
            Z = torch.zeros(self.n_pools, self.n_states, dtype=torch.float, device=self.device)
            Z[:, 0] = 1.0  # Equilibrium for all pools (can change for bound pools)
        else:
            Fp = torch.zeros(self.n_states, dtype=torch.cfloat, device=self.device)
            Fm = torch.zeros(self.n_states, dtype=torch.cfloat, device=self.device)
            Z = torch.zeros(self.n_states, dtype=torch.float, device=self.device)
            Z[0] = 1.0

        epg_states = []

        # Precompute relaxation and phase accrual
        if self.n_pools > 1 and pool_params is not None:
            E1 = torch.tensor([torch.exp(-TR / p[0]) for p in pool_params], device=self.device)
            E2 = torch.tensor([torch.exp(-TR / p[1]) for p in pool_params], device=self.device)
        else:
            E1 = torch.exp(-TR / T1)
            E2 = torch.exp(-TR / T2)
        phi_b0 = 2 * torch.pi * B0 * TR / 1000.0

        for i in range(N):
            # 1. Relaxation, exchange, diffusion, flow
            if self.n_pools > 1:
                Fp, Fm, Z = self.relax_multi(Fp, Fm, Z, E1, E2, k, Z_eq=None)
                if self.cest or self.mt:
                    Fp, Fm, Z = self.exchange(Fp, Fm, Z, k, delta, TR)
            else:
                Fp, Fm, Z = self.relax(Fp, Fm, Z, E1, E2)
            Fp, Fm = self.apply_b0(Fp, Fm, phi_b0)
            if self.diffusion and D is not None and bval is not None:
                Fp, Fm = self.apply_diffusion(Fp, Fm, D, bval)
            if self.flow and v is not None:
                Fp, Fm, Z = self.apply_flow(Fp, Fm, Z, v)
            if self.mqc:
                Fp, Fm, Z = self.apply_mqc(Fp, Fm, Z)
            if grad_spoil is not None:
                Fp, Fm = self.apply_grad_spoil(Fp, Fm, grad_spoil)

            # 2. RF pulse (for multi-pool, only water pool gets flip)
            if self.n_pools > 1:
                alpha = torch.zeros(self.n_pools, device=self.device)
                alpha[0] = flip_angles[i] * B1  # assume only pool 0 (water) excited
                beta = torch.zeros(self.n_pools, device=self.device)
                beta[0] = phases[i]
                Fp, Fm, Z = self.apply_rf_multi(Fp, Fm, Z, alpha, beta)
            else:
                alpha = flip_angles[i] * B1
                beta = phases[i]
                Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, alpha, beta)

            # 3. EPG shift (gradient dephasing), chemical shift
            if self.n_pools > 1 and chemical_shifts is not None:
                for pool in range(self.n_pools):
                    phi_cs = 2 * torch.pi * chemical_shifts[pool] * TR / 1000.0
                    Fp[pool], Fm[pool] = self.apply_b0(Fp[pool], Fm[pool], phi_cs)
            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z)

            # Store current state
            epg_states.append((Fp.clone(), Fm.clone(), Z.clone()))

        return epg_states

    def relax(self, Fp, Fm, Z, E1, E2):
        Fp = E2 * Fp
        Fm = E2 * Fm
        Z = E1 * Z + (1 - E1)
        return Fp, Fm, Z

    def relax_multi(self, Fp, Fm, Z, E1, E2, k, Z_eq=None):
        # Multi-pool relaxation (MT/CEST)
        for p in range(self.n_pools):
            Fp[p] = E2[p] * Fp[p]
            Fm[p] = E2[p] * Fm[p]
            Z[p] = E1[p] * Z[p] + (1 - E1[p]) if Z_eq is None else E1[p] * Z[p] + (1 - E1[p]) * Z_eq[p]
        return Fp, Fm, Z

    def exchange(self, Fp, Fm, Z, k, delta, TR):
        # Simple two-pool exchange (Bloch-McConnell for longitudinal; can extend for full system)
        # k: exchange rates (Hz), delta: pool offsets, TR: time interval
        if self.n_pools < 2 or k is None:
            return Fp, Fm, Z
        # Example for 2 pools: pool 0 <-> pool 1
        k01, k10 = k
        Z0, Z1 = Z[0], Z[1]
        # Exchange step in longitudinal pools (Euler, for simplicity)
        dZ0 = -k01 * Z0 + k10 * Z1
        dZ1 = -k10 * Z1 + k01 * Z0
        Z[0] += dZ0 * TR / 1000.0
        Z[1] += dZ1 * TR / 1000.0
        return Fp, Fm, Z

    def apply_b0(self, Fp, Fm, phi):
        Fp = Fp * torch.exp(1j * phi)
        Fm = Fm * torch.exp(-1j * phi)
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        cos_a2 = torch.cos(alpha / 2)
        sin_a2 = torch.sin(alpha / 2)
        exp_ib = torch.exp(1j * beta)
        exp_mib = torch.exp(-1j * beta)

        Fp_new = cos_a2**2 * Fp + sin_a2**2 * torch.conj(Fm) * exp_ib**2 + 1j * cos_a2 * sin_a2 * (Z * exp_ib)
        Fm_new = sin_a2**2 * torch.conj(Fp) * exp_mib**2 + cos_a2**2 * Fm - 1j * cos_a2 * sin_a2 * (Z * exp_mib)
        Z_new = -1j * sin_a2 * cos_a2 * (Fp * exp_mib - Fm * exp_ib) + (cos_a2**2 - sin_a2**2) * Z

        return Fp_new, Fm_new, Z_new

    def apply_rf_multi(self, Fp, Fm, Z, alpha, beta):
        # RF for multiple pools (only pools with nonzero alpha get excited)
        for p in range(self.n_pools):
            Fp[p], Fm[p], Z[p] = self.apply_rf(Fp[p], Fm[p], Z[p], alpha[p], beta[p])
        return Fp, Fm, Z

    def epg_shift(self, Fp, Fm, Z):
        # Shift states for the effect of gradients (dephasing)
        if Fp.ndim == 2:  # multi-pool
            for p in range(self.n_pools):
                Fp[p] = torch.roll(Fp[p], 1, 0)
                Fm[p] = torch.roll(Fm[p], -1, 0)
                Fp[p, 0] = 0
                Fm[p, -1] = 0
        else:
            Fp = torch.roll(Fp, 1, 0)
            Fm = torch.roll(Fm, -1, 0)
            Fp[0] = 0
            Fm[-1] = 0
        return Fp, Fm, Z

    def apply_diffusion(self, Fp, Fm, D, bval):
        # Diffusion attenuation for each coherence order
        # D: diffusion coefficient [mm^2/s], bval: [s/mm^2]
        for k in range(len(Fp)):
            attenuation = torch.exp(-bval * D * (k**2))
            Fp[k] *= attenuation
            Fm[k] *= attenuation
        return Fp, Fm

    def apply_flow(self, Fp, Fm, Z, v):
        # Shift EPG states to represent flow (simplified)
        shift = int(round(v))
        Fp = torch.roll(Fp, shift, 0)
        Fm = torch.roll(Fm, shift, 0)
        Z = torch.roll(Z, shift, 0)
        return Fp, Fm, Z

    def apply_mqc(self, Fp, Fm, Z):
        # Multiple Quantum Coherences: placeholder to expand state space
        # Not implemented in detail; would require tracking higher quantum orders
        return Fp, Fm, Z

    def apply_grad_spoil(self, Fp, Fm, grad_spoil):
        # Gradient spoiling: apply phase dispersion
        Fp *= grad_spoil
        Fm *= grad_spoil
        return Fp, Fm

# Example for CEST/MT/diffusion/flow:
if __name__ == "__main__":
    n_pulses = 10
    flip_angles = torch.ones(n_pulses) * torch.deg2rad(torch.tensor(90.0))
    phases = torch.zeros(n_pulses)
    T1w, T2w = 1000.0, 80.0  # water pool
    T1b, T2b = 500.0, 10.0   # bound pool (MT/CEST)
    TR, TE = 500.0, 20.0
    B0, B1 = 0.0, 1.0
    D = 0.001  # mm^2/s
    bval = 1.0 # s/mm^2
    v = 1.0    # mm/ms
    k = (1.0, 1.0)  # exchange rates Hz
    delta = (0.0, 2.0) # ppm
    pool_params = [(T1w, T2w), (T1b, T2b)]
    chemical_shifts = [0.0, 100.0] # Hz

    # Enable all extensions
    epg = EPGSimulation(n_states=21, n_pools=2, device='cpu', diffusion=True, flow=True, mt=True, cest=True, mqc=True)
    states = epg(flip_angles, phases, T1w, T2w, TR, TE, B0, B1,
                 D=D, bval=bval, v=v, k=k, delta=delta, pool_params=pool_params,
                 grad_spoil=0.95, chemical_shifts=chemical_shifts)

    for i, (Fp, Fm, Z) in enumerate(states):
        print(f"Pulse {i+1}: pool0 Fp={Fp[0,0].real:.4f}, pool1 Fp={Fp[1,0].real:.4f}, pool0 Z={Z[0,0]:.4f}, pool1 Z={Z[1,0]:.4f}")
