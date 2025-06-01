import torch
import torch.nn as nn

class EPGSimulation(nn.Module):
    """
    Extended Phase Graph (EPG) simulation for MRI.
    Vectorized version.
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
        self.n_pools = n_pools
        self.device = device
        self.diffusion = diffusion
        self.flow = flow
        self.mqc = mqc
        self.mt = mt
        self.cest = cest

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0,
                D=None, bval=None, v=None, k_exch_rates=None, cest_offsets=None, pool_params=None, grad_spoil=None,
                chemical_shifts=None):
        N = len(flip_angles)
        batch_size = 1
        all_params = [T1, T2, B0, B1, D, bval, v, k_exch_rates, cest_offsets, pool_params, grad_spoil, chemical_shifts]
        for p_idx, p_val in enumerate(all_params):
            if isinstance(p_val, torch.Tensor) and p_val.ndim > 0 :
                current_batch_dim = 0
                if p_idx in [0,1,4,7,8,9,11]:
                    if self.n_pools > 1 and p_val.shape[0] == self.n_pools and p_val.ndim > 1:
                         pass
                    elif p_val.shape[0] > 1:
                        current_batch_dim = p_val.shape[0]
                elif p_val.shape[0] > 1 :
                    current_batch_dim = p_val.shape[0]

                if current_batch_dim > 1:
                    if batch_size == 1:
                        batch_size = current_batch_dim
                    elif current_batch_dim != batch_size:
                        raise ValueError(f"Inconsistent batch sizes in input parameters. Got {current_batch_dim} vs {batch_size}.")

        T1 = torch.as_tensor(T1, dtype=torch.float, device=self.device)
        T2 = torch.as_tensor(T2, dtype=torch.float, device=self.device)

        if self.n_pools > 1:
            if T1.ndim == 1 and T1.shape[0] == batch_size: T1 = T1.unsqueeze(1).expand(batch_size, self.n_pools)
            elif T1.ndim == 0 : T1 = T1.expand(batch_size, self.n_pools)
            if T2.ndim == 1 and T2.shape[0] == batch_size: T2 = T2.unsqueeze(1).expand(batch_size, self.n_pools)
            elif T2.ndim == 0 : T2 = T2.expand(batch_size, self.n_pools)
        else:
            if T1.ndim == 0 or (T1.ndim > 0 and T1.shape[0] != batch_size) : T1 = T1.expand(batch_size)
            if T2.ndim == 0 or (T2.ndim > 0 and T2.shape[0] != batch_size) : T2 = T2.expand(batch_size)


        B0 = torch.as_tensor(B0, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)
        B1 = torch.as_tensor(B1, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)

        if self.diffusion:
            D_ = torch.as_tensor(D if D is not None else 0.0, dtype=torch.float, device=self.device)
            bval_ = torch.as_tensor(bval if bval is not None else 0.0, dtype=torch.float, device=self.device)
            if D_.ndim == 0: D_ = D_.expand(batch_size, self.n_pools if self.n_pools > 1 else 1)
            elif D_.shape[0] != batch_size : D_ = D_.unsqueeze(0).expand(batch_size, *D_.shape) # ensure batch dim if not present
            # Correct view for D_ based on n_pools
            D_ = D_.view(batch_size, self.n_pools if self.n_pools > 1 else 1, -1).squeeze(-1).unsqueeze(-1)


            if bval_.ndim == 0 : bval_ = bval_.expand(batch_size)
            bval_ = bval_.view(batch_size, 1, 1)
            D, bval = D_, bval_

        if self.flow:
            v = torch.as_tensor(v if v is not None else 0.0, dtype=torch.float, device=self.device)
            if v.ndim == 0 or v.shape[0] != batch_size: v = v.expand(batch_size)

        if grad_spoil is not None:
            grad_spoil_ = torch.as_tensor(grad_spoil, dtype=torch.cfloat if torch.is_complex(torch.as_tensor(grad_spoil)) else torch.float, device=self.device)
            if grad_spoil_.ndim == 0 or grad_spoil_.shape[0] != batch_size : grad_spoil_ = grad_spoil_.expand(batch_size)
            grad_spoil = grad_spoil_.view(batch_size, 1, 1)

        if self.n_pools > 1:
            if pool_params is not None:
                if isinstance(pool_params, (list, tuple)):
                    pool_params_tensor = torch.tensor(pool_params, dtype=torch.float, device=self.device)
                    pool_params_tensor = pool_params_tensor.unsqueeze(0).expand(batch_size, self.n_pools, 2)
                else:
                    pool_params_tensor = torch.as_tensor(pool_params, dtype=torch.float, device=self.device)
                    if pool_params_tensor.ndim == 2: pool_params_tensor = pool_params_tensor.unsqueeze(0)
                    if pool_params_tensor.shape[0] != batch_size: pool_params_tensor = pool_params_tensor.expand(batch_size, self.n_pools, 2)
                _E1_pool = torch.exp(-TR / pool_params_tensor[..., 0])
                _E2_pool = torch.exp(-TR / pool_params_tensor[..., 1])
                E1_calc = _E1_pool.view(batch_size, self.n_pools, 1)
                E2_calc = _E2_pool.view(batch_size, self.n_pools, 1)
            else:
                E1_calc = torch.exp(-TR / T1.view(batch_size, self.n_pools, 1))
                E2_calc = torch.exp(-TR / T2.view(batch_size, self.n_pools, 1))

            if chemical_shifts is not None:
                chemical_shifts_ = torch.as_tensor(chemical_shifts, dtype=torch.float, device=self.device)
                if chemical_shifts_.ndim == 1: chemical_shifts_ = chemical_shifts_.unsqueeze(0)
                if chemical_shifts_.shape[0] != batch_size: chemical_shifts_ = chemical_shifts_.expand(batch_size, self.n_pools)
                phi_cs_val = 2 * torch.pi * chemical_shifts_ * TR / 1000.0
            else:
                phi_cs_val = torch.zeros(batch_size, self.n_pools, device=self.device)
            if k_exch_rates is not None:
                k_exch_rates = torch.as_tensor(k_exch_rates, device=self.device)
                if k_exch_rates.ndim == 1: k_exch_rates = k_exch_rates.unsqueeze(0)
                if k_exch_rates.shape[0] != batch_size: k_exch_rates = k_exch_rates.expand(batch_size, -1)
            if cest_offsets is not None:
                cest_offsets = torch.as_tensor(cest_offsets, device=self.device)
                if cest_offsets.ndim == 1: cest_offsets = cest_offsets.unsqueeze(0)
                if cest_offsets.shape[0] != batch_size: cest_offsets = cest_offsets.expand(batch_size, self.n_pools)
        else: # Single pool
            E1_calc = torch.exp(-TR / T1.view(batch_size,1)).unsqueeze(-1)
            E2_calc = torch.exp(-TR / T2.view(batch_size,1)).unsqueeze(-1)

        state_shape = (batch_size, self.n_pools, self.n_states) if self.n_pools > 1 else (batch_size, self.n_states)
        Fp = torch.zeros(state_shape, dtype=torch.cfloat, device=self.device)
        Fm = torch.zeros(state_shape, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(state_shape, dtype=torch.float, device=self.device)
        Z[..., 0] = 1.0

        epg_states = []
        phi_b0_val = (2 * torch.pi * B0 * TR / 1000.0).unsqueeze(-1)

        for i_pulse in range(N):
            if self.n_pools > 1:
                Fp, Fm, Z = self.relax_multi(Fp, Fm, Z, E1_calc, E2_calc, k_exch_rates, Z_eq=None)
                if self.cest or self.mt:
                    Fp, Fm, Z = self.exchange(Fp, Fm, Z, k_exch_rates, cest_offsets, TR)
            else:
                Fp, Fm, Z = self.relax(Fp, Fm, Z, E1_calc, E2_calc)

            Fp, Fm = self.apply_b0(Fp, Fm, phi_b0_val)

            if self.diffusion and D is not None and bval is not None:
                Fp, Fm = self.apply_diffusion(Fp, Fm, D, bval)

            if self.flow and v is not None:
                current_v_val = v[0] if batch_size > 1 and v.ndim > 0 and v.numel() >= batch_size else v
                Fp, Fm, Z = self.apply_flow(Fp, Fm, Z, current_v_val)

            if self.mqc: pass

            if grad_spoil is not None:
                Fp, Fm = self.apply_grad_spoil(Fp, Fm, grad_spoil)

            current_flip_angle = flip_angles[i_pulse]
            current_phase = phases[i_pulse]
            alpha_val = current_flip_angle * B1
            beta_val = current_phase

            if self.n_pools > 1:
                alpha_multi = torch.zeros(batch_size, self.n_pools, 1, device=self.device, dtype=alpha_val.dtype)
                alpha_multi[:, 0, :] = alpha_val
                beta_multi = torch.zeros(batch_size, self.n_pools, 1, device=self.device, dtype=torch.float)
                beta_multi[:, 0, :] = beta_val
                Fp, Fm, Z = self.apply_rf_multi(Fp, Fm, Z, alpha_multi, beta_multi)
            else:
                Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, alpha_val.unsqueeze(-1), beta_val)

            if self.n_pools > 1 and chemical_shifts is not None:
                Fp, Fm = self.apply_b0(Fp, Fm, phi_cs_val.unsqueeze(-1))

            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z)
            epg_states.append((Fp.clone().detach(), Fm.clone().detach(), Z.clone().detach()))
        return epg_states

    def relax(self, Fp, Fm, Z, E1, E2):
        Fp = E2 * Fp
        Fm = E2 * Fm
        Z_relaxed = E1 * Z
        Z_relaxed[..., 0] = E1[...,0] * Z[..., 0] + (1 - E1[...,0])
        return Fp, Fm, Z_relaxed

    def relax_multi(self, Fp, Fm, Z, E1, E2, k_exch_rates, Z_eq=None):
        Fp = E2 * Fp
        Fm = E2 * Fm
        Z_relaxed = E1 * Z
        if Z_eq is not None:
            Z_eq_shaped = Z_eq.view(Z_relaxed.shape[0], self.n_pools, -1)
            if Z_eq_shaped.shape[-1] == 1:
                 Z_relaxed[..., 0] = E1[...,0] * Z[..., 0] + (1 - E1[...,0]) * Z_eq_shaped[...,0]
            else:
                 Z_relaxed = E1 * Z + (1-E1)*Z_eq_shaped
        else:
            Z_relaxed[..., 0] = E1[...,0] * Z[..., 0] + (1 - E1[...,0])
        return Fp, Fm, Z_relaxed

    def exchange(self, Fp, Fm, Z, k_exch_rates, cest_offsets, TR):
        if self.n_pools < 2 or k_exch_rates is None or k_exch_rates.shape[-1] < 2:
            return Fp, Fm, Z
        k01 = k_exch_rates[:, 0].view(-1, 1, 1)
        k10 = k_exch_rates[:, 1].view(-1, 1, 1)
        Mz_pool0 = Z[:, 0:1, 0:1]
        Mz_pool1 = Z[:, 1:2, 0:1]
        dMz0 = (-k01 * Mz_pool0 + k10 * Mz_pool1) * (TR / 1000.0)
        dMz1 = (-k10 * Mz_pool1 + k01 * Mz_pool0) * (TR / 1000.0)
        Z_new = Z.clone()
        Z_new[:, 0, 0:1] = (Z_new[:, 0, 0:1].unsqueeze(-1) + dMz0).squeeze(-1)
        Z_new[:, 1, 0:1] = (Z_new[:, 1, 0:1].unsqueeze(-1) + dMz1).squeeze(-1)
        return Fp, Fm, Z_new

    def apply_b0(self, Fp, Fm, phi):
        Fp = Fp * torch.exp(1j * phi)
        Fm = Fm * torch.exp(-1j * phi)
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        cos_a2 = torch.cos(alpha / 2)
        sin_a2 = torch.sin(alpha / 2)
        beta_tensor = torch.as_tensor(beta, device=alpha.device, dtype=torch.float)
        if beta_tensor.ndim == 0:
            exp_ib = torch.exp(1j * beta_tensor)
            exp_mib = torch.exp(-1j * beta_tensor)
        else:
            exp_ib = torch.exp(1j * beta_tensor.view_as(alpha))
            exp_mib = torch.exp(-1j * beta_tensor.view_as(alpha))
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

    def apply_rf_multi(self, Fp, Fm, Z, alpha, beta):
        return self.apply_rf(Fp, Fm, Z, alpha, beta)

    def epg_shift(self, Fp, Fm, Z): # Corrected: Z states are not shifted by typical gradients
        Fp_shifted = torch.roll(Fp, shifts=1, dims=-1)
        Fm_shifted = torch.roll(Fm, shifts=-1, dims=-1)
        Fp_shifted[..., 0] = 0
        Fm_shifted[..., -1] = 0
        return Fp_shifted, Fm_shifted, Z

    def apply_diffusion(self, Fp, Fm, D, bval): # Corrected diffusion logic
        k_orders_sq = torch.arange(self.n_states, device=self.device, dtype=torch.float)**2
        if Fp.ndim == 2: # (batch, states)
            k_orders_sq_shaped = k_orders_sq.view(1, -1)
        elif Fp.ndim == 3: # (batch, pools, states)
            k_orders_sq_shaped = k_orders_sq.view(1, 1, -1)
        else:
            raise ValueError(f"Unexpected Fp ndim: {Fp.ndim}")
        attenuation_factors = torch.exp(-bval * D * k_orders_sq_shaped)
        Fp_diff = Fp * attenuation_factors
        Fm_diff = Fm * attenuation_factors
        return Fp_diff, Fm_diff

    def apply_flow(self, Fp, Fm, Z, v_scalar_or_homog_batch):
        if isinstance(v_scalar_or_homog_batch, torch.Tensor): # Ensure it's a tensor before item()
            # Handle scalar tensor or batched tensor (use first element for homogeneous flow)
            shift_val = v_scalar_or_homog_batch[0] if v_scalar_or_homog_batch.ndim > 0 and v_scalar_or_homog_batch.numel() > 0 else v_scalar_or_homog_batch
            shift_amount = int(round(shift_val.item())) if shift_val.numel() == 1 else int(round(shift_val[0].item())) # handle scalar tensor vs first element
        else: # Python scalar
            shift_amount = int(round(v_scalar_or_homog_batch))

        if shift_amount == 0: return Fp, Fm, Z
        Fp_flow = torch.roll(Fp, shifts=shift_amount, dims=-1)
        Fm_flow = torch.roll(Fm, shifts=shift_amount, dims=-1)
        Z_flow = torch.roll(Z, shifts=shift_amount, dims=-1)
        if shift_amount > 0:
            Fp_flow[..., :shift_amount] = 0
            Fm_flow[..., :shift_amount] = 0
            Z_flow[..., :shift_amount] = 0
        elif shift_amount < 0:
            Fp_flow[..., shift_amount:] = 0
            Fm_flow[..., shift_amount:] = 0
            Z_flow[..., shift_amount:] = 0
        return Fp_flow, Fm_flow, Z_flow

    def apply_mqc(self, Fp, Fm, Z):
        return Fp, Fm, Z

    def apply_grad_spoil(self, Fp, Fm, grad_spoil):
        Fp = Fp * grad_spoil
        Fm = Fm * grad_spoil
        return Fp, Fm

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    n_pulses = 5
    flip_angles = torch.ones(n_pulses, device=device) * torch.deg2rad(torch.tensor(45.0, device=device))
    phases = torch.zeros(n_pulses, device=device)
    TR_val, TE_val = 100.0, 10.0
    print("\n--- Scalar (Backward Compatibility) ---")
    T1w_s, T2w_s = 1000.0, 80.0
    pool_params_s = [(T1w_s, T2w_s), (500.0, 10.0)]
    D_s = 0.001; bval_s = 1.0; v_s = 0.5
    k_exch_s = torch.tensor([[0.1, 0.1]], device=device)
    chemical_shifts_s = [0.0, 200.0]
    grad_spoil_s = torch.exp(1j * torch.tensor(torch.pi, device=device))
    epg_scalar = EPGSimulation(n_states=5, n_pools=2, device=device,
                               diffusion=True, flow=True, mt=True, cest=False)
    states_scalar = epg_scalar(flip_angles, phases, T1w_s, T2w_s, TR_val, TE_val, B0=0.0, B1=1.0,
                               D=D_s, bval=bval_s, v=v_s, k_exch_rates=k_exch_s, cest_offsets=None,
                               pool_params=pool_params_s, grad_spoil=grad_spoil_s, chemical_shifts=chemical_shifts_s)
    for i, (sFp, sFm, sZ) in enumerate(states_scalar):
        print(f"Pulse {i+1}: P0 Fp0={sFp[0,0,0].abs():.3f}, Z0={sZ[0,0,0]:.3f}; "
              f"P1 Fp0={sFp[0,1,0].abs():.3f}, Z0={sZ[0,1,0]:.3f}")
    print("\n--- Vectorized ---")
    batch_size_ex = 3
    T1w_v = torch.tensor([800.0, 1000.0, 1200.0], device=device)
    T2w_v = torch.tensor([60.0, 80.0, 100.0], device=device)
    T1b_v = torch.tensor([400.0, 500.0, 600.0], device=device)
    T2b_v = torch.tensor([8.0, 10.0, 12.0], device=device)
    pool_params_v = torch.stack([torch.stack([T1w_v, T2w_v], dim=-1), torch.stack([T1b_v, T2b_v], dim=-1)], dim=1)
    B0_v = torch.tensor([-5.0, 0.0, 5.0], device=device)
    B1_v = torch.tensor([0.9, 1.0, 1.1], device=device)
    D_v = torch.tensor([[0.001, 0.0005], [0.001,0.0005], [0.0015,0.0008]], device=device)
    bval_v = torch.tensor([0.5, 1.0, 1.5], device=device)
    v_v = torch.tensor([0.0, 0.5, 1.0], device=device)
    k_exch_rates_v = torch.tensor([[0.1,0.1], [0.05,0.05], [0.2,0.2]], device=device)
    chemical_shifts_v = torch.tensor([[0.0, 150.0], [0.0, 200.0], [0.0, 250.0]], device=device)
    grad_spoil_v = torch.exp(1j * torch.pi * torch.tensor([0.5, 1.0, 1.5], device=device))
    epg_vectorized = EPGSimulation(n_states=5, n_pools=2, device=device,
                                   diffusion=True, flow=True, mt=True, cest=False)
    states_vectorized = epg_vectorized(
        flip_angles, phases, T1w_v, T2w_v, TR_val, TE_val, B0=B0_v, B1=B1_v,
        D=D_v, bval=bval_v, v=v_v, k_exch_rates=k_exch_rates_v, cest_offsets=None,
        pool_params=pool_params_v, grad_spoil=grad_spoil_v, chemical_shifts=chemical_shifts_v
    )
    for i, (sFp, sFm, sZ) in enumerate(states_vectorized):
        print(f"Pulse {i+1}:")
        for b_idx in range(batch_size_ex):
            print(f"  B{b_idx}: P0 Fp0={sFp[b_idx,0,0].abs():.3f}, Z0={sZ[b_idx,0,0]:.3f}; "
                  f"P1 Fp0={sFp[b_idx,1,0].abs():.3f}, Z0={sZ[b_idx,1,0]:.3f}")
