import torch
import torch.nn as nn
import math

SCRIPT_VERSION_INFO = "epg_extended_v2_fixes" # Version string

class EPGSimulation(nn.Module): # Keep original class name for consistency with tests
    """
    Extended Phase Graph (EPG) simulation for MRI (Vectorized).
    Supports: MT, CEST, Diffusion, Flow, Off-resonance, etc.
    Order of operations in TR: Effects (Relax,Exch,Diff,Flow) -> B0 -> RF -> ChemShift -> Shift
    """

    def __init__(self, n_states=21, n_pools=1, device='cpu',
                 diffusion=False, flow=False, mqc=False, mt=False, cest=False): # MQC not fully implemented here
        super().__init__()
        self.n_states = n_states # This should be correctly used now for tensor initializations
        self.n_pools = n_pools
        self.device = device
        self.diffusion = diffusion
        self.flow = flow
        self.mqc = mqc
        self.mt = mt
        self.cest = cest

    def _to_batched_tensor(self, value, batch_size, num_elements_per_batch_item=1, dtype=torch.float):
        """Helper to convert/validate parameter shapes for batching."""
        # This helper ensures output is (batch_size, num_elements_per_batch_item) or (batch_size, 1)
        # then the caller can .unsqueeze if further dimensions like (B,P,1) or (B,1,1) are needed.
        if not isinstance(value, torch.Tensor):
            if isinstance(value, (list, tuple)) and len(value) == batch_size * num_elements_per_batch_item:
                 # Complex case like pool_params list of lists, handle with direct reshape after tensor conversion
                 value = torch.tensor(value, dtype=dtype, device=self.device) # Will likely fail for list of lists
            elif isinstance(value, (list, tuple)) and len(value) == batch_size and num_elements_per_batch_item == 1 : # list of scalars for batch
                 value = torch.tensor(value, dtype=dtype, device=self.device)
            elif isinstance(value, (list, tuple)) and num_elements_per_batch_item > 1 and len(value) == num_elements_per_batch_item:
                 # e.g. k_exch_rates = (kf,kb) for scalar batch
                 value = torch.tensor(value, dtype=dtype, device=self.device).unsqueeze(0).expand(batch_size, -1)
            else: # Python scalar
                value = torch.tensor(value, dtype=dtype, device=self.device).expand(batch_size)
        else: # Input is already a tensor
            value = value.to(self.device, dtype=dtype)
            if value.ndim == 0: value = value.expand(batch_size)
            elif value.shape[0] == 1 and batch_size > 1: value = value.expand(batch_size, *value.shape[1:])
            elif value.shape[0] != batch_size:
                 if value.numel() == num_elements_per_batch_item and value.ndim == 1 and batch_size == 1: # (elements,) for B=1
                     value = value.unsqueeze(0) # Becomes (1, elements)
                 elif value.numel() == num_elements_per_batch_item and value.ndim == 1 and batch_size > 1: # (elements,) for B>1, expand
                     value = value.unsqueeze(0).expand(batch_size, -1)
                 else:
                    raise ValueError(f"Tensor param shape {value.shape} not compatible with batch_size {batch_size} for {num_elements_per_batch_item} elements.")

        # Reshape to (batch_size, num_elements_per_batch_item)
        final_shape = (batch_size, num_elements_per_batch_item) if num_elements_per_batch_item > 0 else (batch_size,)
        return value.view(final_shape)


    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0,
                D=None, bval=None, v=None,
                k_exch_rates=None,
                cest_offsets=None,
                pool_params=None,
                pool_fractions=None,
                grad_spoil=None,
                chemical_shifts=None):

        N_pulses = flip_angles.shape[0]
        flip_angles = flip_angles.to(self.device)
        phases = phases.to(self.device)

        batch_size = 1
        ref_param_for_bs = T1 # Default to T1 for batch size inference
        if self.n_pools > 1 and pool_params is not None: ref_param_for_bs = pool_params
        elif self.n_pools > 1 and pool_fractions is not None: ref_param_for_bs = pool_fractions

        if isinstance(ref_param_for_bs, torch.Tensor) and ref_param_for_bs.ndim > 0:
            if ref_param_for_bs.shape[0] > 1 : batch_size = ref_param_for_bs.shape[0]
        elif isinstance(ref_param_for_bs, (list,tuple)) and len(ref_param_for_bs) > 1:
             is_list_of_lists_for_pools = (self.n_pools > 1 and isinstance(ref_param_for_bs[0], (list,tuple)))
             if not is_list_of_lists_for_pools : batch_size = len(ref_param_for_bs)


        B0_val = self._to_batched_tensor(B0, batch_size).unsqueeze(-1) # (B,1,1)
        B1_val = self._to_batched_tensor(B1, batch_size).unsqueeze(-1) # (B,1,1)

        # State vector shapes based on self.n_states (from __init__)
        if self.n_pools > 1:
            state_shape = (batch_size, self.n_pools, self.n_states)
            if pool_params is not None: # (B, P, 2) or (P, 2)
                if not isinstance(pool_params, torch.Tensor): pool_params = torch.tensor(pool_params, dtype=torch.float, device=self.device)
                if pool_params.ndim == 2: pool_params = pool_params.unsqueeze(0).expand(batch_size, self.n_pools, 2)
                T1_p = pool_params[..., 0]; T2_p = pool_params[..., 1] # (B,P)
                E1_calc = torch.exp(-TR / T1_p).unsqueeze(-1) # (B,P,1)
                E2_calc = torch.exp(-TR / T2_p).unsqueeze(-1) # (B,P,1)
            else: # Use main T1, T2 for all pools
                T1_expanded = self._to_batched_tensor(T1, batch_size).unsqueeze(1).expand(batch_size, self.n_pools, 1)
                T2_expanded = self._to_batched_tensor(T2, batch_size).unsqueeze(1).expand(batch_size, self.n_pools, 1)
                E1_calc = torch.exp(-TR / T1_expanded) # (B,P,1)
                E2_calc = torch.exp(-TR / T2_expanded) # (B,P,1)

            if pool_fractions is not None: # (B,P) or (P,)
                current_pool_fracs = self._to_batched_tensor(pool_fractions, batch_size, num_elements_per_batch_item=self.n_pools) # (B,P)
            else: # Default equal fractions
                current_pool_fracs = torch.ones(batch_size, self.n_pools, device=self.device) / self.n_pools
        else: # Single pool
            state_shape = (batch_size, self.n_states)
            T1_main = self._to_batched_tensor(T1, batch_size) # (B,1)
            T2_main = self._to_batched_tensor(T2, batch_size) # (B,1)
            E1_calc = torch.exp(-TR / T1_main).unsqueeze(-1) # (B,1,1) for (B,S) states
            E2_calc = torch.exp(-TR / T2_main).unsqueeze(-1) # (B,1,1)
            current_pool_fracs = torch.ones(batch_size, 1, device=self.device) # (B,1)

        Fp = torch.zeros(state_shape, dtype=torch.cfloat, device=self.device)
        Fm = torch.zeros(state_shape, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(state_shape, dtype=torch.float, device=self.device)

        if self.n_pools > 1: Z[..., 0] = current_pool_fracs # Z[b,p,0] = fracs[b,p]
        else: Z[..., 0] = current_pool_fracs.squeeze(-1)   # Z[b,0] = fracs[b] (since fracs is B,1)

        # Process other optional parameters
        # D (B,P,1) or (B,1,1). bval (B,1,1)
        if self.diffusion:
            D_processed = self._to_batched_tensor(D if D is not None else 0.0, batch_size, self.n_pools if self.n_pools > 1 else 1).unsqueeze(-1)
            bval_processed = self._to_batched_tensor(bval if bval is not None else 0.0, batch_size).unsqueeze(-1).unsqueeze(-1) # Make it (B,1,1,1) for broadcasting with D*k^2
        if self.flow: v_processed = self._to_batched_tensor(v if v is not None else 0.0, batch_size)
        if grad_spoil is not None: grad_spoil_processed = self._to_batched_tensor(grad_spoil, batch_size, dtype=torch.cfloat if torch.is_complex(torch.as_tensor(grad_spoil)) else torch.float).unsqueeze(-1).unsqueeze(-1)
        if k_exch_rates is not None: k_exch_processed = self._to_batched_tensor(k_exch_rates, batch_size, num_elements_per_batch_item=k_exch_rates.shape[-1] if isinstance(k_exch_rates, torch.Tensor) else len(k_exch_rates) if isinstance(k_exch_rates,(list,tuple)) else 2)
        if chemical_shifts is not None and self.n_pools > 1: chemical_shifts_processed = self._to_batched_tensor(chemical_shifts, batch_size, num_elements_per_batch_item=self.n_pools)

        epg_states = []
        phi_b0_main_val = (2 * math.pi * B0_val * TR / 1000.0) # (B,1,1)

        for i_pulse in range(N_pulses):
            # Order: Effects -> B0 -> RF -> ChemShift -> Shift
            if self.n_pools > 1:
                Z_eq_pools = current_pool_fracs # (B,P) for M0 recovery in relax_multi
                Fp, Fm, Z = self.relax_multi(Fp, Fm, Z, E1_calc, E2_calc, k_exch_processed if k_exch_rates is not None else None, Z_eq_pools, TR)
            else:
                Fp, Fm, Z = self.relax(Fp, Fm, Z, E1_calc, E2_calc, current_pool_fracs) # M0_fracs is (B,1)

            if self.diffusion and D is not None and bval is not None: Fp, Fm = self.apply_diffusion(Fp, Fm, D_processed, bval_processed)
            if self.flow and v is not None: Fp, Fm, Z = self.apply_flow(Fp, Fm, Z, v_processed[:,0]) # Homogeneous flow

            Fp, Fm = self.apply_b0(Fp, Fm, phi_b0_main_val)
            if grad_spoil is not None: Fp, Fm = self.apply_grad_spoil(Fp, Fm, grad_spoil_processed)

            current_flip_angle = flip_angles[i_pulse]; current_phase = phases[i_pulse]
            alpha_rf = current_flip_angle * B1_val # (B,1,1)
            beta_rf = current_phase # scalar

            if self.n_pools > 1:
                alpha_actual = torch.zeros(batch_size, self.n_pools, 1, device=self.device, dtype=alpha_rf.dtype)
                alpha_actual[:, 0, :] = alpha_rf.squeeze(-1) # RF on pool 0
                beta_actual = torch.zeros(batch_size, self.n_pools, 1, device=self.device)
                beta_actual[:,0,:] = beta_rf
                Fp, Fm, Z = self.apply_rf_multi(Fp, Fm, Z, alpha_actual, beta_actual)
            else:
                Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, alpha_rf, beta_rf)

            if self.n_pools > 1 and chemical_shifts is not None:
                phi_cs_val = (2 * math.pi * chemical_shifts_processed * TR / 1000.0).unsqueeze(-1) # (B,P,1)
                Fp, Fm = self.apply_b0(Fp, Fm, phi_cs_val)

            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z)
            epg_states.append((Fp.clone().detach(), Fm.clone().detach(), Z.clone().detach()))
        return epg_states

    def relax(self, Fp, Fm, Z, E1, E2, M0_fractions): # M0_fracs (B,1), E1/E2 (B,1,1), States (B,S)
        Fp = Fp * E2.squeeze(-1); Fm = Fm * E2.squeeze(-1) # (B,S)*(B,1) -> (B,S)
        Z_relaxed = Z * E1.squeeze(-1)
        Z_relaxed[..., 0] = Z[..., 0] * E1[...,0].squeeze(-1) + (1 - E1[...,0].squeeze(-1)) * M0_fractions[...,0]
        return Fp, Fm, Z_relaxed

    def relax_multi(self, Fp, Fm, Z, E1, E2, k_rates, M0_fractions, TR_ms):
        # E1/E2 (B,P,1), M0_fracs (B,P), k_rates (B, num_rates), States (B,P,S)
        Fp = Fp * E2; Fm = Fm * E2 # (B,P,S) * (B,P,1) -> (B,P,S)
        Z_relaxed_t1 = Z * E1
        for p_idx in range(self.n_pools): # Apply M0 recovery per pool
            Z_relaxed_t1[:, p_idx, 0] = Z[:, p_idx, 0] * E1[:, p_idx, 0] + \
                                     (1 - E1[:, p_idx, 0]) * M0_fractions[:, p_idx]

        if self.mt and k_rates is not None and self.n_pools == 2: # Simplified 2-pool exchange
            kf = k_rates[:, 0:1].unsqueeze(-1) # (B,1,1) for Z_f (pool 0)
            kb = k_rates[:, 1:2].unsqueeze(-1) # (B,1,1) for Z_b (pool 1)

            Zf_states = Z_relaxed_t1[:,0,:] # (B,S)
            Zb_states = Z_relaxed_t1[:,1,:] # (B,S)

            dZf_dt = -kf * Zf_states + kb * Zb_states # (B,1,1)*(B,S) -> (B,S)
            dZb_dt = +kf * Zf_states - kb * Zb_states

            Z_final = Z_relaxed_t1.clone()
            Z_final[:,0,:] += dZf_dt * (TR_ms / 1000.0)
            Z_final[:,1,:] += dZb_dt * (TR_ms / 1000.0)
            return Fp, Fm, Z_final
        return Fp, Fm, Z_relaxed_t1

    def apply_diffusion(self, Fp, Fm, D, bval):
        # D (B,P,1) or (B,1,1). bval (B,1,1,1). Fp/Fm (B,P,S) or (B,S)
        k_orders_sq = torch.arange(self.n_states, device=self.device, dtype=torch.float)**2
        if Fp.ndim == 2: k_orders_sq = k_orders_sq.view(1, -1)
        else: k_orders_sq = k_orders_sq.view(1, 1, -1)
        # D might be (B,P,1) or (B,1,1). k_orders_sq is (1,1,S) or (1,S)
        # bval is (B,1,1,1)
        # D * k_orders_sq needs care: (B,P,1)*(1,1,S)->(B,P,S) or (B,1,1)*(1,S)->(B,1,S)
        Dk2 = D * k_orders_sq
        attenuation = torch.exp(-bval.squeeze(-1) * Dk2) # bval (B,1,1) * Dk2 (B,P,S or B,1,S)
        return Fp * attenuation, Fm * attenuation

    def apply_flow(self, Fp, Fm, Z, v_batch): # v_batch (B,)
        Fp_f, Fm_f, Z_f = Fp, Fm, Z # Temp assignment
        for i in range(v_batch.shape[0]): # Loop over batch for potentially different shifts
            shift_amount = int(round(v_batch[i].item()))
            if shift_amount == 0: continue
            # Apply roll to specific batch item: Fp[i], Fm[i], Z[i]
            # Dims are (P,S) or (S) within each batch item
            Fp_f[i] = torch.roll(Fp[i], shifts=shift_amount, dims=-1)
            Fm_f[i] = torch.roll(Fm[i], shifts=shift_amount, dims=-1)
            Z_f[i] = torch.roll(Z[i], shifts=shift_amount, dims=-1)
            if shift_amount > 0: Fp_f[i,...,:shift_amount]=0; Fm_f[i,...,:shift_amount]=0; Z_f[i,...,:shift_amount]=0
            elif shift_amount < 0: Fp_f[i,...,shift_amount:]=0; Fm_f[i,...,shift_amount:]=0; Z_f[i,...,shift_amount:]=0
        return Fp_f, Fm_f, Z_f

    def apply_grad_spoil(self, Fp, Fm, grad_spoil_factor): # (B,1,1,1) or similar broadcastable
        return Fp * grad_spoil_factor, Fm * grad_spoil_factor

    def apply_b0(self, Fp, Fm, phi): # phi (B,P,1) or (B,1,1)
        phase = torch.exp(1j * phi)
        return Fp * phase, Fm * torch.conj(phase)

    def apply_rf(self, Fp, Fm, Z, alpha, beta): # alpha (B,S_pools_or_1,1), beta scalar or (B,S_pools_or_1,1)
        c = torch.cos(alpha / 2); s = torch.sin(alpha / 2)
        exp_ib = torch.exp(1j * beta); exp_mib = torch.exp(-1j * beta)
        Zc = Z.to(torch.cfloat)
        Fp_new = c**2*Fp + s**2*torch.conj(Fm)*exp_ib**2 + 1j*c*s*Zc*exp_ib
        Fm_new = s**2*torch.conj(Fp)*exp_mib**2 + c**2*Fm - 1j*c*s*Zc*exp_mib
        Z_new = (-1j*s*c*(Fp*exp_mib - Fm*exp_ib) + (c**2-s**2)*Zc).real
        return Fp_new, Fm_new, Z_new

    def apply_rf_multi(self, Fp, Fm, Z, alpha, beta): # alpha,beta (B,P,1)
        return self.apply_rf(Fp, Fm, Z, alpha, beta)

    def epg_shift(self, Fp, Fm, Z):
        Fp_s = torch.roll(Fp, shifts=1, dims=-1); Fp_s[...,0]=0
        Fm_s = torch.roll(Fm, shifts=-1, dims=-1); Fm_s[...,-1]=0
        Z_s = torch.roll(Z, shifts=1, dims=-1); Z_s[...,0]=0
        return Fp_s, Fm_s, Z_s

if __name__ == "__main__":
    print(f"EPG Extended Vectorized loaded. Version: {SCRIPT_VERSION_INFO}")
    device = torch.device('cpu')
    model = EPGSimulation(n_states=5, n_pools=1, device=device, mt=False, diffusion=False)
    fa = torch.tensor([math.pi/2]*1); ph = torch.zeros_like(fa)
    T1,T2,TR,TE = 1000.,80.,100.,10.
    pool_fracs = torch.tensor([1.0]) # For single pool
    states = model(fa,ph,T1,T2,TR,TE, pool_fractions=pool_fracs)
    print("Minimal single pool run completed.")

    model_mt = EPGSimulation(n_states=5, n_pools=2, device=device, mt=True)
    T1f,T2f,T1b,T2b = 1000.,80.,200.,2.
    kf,kb = 3., 20.
    pf = torch.tensor([[0.85,0.15]]) # Batch size 1, 2 pools
    pp = torch.tensor([[[1000,80],[200,2]]]) # B,P,2
    kr = torch.tensor([[3,20]]) # B,2
    states_mt = model_mt(fa,ph,T1f,T2f,TR,TE, pool_params=pp, pool_fractions=pf, k_exch_rates=kr)
    print("Minimal MT run completed.")
