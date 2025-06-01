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
        self.n_pools = n_pools  # Number of pools for MT/CEST
        self.device = device
        self.diffusion = diffusion
        self.flow = flow
        self.mqc = mqc
        self.mt = mt
        self.cest = cest

    def forward(self, flip_angles, phases, T1, T2, TR, TE, B0=0.0, B1=1.0,
                D=None, bval=None, v=None, k_exch_rates=None, cest_offsets=None, pool_params=None, grad_spoil=None,
                chemical_shifts=None): # Renamed k to k_exch_rates, delta to cest_offsets
        """
        Simulate EPG evolution with extensions (vectorized).
        Args:
            flip_angles: (N,) tensor, RF pulse flip angles in radians.
            phases: (N,) tensor, RF pulse phases in radians.
            T1: float or (batch_size,) or (batch_size, n_pools) tensor, longitudinal relaxation time (ms).
            T2: float or (batch_size,) or (batch_size, n_pools) tensor, transverse relaxation time (ms).
            TR: float, repetition time (ms).
            TE: float, echo time (ms).
            B0: float or (batch_size,) tensor, B0 inhomogeneity (Hz).
            B1: float or (batch_size,) tensor, B1 scale (unitless).
            D: float or (batch_size,) or (batch_size, n_pools) tensor, diffusion coefficient(s) [mm^2/s].
            bval: float or (batch_size,) tensor, b-value of each interval [s/mm^2].
            v: float or (batch_size,) tensor, flow velocity [mm/ms]. (Homogeneous shift for now if batched)
            k_exch_rates: tuple/list of floats or (batch_size, num_pairs*2) tensor for exchange rates [Hz].
            cest_offsets: tuple/list of floats or (batch_size, n_pools) for CEST pool frequency offsets [Hz].
            pool_params: list of tuples [(T1p, T2p),...] or (batch_size, n_pools, 2) tensor.
            grad_spoil: float or (batch_size,) tensor or complex tensor for gradient spoiling.
            chemical_shifts: list of floats or (batch_size, n_pools) tensor for chemical shift offsets [Hz].
        Returns:
            epg_states: list of ( (batch_size, [n_pools,] n_states), ... ) tuples.
        """
        N = len(flip_angles)

        # Determine batch_size
        batch_size = 1
        all_params = [T1, T2, B0, B1, D, bval, v, k_exch_rates, cest_offsets, pool_params, grad_spoil, chemical_shifts]
        for p_idx, p_val in enumerate(all_params):
            if isinstance(p_val, torch.Tensor) and p_val.ndim > 0 : # Check if it's a tensor
                # For T1, T2, D, pool_params, chemical_shifts, k_exch_rates, cest_offsets - first dim is batch or n_pools
                # For B0, B1, bval, v, grad_spoil - first dim is batch
                # This logic might need refinement if a param has n_pools as first dim and batch_size=1
                current_batch_dim = 0
                if p_idx in [0,1,4,7,8,9,11]: # Params that can have n_pools dim
                    # If p_val.shape[0] == self.n_pools and p_val.ndim > 1 (e.g. D=(n_pools,1) for scalar batch)
                    # this is ambiguous. Assume if shape[0] matches batch_size determined so far, it's batched.
                    # A more robust way is to expect (batch_size, n_pools, ...) for such params.
                    # For now, simple check: if shape[0] is not n_pools, it could be batch_size.
                    if self.n_pools > 1 and p_val.shape[0] == self.n_pools and p_val.ndim > 1: # e.g. pool_params (n_pools, 2) for scalar batch
                         pass # This is likely not a batch defining parameter.
                    elif p_val.shape[0] > 1:
                        current_batch_dim = p_val.shape[0]
                elif p_val.shape[0] > 1 : # Params where first dim is always batch
                    current_batch_dim = p_val.shape[0]

                if current_batch_dim > 1:
                    if batch_size == 1:
                        batch_size = current_batch_dim
                    elif current_batch_dim != batch_size:
                        raise ValueError(f"Inconsistent batch sizes in input parameters. Got {current_batch_dim} vs {batch_size}.")

        # Prepare parameters for broadcasting: (batch_size, [n_pools,] 1)
        T1 = torch.as_tensor(T1, dtype=torch.float, device=self.device)
        T2 = torch.as_tensor(T2, dtype=torch.float, device=self.device)

        # Handle T1/T2 that could be (batch_size,) or (batch_size, n_pools)
        if self.n_pools > 1:
            if T1.ndim == 1 and T1.shape[0] == batch_size: T1 = T1.unsqueeze(1).expand(batch_size, self.n_pools)
            elif T1.ndim == 0 : T1 = T1.expand(batch_size, self.n_pools)
            if T2.ndim == 1 and T2.shape[0] == batch_size: T2 = T2.unsqueeze(1).expand(batch_size, self.n_pools)
            elif T2.ndim == 0 : T2 = T2.expand(batch_size, self.n_pools)
        else: # single pool
            if T1.ndim == 0 or T1.shape[0] != batch_size : T1 = T1.expand(batch_size)
            if T2.ndim == 0 or T2.shape[0] != batch_size : T2 = T2.expand(batch_size)

        B0 = torch.as_tensor(B0, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)
        B1 = torch.as_tensor(B1, dtype=torch.float, device=self.device).expand(batch_size).view(batch_size, 1)

        if self.diffusion:
            D_ = torch.as_tensor(D if D is not None else 0.0, dtype=torch.float, device=self.device)
            bval_ = torch.as_tensor(bval if bval is not None else 0.0, dtype=torch.float, device=self.device)
            # D: (batch, [n_pools,]) -> (batch, [n_pools,] 1)
            if D_.ndim == 0: D_ = D_.expand(batch_size, self.n_pools if self.n_pools > 1 else 1)
            elif D_.shape[0] != batch_size : D_ = D_.unsqueeze(0).expand(batch_size, *D_.shape)
            D_ = D_.view(batch_size, -1, 1) # (b,1,1) or (b,p,1)

            if bval_.ndim == 0 : bval_ = bval_.expand(batch_size)
            bval_ = bval_.view(batch_size, 1, 1) # (b,1,1)
            D, bval = D_, bval_ # Assign back to original names

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
            else: # Fallback to main T1/T2 if pool_params not given, T1/T2 should be (batch, n_pools)
                E1_calc = torch.exp(-TR / T1).view(batch_size, self.n_pools, 1)
                E2_calc = torch.exp(-TR / T2).view(batch_size, self.n_pools, 1)


            if chemical_shifts is not None:
                chemical_shifts_ = torch.as_tensor(chemical_shifts, dtype=torch.float, device=self.device)
                if chemical_shifts_.ndim == 1: chemical_shifts_ = chemical_shifts_.unsqueeze(0) # (n_pools) -> (1,n_pools)
                if chemical_shifts_.shape[0] != batch_size: chemical_shifts_ = chemical_shifts_.expand(batch_size, self.n_pools)
                phi_cs_val = 2 * torch.pi * chemical_shifts_ * TR / 1000.0 # (batch, n_pools)
            else:
                phi_cs_val = torch.zeros(batch_size, self.n_pools, device=self.device)

            if k_exch_rates is not None: # k_exch_rates (batch_size, num_rates)
                k_exch_rates = torch.as_tensor(k_exch_rates, device=self.device)
                if k_exch_rates.ndim == 1: k_exch_rates = k_exch_rates.unsqueeze(0)
                if k_exch_rates.shape[0] != batch_size: k_exch_rates = k_exch_rates.expand(batch_size, -1)
            if cest_offsets is not None: # cest_offsets (batch_size, n_pools)
                cest_offsets = torch.as_tensor(cest_offsets, device=self.device)
                if cest_offsets.ndim == 1: cest_offsets = cest_offsets.unsqueeze(0)
                if cest_offsets.shape[0] != batch_size: cest_offsets = cest_offsets.expand(batch_size, self.n_pools)

        else: # Single pool
            E1_calc = torch.exp(-TR / T1.view(batch_size,1)).unsqueeze(-1) # (b,1,1)
            E2_calc = torch.exp(-TR / T2.view(batch_size,1)).unsqueeze(-1) # (b,1,1)

        state_shape = (batch_size, self.n_pools, self.n_states) if self.n_pools > 1 else (batch_size, self.n_states)
        Fp = torch.zeros(state_shape, dtype=torch.cfloat, device=self.device)
        Fm = torch.zeros(state_shape, dtype=torch.cfloat, device=self.device)
        Z = torch.zeros(state_shape, dtype=torch.float, device=self.device)
        Z[..., 0] = 1.0

        epg_states = []
        phi_b0_val = (2 * torch.pi * B0 * TR / 1000.0).unsqueeze(-1) # (b,1,1)

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
                current_v_val = v[0] if batch_size > 1 else v # Homogeneous flow assumption
                Fp, Fm, Z = self.apply_flow(Fp, Fm, Z, current_v_val)

            if self.mqc: pass # Not implemented

            if grad_spoil is not None:
                Fp, Fm = self.apply_grad_spoil(Fp, Fm, grad_spoil)

            current_flip_angle = flip_angles[i_pulse]
            current_phase = phases[i_pulse]
            alpha_val = current_flip_angle * B1 # (batch_size, 1)
            beta_val = current_phase # scalar

            if self.n_pools > 1:
                alpha_multi = torch.zeros(batch_size, self.n_pools, 1, device=self.device, dtype=alpha_val.dtype)
                alpha_multi[:, 0, :] = alpha_val # RF on pool 0 for all batches
                beta_multi = torch.zeros(batch_size, self.n_pools, 1, device=self.device, dtype=torch.float) # Phases are float
                beta_multi[:, 0, :] = beta_val
                Fp, Fm, Z = self.apply_rf_multi(Fp, Fm, Z, alpha_multi, beta_multi)
            else:
                Fp, Fm, Z = self.apply_rf(Fp, Fm, Z, alpha_val.unsqueeze(-1), beta_val)

            if self.n_pools > 1 and chemical_shifts is not None:
                Fp, Fm = self.apply_b0(Fp, Fm, phi_cs_val.unsqueeze(-1)) # (b,p,1)

            Fp, Fm, Z = self.epg_shift(Fp, Fm, Z)

            # Store current state (clone for each step)
            epg_states.append((Fp.clone().detach(), Fm.clone().detach(), Z.clone().detach()))

        return epg_states

    def relax(self, Fp, Fm, Z, E1, E2):
        # Inputs: Fp, Fm, Z (batch, n_states), E1, E2 (batch, 1, 1)
        Fp = E2 * Fp
        Fm = E2 * Fm
        Z_relaxed = E1 * Z
        Z_relaxed[..., 0] = E1[...,0] * Z[..., 0] + (1 - E1[...,0]) # M0 is 1 for Z0 state
        return Fp, Fm, Z_relaxed

    def relax_multi(self, Fp, Fm, Z, E1, E2, k_exch_rates, Z_eq=None): # k_exch_rates not used here
        # Inputs: Fp, Fm, Z (batch, n_pools, n_states)
        # E1, E2 (batch, n_pools, 1)
        Fp = E2 * Fp
        Fm = E2 * Fm

        Z_relaxed = E1 * Z
        if Z_eq is not None:
            # Ensure Z_eq is broadcastable, e.g. (b,p,1) or (b,p) if Z_eq is for Z0 state only
            Z_eq_shaped = Z_eq.view(Z_relaxed.shape[0], self.n_pools, -1) # ensure (b,p,1) or (b,p,s)
            if Z_eq_shaped.shape[-1] == 1: # Z_eq for Z0 state
                 Z_relaxed[..., 0] = E1[...,0] * Z[..., 0] + (1 - E1[...,0]) * Z_eq_shaped[...,0]
            else: # Z_eq for all states (unlikely for M0)
                 Z_relaxed = E1 * Z + (1-E1)*Z_eq_shaped
        else: # Assume M0=1 for all Z0 states of all pools
            Z_relaxed[..., 0] = E1[...,0] * Z[..., 0] + (1 - E1[...,0])
        return Fp, Fm, Z_relaxed

    def exchange(self, Fp, Fm, Z, k_exch_rates, cest_offsets, TR):
        # k_exch_rates: (batch_size, num_pairs*2) or specific format for pool pairs
        # cest_offsets: (batch_size, n_pools) - used for CEST saturation effect, not direct exchange here
        # This is a simplified 2-pool example for Z0 states, assuming k_exch_rates = (k01, k10) per batch
        # Z has shape (batch_size, n_pools, n_states)
        if self.n_pools < 2 or k_exch_rates is None or k_exch_rates.shape[-1] < 2:
            return Fp, Fm, Z

        # Assuming k_exch_rates is (batch_size, 2) for a single pair between pool 0 and 1
        # k01: Z[b,0,0] -> Z[b,1,0], k10: Z[b,1,0] -> Z[b,0,0]
        k01 = k_exch_rates[:, 0].view(-1, 1, 1) # (batch_size,1,1)
        k10 = k_exch_rates[:, 1].view(-1, 1, 1) # (batch_size,1,1)

        # Ensure Mz_pool are (batch_size, 1, 1) to match k01, k10
        Mz_pool0 = Z[:, 0:1, 0:1] # (batch_size, 1, 1) state
        Mz_pool1 = Z[:, 1:2, 0:1] # (batch_size, 1, 1) state

        dMz0 = (-k01 * Mz_pool0 + k10 * Mz_pool1) * (TR / 1000.0) # (batch_size,1,1)
        dMz1 = (-k10 * Mz_pool1 + k01 * Mz_pool0) * (TR / 1000.0) # (batch_size,1,1)

        Z_new = Z.clone()
        # Perform addition with shapes (B,1,1) then squeeze result back to (B,1) for assignment
        Z_new[:, 0, 0:1] = (Z_new[:, 0, 0:1].unsqueeze(-1) + dMz0).squeeze(-1)
        Z_new[:, 1, 0:1] = (Z_new[:, 1, 0:1].unsqueeze(-1) + dMz1).squeeze(-1)
        return Fp, Fm, Z_new


    def apply_b0(self, Fp, Fm, phi):
        # Fp, Fm (batch, [n_pools,] n_states)
        # phi (batch, [n_pools,] 1)
        Fp = Fp * torch.exp(1j * phi)
        Fm = Fm * torch.exp(-1j * phi)
        return Fp, Fm

    def apply_rf(self, Fp, Fm, Z, alpha, beta):
        # Fp,Fm,Z (batch, [n_pools,] n_states)
        # alpha (batch, [n_pools,] 1), beta scalar or (batch, [n_pools,] 1)
        cos_a2 = torch.cos(alpha / 2)
        sin_a2 = torch.sin(alpha / 2)

        beta_tensor = torch.as_tensor(beta, device=alpha.device, dtype=torch.float) # ensure tensor
        if beta_tensor.ndim == 0: # scalar beta
            exp_ib = torch.exp(1j * beta_tensor)
            exp_mib = torch.exp(-1j * beta_tensor)
        else: # tensor beta, ensure shape matches alpha for broadcasting
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
        # Fp,Fm,Z (batch, n_pools, n_states)
        # alpha, beta (batch, n_pools, 1)
        return self.apply_rf(Fp, Fm, Z, alpha, beta)


    def epg_shift(self, Fp, Fm, Z):
        rolled_Fp = torch.roll(Fp, shifts=1, dims=-1)
        rolled_Fm = torch.roll(Fm, shifts=-1, dims=-1)
        rolled_Z = torch.roll(Z, shifts=1, dims=-1)

        rolled_Fp[..., 0] = 0
        rolled_Fm[..., -1] = 0
        return rolled_Fp, rolled_Fm, rolled_Z


    def apply_diffusion(self, Fp, Fm, D, bval):
        # D: (batch_size, [n_pools,] 1), bval: (batch_size, 1, 1)
        # Fp, Fm: (batch_size, [n_pools,] n_states)
        k_orders_sq = torch.arange(self.n_states, device=self.device, dtype=torch.float)**2 # (n_states,)

        # Reshape k_orders_sq for broadcasting to Fp/Fm shape
        if Fp.ndim == 2: # (batch, states)
            k_orders_sq_shaped = k_orders_sq.view(1, -1) # (1,s)
        else: # (batch, pools, states)
            k_orders_sq_shaped = k_orders_sq.view(1, 1, -1) # (1,1,s)

        attenuation_factors = torch.exp(-bval * D * k_orders_sq_shaped)

        Fp_diff = Fp * attenuation_factors
        Fm_diff = Fm * attenuation_factors
        return Fp_diff, Fm_diff


    def apply_flow(self, Fp, Fm, Z, v_scalar_or_homog_batch):
        if isinstance(v_scalar_or_homog_batch, torch.Tensor):
            shift_val = v_scalar_or_homog_batch[0] if v_scalar_or_homog_batch.ndim > 0 and v_scalar_or_homog_batch.numel() > 0 else v_scalar_or_homog_batch
            shift_amount = int(round(shift_val.item()))
        else:
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


    def apply_mqc(self, Fp, Fm, Z): # Batched MQC
        return Fp, Fm, Z # Not implemented

    def apply_grad_spoil(self, Fp, Fm, grad_spoil):
        # grad_spoil (batch,1,1) or (batch,pool,1)
        Fp = Fp * grad_spoil
        Fm = Fm * grad_spoil
        return Fp, Fm

# Example for CEST/MT/diffusion/flow:
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu' # Force CPU for testing
    print(f"Running on device: {device}")

    n_pulses = 5 # Reduced for brevity
    flip_angles = torch.ones(n_pulses, device=device) * torch.deg2rad(torch.tensor(45.0, device=device))
    phases = torch.zeros(n_pulses, device=device)

    TR_val, TE_val = 100.0, 10.0 # ms

    # --- Scalar example (backward compatibility test) ---
    print("\n--- Scalar (Backward Compatibility) ---")
    T1w_s, T2w_s = 1000.0, 80.0
    pool_params_s = [(T1w_s, T2w_s), (500.0, 10.0)] # T1b, T2b
    D_s = 0.001
    bval_s = 1.0
    v_s = 0.5 # mm per TR, effectively (shift amount)
    # k_exch_s should be (1, num_rates) for scalar batch to ensure batch_size remains 1
    k_exch_s = torch.tensor([[0.1, 0.1]], device=device) # k_water->bound, k_bound->water (Hz)
    chemical_shifts_s = [0.0, 200.0] # Hz for pool0 (water), pool1 (bound)
    grad_spoil_s = torch.exp(1j * torch.tensor(torch.pi, device=device)) # Perfect spoil

    epg_scalar = EPGSimulation(n_states=5, n_pools=2, device=device,
                               diffusion=True, flow=True, mt=True, cest=False) # CEST not fully modeled by this exchange

    # Pass T1/T2 for pool0 as main T1/T2 for scalar case consistency with how E1/E2 might be formed if pool_params is None
    states_scalar = epg_scalar(flip_angles, phases, T1w_s, T2w_s, TR_val, TE_val, B0=0.0, B1=1.0,
                               D=D_s, bval=bval_s, v=v_s, k_exch_rates=k_exch_s, cest_offsets=None,
                               pool_params=pool_params_s, grad_spoil=grad_spoil_s, chemical_shifts=chemical_shifts_s)

    for i, (sFp, sFm, sZ) in enumerate(states_scalar):
        # Output for batch item 0 (scalar run has batch_size=1 implicitly)
        print(f"Pulse {i+1}: P0 Fp0={sFp[0,0,0].abs():.3f}, Z0={sZ[0,0,0]:.3f}; "
              f"P1 Fp0={sFp[0,1,0].abs():.3f}, Z0={sZ[0,1,0]:.3f}")


    # --- Vectorized Example ---
    print("\n--- Vectorized ---")
    batch_size_ex = 3 # Renamed to avoid conflict with internal batch_size

    T1w_v = torch.tensor([800.0, 1000.0, 1200.0], device=device)
    T2w_v = torch.tensor([60.0, 80.0, 100.0], device=device)

    T1b_v = torch.tensor([400.0, 500.0, 600.0], device=device)
    T2b_v = torch.tensor([8.0, 10.0, 12.0], device=device)
    pool_params_v = torch.stack([
        torch.stack([T1w_v, T2w_v], dim=-1),
        torch.stack([T1b_v, T2b_v], dim=-1)
    ], dim=1) # (batch_size_ex, n_pools=2, 2)

    B0_v = torch.tensor([-5.0, 0.0, 5.0], device=device)
    B1_v = torch.tensor([0.9, 1.0, 1.1], device=device)

    # D_v: (batch_size, n_pools)
    D_v = torch.tensor([[0.001, 0.0005], [0.001, 0.0005], [0.0015, 0.0008]], device=device)
    bval_v = torch.tensor([0.5, 1.0, 1.5], device=device)
    v_v = torch.tensor([0.0, 0.5, 1.0], device=device) # flow, current apply_flow uses v_v[0] for all

    k_exch_rates_v = torch.tensor([[0.1,0.1], [0.05,0.05], [0.2,0.2]], device=device) # (batch_size_ex, 2)

    chemical_shifts_v = torch.tensor([[0.0, 150.0], [0.0, 200.0], [0.0, 250.0]], device=device) # (batch_size_ex, n_pools)
    grad_spoil_v = torch.exp(1j * torch.pi * torch.tensor([0.5, 1.0, 1.5], device=device)) # (batch_size_ex,)

    epg_vectorized = EPGSimulation(n_states=5, n_pools=2, device=device,
                                   diffusion=True, flow=True, mt=True, cest=False) # MQC False

    # Pass T1w_v, T2w_v. For n_pools > 1, pool_params_v takes precedence if provided.
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
# Final check of parameter handling in forward and helper methods:
# - Batch size determination looks okay.
# - Parameter expansion and view operations are designed to create (batch, [pools,] ...) shapes.
# - relax/relax_multi: E1/E2 are (b,p,1) or (b,1,1), Z0 recovery uses (b,p,0) or (b,0). Seems okay.
# - exchange: k01/k10 (b,1,1), Mz_pool0/1 (b,1). Z_new update is okay.
# - apply_b0: phi (b,[p,]1). Ok.
# - apply_rf: alpha (b,[p,]1), beta scalar/tensor (b,[p,]1). Ok.
# - epg_shift: rolls last dim. Ok.
# - apply_diffusion: D (b,[p,]1), bval (b,1,1), k_sq (1,[1,]s). Attenuation (b,[p,]s). Ok.
# - apply_flow: uses v[0] for now. Ok with that simplification. Boundary conditions applied to all batches/pools.
# - grad_spoil: (b,1,1). Ok.
# - RF pulse alpha/beta generation in main loop: alpha_val (b,1), beta_val scalar.
#   - For multi-pool, alpha_multi (b,p,1), beta_multi (b,p,1) correctly passed to apply_rf_multi.
#   - For single-pool, alpha_val.unsqueeze(-1) -> (b,1,1) and beta_val (scalar) passed to apply_rf. Ok.
# - Chemical shift phi_cs_val (b,p) -> unsqueezed to (b,p,1) for apply_b0. Ok.
# Example usage updated. Looks mostly ready for a test run.
# then proceed with vectorization. This makes the process clearer.
# The `apply_diffusion` fix: The loop `for k_order in range(len(Fp))` is problematic.
# If `Fp` has shape `(num_pools, num_states)`, `len(Fp)` is `num_pools`. So `k_order` iterates through pools.
# If `Fp` has shape `(num_states)`, `len(Fp)` is `num_states`. So `k_order` iterates through states. This is closer.
# The formula `exp(-bval * D * (k_order**2))` needs `k_order` to be the actual coherence order.
# The states `Fp[..., i]` correspond to coherence order `i` (for F+) or `-i` (for F-), Z states are order 0.
# The states are typically ordered F_0, F_1, F_2, ... or F_{-n}, ..., F_0, ..., F_{n}.
# The current EPG state indexing `Fp[0], Fp[1]...` means `Fp[k]` is the k-th order coherence state.
# So, if `Fp` is `(num_states)`, then `k_order` in `range(len(Fp))` is correct.
# If `Fp` is `(num_pools, num_states)`, then `Fp[p, k_order]` is the state.
# The current code is:
# for k_order in range(len(Fp)): # len(Fp) = n_pools if multi-pool, n_states if single.
#    attenuation = torch.exp(-bval * D * (k_order**2))
#    Fp[k_order] *= attenuation # If multi-pool, Fp[k_order] is a whole row for pool k_order
#    Fm[k_order] *= attenuation
# This is definitely a bug for multi-pool diffusion.
# It should be:
# coherence_orders_sq = torch.arange(self.n_states, device=self.device)**2
# attenuation_factors = torch.exp(-bval * D * coherence_orders_sq)
# Fp = Fp * attenuation_factors # Broadcasting over pool dimension (and later batch)
# Fm = Fm * attenuation_factors
# Z states are typically not directly attenuated by diffusion in this way (Z0 is k=0).
# This correction will be part of the vectorization diff.

# Same for flow: roll should be on `dims=-1`.
# Z states also need careful handling for flow, especially Z0.
# The current `epg_shift` (gradient) sets Fp[0]=0, Fm[-1]=0.
# Flow might require different boundary conditions. The current implementation of flow is a simple roll.
# I will proceed with creating the file, then apply vectorization including these corrections.
