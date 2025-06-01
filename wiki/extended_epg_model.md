# Extended Phase Graph (EPG) Model Extensions

The basic Extended Phase Graph (EPG) model provides a robust framework for simulating MRI signal evolution. However, to capture more complex biophysical phenomena and system imperfections, the EPG model can be extended. These extensions often involve adding new states, modifying existing operations, or introducing new parameters.

This document builds upon the [Basic EPG Model](./basic_epg_model.md).

## Common EPG Extensions

Below are descriptions of several common extensions to the EPG model:

### 1. Magnetization Transfer (MT)

*   **Phenomenon:** Magnetization Transfer refers to the exchange of magnetization between protons in free bulk water and protons bound to macromolecules (e.g., proteins, cell membranes). RF pulses primarily saturate the bound pool, and this saturation is transferred to the free water pool via chemical exchange, leading to a reduction in the observable water signal.
*   **EPG Incorporation:**
    *   Typically modeled by introducing a second "bound" pool (or semi-solid pool) alongside the "free" water pool.
    *   Each pool (free and bound) has its own set of EPG states (F+, F-, Z) and its own relaxation parameters (T1f, T2f for free; T1b, T2b for bound).
    *   Exchange terms (governed by exchange rates `kf` and `kb`) are added to the relaxation equations to model the transfer of magnetization (usually longitudinal) between Z<sub>f</sub> and Z<sub>b</sub> states.
    *   RF pulses might directly affect only the free pool, or sometimes a direct saturation effect on the bound pool is also modeled.
*   **Key Parameters:**
    *   `T1f`, `T2f`: Relaxation times of the free pool.
    *   `T1b`, `T2b`: Relaxation times of the bound pool. (T2b is typically very short).
    *   `kf`: Exchange rate from free to bound pool (e.g., water to macromolecule).
    *   `kb`: Exchange rate from bound to free pool (e.g., macromolecule to water). Often `kf * M0f = kb * M0b` where M0f and M0b are equilibrium magnetizations or pool sizes.
    *   `wf`, `wb` (or `M0f`, `M0b`): Equilibrium magnetization fractions or relative sizes of the free and bound pools.

### 2. Chemical Exchange Saturation Transfer (CEST)

*   **Phenomenon:** CEST is a technique sensitive to slow-to-intermediate chemical exchange between bulk water protons and solute protons that have a different resonance frequency (e.g., amide protons in proteins, glycosaminoglycans). A frequency-selective RF saturation pulse is applied at the resonance frequency of the solute protons. This saturation is then transferred to water protons via chemical exchange, causing a detectable decrease in the water signal.
*   **EPG Incorporation:**
    *   Similar to MT, CEST is modeled using multiple pools (at least two: water and the exchanging solute pool). More pools can be added for multiple exchanging species.
    *   Each pool has its own EPG states and relaxation parameters.
    *   Exchange terms are included in the relaxation-exchange matrix.
    *   Crucially, the RF pulse operation must account for the frequency offset of the saturation pulse and the chemical shift of each pool. The RF pulse might selectively saturate one pool while minimally affecting others, depending on its bandwidth and frequency.
*   **Key Parameters:**
    *   Parameters for each pool: `T1i`, `T2i`, `wi` (equilibrium fraction for pool `i`).
    *   `k_ij`: Exchange rates between pools `i` and `j`.
    *   `delta_i` or `chemical_shift_i`: Frequency offset (chemical shift) of each pool `i` relative to a reference (often water).
    *   RF pulse parameters: flip angle, phase, duration, and importantly, frequency offset of the saturation pulse.

### 3. Diffusion

*   **Phenomenon:** Diffusion refers to the random thermal motion of water molecules. In the presence of magnetic field gradients, this motion causes phase dispersion of transverse magnetization, leading to signal attenuation.
*   **EPG Incorporation:**
    *   Diffusion effects are incorporated by applying an attenuation factor to the F+ and F- states.
    *   The attenuation for a coherence order `k` is typically `exp(-b * D * k^2)`, where `D` is the diffusion coefficient and `b` is the b-value determined by the gradient strength, duration, and timings.
    *   This attenuation is applied after each gradient segment or effectively over an interval TR if b-value for that TR is known. Z states are generally not directly attenuated by this `k^2` dependent term.
*   **Key Parameters:**
    *   `D`: Diffusion coefficient (e.g., in mm<sup>2</sup>/s). Can be a scalar or a tensor for anisotropic diffusion.
    *   `bval` or `b`: The b-value (e.g., in s/mm<sup>2</sup>), which summarizes the strength and timing of diffusion-sensitizing gradients.

### 4. Flow/Motion

*   **Phenomenon:** Macroscopic flow (e.g., blood flow) or bulk motion can cause phase shifts and misregistration of magnetization, affecting the signal.
*   **EPG Incorporation:**
    *   **Phase Accrual:** Flow along a gradient imparts a phase to the magnetization. This can be added to the `apply_b0` step or as a separate phase term, proportional to velocity and gradient area.
    *   **EPG State Shifting (Simplified model for through-plane flow):** For flow perpendicular to the imaging plane in the presence of slice-select or crusher gradients, flow can be modeled as an additional shift in coherence order `k`, similar to gradient dephasing. The amount of shift depends on the velocity and gradient properties. This is a simplification as flow can also bring fresh, unperturbed spins into the voxel.
*   **Key Parameters:**
    *   `v`: Flow velocity (e.g., in mm/s or mm/TR).
    *   Gradient information (implicitly, as it affects phase or effective shift).

### 5. Multiple Quantum Coherences (MQC)

*   **Phenomenon:** In spin systems with coupled spins (J-coupling) or in tissues with residual quadrupolar interactions, RF pulses can generate higher-order coherences beyond the usual single quantum coherences (F+ & F- correspond to &Delta;p = &plusmn;1). These are Multiple Quantum Coherences (MQC), such as double quantum (DQ, &Delta;p = &plusmn;2) or triple quantum (TQ, &Delta;p = &plusmn;3) coherences.
*   **EPG Incorporation:**
    *   The F states are expanded to include an additional dimension for the MQC order `q`. So, states become `F_k,q`.
    *   The RF pulse operator becomes more complex as it now also causes transitions between different MQC orders `q`. For example, a 90&deg; pulse can convert ZQ (longitudinal) to SQ, and another 90&deg; can convert SQ to ZQ, DQ, etc.
    *   Gradients affect the spatial order `k` based on the MQC order `q`: a gradient `G` shifts `F_k,q` to `F_{k+qG},q`. This means DQ coherences dephase twice as fast as SQ coherences.
*   **Key Parameters:**
    *   `max_mqc_order`: The maximum MQC order `q` to be tracked in the simulation.
    *   J-coupling constants (if modeling spin-spin coupling explicitly, though often effective MQC transfer functions are used).

### 6. Off-Resonance (&Delta;B0) and Chemical Shift

*   **Phenomenon:** Static magnetic field inhomogeneities (&Delta;B0) or different chemical environments (chemical shift) cause spins to precess at slightly different frequencies. This leads to phase accumulation during time intervals.
*   **EPG Incorporation:**
    *   This is handled by the `apply_b0` operation (or a similar dedicated phase accrual step).
    *   An additional phase `phi = 2 * pi * offset_frequency * dt` is applied to all F+ states, and its conjugate to F- states.
    *   `offset_frequency` can be due to global B0 inhomogeneity or specific chemical shifts of different species/pools. If multiple pools with different chemical shifts are present, each pool's F states will acquire phase according to its specific offset.
*   **Key Parameters:**
    *   `B0` or `offset_frequency`: The off-resonance frequency in Hz.
    *   `chemical_shifts`: An array of frequency offsets if multiple pools with distinct shifts are modeled.

### 7. Gradient Imperfections

*   **Phenomenon:** Real gradients may not be perfectly shaped (e.g., due to eddy currents) or may have slight miscalibrations. Crusher or spoiler gradients might not perfectly dephase all transverse magnetization.
*   **EPG Incorporation:**
    *   **Imperfect spoiling:** Can be modeled by not completely zeroing higher-order F states after a spoiler, or by applying a phase distribution (effectively reducing the magnitude of higher-order states).
    *   **Eddy currents:** Can be modeled as additional, time-varying gradient fields that contribute to dephasing and phase shifts. This can be complex to incorporate accurately.
    *   More simply, a `grad_spoil` factor (scalar or complex) can be applied to transverse states to simulate imperfect spoiling.
*   **Key Parameters:**
    *   `grad_spoil`: A factor (e.g., 0 to 1, or a complex number for phase effects) applied to F states during spoiling.
    *   Detailed characterization of gradient deviations if more accurate modeling is needed.

### 8. Slice Profile Effects

*   **Phenomenon:** Real-world RF pulses do not excite perfectly rectangular slices. The actual flip angle often varies across the slice thickness, and spins outside the nominal slice may be partially excited.
*   **EPG Incorporation:** One common method is to discretize the slice into several sub-slices, each experiencing a different effective flip angle based on its position within the RF pulse's profile. The EPG simulation is then run for each sub-slice (often in parallel using batching), and the total signal is an average or sum over these sub-slices. For more details, see [[Slice Profile EPG Models|./epg_slice_profiles.md]].
*   **Key Parameters:**
    *   `slice_profile_factors`: A set of scaling factors representing the effective flip angle at different points across the slice.

These extensions can be combined to create sophisticated EPG models that simulate a wide range of MRI physics and artifacts.
