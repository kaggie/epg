# EPG Simulation with Slice Profile Effects

## Importance of Slice Profiles

In Magnetic Resonance Imaging (MRI), radiofrequency (RF) pulses are used to excite a slice of tissue. Ideally, these pulses would excite all spins within the slice perfectly uniformly and no spins outside the slice. However, real RF pulses have non-ideal shapes (e.g., sinc, Gaussian) and durations, leading to:

1.  **Non-uniform Excitation:** Spins at different positions within the slice experience different effective flip angles. Spins near the center of the slice might experience the nominal flip angle, while spins near the edges experience lower flip angles.
2.  **Imperfect Slice Boundaries:** Spins just outside the intended slice boundaries might still be partially excited.

These slice profile imperfections can significantly impact the MRI signal, image contrast, and quantitative measurements, especially in sequences that use many RF pulses (e.g., Fast Spin Echo / Turbo Spin Echo, SSFP) or when trying to achieve precise flip angles for quantification.

## Simulating Slice Profiles with EPG

The Extended Phase Graph (EPG) model can be adapted to simulate the effects of non-ideal slice profiles. A common approach, implemented in this project, involves:

1.  **Discretizing the Slice:** The excited slice is modeled as a collection of thin "sub-slices" or isochromats along the slice selection direction.
2.  **Assigning Effective Flip Angles:** Each sub-slice is assigned a specific effective flip angle. This is determined by the nominal flip angle of the RF pulse and a `slice_profile_factor` corresponding to that sub-slice's position. The `slice_profile_factors` represent the normalized B1 amplitude (or effective flip angle scaling) across the slice. For example, a factor of 1.0 means the nominal flip angle is achieved, while a factor of 0.5 means half the nominal flip angle.
3.  **Parallel EPG Simulations:** An independent EPG simulation is effectively run for each sub-slice, using its specific effective flip angle for all RF pulses in the sequence. Other parameters (T1, T2, B0) are often assumed to be uniform across the slice for simplicity in this model.
    *   This is achieved efficiently by leveraging the batching capability of the vectorized EPG codes, where each sub-slice is treated as an item in a batch.

## Key Inputs

*   **`nominal_flip_angles_rad` (1D Tensor):** The sequence of nominal flip angles intended by the pulse sequence design.
*   **`slice_profile_factors` (1D Tensor):** A tensor where each element represents the scaling factor for the nominal flip angles for a particular sub-slice. The length of this tensor (`num_sub_slices`) determines how many discrete points are used to represent the slice profile. For an ideal "rectangular" slice profile, all factors would be 1.0. For a more realistic profile, these factors would vary (e.g., higher in the center, lower at the edges).
*   Other standard EPG parameters (T1, T2, TR, TE, phases, B0, global B1 scaling) are also required. T1, T2, and B0 are typically assumed to be uniform across the sub-slices in this model. The `global_B1_scale` further modulates all effective flip angles.

## Outputs

The simulation yields:

1.  **Per-Sub-Slice EPG States:** For each TR in the sequence, the full EPG state vectors (Fp, Fm, Z) are available for each sub-slice. This allows detailed inspection of how magnetization evolves differently across the slice. The shape of these state tensors will typically be `(num_sub_slices, n_epg_states)`.
2.  **Averaged Signal:** A method is usually provided to calculate the net signal from the entire slice at each time point. This is done by:
    *   Extracting the relevant EPG state for signal calculation (e.g., the magnitude of F<sub>k=1</sub>, which is the shifted F<sub>0</sub> state, `abs(Fp[sub_slice, 1])`).
    *   Averaging these signal contributions across all sub-slices.
    This results in a single time series representing the overall signal evolution from the slice, incorporating the slice profile effects.

## Implementation Example

An example implementation of this approach can be found in:
*   [`vectorized-sims/epg_mri_slice_profile.py`](../vectorized-sims/epg_mri_slice_profile.py)

This method provides a practical way to assess the impact of slice profile imperfections on MRI sequence performance and signal behavior.
