# Magnetization Transfer (MT) EPG Model

Magnetization Transfer (MT) is an MRI phenomenon that provides contrast based on the interaction between mobile bulk water protons (free pool) and protons bound to macromolecules or within semi-solid structures (bound pool). The Extended Phase Graph (EPG) algorithm can be adapted to simulate MT effects, typically by using a two-pool model.

This document builds upon the [Basic EPG Model](./basic_epg_model.md) and concepts in [Extended EPG Model](./extended_epg_model.md).

## Basics of Magnetization Transfer

*   **Two Pools:** Biological tissues contain at least two distinct populations of protons:
    1.  **Free Pool:** Protons in bulk water, characterized by relatively long T1 and T2 relaxation times. These are the protons primarily observed in standard MRI.
    2.  **Bound Pool:** Protons bound to macromolecules (e.g., proteins, lipids, collagen) or part of semi-solid cellular structures. These protons have very short T2 relaxation times (microseconds to milliseconds) and are thus typically not directly visible with conventional MRI sequences due to their rapid signal decay. They have a broad resonance lineshape.

*   **Magnetization Exchange:** Protons in the free and bound pools can exchange magnetization through various mechanisms, including chemical exchange (e.g., proton exchange between water molecules and hydroxyl or amine groups on macromolecules) and dipolar interactions.

*   **MT Effect:** The MT effect is typically observed by applying an off-resonance RF pulse. This RF pulse is designed to preferentially saturate the magnetization of the bound pool (due to its broad lineshape) while having minimal direct effect on the narrow resonance of the free water pool. The saturation of the bound pool is then transferred to the free pool via the exchange mechanisms. This results in a reduction of the observable water signal, and the degree of this reduction (the MT ratio or MTR) provides information about the concentration and properties of the macromolecular environment.

## Two-Pool Model in EPG for MT

To simulate MT effects with EPG, a two-pool model is commonly employed:

1.  **Separate EPG States for Each Pool:**
    *   **Free Pool (f):** Has its own set of EPG states: F<sup>+</sup><sub>f,k</sub>, F<sup>-</sup><sub>f,k</sub>, and Z<sub>f,k</sub>.
    *   **Bound Pool (b):** Also has its own set of EPG states: F<sup>+</sup><sub>b,k</sub>, F<sup>-</sup><sub>b,k</sub>, and Z<sub>b,k</sub>. However, due to the extremely short T2 of the bound pool (T2b), its transverse states (F<sup>+</sup><sub>b,k</sub>, F<sup>-</sup><sub>b,k</sub>) are often assumed to be zero or decay almost instantaneously. Thus, some MT models might only explicitly track Z<sub>b,k</sub> or even just Z<sub>b,0</sub> for the bound pool if its spatial coherence is neglected. More complete models will track all states.

2.  **Relaxation Parameters:** Each pool has its own distinct relaxation times:
    *   `T1f`, `T2f`: For the free pool.
    *   `T1b`, `T2b`: For the bound pool. T2b is characteristically very short.

## Role of Exchange Rates (kf, kb)

The two pools are coupled through magnetization exchange, primarily affecting the longitudinal (Z) states. This exchange is governed by rate constants:

*   **`kf` (or k<sub>f&rarr;b</sub>):** The rate constant for magnetization transfer from the free pool to the bound pool.
*   **`kb` (or k<sub>b&rarr;f</sub>):** The rate constant for magnetization transfer from the bound pool to the free pool.

These rates are incorporated into the EPG evolution equations, typically as part of a combined relaxation-exchange operator that updates the Z<sub>f,k</sub> and Z<sub>b,k</sub> states. In a simple Bloch-McConnell two-site exchange model, the change in longitudinal magnetization includes terms like:
`dZ_f/dt = ... - kf * Z_f + kb * Z_b`
`dZ_b/dt = ... - kb * Z_b + kf * Z_f`
These are then solved (often using an Euler step or matrix exponential) along with T1 relaxation terms over each time interval (e.g., TR).

## RF Pulse Effects in MT EPG

*   **Selective Excitation (Common Model):** In many MT EPG models, on-resonance RF pulses (used for excitation or echo formation) are assumed to primarily affect the **free water pool**. The direct effect on the bound pool by these pulses is often neglected due to its very short T2 and broad linewidth. So, the RF rotation matrix is applied only to the F<sup>+</sup><sub>f,k</sub>, F<sup>-</sup><sub>f,k</sub>, and Z<sub>f,k</sub> states.
*   **Off-Resonance Saturation Pulse:** If simulating the MT preparation pulse itself (the off-resonance saturation pulse), its effect is primarily on the Z<sub>b,0</sub> state of the bound pool, reducing its magnitude. This saturation is then propagated to the free pool via the `kb` exchange term. Some models might also include direct partial saturation of the free pool if the MT pulse has non-negligible amplitude at the water resonance frequency.

## Gradient Dephasing (Shifting)

Gradient dephasing operations (shifts in `k`-space) are typically applied to both pools independently, affecting their respective F<sup>+</sup>, F<sup>-</sup>, and Z states if these are tracked.

## Key Parameters for MT EPG Simulations

*   **`T1f`, `T2f`**: Longitudinal and transverse relaxation times of the free pool (ms).
*   **`T1b`, `T2b`**: Longitudinal and transverse relaxation times of the bound pool (ms). (T2b is very short, e.g., 10-100 &mu;s).
*   **`kf`**: Exchange rate from free pool to bound pool (Hz or s<sup>-1</sup>).
*   **`kb`**: Exchange rate from bound pool to free pool (Hz or s<sup>-1</sup>).
    *   Alternatively, one of the rates (e.g., `kf`) and the bound pool fraction (`wb` or `M0b`) are specified, and the other rate is derived assuming equilibrium: `kf * wf = kb * wb`.
*   **`wf`, `wb`**: Equilibrium magnetization fractions of the free and bound pools, respectively (where `wf + wb = 1`). `wf` is often called M0f and `wb` is M0b.
*   **Pulse sequence parameters**: `flip_angles`, `phases`, `TR`, `TE` as in basic EPG.
*   **RF pulse for MT saturation (if modeled explicitly):** Flip angle, duration, shape, and frequency offset.
*   **`B0`, `B1`**: Off-resonance and B1 scaling factor, as in basic EPG.

## GPU Implementation Considerations

Simulating MT EPG models, especially vectorized versions that handle batches of different tissue parameters, can be computationally intensive. GPU implementations are highly beneficial.
*   **Tensor Operations:** All EPG states (F+, F-, Z for both pools) and parameters should be represented as PyTorch tensors.
*   **Device Placement:** Ensure all tensors are moved to the designated GPU device at the beginning of the simulation.
*   **Broadcasting:** Leverage PyTorch's broadcasting capabilities to efficiently apply operations (relaxation, exchange, RF pulses, shifts) across the `n_states` dimension and, importantly, across the `batch_size` dimension in vectorized implementations. For instance, relaxation parameters like `E1f` (derived from `T1f`) can be shaped as `(batch_size, 1)` to broadcast with state vectors of shape `(batch_size, n_states)`.
*   **Efficiency:** Matrix operations for combined relaxation and exchange can be pre-calculated if parameters are constant over TRs, or efficiently computed on the GPU if they change.

By using a two-pool EPG model, researchers can simulate and analyze the complex signal behavior in MT-weighted MRI sequences, aiding in sequence design and interpretation of tissue contrast.
