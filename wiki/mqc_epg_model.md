# Multiple Quantum Coherence (MQC) EPG Model

The Extended Phase Graph (EPG) model can be further extended to simulate Multiple Quantum Coherences (MQC). This is particularly relevant for sequences designed to exploit or filter these higher-order coherences, which can arise in J-coupled spin systems or systems with quadrupolar interactions.

This document builds upon the [Basic EPG Model](./basic_epg_model.md) and concepts in [Extended EPG Model](./extended_epg_model.md).

## What are Multiple Quantum Coherences?

In standard MRI, we typically observe single-quantum coherences (SQC). These correspond to transitions where the magnetic quantum number `m` changes by &Delta;m = &plusmn;1. The F<sup>+</sup> and F<sup>-</sup> states in the basic EPG model represent these SQCs.

Multiple Quantum Coherences (MQCs) involve transitions with &Delta;m &ne; &plusmn;1. For example:
*   **Zero-Quantum Coherences (ZQC):** &Delta;m = 0. These are not directly observable but can evolve and be converted to SQC.
*   **Double-Quantum Coherences (DQC):** &Delta;m = &plusmn;2.
*   **Triple-Quantum Coherences (TQC):** &Delta;m = &plusmn;3.
*   And so on.

MQCs are not directly detectable by standard MRI coils (which are sensitive to &Delta;m = &plusmn;1). However, they can be created, evolve, and then be converted back into observable SQCs by subsequent RF pulses. Their unique evolution characteristics, particularly their sensitivity to phase accrual from gradients and off-resonance, can be exploited for specific imaging contrasts or artifact reduction.

## Adapting EPG for MQC

To track MQC within the EPG formalism, the state representation is expanded:

1.  **Expanded F States:** The transverse magnetization states (F states) are augmented with an additional index, `q`, representing the quantum coherence order.
    *   So, instead of just F<sub>k</sub><sup>+</sup> and F<sub>k</sub><sup>-</sup> (representing SQC with spatial order `k`), we have F<sub>k,q</sub>.
    *   `k` remains the spatial dephasing order (as in basic EPG).
    *   `q` is the quantum coherence order:
        *   `q = +1`: Positive single-quantum coherence (like F<sup>+</sup>).
        *   `q = -1`: Negative single-quantum coherence (like F<sup>-</sup>).
        *   `q = +2`: Positive double-quantum coherence.
        *   `q = -2`: Negative double-quantum coherence.
        *   And so on, up to a `max_mqc_order`.
    *   Often, F<sub>k,0</sub> is not explicitly stored as it would represent zero-quantum transverse coherence, which behaves differently or is part of the Z states in some formalisms. The MQC F-states usually represent orders `q &ne; 0`.

2.  **Z States:** The longitudinal magnetization (Z<sub>k</sub>) typically still corresponds to `q=0` (zero quantum order, as it's longitudinal). However, the RF pulse operations will show how Z<sub>k</sub> states are converted to and from F<sub>k,q</sub> states.

The state vector for transverse magnetization `F` thus becomes a 2D array (or 3D if batched) indexed by `(k, q_idx)`, where `q_idx` maps the actual quantum order `q` (e.g., -2, -1, +1, +2) to an array index (e.g., 0, 1, 2, 3).

## Impact of RF Pulses on MQC States

RF pulses are crucial for MQC as they create coherences of different orders and interconvert them.
*   A single RF pulse applied to equilibrium magnetization (Z<sub>0</sub>) primarily creates SQC (F<sub>0,+1</sub> and F<sub>0,-1</sub>).
*   Subsequent RF pulses can convert these SQCs into ZQCs, DQCs, and other MQCs. For example, a 90&deg; pulse can convert SQC into a mixture of ZQC and DQC. Another 90&deg; pulse can convert DQC back into observable SQC.
*   The exact mixing behavior depends on the flip angle, phase, and the initial state of the MQC orders. The mathematical description involves rotation matrices that are more complex than the basic EPG RF operator, as they need to describe transitions between different `q` values.
*   Specific pulse sequences (e.g., two pulses with appropriate phases) are designed to selectively create or convert specific MQC orders.

## Off-Resonance and Gradient Effects on MQC

A key characteristic of MQC is how different orders `q` evolve under off-resonance and gradients:

*   **Off-Resonance (B0 or chemical shift):** If there's an off-resonance frequency `&Delta;&omega;`, a coherence of order `q` will accumulate phase at a rate of `q * &Delta;&omega;`.
    *   This means DQCs (&plusmn;2) accumulate phase twice as fast as SQCs (&plusmn;1).
    *   ZQC (q=0) does not accumulate phase due to off-resonance.
    *   In EPG, this is implemented by multiplying F<sub>k,q</sub> by `exp(i * q * &Delta;&omega; * t)`.

*   **Gradient Dephasing:** When a gradient `G` is applied, the spatial phase evolution also depends on the MQC order `q`.
    *   The EPG shift operation changes the spatial order `k`. For an MQC state F<sub>k,q</sub>, a gradient impulse might effectively shift `k` by an amount proportional to `q`. For example, a gradient that shifts SQC by `&Delta;k=1` would shift DQC by `&Delta;k=2`.
    *   A common simplified EPG shift for MQC is to shift each F<sub>k,q</sub> to F<sub>k+1,q</sub>, effectively assuming the gradient strength is normalized for SQC, and the `q`-dependent dephasing is handled elsewhere or implicitly by how `k` is defined. More rigorous models (like those by Weigel) explicitly show `k` changing by `q * gradient_integral`.
    *   Longitudinal states Z<sub>k</sub> (implicitly q=0) are typically not affected by gradients in the same way as transverse MQC states, or their shift depends on the pathway through which they were created from transverse states. In many MQC EPG models, Z states are not shifted by gradients, or only the F states are explicitly shifted with their `k` index.

## Relaxation

*   Transverse MQC states (F<sub>k,q</sub>) decay with T2 relaxation. The T2 relaxation rate might differ for different quantum orders in some complex systems, but often a single T2 is assumed.
*   Longitudinal states (Z<sub>k</sub>) recover with T1 relaxation.

## Key Parameters for MQC EPG Simulations

In addition to basic EPG parameters (flip angles, phases, T1, T2, TR, etc.):

*   **`max_mqc_order`**: Defines the maximum quantum coherence order `q` to be simulated (e.g., `max_mqc_order = 2` would typically track q = -2, -1, +1, +2, and Z states for q=0). This determines the size of the MQC dimension in the `F` state matrix.
*   **RF pulse details**: Precise flip angles and phases are critical as they control the efficiency of MQC creation and conversion.
*   **Gradient information**: Strengths and timings of gradients are essential as MQC orders evolve differently under gradients.

Simulating MQC with EPG allows for the design and understanding of pulse sequences that can, for example, select for DQC signals (which might have different contrast properties than SQC) or use MQC pathways for specific purposes like background suppression.
