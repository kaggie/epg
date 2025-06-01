# Basic Extended Phase Graph (EPG) Model

## Introduction

The Extended Phase Graph (EPG) algorithm is a powerful tool in Magnetic Resonance Imaging (MRI) for simulating the behavior of magnetization in response to a sequence of radiofrequency (RF) pulses and magnetic field gradients. Unlike traditional Bloch equation simulations that track the bulk magnetization vector, EPG models the distribution of magnetization into different "states" based on their phase history. This makes EPG particularly efficient for understanding and predicting signal evolution in sequences with complex gradient waveforms, such as gradient echo and spin echo sequences, and for analyzing effects like stimulated echoes.

## Core Components: EPG States

The EPG model describes the magnetization in terms of three main types of states, typically represented as vectors or coefficients:

1.  **F+ States (F<sub>k</sub><sup>+</sup>):** These represent transverse magnetization states that have a positive dephasing order `k`. The order `k` (an integer) indicates how many full cycles of dephasing (e.g., 2&pi; radians) a particular magnetization component has experienced due to gradients. F+ states are often associated with "spin echo" pathways. `F_0^+` is the transverse magnetization that has not yet dephased due to gradients.

2.  **F- States (F<sub>k</sub><sup>-</sup>):** These represent transverse magnetization states that have a negative dephasing order `k`. They are the complex conjugates of the F+ states if only considering simple dephasing. F- states are also crucial for echo formation. `F_0^-` is the conjugate of `F_0^+`.

3.  **Z States (Z<sub>k</sub>):** These represent longitudinal magnetization states. `Z_0` is the primary longitudinal magnetization component (M<sub>z</sub>). Higher-order Z states (Z<sub>k</sub> for k > 0) can arise in sequences with multiple RF pulses and represent longitudinal magnetization that has been dephased and then stored along the longitudinal axis. These are important for understanding phenomena like stimulated echoes.

Each state `F_k^+`, `F_k^-`, and `Z_k` holds a complex coefficient representing the magnitude and phase of that specific magnetization configuration. The index `k` is the "spatial order" or "coherence order," indicating the spatial frequency of the magnetization along the gradient axis.

## Key Operations

The EPG simulation evolves these states through a series of operations corresponding to events in an MRI pulse sequence:

1.  **RF Pulses:**
    *   An RF pulse rotates the magnetization. In the EPG model, this means it redistributes magnetization between F+, F-, and Z states.
    *   The effect of an RF pulse with flip angle &alpha; and phase &phi; is described by a rotation matrix that mixes coefficients between different states and orders. For example, an RF pulse can convert Z<sub>0</sub> magnetization into F<sub>0</sub><sup>+</sup> and F<sub>0</sub><sup>-</sup> states, and it can also convert existing F states into Z states or other F states, potentially changing their coherence order `k`.

2.  **Relaxation:**
    *   **T1 (Longitudinal) Relaxation:** Affects the Z states, causing them to return towards their equilibrium value (typically Z<sub>0</sub> recovering towards M<sub>0</sub>).
    *   **T2 (Transverse) Relaxation:** Affects the F+ and F- states, causing them to decay.
    *   Relaxation is applied to all states, with different rates for longitudinal (T1) and transverse (T2) components. `E1 = exp(-TR/T1)` and `E2 = exp(-TR/T2)` are commonly used decay factors over a period TR.

3.  **Gradient Dephasing (Shifting):**
    *   Application of a gradient causes magnetization with transverse components to acquire phase based on its spatial position. In EPG, this is modeled as a "shift" operation on the F+ and F- states.
    *   A gradient causing one full dephasing cycle (2&pi;) across a voxel increments the order `k` of F+ states by one (F<sub>k</sub><sup>+</sup> &rarr; F<sub>k+1</sub><sup>+</sup>) and decrements the order `k` of F- states by one (F<sub>k</sub><sup>-</sup> &rarr; F<sub>k-1</sub><sup>-</sup>).
    *   The Z states are also typically shifted (Z<sub>k</sub> &rarr; Z<sub>k+1</sub>), representing the dephasing of any transverse magnetization that was subsequently stored as longitudinal magnetization.
    *   The state F<sub>0</sub><sup>+</sup> (freshly created transverse magnetization from Z<sub>0</sub> by an RF pulse) is moved to F<sub>1</sub><sup>+</sup>. Similarly, F<sub>0</sub><sup>-</sup> moves to F<sub>-1</sub><sup>-</sup>. The new F<sub>0</sub><sup>+</sup> and F<sub>0</sub><sup>-</sup> states are typically set to zero after a gradient unless immediately repopulated by an RF pulse acting on Z<sub>0</sub>. The signal echo is usually observed when dephasing causes population to return to the F<sub>0</sub><sup>+</sup> or F<sub>0</sub><sup>-</sup> state (e.g., an F<sub>1</sub><sup>+</sup> state being shifted by a negative gradient back to F<sub>0</sub><sup>+</sup>).

## Input Parameters for Basic EPG Simulation

A typical basic EPG simulation requires the following parameters:

*   **`flip_angles`**: A sequence (array or list) of RF pulse flip angles in radians. Each element corresponds to an RF pulse in the sequence.
*   **`phases`**: A sequence of RF pulse phases in radians, corresponding to each flip angle.
*   **`T1`**: Longitudinal relaxation time (in milliseconds).
*   **`T2`**: Transverse relaxation time (in milliseconds).
*   **`TR`**: Repetition Time (in milliseconds), the time interval between successive RF pulses or blocks of pulses. Relaxation and state evolution are often calculated per TR.
*   **`TE`**: Echo Time (in milliseconds). While not always directly used in the EPG state evolution step-by-step, it's crucial for determining when the signal (often F<sub>0</sub><sup>+</sup> or F<sub>0</sub><sup>-</sup>) is "readout" or observed.
*   **`B0` (optional)**: Off-resonance frequency in Hz. This causes an additional phase accrual for transverse states (F+ and F-) during time intervals (like TR or TE).
*   **`B1` (optional)**: B1 scaling factor (unitless). Represents imperfections in the RF pulse amplitude, effectively scaling the applied flip angles. A B1 of 1.0 means a perfect pulse.

## Output

The primary output of an EPG simulation is the sequence of EPG states (the set of F+, F-, and Z coefficients) at each step of the pulse sequence (e.g., after each RF pulse and subsequent gradient/relaxation period).
Typically, the transverse magnetization in the F<sub>0</sub><sup>+</sup> state (or sometimes F<sub>0</sub><sup>-</sup>) at the desired echo time (TE) is used to calculate the MRI signal.

By tracking these states, EPG allows for accurate simulation of complex MRI sequences and the prediction of signal amplitudes and characteristics.
