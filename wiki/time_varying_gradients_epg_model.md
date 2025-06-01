# EPG Model for Time-Varying Gradients

This document describes the Extended Phase Graph (EPG) model adapted for simulating MRI sequences with time-varying gradients. This capability allows for more accurate modeling of complex gradient waveforms that do not conform to simple rectangular pulse assumptions.

## Core Concept

The core EPG formalism is extended by allowing the gradient-induced phase accumulation and corresponding EPG state shifts to be calculated based on a defined gradient waveform, typically sampled over the repetition time (TR) or relevant gradient event duration.

## Parameters

The primary new parameter introduced in `epg_mri_time_varying_gradients.py` is:

*   `gradient_waveform`: A tensor or array-like structure defining the amplitude and timing of the gradient moments. The exact structure and interpretation (e.g., phase per TR, phase per dt) will depend on the implementation.
    *   *(Placeholder: Further details on waveform specification and units to be added here as the implementation is developed.)*

Other standard EPG parameters (T1, T2, flip angles, phases, TR, TE, B0, B1) remain applicable.

## Implementation Details

*(Placeholder: Details on how the `epg_shift` operation is modified to use the `gradient_waveform` will be described here. This might involve integrating the waveform over time intervals or applying discrete phase shifts corresponding to waveform samples.)*

## Example Usage

*(Placeholder: Example code snippets demonstrating how to define a `gradient_waveform` and use it with the `EPGSimulationTimeVaryingGradients` class will be provided here.)*

---

*This model is currently under development. The implementation in `epg_mri_time_varying_gradients.py` provides the basic structure, and the detailed time-varying gradient logic is yet to be fully implemented.*
