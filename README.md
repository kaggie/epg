# EPG: Extended Phase Graph Simulator for MRI

**epg** is a Python library for simulation of MRI signal evolution using the Extended Phase Graph (EPG) formalism. It supports single- and multi-pool models, magnetization transfer (MT), chemical exchange (CEST), multiple quantum coherences (MQC), and more. The package is designed for research and educational use in quantitative MRI, pulse sequence development, and relaxation/exchange studies.

---

## Scientific Context

The EPG formalism is a powerful and efficient tool to model the evolution of magnetization under the action of arbitrary RF pulse sequences, relaxation, exchange, gradients, and other physical effects in MR imaging. It enables simulation of complex sequences (spin echo, gradient echo, bSSFP, MT, CEST, etc.) and is foundational for sequence optimization, quantitative mapping, and understanding the effects of non-idealities in MRI.

Key references:
- Hennig J. Echoes—how to generate, recognize, use or avoid them in MR imaging sequences. Part I: Fundamental and not so fundamental properties of spin echoes. Concepts Magn Reson. 1991;3:125-143.
- Weigel M. Extended phase graphs: dephasing, RF pulses, and echoes—pure and simple. J Magn Reson Imaging. 2015;41(2):266-295.
- Zur Y, Stokar S, Bendel P. An analysis of fast imaging sequences with steady-state transverse magnetization refocusing. Magn Reson Med. 1988;6(2):175-193.
- Gloor M, Scheffler K, Bieri O. Quantitative magnetization transfer imaging using balanced SSFP. Magn Reson Med. 2008;60(3):691-700.

---

## Installation

**Dependencies:**  
- Python >= 3.8
- numpy
- matplotlib
- torch (for GPU support)

**Install via pip:**
```bash
pip install git+https://github.com/kaggie/epg.git
```

**Or clone and install locally:**
```bash
git clone https://github.com/kaggie/epg.git
cd epg
pip install -e .
```

---

## Usage Example

### Basic EPG Simulation

```python
import torch
from epg_mri import EPGSimulation

n_pulses = 10
flip_angles = torch.ones(n_pulses) * torch.deg2rad(torch.tensor(90.0))
phases = torch.zeros(n_pulses)
T1, T2 = 1000.0, 80.0  # ms
TR, TE = 500.0, 20.0   # ms

epg = EPGSimulation(n_states=21)
states = epg(flip_angles, phases, T1, T2, TR, TE)

for i, (Fp, Fm, Z) in enumerate(states):
    print(f"Pulse {i+1}: Fp={Fp[0].real:.4f}, Z={Z[0]:.4f}")
```

### Magnetization Transfer (MT) Simulation

```python
from epg_mri_mt import EPGSimulationMT

epg_mt = EPGSimulationMT(n_states=21)
states = epg_mt(
    flip_angles, phases,
    T1f=1000, T2f=80, T1b=1000, T2b=10,
    kf=3.0, kb=6.0,
    TR=500, TE=20,
    wf=0.9, wb=0.1
)
```

### GPU Acceleration

```python
epg = EPGSimulationMT(n_states=21, device='cuda')
# All tensors and computation will run on GPU if available
```

### Pulse Sequence Visualization

```python
from epg_plotting_tools import plot_pulse_sequence
plot_pulse_sequence(flip_angles, phases, TR=5.0)
```

---

## API Documentation

- **EPGSimulation**: Standard EPG for single-pool systems.
    - `forward(flip_angles, phases, T1, T2, TR, TE, B0=0., B1=1.0)`
- **EPGSimulationMT**: Two-pool Magnetization Transfer simulation.
    - `forward(flip_angles, phases, T1f, T2f, T1b, T2b, kf, kb, TR, TE, B0=0., B1=1.0, wf=1.0, wb=0.1)`
- **EPGSimulationMQC**: EPG with Multiple Quantum Coherences.
    - `forward(flip_angles, phases, T1, T2, TR, TE, B0=0., B1=1.0)`
- **plot_epg_evolution, plot_pulse_sequence**: Visualization utilities.
