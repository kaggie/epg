import unittest
import torch
import sys
import os
import math

# Add vectorized-sims directory to sys.path for module import
current_dir = os.path.dirname(os.path.abspath(__file__))
vectorized_sims_dir = os.path.join(current_dir, '..', 'vectorized-sims')
sys.path.insert(0, vectorized_sims_dir)

MODULE_FOUND = False
IMPORT_ERROR = None

try:
    from epg_mri_time_varying_gradients import EPGSimulationTimeVaryingGradients
    MODULE_FOUND = True
except ImportError as e:
    IMPORT_ERROR = e

class TestEPGSimulationTimeVaryingGradients(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if MODULE_FOUND:
            print(f"Successfully imported EPGSimulationTimeVaryingGradients from {vectorized_sims_dir}.")
        else:
            print(f"Failed to import EPGSimulationTimeVaryingGradients. Error: {IMPORT_ERROR}")
            if IMPORT_ERROR and hasattr(IMPORT_ERROR, 'path'):
                 print(f"Module search path: {sys.path}")
                 print(f"Attempted import path: {IMPORT_ERROR.path}")

    def setUp(self):
        self.skipTestIfModuleNotFound()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_states_default = 21
        self.batch_size = 1 # For most basic tests

        # Standard parameters for a simple run
        self.T1 = torch.tensor([1000.0], device=self.device) # ms
        self.T2 = torch.tensor([100.0], device=self.device)  # ms
        self.TR = 500.0  # ms
        self.TE = 10.0   # ms, not directly used in basic EPG loop but good to have
        self.B0 = torch.tensor([0.0], device=self.device)    # Hz
        self.B1 = torch.tensor([1.0], device=self.device)    # Scaling factor

        # RF Pulses (e.g., one 90-degree pulse)
        self.n_pulses = 1
        self.flip_angles = torch.deg2rad(torch.tensor([90.0] * self.n_pulses, device=self.device))
        self.phases = torch.zeros(self.n_pulses, device=self.device)

    def skipTestIfModuleNotFound(self):
        if not MODULE_FOUND:
            self.skipTest(f"Module EPGSimulationTimeVaryingGradients not found. Skipping tests. Error: {IMPORT_ERROR}")

    def test_module_import(self):
        self.assertTrue(MODULE_FOUND, msg=f"Module import failed: {IMPORT_ERROR}")
        self.assertIsNone(IMPORT_ERROR, msg=f"Import error encountered: {IMPORT_ERROR}")

    def test_initialization(self):
        # Test default initialization
        model = EPGSimulationTimeVaryingGradients(device=self.device)
        self.assertIsNotNone(model, "Model should not be None after default initialization.")
        self.assertEqual(model.n_states, self.n_states_default, "Default n_states not set correctly.")
        self.assertIsNone(model.gradient_waveform, "Default gradient_waveform should be None.")

        # Test initialization with n_states
        custom_n_states = 15
        model_custom_states = EPGSimulationTimeVaryingGradients(n_states=custom_n_states, device=self.device)
        self.assertEqual(model_custom_states.n_states, custom_n_states, "Custom n_states not set correctly.")

        # Test initialization with a sample gradient_waveform
        sample_waveform = torch.tensor([1.0, -1.0, 0.0], device=self.device)
        model_with_waveform = EPGSimulationTimeVaryingGradients(gradient_waveform=sample_waveform, device=self.device)
        self.assertIsNotNone(model_with_waveform.gradient_waveform, "gradient_waveform should be stored.")
        self.assertTrue(torch.equal(model_with_waveform.gradient_waveform, sample_waveform),
                        "Stored gradient_waveform does not match input.")

    def test_simulation_basic_run(self):
        model = EPGSimulationTimeVaryingGradients(n_states=self.n_states_default, device=self.device)

        states = model(self.flip_angles, self.phases, self.T1, self.T2, self.TR, self.TE, self.B0, self.B1)

        self.assertIsNotNone(states, "Simulation output states should not be None.")
        self.assertEqual(len(states), self.n_pulses, "Number of state entries should match number of pulses.")

        Fp, Fm, Z = states[0] # Check the first state tuple
        self.assertIsInstance(Fp, torch.Tensor, "Fp should be a tensor.")
        self.assertIsInstance(Fm, torch.Tensor, "Fm should be a tensor.")
        self.assertIsInstance(Z, torch.Tensor, "Z should be a tensor.")

        expected_shape = (self.batch_size, self.n_states_default)
        self.assertEqual(Fp.shape, expected_shape, f"Fp shape incorrect. Expected {expected_shape}, got {Fp.shape}")
        self.assertEqual(Fm.shape, expected_shape, f"Fm shape incorrect. Expected {expected_shape}, got {Fm.shape}")
        self.assertEqual(Z.shape, expected_shape, f"Z shape incorrect. Expected {expected_shape}, got {Z.shape}")

    def test_simulation_with_gradient_waveform_placeholder(self):
        model = EPGSimulationTimeVaryingGradients(n_states=self.n_states_default, device=self.device)

        # Define a sample gradient waveform to be passed as an argument
        # The actual structure of this waveform will depend on how it's used in epg_shift.
        # For this placeholder test, its content doesn't matter as much as its presence.
        gradient_waveform_arg = torch.tensor([0.5, 0.5] * self.n_pulses, device=self.device)
                                         # Example: could be (n_pulses) or (n_pulses, n_gradient_points)
                                         # For the current epg_shift, it expects one value per pulse if used.

        states = model(self.flip_angles, self.phases, self.T1, self.T2, self.TR, self.TE,
                       self.B0, self.B1, gradient_waveform=gradient_waveform_arg)

        self.assertIsNotNone(states, "Simulation output states should not be None when gradient_waveform arg is passed.")
        self.assertEqual(len(states), self.n_pulses, "Number of state entries should match number of pulses.")

        Fp, Fm, Z = states[0]
        expected_shape = (self.batch_size, self.n_states_default)
        self.assertEqual(Fp.shape, expected_shape, "Fp shape incorrect with gradient_waveform arg.")
        self.assertEqual(Fm.shape, expected_shape, "Fm shape incorrect with gradient_waveform arg.")
        self.assertEqual(Z.shape, expected_shape, "Z shape incorrect with gradient_waveform arg.")

        # Test that providing None to forward also works (should use self.gradient_waveform, which might be None)
        model_with_default_waveform = EPGSimulationTimeVaryingGradients(
            gradient_waveform=torch.tensor([0.1]*self.n_pulses, device=self.device), device=self.device)
        states_none_arg = model_with_default_waveform(
            self.flip_angles, self.phases, self.T1, self.T2, self.TR, self.TE, self.B0, self.B1, gradient_waveform=None)
        self.assertIsNotNone(states_none_arg, "Simulation failed when gradient_waveform=None in forward and self.gradient_waveform exists.")
        self.assertEqual(len(states_none_arg), self.n_pulses)


if __name__ == '__main__':
    unittest.main()
