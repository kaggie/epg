import unittest
import torch
import sys
import os
import math

# Add the vectorized-sims directory to sys.path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vectorized-sims')))

IMPORT_ERROR = None
MODULE_FOUND = False
EPGSimulation = None
SCRIPT_VERSION_LOADED = None

try:
    # Import the module itself to access module-level variables
    import epg_mri_vectorized as epg_mri_module_to_test
    EPGSimulation = epg_mri_module_to_test.EPGSimulation # Get the class
    MODULE_FOUND = True
    try:
        SCRIPT_VERSION_LOADED = epg_mri_module_to_test.SCRIPT_VERSION_INFO
    except AttributeError:
        SCRIPT_VERSION_LOADED = "Not found"
except ImportError as e:
    IMPORT_ERROR = e
    SCRIPT_VERSION_LOADED = "Module import failed"
except Exception as e:
    IMPORT_ERROR = e
    SCRIPT_VERSION_LOADED = f"Other import error: {e}"


class TestEPGSimulationVectorized(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"Attempting to load EPGSimulation module...")
        if not MODULE_FOUND:
            print(f"Critical: EPGSimulation module not found due to: {IMPORT_ERROR}. Most tests will be skipped.")
        else:
            print(f"EPGSimulation module imported successfully.")
            print(f"SCRIPT_VERSION_INFO found in imported module: '{SCRIPT_VERSION_LOADED}'")
            print(f"Tests are written expecting operational order: RF->Relax->B0->Shift (consistent with v2).")


    def setUp(self):
        self.skipTestIfModuleNotFound()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_states = 5
        self.epg_model = EPGSimulation(n_states=self.n_states, device=self.device)

        self.T1_val = 1000.0
        self.T2_val = 100.0
        self.TR_val = 500.0
        self.short_TR_val = 100.0
        self.TE_val = 20.0
        self.B0_val = 0.0
        self.B1_val = 1.0

        self.T1 = torch.tensor([self.T1_val], device=self.device)
        self.T2 = torch.tensor([self.T2_val], device=self.device)
        self.B0 = torch.tensor([self.B0_val], device=self.device)
        self.B1 = torch.tensor([self.B1_val], device=self.device)

        self.T1_batch = torch.tensor([self.T1_val, self.T1_val - 200.0], device=self.device)
        self.T2_batch = torch.tensor([self.T2_val, self.T2_val - 20.0], device=self.device)
        self.B0_batch = torch.tensor([self.B0_val, 5.0], device=self.device)
        self.B1_batch = torch.tensor([self.B1_val, 0.9], device=self.device)

    def skipTestIfModuleNotFound(self):
        if not MODULE_FOUND:
            self.skipTest(f"Skipping test: EPGSimulation module not found.")

    def test_module_import_and_version(self):
        """Test module import and presence of SCRIPT_VERSION_INFO."""
        self.assertTrue(MODULE_FOUND, msg=f"Failed to import EPGSimulation. Error: {IMPORT_ERROR}")
        self.assertIsNotNone(SCRIPT_VERSION_LOADED, "SCRIPT_VERSION_LOADED should be set.")
        self.assertEqual(SCRIPT_VERSION_LOADED, "epg_mri_v2_RF_first",
                         f"SCRIPT_VERSION_INFO mismatch. Expected 'epg_mri_v2_RF_first', got '{SCRIPT_VERSION_LOADED}'")


    def test_initialization(self):
        self.assertIsNotNone(self.epg_model)
        self.assertEqual(self.epg_model.n_states, self.n_states)

    def test_single_90_degree_pulse(self):
        flip_angles = torch.tensor([math.pi / 2], device=self.device)
        phases = torch.tensor([0.0], device=self.device)
        states = self.epg_model(flip_angles, phases, self.T1, self.T2, self.TR_val, self.TE_val, B0=self.B0, B1=self.B1)
        Fp_actual, Fm_actual, Z_actual = states[0]

        E1_TR = math.exp(-self.TR_val / self.T1_val)
        E2_TR = math.exp(-self.TR_val / self.T2_val)

        # After RF (Z0=0, F0=0.5j) -> Relax (Z0=1-E1, F0=0.5j*E2) -> B0 (no change if B0=0) -> Shift (Z not shifted, F0 shifted to F1, new F0=0)
        expected_Z0_final = 1.0 - E1_TR
        expected_F1_complex = 0.5j * E2_TR

        self.assertAlmostEqual(Z_actual[0,0].item(), expected_Z0_final, places=3,
                               msg=f"Z[0,0] (Z0_postRF_relax). Expected {expected_Z0_final:.3f}, Got {Z_actual[0,0].item()}")
        self.assertTrue(torch.allclose(Fp_actual[0,0], torch.tensor(0.0j, device=self.device), atol=1e-4), msg="Fp[0,0] after shift should be 0")
        self.assertAlmostEqual(Z_actual[0,1].item(), 0.0, places=4,
                               msg=f"Z[0,1] (Z1 state should be 0 as Z not shifted). Got {Z_actual[0,1].item()}")
        self.assertTrue(torch.allclose(Fp_actual[0,1], torch.tensor(expected_F1_complex, device=self.device, dtype=torch.cfloat), atol=1e-3),
                        msg=f"Fp[0,1] (shifted F0_postRF_relax_B0). Expected {expected_F1_complex:.4f}, Got {Fp_actual[0,1]}")

    def test_scalar_vs_batched_equivalence(self):
        flip_angles = torch.tensor([math.pi / 2, math.pi / 4], device=self.device)
        phases = torch.tensor([0.0, math.pi / 2], device=self.device)
        states_s = self.epg_model(flip_angles, phases, self.T1, self.T2, self.TR_val, self.TE_val, B0=self.B0, B1=self.B1)
        Fp_s_final, _, Z_s_final = states_s[-1]

        T1_b = torch.tensor([self.T1_val, self.T1_val - 200], device=self.device)
        T2_b = torch.tensor([self.T2_val, self.T2_val - 20], device=self.device)
        B0_b = torch.tensor([self.B0_val, 5.0], device=self.device)
        B1_b = torch.tensor([self.B1_val, 0.9], device=self.device)

        states_bn = self.epg_model(flip_angles, phases, T1_b, T2_b, self.TR_val, self.TE_val, B0=B0_b, B1=B1_b)
        Fp_bn_final, _, Z_bn_final = states_bn[-1]

        self.assertTrue(torch.allclose(Fp_s_final[0], Fp_bn_final[0], atol=1e-6))
        self.assertTrue(torch.allclose(Z_s_final[0], Z_bn_final[0], atol=1e-6))

    def test_t1_relaxation(self):
        flip_angles = torch.tensor([math.pi / 2], device=self.device)
        phases = torch.tensor([0.0], device=self.device)
        long_TR = self.T1_val * 5
        states = self.epg_model(flip_angles, phases, self.T1, self.T2, long_TR, self.TE_val, B0=self.B0, B1=self.B1)
        _, _, Z_final = states[0]
        E1_calc = torch.exp(-long_TR / self.T1[0])
        # After RF (Z0=0) -> Relax (Z0=1-E1_long_TR) -> B0 -> Shift (Z not shifted)
        expected_Z0_recovered = (1.0 - E1_calc.item())
        self.assertAlmostEqual(Z_final[0,0].item(), expected_Z0_recovered, places=3,
                               msg=f"Z[0,0] (recovered Z0). Expected {expected_Z0_recovered:.3f}, got {Z_final[0,0].item():.3f}")
        self.assertAlmostEqual(Z_final[0,1].item(), 0.0, places=3, msg="Z[0,1] (Z1 state should be 0 as Z not shifted).")

    def test_t2_relaxation(self):
        flip_angles = torch.tensor([math.pi / 2], device=self.device)
        phases = torch.tensor([0.0], device=self.device)
        states = self.epg_model(flip_angles, phases, self.T1, self.T2, self.TR_val, self.TE_val, B0=self.B0, B1=self.B1)
        Fp_final, _, _ = states[0]
        F0_postRF_mag = 0.5
        E2_calc = torch.exp(-self.TR_val / self.T2[0])
        expected_Fp1_mag = F0_postRF_mag * E2_calc.item()
        self.assertAlmostEqual(torch.abs(Fp_final[0,1]).item(), expected_Fp1_mag, places=3,
                               msg=f"Fp[0,1] magnitude after TR. Expected ~{expected_Fp1_mag:.3f}, got {torch.abs(Fp_final[0,1]).item():.3f}")

    def test_off_resonance_effect(self):
        flip_angles = torch.tensor([math.pi / 4], device=self.device)
        phases = torch.tensor([0.0], device=self.device)
        test_B0_offset_val = 5.0
        test_TR_val = self.short_TR_val
        test_B0_param = torch.tensor([test_B0_offset_val], device=self.device)

        states_ref = self.epg_model(flip_angles, phases, self.T1, self.T2, test_TR_val, self.TE_val, B0=self.B0, B1=self.B1)
        Fp_ref, _, _ = states_ref[0]
        states_offres = self.epg_model(flip_angles, phases, self.T1, self.T2, test_TR_val, self.TE_val, B0=test_B0_param, B1=self.B1)
        Fp_offres, _, _ = states_offres[0]
        expected_b0_phase_shift = (2 * math.pi * test_B0_offset_val * test_TR_val / 1000.0)

        # Phase of F0 created by RF (alpha=pi/4, beta=0) on Z0=1:
        # Fp_new = ... + 1j * cos(alpha/2) * sin(alpha/2) * Z
        # Initial phase is pi/2.
        phase_RF = math.pi/2
        angle_ref = torch.angle(Fp_ref[0,1]).item() if torch.abs(Fp_ref[0,1]) > 1e-6 else phase_RF
        angle_offres = torch.angle(Fp_offres[0,1]).item() if torch.abs(Fp_offres[0,1]) > 1e-6 else phase_RF
        actual_phase_diff = (angle_offres - angle_ref)
        # Normalize phase difference to [-pi, pi]
        actual_phase_diff = (actual_phase_diff + math.pi) % (2 * math.pi) - math.pi

        # Allow for positive or negative phase shift of the same magnitude
        if not (math.isclose(actual_phase_diff, expected_b0_phase_shift, abs_tol=1e-3) or \
                math.isclose(actual_phase_diff, -expected_b0_phase_shift, abs_tol=1e-3)):
            self.fail(msg=f"Phase difference in Fp1. Expected magnitude {expected_b0_phase_shift:.3f}, got {actual_phase_diff:.3f}")

if __name__ == '__main__':
    # This top-level print will show up during test discovery if directly run
    print(f"Attempting to run tests for EPG MRI Vectorized. Loaded SCRIPT_VERSION_INFO: '{SCRIPT_VERSION_LOADED}'")
    if not MODULE_FOUND:
        print(f"Critical: EPGSimulation module not found. Cannot run tests. Import error: {IMPORT_ERROR}")

    unittest.main()
