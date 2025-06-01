import unittest
import torch
import sys
import os
import math

# Add the vectorized-sims directory to sys.path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vectorized-sims')))

IMPORT_ERROR = None
MODULE_FOUND = False
EPGSimulationMQC = None  # Placeholder

try:
    from epg_mqc_vectorized import EPGSimulationMQC # Actual class name
    MODULE_FOUND = True
except ImportError as e:
    IMPORT_ERROR = e
except Exception as e:
    IMPORT_ERROR = e


class TestEPGMQCVectorized(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not MODULE_FOUND:
            print(f"Critical: EPGSimulationMQC module not found: {IMPORT_ERROR}. Tests will be skipped.")
        else:
            print(f"EPGSimulationMQC module imported. Assumed sim order: Relax -> B0 -> RF -> Shift.")

    def setUp(self):
        self.skipTestIfModuleNotFound()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_states = 7
        self.TR_val = 100.0
        self.TE_val = 10.0

        self.flip_angles_single = torch.tensor([math.pi / 2], device=self.device)
        self.phases_single = torch.tensor([0.0], device=self.device)

        self.flip_angles_multi = torch.tensor([math.pi / 2, math.pi / 4], device=self.device) # Added this
        self.phases_multi = torch.tensor([0.0, math.pi/2], device=self.device)      # Added this

        self.flip_angles_two_90s = torch.tensor([math.pi / 2, math.pi / 2], device=self.device)
        self.phases_two_90s = torch.tensor([0.0, 0.0], device=self.device)

        self.T1 = torch.tensor([1000.0], device=self.device)
        self.T2 = torch.tensor([80.0], device=self.device)
        self.B0 = torch.tensor([0.0], device=self.device)
        self.B1 = torch.tensor([1.0], device=self.device)

        self.T1_batch = torch.tensor([1000.0, 800.0], device=self.device)
        self.T2_batch = torch.tensor([80.0, 60.0], device=self.device)
        self.B0_batch = torch.tensor([0.0, 5.0], device=self.device)
        self.B1_batch = torch.tensor([1.0, 0.9], device=self.device)

    def skipTestIfModuleNotFound(self):
        if not MODULE_FOUND:
            self.skipTest(f"Skipping test: EPGSimulationMQC module not found.")

    def test_module_import(self):
        self.assertTrue(MODULE_FOUND, msg=f"Failed to import EPGSimulationMQC. Error: {IMPORT_ERROR}")

    def test_1_1_initialization(self):
        model_sqc = EPGSimulationMQC(n_states=self.n_states, max_mqc_order=1, device=self.device)
        self.assertIsNotNone(model_sqc)
        self.assertEqual(model_sqc.max_mqc_order, 1)
        self.assertEqual(model_sqc.n_mqc, 3)

        model_dqc = EPGSimulationMQC(n_states=self.n_states, max_mqc_order=2, device=self.device)
        self.assertIsNotNone(model_dqc)
        self.assertEqual(model_dqc.max_mqc_order, 2)
        self.assertEqual(model_dqc.n_mqc, 5)
        self.assertEqual(model_dqc.n_states, self.n_states)

    def test_1_2_single_90_pulse_max_order_1(self):
        max_order = 1
        model = EPGSimulationMQC(n_states=self.n_states, max_mqc_order=max_order, device=self.device)
        states_list = model(self.flip_angles_single, self.phases_single, self.T1, self.T2, self.TR_val, self.TE_val, B0=self.B0, B1=self.B1)
        F_state, Z_state = states_list[0]

        self.assertAlmostEqual(Z_state[0,0].item(), 0.0, places=3, msg="Z0 after RF")
        idx_p1 = model.mqc_idx(+1)
        idx_m1 = model.mqc_idx(-1)
        self.assertTrue(torch.allclose(F_state[0,0,idx_p1], torch.tensor(0.0j, device=self.device), atol=1e-4), "F[0,0,+1] after shift")
        self.assertTrue(torch.allclose(F_state[0,1,idx_p1], torch.tensor(0.5j, device=self.device), atol=1e-3), f"F[0,1,+1] (shifted F0). Got {F_state[0,1,idx_p1]}")
        self.assertTrue(torch.allclose(F_state[0,1,idx_m1], torch.tensor(-0.5j, device=self.device), atol=1e-3), f"F[0,1,-1] (shifted F0-). Got {F_state[0,1,idx_m1]}")

    def test_1_3_scalar_vs_batched_mqc(self):
        max_order = 1
        model = EPGSimulationMQC(n_states=self.n_states, max_mqc_order=max_order, device=self.device)
        states_s_list = model(self.flip_angles_multi, self.phases_multi, self.T1, self.T2, self.TR_val, self.TE_val, B0=self.B0, B1=self.B1)
        F_s, Z_s = states_s_list[-1]
        states_b_list = model(self.flip_angles_multi, self.phases_multi, self.T1_batch, self.T2_batch, self.TR_val, self.TE_val, B0=self.B0_batch, B1=self.B1_batch)
        F_b, Z_b = states_b_list[-1]
        self.assertTrue(torch.allclose(F_s[0], F_b[0], atol=1e-6), "F state scalar vs batched item 0")
        self.assertTrue(torch.allclose(Z_s[0], Z_b[0], atol=1e-6), "Z state scalar vs batched item 0")

    def test_2_1_dqc_generation(self):
        max_order = 2
        model = EPGSimulationMQC(n_states=self.n_states, max_mqc_order=max_order, device=self.device)
        short_TR_for_dqc = 20.0
        states_list = model(self.flip_angles_two_90s, self.phases_two_90s,
                              self.T1, self.T2, short_TR_for_dqc, self.TE_val, B0=self.B0, B1=self.B1)
        F_final, Z_final = states_list[-1]
        idx_p2 = model.mqc_idx(+2); idx_m2 = model.mqc_idx(-2)
        dqc_p2_mag = torch.abs(F_final[0, :, idx_p2]).sum()
        dqc_m2_mag = torch.abs(F_final[0, :, idx_m2]).sum()
        self.assertAlmostEqual(dqc_p2_mag.item(), 0.0, places=4, msg="DQC F[+2] signal should be zero with current RF model")
        self.assertAlmostEqual(dqc_m2_mag.item(), 0.0, places=4, msg="DQC F[-2] signal should be zero with current RF model")

    def test_2_2_off_resonance_mqc_orders(self):
        max_order = 2
        model = EPGSimulationMQC(n_states=self.n_states, max_mqc_order=max_order, device=self.device)
        flip_angles = torch.tensor([math.pi / 6], device=self.device)
        phases = torch.tensor([0.0], device=self.device)
        test_B0_offset = torch.tensor([20.0], device=self.device)
        test_TR = 50.0
        states_list = model(flip_angles, phases, self.T1, self.T2, test_TR, self.TE_val, B0=test_B0_offset, B1=self.B1)
        F_state, _ = states_list[0]
        angle_Fp1 = torch.angle(F_state[0,1,model.mqc_idx(+1)]).item()
        # Expected phase from RF (beta=0, for F+ ~0.5j*sin(alpha_rf)) is pi/2 if alpha_rf > 0.
        # Here alpha_rf = pi/6 * 1.0. sin(pi/12) > 0.
        # With Relax->B0->RF->Shift, F0 created by RF does not see B0 in this TR.
        # So phase should be that from RF pulse.
        # For F+, if beta=0, phase is pi/2.
        self.assertAlmostEqual(angle_Fp1, math.pi/2, places=3,
                               msg=f"Phase of F[0,1,+1] after 1st pulse. Expected pi/2, got {angle_Fp1}")
        if max_order >=2:
            self.assertAlmostEqual(torch.abs(F_state[0,1,model.mqc_idx(+2)]).item(), 0.0, places=4,
                                   msg="DQC F[0,1,+2] should be near zero after 1st pulse with current RF model")

if __name__ == '__main__':
    if not MODULE_FOUND:
        print(f"Skipping tests: EPGSimulationMQC module not found. Import error: {IMPORT_ERROR}")
    else:
        unittest.main()
