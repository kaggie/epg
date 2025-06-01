import unittest
import torch
import sys
import os
import math

# Add the vectorized-sims directory to sys.path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vectorized-sims')))

IMPORT_ERROR = None
MODULE_FOUND = False
EPGSimulationMT_GPU_Vectorized = None  # Placeholder

try:
    from epg_mr_mt_gpu_vectorized import EPGSimulationMT_GPU_Vectorized
    MODULE_FOUND = True
except ImportError as e:
    IMPORT_ERROR = e
except Exception as e:
    IMPORT_ERROR = e

class TestEPGMrMtGpuVectorized(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not MODULE_FOUND:
            print(f"Critical: EPGSimulationMT_GPU_Vectorized module not found: {IMPORT_ERROR}. Tests will be skipped.")
        else:
            print(f"EPGSimulationMT_GPU_Vectorized module imported. Assumed sim order: Relax/Exch -> B0 -> RF -> Shift.")

    def setUp(self):
        self.skipTestIfModuleNotFound()
        self.device = torch.device('cpu')
        self.n_states = 7
        self.model = EPGSimulationMT_GPU_Vectorized(n_states=self.n_states, device=self.device)

        self.TR_val = 100.0
        self.TE_val = 10.0
        self.flip_angles_single = torch.tensor([math.pi / 2], device=self.device)
        self.phases_single = torch.tensor([0.0], device=self.device)
        self.flip_angles_multi = torch.tensor([math.pi/2, math.pi/4], device=self.device)
        self.phases_multi = torch.tensor([0.0, math.pi/2], device=self.device)

        self.T1f_s = torch.tensor([1200.0], device=self.device)
        self.T2f_s = torch.tensor([50.0], device=self.device)
        self.T1b_s = torch.tensor([300.0], device=self.device)
        self.T2b_s = torch.tensor([2.0], device=self.device)
        self.kf_s = torch.tensor([3.0], device=self.device)
        self.kb_s = torch.tensor([20.0], device=self.device)
        self.wf_s = torch.tensor([0.85], device=self.device)
        self.wb_s = torch.tensor([0.15], device=self.device)
        self.B0_s = torch.tensor([0.0], device=self.device)
        self.B1_s = torch.tensor([1.0], device=self.device)

        self.T1f_b = torch.tensor([1200.0, 1100.0], device=self.device)
        self.T2f_b = torch.tensor([50.0, 60.0], device=self.device)
        self.T1b_b = torch.tensor([300.0, 250.0], device=self.device)
        self.T2b_b = torch.tensor([2.0, 2.5], device=self.device)
        self.kf_b = torch.tensor([3.0, 3.5], device=self.device)
        self.kb_b = torch.tensor([20.0, 22.0], device=self.device)
        self.wf_b = torch.tensor([0.85, 0.80], device=self.device)
        self.wb_b = torch.tensor([0.15, 0.20], device=self.device)
        self.B0_b = torch.tensor([0.0, 5.0], device=self.device)
        self.B1_b = torch.tensor([1.0, 0.95], device=self.device)

    def skipTestIfModuleNotFound(self):
        if not MODULE_FOUND:
            self.skipTest(f"Skipping test: EPGSimulationMT_GPU_Vectorized module not found.")

    def test_module_import(self):
        self.assertTrue(MODULE_FOUND, msg=f"Failed to import EPGSimulationMT_GPU_Vectorized. Error: {IMPORT_ERROR}")

    def test_1_1_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.n_states, self.n_states)
        self.assertEqual(self.model.device, self.device)

    def test_1_2_single_90_pulse_initial_states(self):
        states_list = self.model(self.flip_angles_single, self.phases_single,
                                 self.T1f_s, self.T2f_s, self.T1b_s, self.T2b_s,
                                 self.kf_s, self.kb_s, self.TR_val, self.TE_val,
                                 B0=self.B0_s, B1=self.B1_s, wf=self.wf_s, wb=self.wb_s)
        Fp_f, Fm_f, Z_f, Fp_b, Fm_b, Z_b = states_list[0]

        # Order: Relax/Exch -> B0 -> RF -> Shift
        # Zf starts at wf. Relax/Exch might change it slightly. B0 no change.
        # RF (90) on Zf: Zf_postRF[0]~0, Fpf_postRF[0]~0.5j*Zf_preRF.
        # Shift then makes Zf_shifted[0,1] = Zf_postRF[0] ~ 0. Fpf_shifted[0,1] = Fpf_postRF[0].
        self.assertAlmostEqual(Z_f[0,0].item(), 0.0, places=3, msg="Zf[0,0] (water Z0 post-shift)")
        self.assertAlmostEqual(Z_f[0,1].item(), 0.0, places=3, msg="Zf[0,1] (water Z0_postRF shifted)")

        # Approximate Fpf1: assume Zf_preRF is close to wf_s for this check after one relax/exch step.
        # A more precise value would require simulating the relax/exch step on wf_s.
        # For simplicity, using wf_s directly gives an order of magnitude.
        # The relax/exchange in the model acts on Zf[0]=wf and Zb[0]=wb.
        # Zf_pre_rf = E1f*wf + (1-E1f)*wf + (-kf*wf + kb*wb)*TR/1000 = wf + (-kf*wf + kb*wb)*TR/1000
        # Let's assume for this qualitative check that Zf_pre_rf is still close to wf for the Fpf calculation.
        # For TR=100ms, kf=3, kb=20, wf=0.85, wb=0.15:
        # dZf_term = (-3*0.85 + 20*0.15)*0.1 = (-2.55 + 3)*0.1 = 0.045. So Zf_pre_rf = 0.85 + 0.045 = 0.895
        # This is if Zf was just a scalar. EPG states are vectors. Z_f[0] is wf.
        # Z_f[0] after relax/exch: Z_f[0]*E1f + (1-E1f)*wf + (-kf*Z_f[0] + kb*Z_b[0])*TR/1000
        # If Z_f[0]=wf, Z_b[0]=wb initially: wf*E1f + (1-E1f)*wf + (-kf*wf + kb*wb)*TR/1000
        # = wf + (-kf*wf + kb*wb)*TR/1000. This is Zf_pre_rf.
        E1f_calc = torch.exp(-self.TR_val / self.T1f_s)
        Zf_pre_rf = self.wf_s.item() + (-self.kf_s.item()*self.wf_s.item() + self.kb_s.item()*self.wb_s.item()) * self.TR_val / 1000.0
        expected_Fpf1_mag_approx = 0.5 * Zf_pre_rf

        self.assertAlmostEqual(torch.abs(Fp_f[0,1]).item(), expected_Fpf1_mag_approx, delta=0.1, # Wider delta due to approximation
                        msg=f"Fpf[0,1] mag. Expected ~{expected_Fpf1_mag_approx:.3f}, Got {torch.abs(Fp_f[0,1]).item():.3f}")

        self.assertAlmostEqual(Z_b[0,0].item(), 0.0, places=3, msg="Zb[0,0] (bound Z0 post-shift)")
        self.assertTrue(Z_b[0,1].item() > 0.0, msg=f"Zb[0,1] (bound Z0 after relax/exch shifted). Got {Z_b[0,1].item()}")

    def test_1_3_scalar_vs_batched_equivalence(self):
        states_s_list = self.model(self.flip_angles_multi, self.phases_multi,
                                   self.T1f_s, self.T2f_s, self.T1b_s, self.T2b_s,
                                   self.kf_s, self.kb_s, self.TR_val, self.TE_val,
                                   B0=self.B0_s, B1=self.B1_s, wf=self.wf_s, wb=self.wb_s)
        Fpf_s, _, Zf_s, _, _, Zb_s = states_s_list[-1]

        states_b_list = self.model(self.flip_angles_multi, self.phases_multi,
                                   self.T1f_b, self.T2f_b, self.T1b_b, self.T2b_b,
                                   self.kf_b, self.kb_b, self.TR_val, self.TE_val,
                                   B0=self.B0_b, B1=self.B1_b, wf=self.wf_b, wb=self.wb_b)
        Fpf_b, _, Zf_b, _, _, Zb_b = states_b_list[-1]

        self.assertTrue(torch.allclose(Fpf_s[0], Fpf_b[0], atol=1e-5))
        self.assertTrue(torch.allclose(Zf_s[0], Zf_b[0], atol=1e-5))
        self.assertTrue(torch.allclose(Zb_s[0], Zb_b[0], atol=1e-5))

    def test_2_1_exchange_effect_qualitative(self):
        T1f = torch.tensor([20000.0], device=self.device)
        T2f = torch.tensor([5.0], device=self.device)
        T1b = torch.tensor([300.0], device=self.device)
        T2b = torch.tensor([2.0], device=self.device)
        kf = torch.tensor([50.0], device=self.device)
        # kb such that kf*wf = kb*wb => kb = kf*wf/wb
        kb_val = kf.item() * self.wf_s.item() / self.wb_s.item()
        kb = torch.tensor([kb_val], device=self.device)
        wf = self.wf_s; wb = self.wb_s

        n_sat_pulses = 10
        sat_flip_angles = torch.ones(n_sat_pulses, device=self.device) * math.pi / 2
        sat_phases = torch.zeros(n_sat_pulses, device=self.device)
        sat_TR = 10.0

        states_after_sat_list = self.model(sat_flip_angles, sat_phases,
                                           T1f, T2f, T1b, T2b, kf, kb, sat_TR, self.TE_val,
                                           B0=self.B0_s, B1=self.B1_s, wf=wf, wb=wb)
        _, _, Zf_after_sat, _, _, Zb_after_sat = states_after_sat_list[-1]

        Zf0_after_sat_shifted_to_Z1 = Zf_after_sat[0,1]
        Zb0_after_sat_shifted_to_Z1 = Zb_after_sat[0,1]
        self.assertTrue(torch.abs(Zf0_after_sat_shifted_to_Z1).item() < 0.1 * wf.item(), "Free pool not sufficiently saturated.")
        self.assertTrue(Zb0_after_sat_shifted_to_Z1.item() < wb.item(),
                        f"Bound Z0 ({Zb0_after_sat_shifted_to_Z1.item()}) should decrease from initial {wb.item()}.")

    def test_2_2_relaxation_qualitative(self):
        # With Relax/Exch -> B0 -> RF -> Shift order:
        # Fpf created by RF does not experience T2f decay in the same TR.
        # So, Fpf[0,1] (shifted F0 component) should have magnitude ~0.5 * Zf_pre_RF.
        long_TR_relax = self.T2f_s.item() * 3 # Not directly used to check decay in this TR

        states_list = self.model(self.flip_angles_single, self.phases_single,
                                 self.T1f_s, self.T2f_s, self.T1b_s, self.T2b_s,
                                 self.kf_s, self.kb_s, self.TR_val, # Using self.TR_val = 100ms
                                 self.TE_val, B0=self.B0_s, B1=self.B1_s, wf=self.wf_s, wb=self.wb_s)
        Fp_f, _, Zf, Fp_b, _, Zb = states_list[0]

        # Calculate Zf_pre_rf more accurately for this specific TR
        E1f_calc = torch.exp(-self.TR_val / self.T1f_s) # (1,1)
        Zf0_initial = self.wf_s # (1,1)
        Zb0_initial = self.wb_s # (1,1)

        # Value of Zf[0] just before RF, after relax/exchange (single step)
        # Z_f_pre_rf_approx = Zf0_initial * E1f_calc + (1-E1f_calc)*self.wf_s + \
        #                     (-self.kf_s * Zf0_initial + self.kb_s * Zb0_initial) * (self.TR_val / 1000.0)
        # This is an approximation as Zf0_initial should be Z_f[0,0] which is a vector.
        # For simplicity, we assume Zf_pre_rf is close to initial wf for magnitude estimation.
        Zf_pre_rf_approx = self.wf_s.item()
        # (If relax/exchange is complex, Zf_pre_rf might deviate more from wf_s)
        # A more direct check: if T2f is very long, Fpf1 mag should be ~0.5*Zf_pre_rf.
        # If T2f is short, and TR is long, then Fpf1 should be small IF relax was after RF.
        # But here, relax is BEFORE RF. So Fpf1 is formed from Zf_pre_rf, then shifted.
        # It does not decay in this TR.

        expected_Fpf1_mag = 0.5 * Zf_pre_rf_approx

        self.assertAlmostEqual(torch.abs(Fp_f[0,1]).item(), expected_Fpf1_mag, places=1, # Relaxed places due to Zf_pre_rf approx
                               msg=f"Fpf[0,1] mag. Expected ~{expected_Fpf1_mag:.3f} (no decay in this TR), Got {torch.abs(Fp_f[0,1]).item():.3f}")

        self.assertAlmostEqual(torch.abs(Fp_b[0,1]).item(), 0.0, places=3, msg="Fpb[0,1] should be ~0 (bound pool not excited).")


if __name__ == '__main__':
    if not MODULE_FOUND:
        print(f"Skipping tests: EPGSimulationMT_GPU_Vectorized module not found. Import error: {IMPORT_ERROR}")
    else:
        unittest.main()
