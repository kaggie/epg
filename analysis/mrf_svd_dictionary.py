import torch
import numpy as np
import os
import sys
import math

# Add project root and vectorized-sims to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vectorized-sims')))

EPGSimulation = None # Placeholder

def generate_mrf_dictionary(param_sets, flip_angle_series_rad, TR_ms,
                            epg_simulator_class, sequence_phases_rad=None, TE_ms=None,
                            n_states=21, default_B0=0.0, default_B1=1.0):
    if epg_simulator_class is None:
        raise ValueError("epg_simulator_class must be provided.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for EPG simulations.")

    # Ensure n_states is at least 2 to access state k=1 (Fp[...,1])
    if n_states < 2:
        print(f"Warning: n_states is {n_states}, but MRF signal is typically from Fp[...,1] (shifted F0). Adjusting n_states to 2 for this.")
        n_states = 2 # Minimum to have a k=1 state

    epg_model = epg_simulator_class(n_states=n_states, device=device)

    num_timepoints = len(flip_angle_series_rad)
    if sequence_phases_rad is None:
        sequence_phases_rad = torch.zeros(num_timepoints, device=device)
    if TE_ms is None: TE_ms = TR_ms

    flip_angle_series_rad = flip_angle_series_rad.to(device)
    sequence_phases_rad = sequence_phases_rad.to(device)

    dictionary_signals = []
    print(f"Generating dictionary for {len(param_sets)} parameter sets using {num_timepoints} timepoints...")

    for i, p_set in enumerate(param_sets):
        print(f"  Simulating entry {i+1}/{len(param_sets)}: T1={p_set['T1']:.0f}, T2={p_set['T2']:.0f}, "
              f"B0={p_set.get('B0', default_B0):.1f}, B1={p_set.get('B1', default_B1):.2f}")

        T1 = torch.tensor([p_set['T1']], device=device)
        T2 = torch.tensor([p_set['T2']], device=device)
        B0 = torch.tensor([p_set.get('B0', default_B0)], device=device)
        B1 = torch.tensor([p_set.get('B1', default_B1)], device=device)

        states_list = epg_model(flip_angle_series_rad, sequence_phases_rad,
                                T1, T2, TR_ms, TE_ms, B0=B0, B1=B1)

        # MRF signal is typically the F0 state. After shift, F0 is moved to F1 (index 1).
        # Fp has shape (batch_size=1, n_states) for each item in states_list.
        # We take the complex value of Fp[0,1].
        if n_states < 2 : # Should not happen due to check above, but defensive
            signal_idx_to_use = 0
            print("Warning: n_states < 2, using Fp[0,0] which might be zeroed by shift.")
        else:
            signal_idx_to_use = 1 # F_k=1 state, which holds the shifted F_k=0 component
                                  # Or, if TE is explicitly handled, one might take F_k=0 before shift.
                                  # For now, taking F_k=1 from states_list (which are post-shift).

        entry_signal_series = torch.tensor([s[0][0, signal_idx_to_use] for s in states_list], dtype=torch.cfloat, device=device)
        dictionary_signals.append(entry_signal_series)

    if not dictionary_signals:
        return torch.empty(num_timepoints, 0, device=device), param_sets

    dictionary_matrix = torch.stack(dictionary_signals, dim=1)
    print(f"Dictionary generation complete. Matrix shape: {dictionary_matrix.shape}")
    return dictionary_matrix, param_sets


def perform_svd(dictionary_matrix):
    if not isinstance(dictionary_matrix, torch.Tensor):
        try:
            dictionary_matrix = torch.from_numpy(dictionary_matrix).to(torch.cfloat) # Ensure complex if converting
        except Exception as e:
            print(f"Could not convert dictionary_matrix to PyTorch complex tensor: {e}")
            return None, None, None

    if dictionary_matrix.numel() == 0:
        print("Dictionary matrix is empty, skipping SVD.")
        return None, None, None

    current_device = dictionary_matrix.device
    # Ensure dictionary is complex for SVD, as signals are complex
    if not dictionary_matrix.is_complex():
        print("Warning: Dictionary matrix for SVD is not complex. Converting to complex.")
        dictionary_matrix = dictionary_matrix.to(torch.cfloat)

    print(f"Performing SVD on dictionary (shape: {dictionary_matrix.shape}, device: {current_device})...")

    try:
        U, S, Vh = torch.linalg.svd(dictionary_matrix, full_matrices=False)
        print("SVD complete.")
        print(f"  U shape: {U.shape} (complex: {U.is_complex()}), S shape: {S.shape} (real), Vh shape: {Vh.shape} (complex: {Vh.is_complex()})")
        return U, S, Vh
    except Exception as e:
        print(f"Error during SVD: {e}")
        return None, None, None


if __name__ == '__main__':
    print("MRF SVD Dictionary Generation Script")
    print("------------------------------------")

    try:
        from epg_mri_vectorized import EPGSimulation as ImportedEPGSimulator
        print("Successfully imported EPGSimulation from epg_mri_vectorized.")
        EPGSimulation = ImportedEPGSimulator
    except Exception as import_err:
        print(f"Could not import EPG simulator. Script will not run dictionary generation. Error: {import_err}")
        EPGSimulation = None

    if EPGSimulation is None:
        print("Cannot run example: EPGSimulation class not available.")
    else:
        example_param_sets = [
            {'T1': 600.0, 'T2': 60.0, 'B0': 0.0},
            {'T1': 1000.0, 'T2': 100.0, 'B0': 0.0},
            {'T1': 1400.0, 'T2': 140.0, 'B0': 0.0},
            {'T1': 2000.0, 'T2': 200.0, 'B0': 5.0},
        ]

        num_mrf_timepoints = 50
        TR_ms_example = 12.0
        TE_ms_example = TR_ms_example / 2.0

        default_tensor_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {default_tensor_device} for sequence parameters in example.")

        base_fas = torch.linspace(10, 70, num_mrf_timepoints // 2, device=default_tensor_device)
        fa_pattern = torch.cat((base_fas, base_fas.flip(0)))
        if num_mrf_timepoints % 2 != 0:
            fa_pattern = torch.cat((fa_pattern, torch.tensor([base_fas.mean()], device=default_tensor_device)))
        flip_angle_series_deg = fa_pattern + torch.rand(num_mrf_timepoints, device=default_tensor_device) * 10 - 5
        flip_angle_series_deg = torch.clamp(flip_angle_series_deg, 5, 85)
        flip_angle_series_rad_ex = torch.deg2rad(flip_angle_series_deg)

        phases_rad_ex = (torch.arange(num_mrf_timepoints, device=default_tensor_device).float()**2) * (math.pi / num_mrf_timepoints)

        print(f"\nSimulating with {num_mrf_timepoints} TRs of {TR_ms_example:.1f}ms each.")

        # Ensure n_states >= 2 for signal extraction from Fp[...,1]
        n_states_for_mrf = max(11, 2) # Default to 11, but ensure at least 2

        generated_dictionary, generated_params = generate_mrf_dictionary(
            param_sets=example_param_sets,
            flip_angle_series_rad=flip_angle_series_rad_ex,
            TR_ms=TR_ms_example,
            epg_simulator_class=EPGSimulation,
            sequence_phases_rad=phases_rad_ex,
            TE_ms=TE_ms_example,
            n_states=n_states_for_mrf
        )

        if generated_dictionary.numel() > 0:
            print(f"\nGenerated Dictionary Shape: {generated_dictionary.shape}")
            print(f"Parameters used for first {min(3, len(generated_params))} entries:")
            for i in range(min(3, len(generated_params))):
                print(f"  Entry {i}: {generated_params[i]}")

            print("\nPerforming SVD on the generated dictionary...")
            U, S, Vh = perform_svd(generated_dictionary.to(default_tensor_device))

            if U is not None and S is not None and Vh is not None:
                print(f"Top 5 singular values: {S[:min(5, S.shape[0])]}") # Print available singular values

                k = min(10, U.shape[1], Vh.shape[0])
                print(f"\nReconstructing first dictionary entry using top {k} components...")

                U_k = U[:, :k]
                # S is 1D (singular values), convert to complex diagonal matrix for multiplication
                S_k_diag = torch.diag(S[:k]).to(torch.cfloat)
                Vh_k = Vh[:k, :]

                reconstructed_signal_0_approx = U_k @ S_k_diag @ Vh_k[:, 0]
                original_signal_0 = generated_dictionary[:, 0].to(default_tensor_device) # Ensure same device

                reconstruction_error = torch.norm(original_signal_0 - reconstructed_signal_0_approx)
                print(f"Reconstruction error for first signal (norm of difference): {reconstruction_error.item():.4e}")
                print(f"Original signal (first 5 pts): {original_signal_0[:5]}")
                print(f"Reconstructed signal (first 5 pts): {reconstructed_signal_0_approx[:5]}")
        else:
            print("SVD skipped or dictionary generation failed.")

    print("\nScript execution finished.")
