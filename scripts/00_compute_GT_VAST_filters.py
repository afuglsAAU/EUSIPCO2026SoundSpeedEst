
"""
Compute Ground Truth (GT) VAST filters for all simulated sound speeds.

This script:
1) Computes covariance matrices (R_B, R_D, r_B) for each sound speed
2) VAST filters for all ranks in shared_params['vast_ranks'] using joint diagonalization

Folder conventions:
- Ground truth RIRs are expected in:
    data_path/simulated_RIRs/GT/{array_setup}_RT60_{rt60}/
- Covariances are stored under:
    covariance_path/GT_filters/
- Filters are stored under:
    filter_files_path/RT60_{rt60}/GT_VAST_filters/
"""
import os
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
# os.environ["OPENBLAS_NUM_THREADS"] = "6"
import numpy as np
import scipy.io as sio
import sys
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dotenv import find_dotenv, dotenv_values
from datetime import datetime
import gc

Loc_env = find_dotenv('.env')
CONFIG_ENV = dotenv_values(Loc_env)
sys.path.append(CONFIG_ENV['MainCodePath'])

from src.utils.config_loader import load_config_module, get_shared_paths
from src.algorithms.filter_generation import fit_vast_closed_form
from src.algorithms.filter_helpers import compute_covariance_matrices, diagonalize_matrices
from src.utils.simdata import load_gt_rirs_speed




def get_covar_name(sound_speed, params):
    """
    Generate covariance matrix filename suffix based on parameters.

    Args:
        sound_speed (int): Sound speed in m/s.
        params (dict): Parameter dictionary containing 'array_setup', 'rt60', 'J', 'K'.

    Returns:
        str: Covariance matrix filename suffix.
    """
    J = params['J']
    K = params['K']

    covar_name = f'{params["array_setup"]}_RT60_{params["rt60"]}_speed_{sound_speed:d}_J_{J}_K_{K}_GT'  # Ground thruth, so suffix with _GT. Add "interp_from_xxx" when SICERING from other speeds

    return covar_name

def check_covariance_matrices_exist(covariance_path, sound_speed, params):
    """
    Check if covariance matrices exist for a given sound speed.

    Args:
        covariance_path (Path): Path to covariance matrices.
        sound_speed (int): Sound speed in m/s.
        params (dict): Parameter dictionary containing 'array_setup', 'rt60', 'J', 'K'.

    Returns:
        b bool: True if all matrices (R_B, R_D, r_B) exist, False otherwise.
    """

    covar_name = get_covar_name(sound_speed, params)

    str_save_R_B = covariance_path.joinpath(f'R_B_{covar_name}.npz')
    str_save_R_D = covariance_path.joinpath(f'R_D_{covar_name}.npz')
    str_save_r_B = covariance_path.joinpath(f'cross_r_B_{covar_name}.npz')

    return str_save_R_B.exists() and str_save_R_D.exists() and str_save_r_B.exists()


def compute_covariance_for_speed(sound_speed, paths, covariance_path, params):
    """
    Compute and save covariance matrices for a given sound speed.

    Args:
        sound_speed (int): Sound speed in m/s
        paths (dict): Dictionary of paths including 'data_path'
        covariance_path (Path): Path to save covariance matrices
        params (dict): Parameter dictionary containing J, ref_source, model_delay, etc.

    Returns:
        tuple: (sound_speed, success) - speed and success flag
    """
    try:
        rirs = load_gt_rirs_speed(paths, params, int(sound_speed))
        imp_resp_bz = rirs['BZ']
        imp_resp_dz = rirs['DZ']

        # Compute and save covariance matrices
        covar_name = get_covar_name(sound_speed, params)
        compute_covariance_matrices(
            imp_resp_bz, imp_resp_dz,
            params['J'], params['ref_source'], params['model_delay'],
            covariance_path, covar_name, covar_name,
            verbose=False
        )

        return sound_speed, True

    except Exception as e:
        print(f"Error computing covariance for speed {sound_speed}: {e}")
        import traceback
        traceback.print_exc()
        return sound_speed, False




def compute_filters_worker(sound_speed, covariance_path, output_path, params):
    """
    Combined worker function that loads covariance matrices and computes VAST filters.

    This function encapsulates both the loading and computation steps, eliminating
    the need for a separate matrix loader thread.

    Args:
        sound_speed (int): Sound speed in m/s
        covariance_path (Path): Path to covariance matrices
        output_path (Path): Path to save output filter .npz files.
        params (dict): Parameter dictionary containing J, K, vast_ranks, mu, use_lsp, etc.

    Returns:
        tuple: (sound_speed, success) - speed and success flag
    """
    try:
        pid = os.getpid()

        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path.joinpath(f'GT_VAST_filters_speed_{sound_speed:d}_J_{params["J"]}.npz')
        if output_file.exists():
            print(f"Worker {pid}: Filters already exist for speed {sound_speed} m/s, skipping computation.")
            return sound_speed, True

        # Load covariance matrices
        covar_name = get_covar_name(sound_speed, params)

        str_save_R_B = covariance_path.joinpath(f'R_B_{covar_name}.npz')
        str_save_R_D = covariance_path.joinpath(f'R_D_{covar_name}.npz')
        str_save_r_B = covariance_path.joinpath(f'cross_r_B_{covar_name}.npz')

        print(f"Worker {pid}: Loading covariance matrices for speed {sound_speed} m/s...")
        start_load = datetime.now()
        R_B = np.load(str_save_R_B)['R_B']
        r_B = np.load(str_save_r_B)['r_B']
        R_D = np.load(str_save_R_D)['R_D']
        end_load = datetime.now()
        elapsed_load = end_load - start_load
        print(f"Worker {pid}: Loaded covariance matrices for speed {sound_speed} m/s in {elapsed_load}")
        J = params['J']
        L = len(params['use_lsp'])
        mu = params['mu']
        vast_ranks = params['vast_ranks']

        # Perform joint diagonalization of R_B and R_D
        print(f"Worker {pid}: Diagonalizing matrices for speed {sound_speed} m/s...")
        start_diag = datetime.now()
        eig_vec, eig_val = diagonalize_matrices(R_B, R_D, descend=True)
        del R_B, R_D  # Free memory
        gc.collect()
        end_diag = datetime.now()
        elapsed_diag = end_diag - start_diag
        print(f"Worker {pid}: Diagonalized matrices for speed {sound_speed} m/s in {elapsed_diag}")
        filters_array = np.zeros((len(vast_ranks), L, J))


        # Compute filters for each VAST rank
        print(f"Worker {pid}: Computing filters for speed {sound_speed} m/s...")
        start_filters = datetime.now()
        for rank_idx, rank in enumerate(vast_ranks):
            rank = int(rank)

            # Ensure rank doesn't exceed full dimension
            rank = min(rank, eig_vec.shape[1])

            # Compute VAST filter using closed-form solution
            q_vast = fit_vast_closed_form(rank, mu, r_B, J, L, eig_vec, eig_val, mat_out=True)

            filters_array[rank_idx] = q_vast

        np.savez_compressed(str(output_file), filters=filters_array, ranks=vast_ranks)
        end_filters = datetime.now()
        elapsed_filters = end_filters - start_filters
        print(f"Worker {pid}: Computed and saved filters for speed {sound_speed} m/s in {elapsed_filters}")
        return sound_speed, True

    except Exception as e:
        print(f"Error processing speed {sound_speed}: {e}")
        import traceback
        traceback.print_exc()
        return sound_speed, False





def main():
    """
    Main function to compute GT VAST filters for all sound speeds.

    Workflow:
    1) Check which covariance matrices exist for each speed
    2) Compute missing covariance matrices (parallel or sequential)
    3) Compute VAST filters for all ranks using joint diagonalization (parallel or sequential)

    Command-line arguments:
        --config (Path, required): Path to config module (e.g., configs/config_RT60_0.1.py)
        --parallel (flag): Enable parallel processing
        --max-workers (int): Maximum number of workers for parallel processing (default: 8)
    """

    import argparse
    parser = argparse.ArgumentParser(description='Compute GT VAST filters for all sound speeds using a selected config file.')
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to config module (e.g., configs/config_RT60_0.1.py)')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='Max workers for parallel processing')
    args = parser.parse_args()

    cfg = load_config_module(args.config)
    paths = get_shared_paths(config_env=CONFIG_ENV)
    params = cfg.shared_params.copy()

    sound_speeds = params['sound_speeds'].astype(int)
    covariance_path = paths['covariance_path'].joinpath('GT_filters')
    # Save under RT60-specific subfolder
    output_path = paths['filter_files_path'].joinpath(f"RT60_{params['rt60']}", 'GT_VAST_filters')

    print(f"\n{'='*70}")
    print(f"Computing GT VAST filters for {len(sound_speeds)} sound speeds")
    print(f"Sound speeds: {sound_speeds[0]} to {sound_speeds[-1]} m/s")
    print(f"VAST ranks: {len(params['vast_ranks'])} ranks from {params['vast_ranks'][0]:.0f} to {params['vast_ranks'][-1]:.0f}")
    print(f"Covariance path: {covariance_path}")
    print(f"Output path: {output_path}")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    # Step 1: Check for existing covariance matrices
    print("Step 1: Checking for existing covariance matrices...")
    missing_speeds = []
    existing_speeds = []

    for speed in sound_speeds:
        if check_covariance_matrices_exist(covariance_path, int(speed), params):
            existing_speeds.append(speed)
        else:
            missing_speeds.append(speed)

    print(f"  Found covariance matrices for {len(existing_speeds)} speeds")
    print(f"  Need to compute covariance matrices for {len(missing_speeds)} speeds")

    # Step 2: Compute missing covariance matrices
    if missing_speeds:
        print(f"\nStep 2: Computing missing covariance matrices...")

        if args.parallel:
            workers = min(len(missing_speeds), args.max_workers)
            print(f"  Using {workers} workers for parallel processing")

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(compute_covariance_for_speed, int(speed), paths,
                                  covariance_path, params): int(speed)
                    for speed in missing_speeds
                }

                for future in tqdm.tqdm(as_completed(futures), total=len(futures),
                                       desc="  Computing covariances"):
                    speed = futures[future]
                    try:
                        speed_result, success = future.result()
                        if not success:
                            print(f"    ✗ Speed {speed} m/s: Failed to compute covariance")
                    except Exception as exc:
                        print(f"    ✗ Speed {speed} m/s: Exception - {exc}")
        else:
            print("  Using sequential processing")
            for speed in tqdm.tqdm(missing_speeds, desc="  Computing covariances"):
                speed_result, success = compute_covariance_for_speed(
                    int(speed), paths, covariance_path, params
                )
                if not success:
                    print(f"    ✗ Speed {speed}: Failed to compute covariance")

        print("  All covariance matrices computed!")
    else:
        print("  All covariance matrices already exist!")

    # Step 3: Compute VAST filters with combined loading and computation
    print(f"\nStep 3: Computing VAST filters for all sound speeds...")

    if args.parallel:
        workers = min(len(sound_speeds), args.max_workers)
        print(f"  Using {workers} workers for parallel processing...")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(compute_filters_worker, int(speed), covariance_path,
                              output_path, params): int(speed)
                for speed in sound_speeds
            }

            successful = 0
            failed = 0

            with tqdm.tqdm(as_completed(futures), total=len(futures),
                          desc='  Computing filters') as pbar:
                for future in pbar:
                    speed = futures[future]
                    try:
                        speed_result, success = future.result()
                        if success:
                            successful += 1
                            print(f"    ✓ Speed {speed_result} m/s: filters computed")
                        else:
                            failed += 1
                            print(f"    ✗ Speed {speed}: Failed to compute filters")
                    except Exception as exc:
                        failed += 1
                        print(f"    ✗ Speed {speed}: Exception - {exc}")

    else:
        print("  Using sequential processing")
        successful = 0
        failed = 0

        for speed in tqdm.tqdm(sound_speeds, desc="  Computing filters"):
            speed_result, success = compute_filters_worker(
                int(speed), covariance_path, output_path, params
            )
            if success:
                successful += 1
                print(f"    ✓ Speed {speed_result} m/s: filters computed")
            else:
                failed += 1
                print(f"    ✗ Speed {speed}: Failed to compute filters")

    end_time = datetime.now()
    elapsed = end_time - start_time

    print("\n" + "="*70)
    print(f"Computation finished in {elapsed}")
    print(f"Successfully computed filters for {successful}/{len(sound_speeds)} speeds")
    if failed > 0:
        print(f"Failed for {failed}/{len(sound_speeds)} speeds")
    print("="*70)


if __name__ == "__main__":
    main()
