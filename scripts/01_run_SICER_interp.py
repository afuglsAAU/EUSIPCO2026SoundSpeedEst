
"""
SICER Speed-change corrected IR synthesis using split per-speed RIR files.
Method proposed in [1] and implemented in MATLAB by the authors, adapted here for Python with batch processing.

This module generates speed-corrected impulse responses (IRs) by loading baseline
RIRs from split per-speed .mat files and applying speed interpolation to synthesize
IRs for target speeds. The output is compatible with the original MATLAB pipeline.

Main entry point:
    main(): Parses command-line arguments and orchestrates speed interpolation
            across multiple base speeds.

[1] S. S. Bhattacharjee, J. R. Jensen, and M. G. Christensen, “Sound Speed Perturbation Robust Audio: Impulse Response Correction and Sound Zone Control,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 2008–2020, 2025, doi: 10.1109/TASLPRO.2025.3570949.

"""
import argparse
from pathlib import Path
import numpy as np
import tqdm
import scipy.io as sio
from scipy.io import savemat
from src.utils.speed_change import speed_up_batch, speed_down_batch
import sys
from dotenv import find_dotenv, dotenv_values
# Local imports from project
Loc_env = find_dotenv('.env')
CONFIG_ENV = dotenv_values(Loc_env)
_main_path = CONFIG_ENV.get('MainCodePath')
if isinstance(_main_path, str) and _main_path:
    sys.path.append(_main_path)
from src.utils.config_loader import load_config_module, get_shared_paths



def _load_split_rirs_for_speed(data_path: Path, rt60: float, sound_speed: int, K: int, array_setup: str) -> tuple[np.ndarray, np.ndarray]:
    """Load split RIRs saved by split_matlab_data.py for a given speed.

    Returns
    -------
    imp_resp_bz : ndarray, shape (M_b, L, K)
    imp_resp_dz : ndarray, shape (M_d, L, K)
    """
    mat_file = data_path.joinpath(
        f"simulated_RIRs/{array_setup}_RT60_{rt60}_speed_{sound_speed:d}.mat"
    )
    if not mat_file.exists():
        raise FileNotFoundError(f"RIR file not found: {mat_file}")

    data = sio.loadmat(str(mat_file))
    rirs = data["RIRs"]
    # Stored as MATLAB structs; access via [0,0]
    imp_resp_bz = np.asarray(rirs["BZ"][0, 0])[:, :, :K]  # mics, LSPs, samples
    imp_resp_dz = np.asarray(rirs["DZ"][0, 0])[:, :, :K]  # mics, LSPs, samples
    return imp_resp_bz, imp_resp_dz


def run_speed_interp_from_split(base_speed: int,
                                target_speeds: np.ndarray,
                                output_path: Path | None,
                                paths: dict,
                                params: dict) -> Path:
    """Generate speed-corrected IRs using split per-speed .mat files.

    - Loads baseline IRs from the split file for `base_speed`.
    - Applies speed_up/down batch interpolation to synthesize IRs for each target speed.
    - Saves a .mat compatible with the original MATLAB pipeline.
    """
    # Require explicit params and paths provided from selected config
    if params is None or paths is None:
        raise ValueError("'params' and 'paths' must be provided from the selected config")

    K = int(params['K'])
    L = int(params['L'])
    rt60 = params['rt60']

    # Load baseline RIRs: shapes (M, L, K)
    imp_resp_bz_base, imp_resp_dz_base = _load_split_rirs_for_speed(paths['data_path'], rt60, int(base_speed), K, params['array_setup'])

    M_b = imp_resp_bz_base.shape[0]
    M_d = imp_resp_dz_base.shape[0]

    # Prepare segments for batch processing: (n, L*M)
    n = K
    bz_segments = imp_resp_bz_base.transpose(2, 1, 0).reshape(n, L * M_b)
    dz_segments = imp_resp_dz_base.transpose(2, 1, 0).reshape(n, L * M_d)

    # Output arrays: [T, L*K, M]
    target_speeds = np.asarray(target_speeds, dtype=float)
    perturbations = target_speeds - float(base_speed)
    T = target_speeds.size
    imp_resp_bz_corrected = np.zeros((T, M_b, L, n), dtype=float)
    imp_resp_dz_corrected = np.zeros((T, M_d, L, n), dtype=float)

    # Determine output directory
    default_out_dir = paths['data_path'].joinpath(
        f"simulated_RIRs/SICER_corrected/{params['array_setup']}_RT60_{rt60}"
    )
    out_dir = Path(output_path) if (output_path is not None) else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for tp in tqdm.tqdm(range(T), desc="Speed interpolation (split files)"):
        new_speed = float(target_speeds[tp])
        if new_speed > base_speed:
            bz_out_mat = speed_up_batch(bz_segments, n, base_speed, new_speed)
            dz_out_mat = speed_up_batch(dz_segments, n, base_speed, new_speed)
        elif new_speed < base_speed:
            bz_out_mat = speed_down_batch(bz_segments, n, base_speed, new_speed)
            dz_out_mat = speed_down_batch(dz_segments, n, base_speed, new_speed)
        else:
            bz_out_mat = bz_segments
            dz_out_mat = dz_segments

        # Reshape back to [M, L, n]
        bz_out = bz_out_mat.reshape(n, L, M_b).transpose(2, 1, 0)
        dz_out = dz_out_mat.reshape(n, L, M_d).transpose(2, 1, 0)

        imp_resp_bz_corrected[tp, :, :] = bz_out
        imp_resp_dz_corrected[tp, :, :] = dz_out

        # Save per target speed into out_dir
        out_file = out_dir.joinpath(
            f"init_{int(base_speed)}_to_{int(new_speed)}.mat"
        )
        savemat(str(out_file), {
            "RIRs_corrected" : {
                'BZ': bz_out,
                'DZ': dz_out
            },
            "old_speed": float(base_speed),
            "new_speed": float(target_speeds[tp]),
            "K": int(n),
        })
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Speed-change corrected IR synthesis using split per-speed RIR files"
    )
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to config module (e.g., exp1_SICER/config_RT60_0.1.py)')
    parser.add_argument("--base-speed", type=int, default=None,
                        help="Baseline sound speed to load from split files (default: config base_speed)")
    parser.add_argument("--base-speeds", type=str, default=None,
                        help="Multiple baseline speeds: 'config' to use config sound_speeds or comma-separated list, e.g., 333,343,353")
    parser.add_argument("--speeds", type=str, default="config",
                        help="Target speeds. 'config' uses config sound_speeds; or comma-separated list, e.g., 333,343,353")
    parser.add_argument("--output", type=Path, default=None, help="Optional output .mat path")
    args = parser.parse_args()

    # Env and config

    cfg = load_config_module(Path(args.config))
    paths = get_shared_paths(config_env=CONFIG_ENV)
    params = cfg.shared_params.copy()

    # Resolve list of base speeds
    if args.base_speeds is not None:
        if args.base_speeds.lower() in ("config", "all"):
            base_speeds = params['sound_speeds'].astype(int)
        else:
            base_speeds = np.array([int(s) for s in args.base_speeds.split(',') if s.strip() != ""], dtype=int)
    else:
        single_base = int(params['base_speed']) if args.base_speed is None else int(args.base_speed)
        base_speeds = np.array([single_base], dtype=int)

    # Resolve target speeds
    if args.speeds == "config":
        target_speeds = params['sound_speeds'].astype(int)
    else:
        target_speeds = np.array([int(s) for s in args.speeds.split(',') if s.strip() != ""], dtype=int)

    all_out_dirs = []
    for b in tqdm.tqdm(base_speeds, desc="Base speeds"):
        out_dir = run_speed_interp_from_split(int(b), target_speeds, output_path=args.output, paths=paths, params=params)
        all_out_dirs.append(out_dir)

    print("Saved folders:")
    for d in all_out_dirs:
        print(f" - {d}")


if __name__ == "__main__":
    main()
