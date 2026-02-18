import numpy as np
import scipy.io as sio
import pandas as pd
import warnings
from pathlib import Path
from .io import audioread



# ------------------------------------------------------------
# Index handling
# ------------------------------------------------------------
def load_index(csv_path="results_index.csv"):
    df = pd.read_csv(csv_path)
    return df



# ------------------------------------------------------------
# Results loading
# ------------------------------------------------------------
def load_results(row):
    """
    row: one row of the index DataFrame
    """
    path = Path(row["results_npz"])
    data = np.load(path)

    return {
        "est_speed": data["est_speed_track"],
        "true_speed": data["true_speed"],
        "speed_change_frames": data.get("speed_change_frames", None),
        "AC": data.get("AC_performance_track", None),
        "nSDP": data.get("nSDP_performance_track", None),
        "input_power": data.get("input_signal_power", None),
        "input_signal_array": data.get("input_signal_array", None),
    }


def load_gt_rirs_speed(paths: dict, params: dict, speed: int) -> dict:
    """Load Ground-Truth RIRs for a given sound speed and trim to K.

    Returns dict with keys 'BZ' and 'DZ' shaped as (M, L, K).
    """
    rt60 = params['rt60']
    K = int(params['K'])
    setup = params['array_setup']
    rir_file = Path(paths['data_path']).joinpath(
        f"simulated_RIRs/{setup}_RT60_{rt60}_speed_{int(speed):d}.mat"
    )
    if not rir_file.exists():
        raise FileNotFoundError(f"RIR file not found for speed {speed}: {rir_file}")
    data = sio.loadmat(str(rir_file))
    rirs = data['RIRs']
    imp_resp_bz = np.asarray(rirs['BZ'][0, 0])[:, :, :K].astype(np.float64)
    imp_resp_dz = np.asarray(rirs['DZ'][0, 0])[:, :, :K].astype(np.float64)
    return {'BZ': imp_resp_bz, 'DZ': imp_resp_dz}


def load_sicer_corrected_rirs(paths: dict, params: dict, base_speed: int, target_speed: int) -> dict:
    """Load SICER-corrected RIRs for a (base → target) speed pair and trim to K."""
    rt60 = params['rt60']
    K = int(params['K'])
    setup = params['array_setup']
    base_dir = Path(paths['data_path']).joinpath(
        f"simulated_RIRs/SICER_corrected/{setup}_RT60_{rt60}"
    )
    mat_file = base_dir.joinpath(f"init_{int(base_speed)}_to_{int(target_speed)}.mat")
    if not mat_file.exists():
        raise FileNotFoundError(f"Corrected IRs not found for {base_speed}→{target_speed}: {mat_file}")
    data = sio.loadmat(str(mat_file))
    RIRs = data['RIRs_corrected']
    imp_bz = np.asarray(RIRs['BZ'][0, 0])[:, :, :K]
    imp_dz = np.asarray(RIRs['DZ'][0, 0])[:, :, :K]
    return {'BZ': imp_bz, 'DZ': imp_dz}


def load_input_signal(paths: dict, params: dict):
    """Load or generate input signal based on params.

    Supports: 'audio', 'dirac' (impulse), 'white' (WGN).
    Returns (signal, label).
    """
    fs = int(params['fs'])
    input_type = str(params.get('input_signal', 'audio')).lower()

    duration = params['input_duration']

    if input_type == 'audio':
        audio_name = params.get('input_audio')
        if audio_name is None:
            raise ValueError("params['input_audio'] must be set for input_signal='audio'")
        sig, fs1 = audioread(Path(paths['input_audio_path']).joinpath(audio_name), fs)
        if fs1 != fs:
            raise ValueError(f"Mismatch in sampling frequency of input (fs={fs1}) and params fs={fs}")
        if len(sig) < int(duration * fs):
            # Adjust duration to actual signal length if shorter than configured
            print("Warning: input audio shorter than configured duration, adjusting duration accordingly.")
            params['input_duration'] = len(sig) / fs
        sig = sig[: int(params['input_duration'] * fs)]
        return sig.astype(np.float64), f"audio_{audio_name}"

    if input_type in ('white', 'white_noise', 'noise'):
        N = int(duration * fs)
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(N).astype(np.float64)
        return sig, 'white'

    if input_type in ('dirac', 'impulse', 'delta'):
        N = int(duration * fs)
        sig = np.zeros(N, dtype=np.float64)
        sig[0] = 1.0
        return sig, 'dirac'

    raise ValueError(f"Unsupported input_signal type: {input_type}")



def load_vast_filters_for_speed(sound_speed, paths, params):
    """Load GT VAST filters for a given speed.

    Supports consolidated and per-rank files.
    Returns (filters, ranks): filters shape (R, L, J), ranks is list[int] length R.
    """
    L = len(params['use_lsp'])
    J = params['J']
    ranks = list(map(int, np.asarray(params['vast_ranks']).astype(int).tolist()))
    rt60 = params.get('rt60')
    J = params.get('J')
    out_dir_rt60 = paths['filter_files_path'].joinpath( f'RT60_{rt60}', 'GT_VAST_filters')
    consolidated = out_dir_rt60.joinpath(f'GT_VAST_filters_speed_{int(sound_speed):d}_J_{int(J)}.npz')
    if consolidated.exists():
        data = np.load(str(consolidated))
        if 'filters' in data:
            filters = np.asarray(data['filters'])
        else:
            raise KeyError(f"'filters' key missing in {consolidated}")
        if 'ranks' in data:
            loaded_ranks = list(np.asarray(data['ranks']).astype(int))
        else:
            warnings.warn(f"Consolidated file missing 'ranks' key: {consolidated}. Using configuration file ranks.")
            if len(ranks) != filters.shape[0]:
                raise ValueError(f"Number of ranks in config ({len(ranks)}) does not match number of filters in file ({filters.shape[0]}). Unsafe to proceed.")
            loaded_ranks = ranks
        return filters, loaded_ranks
    else:
        raise FileNotFoundError(f"GT VAST filters not found for speed {sound_speed}: {consolidated}")




def load_sicer_filters_for_pair(new_speed, old_speed, paths, params):
    """Load SICER VAST filters for a specific (old -> new) speed pair.

    The SICER computation saves consolidated files per pair containing all ranks.
    Returns (filters, ranks): filters shape (R, L, J), ranks is list[int] length R.
    """
    L = len(params['use_lsp'])
    J = params['J']
    ranks = list(map(int, np.asarray(params['vast_ranks']).astype(int).tolist()))
    rt60 = params.get('rt60')
    J = params.get('J')
    out_dir_rt60 = paths['filter_files_path'].joinpath(f'RT60_{rt60}', 'SICER_VAST_filters')
    consolidated = out_dir_rt60.joinpath(
        f'SICER_VAST_filters_init_speed_{int(old_speed):d}_interp_to_{int(new_speed):d}_J_{int(J)}.npz'
    )

    if not consolidated.exists():
        raise FileNotFoundError(
            f"SICER filters not found for {old_speed}→{new_speed}: {consolidated}"
        )
    data = np.load(str(consolidated))
    if 'filters' not in data:
        raise KeyError(f"'filters' key missing in {consolidated}")

    filters = np.asarray(data['filters'])  # (R, L, J)
    if 'ranks' in data:
        loaded_ranks = list(np.asarray(data['ranks']).astype(int))
    else:
        warnings.warn(f"'ranks' key missing in {consolidated}. Using configuration file ranks.")
        loaded_ranks = ranks

    return filters, loaded_ranks