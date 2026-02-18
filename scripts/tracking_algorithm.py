

"""
Sound Speed Tracking Algorithm for SZC (Source Zone Control) with VAST Filters.

This module implements frame-wise sound speed estimation using SICER-based IR correction
and optimization over observation signals. It supports multiple speed estimation strategies
(Newton, bisection, grid search) and adaptive filter updates for robust audio control.

Key Components:
    - Speed Estimators: Newton/bisection/grid-based methods to minimize MSE between
      predicted and observed microphone signals.
    - SICER Correction: Speed-change-independent room response (SICER) correction applied
      to impulse responses for on-the-fly speed hypothesis testing.
    - Filter Management: Dynamic VAST filter computation and caching for estimated speeds,
      with support for both offline precomputed and online adaptive modes.
    - Performance Metrics: Acoustic contrast (AC) and normalized signal distortion (nSDP)
      tracking alongside speed estimation.

Main Entry Point:
    run_speed_tracking(): Performs frame-wise speed tracking and filter adaptation with
    optional result saving and visualization.

Configuration:
    - Experiment parameters loaded from config modules.
    - Supports baseline modes (GT/SICER precomputed filters), active loudspeaker subsets,
      and observation-only microphone subsets for efficiency.
    - Adaptive search windows and filter update criteria for online operation.
"""

import os
# os.environ["OMP_NUM_THREADS"] = "32"
# os.environ["MKL_NUM_THREADS"] = "32"
import json
import numpy as np
import tqdm
import scipy.io as sio
import argparse
import sys
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dotenv import find_dotenv, dotenv_values
from pathlib import Path
from scipy.fft import rfft, irfft, next_fast_len
from scipy.optimize import newton as scipy_newton

# Local imports from project
Loc_env = find_dotenv('.env')
CONFIG_ENV = dotenv_values(Loc_env)
_main_path = CONFIG_ENV.get('MainCodePath')
if isinstance(_main_path, str) and _main_path:
    sys.path.append(_main_path)

from src.utils.config_loader import load_config_module, get_shared_paths
from src.utils.speed_change import speed_up_batch, speed_down_batch
from src.utils.simdata import load_gt_rirs_speed, load_input_signal, load_vast_filters_for_speed, load_sicer_filters_for_pair
from src.utils.pressure import compute_mic_signal_frame_fft_overlap, compute_lsp_signal_frame
from src.algorithms.filter_helpers import diagonalize_matrices, get_zone_convolution_mat
from src.algorithms.filter_generation import fit_vast_closed_form


def _sicer_apply_3d(arr: np.ndarray, old_speed: float, new_speed: float, K: int) -> np.ndarray:
    """Apply SICER speed change to a 3D IR tensor of shape (M, L, K).

    Flattens to (K, L*M), applies speed_up/down in segment domain, reshapes back.
    """
    arr = np.asarray(arr)
    M, L, K_ = arr.shape
    assert K_ == K, "Last dimension must match provided K"
    segments = arr.transpose(2, 1, 0).reshape(K, L * M)
    if float(new_speed) > float(old_speed):
        out_mat = speed_up_batch(segments, K, float(old_speed), float(new_speed))
    elif float(new_speed) < float(old_speed):
        out_mat = speed_down_batch(segments, K, float(old_speed), float(new_speed))
    else:
        out_mat = segments
    return out_mat.reshape(K, L, M).transpose(2, 1, 0)


def _load_gt_rirs_speed_cached(paths: Dict, params: Dict, speed: int, cache: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Load GT RIRs for a given speed with a simple in-memory cache."""
    s = int(speed)
    if s not in cache:
        cache[s] = load_gt_rirs_speed(paths, params, s)
    return cache[s]


def _apply_sicer_to_obs_irs(init_obs_IRs: Dict[str, np.ndarray], old_speed: float, new_speed: float, params: Dict) -> Dict[str, np.ndarray]:
    """Apply SICER speed-change to observed IRs only (per zone).

    init_obs_IRs: dict zone -> (M_t, L, K)
    Returns dict zone -> (M_t, L, K)
    """
    out: Dict[str, np.ndarray] = {}
    for zone_key, zone_arr in init_obs_IRs.items():
        M, L, K = np.asarray(zone_arr).shape
        out[zone_key] = _sicer_apply_3d(zone_arr, old_speed, new_speed, K)
    return out


def _apply_sicer_to_full_rirs(rirs_full: Dict[str, np.ndarray], old_speed: float, new_speed: float, params: Dict) -> Dict[str, np.ndarray]:
    """Apply SICER to full RIR dict (all mics, all LSP) for BZ and DZ."""
    corrected: Dict[str, np.ndarray] = {}
    for zone_key in ('BZ', 'DZ'):
        if zone_key not in rirs_full:
            continue
        M, L, K = np.asarray(rirs_full[zone_key]).shape
        corrected[zone_key] = _sicer_apply_3d(rirs_full[zone_key], old_speed, new_speed, K)
    return corrected


def _mse_frames(pred_frames: Dict[str, np.ndarray], obs_frames: Dict[str, np.ndarray]) -> float:
    num = 0.0
    den = 0
    for k in obs_frames.keys():
        if k not in pred_frames:
            continue
        y = np.asarray(obs_frames[k])
        yhat = np.asarray(pred_frames[k])

        # Normalized MSE
        norm = np.sqrt(np.mean(y**2)) + 1e-9
        y_n = y / norm
        yhat_n = yhat / norm

        num += float(np.mean((yhat_n - y_n) ** 2))

        # num += float(np.mean((yhat - y) ** 2))

        den += 1
    return num / max(1, den)


def _mse_for_speed_grid_item(
    s: float,
    obs_frames: Dict[str, np.ndarray],
    lsp_curr: np.ndarray,
    lsp_fft_cached: np.ndarray,
    lsp_fft_prev: np.ndarray,
    base_obs_IRs: Dict[str, np.ndarray],
    old_speed_base: float,
    params: Dict,
    n_fft: int,
    frame_size: int,
    K: int,
) -> float:
    """Compute MSE for a single grid sample speed `s`.

    Mirrors the per-speed logic used inside `_grid_search_speed_for_frame` without
    relying on local closures, so it can be executed in a worker thread/process.
    """
    # SICER-correct observed IRs to candidate speed
    est_obs_IRs = _apply_sicer_to_obs_irs(base_obs_IRs, old_speed_base, float(s), params)

    # Predict frames with overlap (do not mutate prev states)
    pred_frames: Dict[str, np.ndarray] = {}
    for t in obs_frames.keys():
        IR_est_fft = rfft(est_obs_IRs[t], n=n_fft, axis=2)  # shape (M, L, F)

        # previous-frame contribution
        prev_est = np.einsum('mlf,lf->mf', IR_est_fft, lsp_fft_prev)
        prev_est_full = irfft(prev_est, n=n_fft, axis=1)
        out_len = frame_size + K - 1
        prev_est_full = prev_est_full[:, :out_len]
        prev_est_tail = np.zeros((IR_est_fft.shape[0], frame_size))
        prev_est_tail[:, :K-1] = prev_est_full[:, frame_size:]

        # current-frame prediction
        pred_t, _ = compute_mic_signal_frame_fft_overlap(
            IR_est_fft,
            lsp_curr,
            prev_est_tail,
            frame_size,
            K,
            n_fft=n_fft,
            lsp_fft=lsp_fft_cached,
        )
        pred_frames[t] = pred_t

    return _mse_frames(pred_frames, obs_frames)


def _bisect_speed_for_frame(obs_frames: Dict[str, np.ndarray],
                            lsp_curr: np.ndarray,
                            lsp_fft_cached,
                            base_obs_IRs: Dict[str, np.ndarray],
                            old_speed_base: float,
                            params: Dict,
                            n_fft,
                            frame_size: int,
                            K: int,
                            lsp_fft_prev,
                            verbose: bool = False) -> Tuple[float, Dict[str, np.ndarray], List[float], List[float]]:
    """Continuous bisection-like search for the frame's speed minimizing MSE.

    - Base IRs: init_obs_IRs at base speed `old_speed_base`.
    - Variable: new_speed in [search_min_speed, search_max_speed].
    - Objective: predicted mic frame (FFT overlap) vs obs_frames.
    """
    bis = params.get('bisect', {})
    s_min = float(params['search_min_speed'])
    s_max = float(params['search_max_speed'])
    tol = float(bis['tol'])
    max_iters = int(bis['max_iters'])
    neighbor_frac = float(bis['neighbor_frac'])
    refine_samples = int(bis['refine_samples'])
    stagnation_patience = int(bis['early_stop_patience'])

    lo = min(s_min, s_max)
    hi = max(s_min, s_max)
    cache: Dict[float, float] = {}

    def mse_for(s: float) -> float:
        key = float(np.round(s, 9))
        if key in cache:
            return cache[key]
        val = _mse_for_speed_grid_item(
            float(s),
            obs_frames,
            lsp_curr,
            lsp_fft_cached,
            lsp_fft_prev,
            base_obs_IRs,
            old_speed_base,
            params,
            int(n_fft),
            int(frame_size),
            int(K),
        )
        cache[key] = val
        return val

    def d_cost(s):
        h = 0.1
        return (mse_for(s + h) - mse_for(s - h)) / (2 * h)

    def cost_for(s):
        val = d_cost(s)
        return val


    # Evaluate endpoints
    _ = cost_for(lo)
    _ = cost_for(hi)
    it = 0
    stagn = 0
    # eps = max(1e-9, 1e-3 * tol)
    # while (hi - lo) > tol and it < max_iters:
    #     lo0, hi0 = lo, hi
    #     mid = 0.5 * (lo + hi)
    #     step = max(tol * 0.5, neighbor_frac * (hi - lo))
    #     sL = max(lo, mid - step)
    #     sR = min(hi, mid + step)
    #     mL = cost_for(sL)
    #     mM = cost_for(mid)
    #     mR = cost_for(sR)
    #     # if verbose:
    #     #     print(f"  iter {it:02d}: [lo,mid,hi]=[{lo:.3f},{mid:.3f},{hi:.3f}], fL={mL:.3e}, fM={mM:.3e}, fR={mR:.3e}")
    #     if mL < mM and mL <= mR:
    #         hi = mid
    #     elif mR < mM and mR <= mL:
    #         lo = mid
    #     else:
    #         lo, hi = sL, sR
    #     it += 1
    #     if abs(lo - lo0) <= eps and abs(hi - hi0) <= eps:
    #         stagn += 1
    #         if stagn >= stagnation_patience:
    #             if verbose:
    #                 print(f"  ! Early stopping: interval stagnated for {int(stagnation_patience)} iterations")
    #             break
    #     else:
    #         stagn = 0

    # Gradient-based bisection
    while (hi - lo) > tol and it < max_iters:
         # Find middle point
        c = (lo+hi)/2

        # Check if middle point is root
        if (cost_for(c) == 0.0):
            break

        # Decide the side to repeat the steps
        if (cost_for(c)*cost_for(lo) < 0):
            hi = c
        else:
            lo = c

    # Final refinement
    samples = np.linspace(lo, hi, num=max(3, refine_samples))
    vals = [cost_for(float(s)) for s in samples]
    best_idx = int(np.argmin(vals))
    s_best = float(samples[best_idx])
    # Build estimated IRs at best speed
    est_obs_IRs_best = _apply_sicer_to_obs_irs(base_obs_IRs, old_speed_base, s_best, params)
    if verbose:
        print(f"  ✓ Final bracket [{lo:.3f}, {hi:.3f}], best={s_best:.3f}, mse={vals[best_idx]:.6e}")
    grid_samples = cache.keys()
    grid_vals = cache.values()
    return s_best, est_obs_IRs_best, list(grid_vals), list(grid_samples)


def _grid_search_speed_for_frame(obs_frames: Dict[str, np.ndarray],
                            lsp_curr: np.ndarray,
                            lsp_fft_cached,
                            base_obs_IRs: Dict[str, np.ndarray],
                            old_speed_base: float,
                            params: Dict,
                            n_fft,
                            frame_size: int,
                            K: int,
                            fno: int,
                            lsp_fft_prev,
                            verbose: bool = False) -> Tuple[float, Dict[str, np.ndarray], List[float], List[float]]:
    """Grid search for the frame's speed minimizing MSE.

    - Base IRs: init_obs_IRs at base speed `old_speed_base`.
    - Variable: new_speed in [search_min_speed, search_max_speed].
    - Objective: predicted mic frame (FFT overlap) vs obs_frames.
    """
    grid_params = params.get('grid_search', {})
    s_min = float(params['search_min_speed'])
    s_max = float(params['search_max_speed'])
    if fno == 0:
        tol = float(grid_params['tol_init'])
    else:
        tol = float(grid_params['tol'])

    lo = min(s_min, s_max)
    hi = max(s_min, s_max)
    cache: Dict[float, float] = {}

    def mse_for(s: float) -> float:
        key = float(np.round(s, 9))
        if key in cache:
            return cache[key]
        val = _mse_for_speed_grid_item(
            float(s),
            obs_frames,
            lsp_curr,
            lsp_fft_cached,
            lsp_fft_prev,
            base_obs_IRs,
            old_speed_base,
            params,
            int(n_fft),
            int(frame_size),
            int(K),
        )
        cache[key] = val
        return val

    # Grid sampling refinement (optionally parallel)
    samples = np.arange(lo, hi, tol)

    workers = int(params.get('grid_workers', 1) or 1)
    backend = str(params.get('grid_backend', 'thread')).lower()

    if workers > 1 and len(samples) > 1:
        try:
            if backend == 'process':
                # Note: large arrays will be pickled; use with small worker counts
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    vals = list(ex.map(
                        _mse_for_speed_grid_item,
                        [float(s) for s in samples],
                        [obs_frames] * len(samples),
                        [lsp_curr] * len(samples),
                        [lsp_fft_cached] * len(samples),
                        [lsp_fft_prev] * len(samples),
                        [base_obs_IRs] * len(samples),
                        [old_speed_base] * len(samples),
                        [params] * len(samples),
                        [int(n_fft)] * len(samples),
                        [int(frame_size)] * len(samples),
                        [int(K)] * len(samples),
                    ))
            else:
                # Default: threads to avoid pickling overhead on large arrays
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [
                        ex.submit(
                            _mse_for_speed_grid_item,
                            float(s),
                            obs_frames,
                            lsp_curr,
                            lsp_fft_cached,
                            lsp_fft_prev,
                            base_obs_IRs,
                            old_speed_base,
                            params,
                            int(n_fft),
                            int(frame_size),
                            int(K),
                        ) for s in samples
                    ]
                    vals = [f.result() for f in futures]
        except Exception:
            # Fallback to serial on any pool error
            vals = [mse_for(float(s)) for s in samples]
    else:
        vals = [mse_for(float(s)) for s in samples]
    best_idx = int(np.argmin(vals))
    s_best = float(samples[best_idx])
    # Build estimated IRs at best speed
    est_obs_IRs_best = _apply_sicer_to_obs_irs(base_obs_IRs, old_speed_base, s_best, params)
    if verbose:
        print(f"  ✓ Grid range [{lo:.3f}, {hi:.3f}], best={s_best:.3f}, mse={vals[best_idx]:.6e}")
    return s_best, est_obs_IRs_best, vals, samples.tolist()


def _newton_speed_for_frame(obs_frames: Dict[str, np.ndarray],
                            lsp_curr: np.ndarray,
                            lsp_fft_cached,
                            base_obs_IRs: Dict[str, np.ndarray],
                            old_speed_base: float,
                            params: Dict,
                            n_fft,
                            frame_size: int,
                            K: int,
                            lsp_fft_prev,
                            verbose: bool = False) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    SciPy-Newton based search for frame speed minimizing MSE by finding the root
    of the numerical derivative d/ds MSE(s) over the search interval.

    - Uses on-the-fly SICER correction of observed IRs from `old_speed_base` to candidate `s`.
    - Predicts mic frames via FFT-overlap and computes MSE vs observed frames.
    - Applies Newton method on the derivative, with central differences and fallback.
    """
    # Config
    newton_params = params.get('newton', {})
    s_min = float(params['search_min_speed'])
    s_max = float(params['search_max_speed'])
    tol = float(newton_params.get('tol', 0.1))
    max_iters = int(newton_params.get('max_iters', 30))
    deriv_step = float(newton_params.get('deriv_step', 0.5))

    lo = min(s_min, s_max)
    hi = max(s_min, s_max)

    cache: Dict[float, float] = {}

    def mse_for(s: float) -> float:
        key = float(np.round(s, 9))
        if key in cache:
            return cache[key]
        val = _mse_for_speed_grid_item(
            float(s),
            obs_frames,
            lsp_curr,
            lsp_fft_cached,
            lsp_fft_prev,
            base_obs_IRs,
            old_speed_base,
            params,
            int(n_fft),
            int(frame_size),
            int(K),
        )
        cache[key] = val
        return val

    def cost_for(s):
        val = mse_for(s)
        return val

    def d_cost(s):
        h = deriv_step
        return (cost_for(s + h) - cost_for(s - h)) / (2 * h)


    def dd_cost(s):
        h = deriv_step
        return (cost_for(s + h) - 2 * cost_for(s) + cost_for(s - h)) / (h ** 2)


    # Initial guess: midpoint of search interval or old base speed clamped
    # x0 = float(np.clip(0.5 * (lo + hi), lo, hi))
    xs = np.linspace(lo, hi, num=10)
    ms = np.array([cost_for(float(x)) for x in xs], dtype=np.float64)
    x0 = float(xs[int(np.argmin(ms))])
    if verbose:
        print(f"  • Newton init: x0={x0:.3f}, mse0={cost_for(x0):.6e}")

    try:
        s_root = float(scipy_newton(func=d_cost, x0=x0, fprime=dd_cost, tol=float(tol), maxiter=int(max_iters)))
        if not (lo <= s_root <= hi):
            s_root = float(min(max(s_root, lo), hi))
        if verbose:
            print(f"  ✓ Newton: root at s={s_root:.3f}")
    except Exception as e:
        if verbose:
            print(f"  ✗ Newton failed: {e}; falling back to grid minimum")
        # Fallback: uniform sampling across interval
        samples = np.linspace(lo, hi, num=max(5, int(params.get('newton', {}).get('refine_samples', 21))))
        vals = [cost_for(float(s)) for s in samples]
        s_root = float(samples[int(np.argmin(vals))])

    # Build estimated IRs at best speed
    est_obs_IRs_best = _apply_sicer_to_obs_irs(base_obs_IRs, old_speed_base, s_root, params)
    return s_root, est_obs_IRs_best



def run_speed_tracking(paths: Dict, params: Dict, save_plots: bool = True,
                       save_results: bool = True, verbose: bool = False) -> Dict:
    """
    Frame-wise sound speed tracking using an estimation function.

    For each frame, uses a (to-be-implemented) estimator to predict the
    current sound speed and an estimated set of IRs to all mics (control and
    observation), based on observed mic signals, known loudspeaker outputs,
    and the initial true starting IR.

    Loudspeaker outputs are treated as known: a single loudspeaker
    (indexed by 'corr_lsp_idx') emits the input frame; others are zero.

    Args:
        paths: Shared paths including at least `results_path`, `input_audio_path`, and dataset paths.
        params: Experiment parameters used by `setup.initialize_parameters` and tracking keys:
                - 'obs_mic_types': list of mic groups to observe, e.g., ['BZ','DZ']
                - 'obs_mic_indices': dict with indices for each mic type
                - 'frame_duration', 'fs', 'input_duration', 'L', 'J', 'K'
                - 'speed_change_start_speed': initial true starting speed (IR known)
                - 'true_speed': ground-truth speed used to synthesize observations (simulated)
                - 'corr_lsp_idx': index of the loudspeaker that emits the known signal
        show_plots: Show tracking plots after processing.
        save_results: Save tracking results into `paths['results_path']`.

    Returns:
        results dict containing:
        - 'est_speed_track': array [n_frames] of estimated speed per frame
        - 'true_speed': ground-truth speed used to synthesize observations
        - 'corr_lsp_idx': loudspeaker index used for the known output
    """

    # Input signal and basic checks via setup
    # We reuse the audio load from setup but bypass IR loading (we need GT per-speed)
    input_signal, input_label = load_input_signal(paths, params)

    # Frame setup and data load
    _compute_frame_params_inplace(params)


    # Prepare I/O and parameters
    frame_size = params['frame_size']
    n_frames = params['frame_count']
    L, J = params['L'], params['J']

    # Active loudspeakers configuration (subset that emits sound)
    active_lsp_param = params.get('active_lsp_indices', params.get('use_lsp', np.arange(L)))
    active_lsp_indices = np.array(active_lsp_param, dtype=int)
    active_lsp_indices = np.unique(active_lsp_indices)
    active_lsp_indices = active_lsp_indices[(active_lsp_indices >= 0) & (active_lsp_indices < L)]
    # Helper to mask filters to active LSPs only
    def _mask_filter_rows(q: np.ndarray) -> np.ndarray:
        qm = np.zeros_like(q)
        qm[active_lsp_indices] = q[active_lsp_indices]
        return qm


    # ---- Simple cache for GT RIR loads to avoid duplicates ----
    rir_cache: Dict[int, Dict[str, np.ndarray]] = {}

    obs_lsp_idx = params.get('obs_lsp_indices', np.arange(L))
    start_speed = int(params.get('speed_change_start_speed', 333))
    true_speed = start_speed + np.zeros(n_frames, dtype=int)


    # Observation mic selection
    obs_mic_indices = params.get('obs_mic_indices', {})

    # Load IRs: initial true starting IRs (assumed known) and GT IRs for synthesizing observations
    gt_IRs_true_full = _load_gt_rirs_speed_cached(paths, params, start_speed, rir_cache)
    # Zones available
    full_mic_zones = ['BZ', 'DZ']
    # Precompute IR FFTs for true speed (full zones)
    K = int(params['K'])
    frame_size = params['frame_size']
    n_fft = next_fast_len(frame_size + K - 1)
    # FFT length for LSP filtering (input x q)
    n_fft_lsp = next_fast_len(frame_size + J - 1)

    # Determine mic indices to compute per zone: either all mics or only observation mics
    compute_obs_only = bool(params.get('compute_obs_only', False))
    mic_indices_compute = {}
    for t in full_mic_zones:
        if t in gt_IRs_true_full:
            if compute_obs_only and t in obs_mic_indices:
                mic_indices_compute[t] = np.asarray(obs_mic_indices[t])
            else:
                mic_indices_compute[t] = np.arange(gt_IRs_true_full[t].shape[0])

    # Build FFT tensors for the selected mic indices per zone
    IR_fft_compute = {t: rfft(gt_IRs_true_full[t][mic_indices_compute[t]], n=n_fft, axis=2) for t in mic_indices_compute.keys()}

    # Observation IRs (for estimators' SICER corrections)
    obs_IRs_true = {t: gt_IRs_true_full[t][obs_mic_indices[t]] for t in obs_mic_indices.keys() if t in gt_IRs_true_full}

    # Precompute mapping from computed mic indices to observation subset positions (for estimator inputs)
    obs_pos_in_compute = {}
    for t in obs_mic_indices.keys():
        if t in mic_indices_compute:
            comp = mic_indices_compute[t]
            obs = np.asarray(obs_mic_indices[t])
            mask = np.isin(comp, obs)
            pos = np.flatnonzero(mask)
            obs_pos_in_compute[t] = pos

    # Overlap-add states for computed microphones per zone
    prev_mic_true = {t: np.zeros((IR_fft_compute[t].shape[0], frame_size)) for t in IR_fft_compute.keys()}
    lsp_prev = np.zeros((L, J - 1))

    # Initialize loudspeaker frames
    lsp_curr = np.zeros((L, frame_size))
    lsp_prev = np.zeros((L, frame_size))
    lsp_fft_curr = np.zeros((L, n_fft // 2 + 1))

    # Initialize filter as the GT filter for the starting speed
    current_filter, loaded_ranks = load_vast_filters_for_speed(start_speed, paths, params)  # (ranks, L, J)
    if params['track_vast_rank'] not in loaded_ranks:
        raise ValueError(f"Requested VAST rank {params['track_vast_rank']} not found for speed {start_speed} (available: {loaded_ranks})")
    rank_idx = loaded_ranks.index(params['track_vast_rank'])
    current_filter = current_filter[rank_idx]  # (L, J)
    if not params['apply_filter']:
        current_filter = np.zeros((L, J))  # For no-control baseline
        current_filter[:, 0] = 1.0  # Dirac filter on all LSPs

    # Apply active-loudspeaker mask and precompute FFT for LSP generation (updated if filter changes)
    current_filter = _mask_filter_rows(current_filter)
    q_fft_curr = rfft(current_filter, n=n_fft_lsp, axis=1)
    current_filter_speed = float(start_speed)

    # Allocate output buffers for signals (either all mics or observation-only)
    total_len = n_frames * frame_size
    mic_signals = {}
    for t in IR_fft_compute.keys():
        mic_signals[t] = np.zeros((IR_fft_compute[t].shape[0], total_len), dtype=np.float64)
    # Desired signals for BZ (if available) using delta filter with model_delay
    desired_bz_signals = None
    prev_desired_bz = None
    desired_lsp_idx = int(params.get('ref_source', 0))
    model_delay = int(params.get('model_delay', 0))
    # LSP OLA buffer for desired-only filtering
    lsp_prev_des = np.zeros((L, frame_size), dtype=np.float64)
    # Build desired filter (delta at model_delay), then mask to active LSPs
    J = int(params['J'])
    q_desired = np.zeros((L, J), dtype=np.float64)
    if 0 <= model_delay < J and 0 <= desired_lsp_idx < L:
        q_desired[desired_lsp_idx, model_delay] = 1.0
    q_desired = _mask_filter_rows(q_desired)
    # Precompute desired filter FFT (constant)
    q_fft_desired = rfft(q_desired, n=n_fft_lsp, axis=1)
    if 'BZ' in IR_fft_compute:
        desired_bz_signals = np.zeros((IR_fft_compute['BZ'].shape[0], total_len), dtype=np.float64)
        prev_desired_bz = np.zeros((IR_fft_compute['BZ'].shape[0], frame_size), dtype=np.float64)

    # Bisection base switching
    old_speed_base = float(params['estimator_base_speed'])
    if old_speed_base != float(start_speed):
        IRs_estimator_start_speed = _load_gt_rirs_speed_cached(paths, params, int(round(old_speed_base)), rir_cache)
        base_obs_IRs = {t: IRs_estimator_start_speed[t][obs_mic_indices[t]] for t in obs_mic_indices.keys()}
    else:
        base_obs_IRs = obs_IRs_true.copy()

    # Search wnindow initial config
    adaptive_base = params['adaptive_base']
    adaptive_window = params['adaptive_window']
    if not adaptive_window:
        if verbose:
            print("-> Using fixed search window for speed estimation during tracking. Tolerance:", params['grid_search']['tol_init'])
        params['grid_search']['tol'] = params['grid_search']['tol_init']
    search_min_speed_0 = params['search_min_speed']
    search_max_speed_0 = params['search_max_speed']

    if verbose:
        if adaptive_base:
            print("-> Using adaptive base for SICER IRs during tracking")
        if adaptive_window:
            print("-> Using adaptive search window for speed estimation during tracking")
        if len(active_lsp_indices) < L:
            print("-> Using active loudspeaker subset for LSP generation during tracking:", active_lsp_indices.tolist())
        print("-> Using observation mics for speed estimation:", obs_mic_indices)

    # Tracking outputs
    est_speed_track = np.empty(n_frames)
    est_speed_track.fill(np.nan)
    # Keep track of exactly what speed is used for the current filter
    filter_speed_track = np.empty(n_frames)
    filter_speed_track.fill(np.nan)
    input_signal_frame_power = np.zeros(n_frames)
    grid_val_track = []
    grid_sample_track = []
    AC_performance_track = np.zeros(n_frames)
    nSDP_performance_track = np.zeros(n_frames)

    # Baseline mode setup (use GT filters or SICER filters, no estimation)
    if params.get('gt_baseline', False) and params.get('sicer_baseline', False):
        raise ValueError("Cannot use both 'gt_baseline' and 'sicer_baseline' modes simultaneously")
    if params.get('gt_baseline', False):
        baseline_method = "GT"
    elif params.get('sicer_baseline', False):
        baseline_method = "SICER"

    baseline_mode = bool(params.get('gt_baseline', False)) or bool(params.get('sicer_baseline', False))

    # Frame loop (optionally disable progress bar for parallel runs)
    _disable_pb = bool(params.get('disable_progressbar', False) or params.get('quiet', False))
    for fno in tqdm.tqdm(range(n_frames), desc="Speed estimation tracking", disable=_disable_pb):
        f_start = fno * frame_size
        f_end = (fno + 1) * frame_size
        input_frame = input_signal[f_start:f_end]
        input_signal_frame_power[fno] = float(np.mean(input_frame ** 2))

        # True speed update based on schedule
        if fno in params['speed_change_frames']:
            true_speed[fno:] += params['speed_change']

            # Update GT IRs for zones and recompute FFTs for selected mic indices
            gt_IRs_true_full = _load_gt_rirs_speed_cached(paths, params, int(true_speed[fno]), rir_cache)
            IR_fft_compute = {t: rfft(gt_IRs_true_full[t][mic_indices_compute[t]], n=n_fft, axis=2) for t in mic_indices_compute.keys()}

            # In baseline mode, switch to GT or SICER filter for the new true speed
            if baseline_mode and params['apply_filter']:
                current_filter_speed = int(true_speed[fno])
                if baseline_method == "GT":
                    filters_baseline, loaded_ranks = load_vast_filters_for_speed(current_filter_speed, paths, params)
                elif baseline_method == "SICER":
                    filters_baseline, loaded_ranks = load_sicer_filters_for_pair(current_filter_speed, old_speed_base, paths, params)
                else:
                    raise ValueError(f"Unknown baseline method: {baseline_method}")

                if params['track_vast_rank'] not in loaded_ranks:
                    raise ValueError(f"Requested VAST rank {params['track_vast_rank']} not found for speed {start_speed} (available: {loaded_ranks})")
                r_idx = loaded_ranks.index(params['track_vast_rank'])
                current_filter = _mask_filter_rows(filters_baseline[r_idx])
                q_fft_curr = rfft(current_filter, n=n_fft_lsp, axis=1)


        # Compute all loudspeaker frame using current filter and input signal
        lsp_fft_prev = lsp_fft_curr.copy()
        # Precompute input FFT for LSP filtering once per frame
        S_fft_lsp = rfft(input_frame, n=n_fft_lsp)
        lsp_curr, lsp_prev = compute_lsp_signal_frame(
            current_filter, input_frame, params, lsp_prev,
            q_fft=q_fft_curr, S_fft=S_fft_lsp, n_fft=n_fft_lsp
        )

        # Compute LSP FFT once per frame and reuse
        lsp_fft_curr = rfft(lsp_curr, n=n_fft, axis=1)

        # Compute frames for selected mics per zone, store, and derive observation frames
        obs_true_frames = {}
        for t in IR_fft_compute.keys():
            mic_frame, prev_mic_true[t] = compute_mic_signal_frame_fft_overlap(
                IR_fft_compute[t], lsp_curr, prev_mic_true[t], frame_size, K, n_fft=n_fft, lsp_fft=lsp_fft_curr
            )
            mic_signals[t][:, f_start:f_end] = mic_frame
            # If this zone is used for estimation, select observation subset positions
            if t in obs_mic_indices.keys() and t in obs_pos_in_compute:
                obs_true_frames[t] = mic_frame[obs_pos_in_compute[t], :]

        # Performance metrics (AC, nSDP) at current true speed, with estimated filter from previous frame
        if not compute_obs_only:  # Only track performance when computing BZ and DZ mics
            BZ_frame = mic_signals['BZ'][:, f_start:f_end]
            DZ_frame = mic_signals['DZ'][:, f_start:f_end]
            # AC computation
            AC_perf = 10*np.log10(np.mean(np.abs(BZ_frame)**2) / np.mean(np.abs(DZ_frame)**2))
            AC_performance_track[fno] = AC_perf

            # Desired BZ signals via delta filter (model_delay) and standard pipeline
            # Only need to track when not computing obs-only mics
            if desired_bz_signals is not None:
                lsp_des_curr, lsp_prev_des = compute_lsp_signal_frame(
                    q_desired, input_frame, params, lsp_prev_des,
                    q_fft=q_fft_desired, S_fft=S_fft_lsp, n_fft=n_fft_lsp
                )
                lsp_des_fft = rfft(lsp_des_curr, n=n_fft, axis=1)
                des_frame_bz, prev_desired_bz = compute_mic_signal_frame_fft_overlap(
                    IR_fft_compute['BZ'], lsp_des_curr, prev_desired_bz, frame_size, K, n_fft=n_fft, lsp_fft=lsp_des_fft
                )
                desired_bz_signals[:, f_start:f_end] = des_frame_bz

                # nSDP computation
                err_frame = BZ_frame - des_frame_bz
                err_power = np.mean(err_frame ** 2)
                des_power = np.mean(des_frame_bz ** 2)
                des_power = max(des_power, 1e-12)  # avoid div by zero
                nSDP_perf = 10 * np.log10(err_power / des_power)
                nSDP_performance_track[fno] = nSDP_perf


        # Assuming time-sync issues are handled, and sound travels instantly

        # # Estimate current speed and IRs via continuous bisection on previous frame
        if fno > 0 and adaptive_window:
            # Search window centered around last estimate
            middle_est = np.nanmean(est_speed_track[max(0, fno - params['search_middle_lookback']):fno])
            width = params['search_window_width'] / 2
            params['search_min_speed'] = max(middle_est - width, search_min_speed_0)
            params['search_max_speed'] = min(middle_est + width, search_max_speed_0)


        # Estimate current speed and IRs via selected estimator (default: Newton)
        if not baseline_mode:
            estimator = str(params.get('estimator', 'newton')).lower()
            grid_vals, grid_samples = [], []
            if estimator == 'bisect':
                est_speed, est_obs_IRs, grid_vals, grid_samples = _bisect_speed_for_frame(
                    obs_true_frames,
                    lsp_curr,
                    lsp_fft_curr,
                    base_obs_IRs,
                    old_speed_base,
                    params,
                    n_fft,
                    frame_size,
                    K,
                    lsp_fft_prev,
                    verbose=False,
                )
            elif estimator == 'grid':
                est_speed, est_obs_IRs, grid_vals, grid_samples = _grid_search_speed_for_frame(
                    obs_true_frames,
                    lsp_curr,
                    lsp_fft_curr,
                    base_obs_IRs,
                    old_speed_base,
                    params,
                    n_fft,
                    frame_size,
                    K,
                    fno,
                    lsp_fft_prev,
                    verbose=False,
                )
            else:
                est_speed, est_obs_IRs = _newton_speed_for_frame(
                    obs_true_frames,
                    lsp_curr,
                    lsp_fft_curr,
                    base_obs_IRs,
                    old_speed_base,
                    params,
                    n_fft,
                    frame_size,
                    K,
                    lsp_fft_prev,
                    verbose=False,
                )
            if verbose:
                print(f"Frame {fno}: estimated speed = {est_speed:.2f} m/s (true: {true_speed[fno]} m/s)")
            est_speed_track[fno] = float(est_speed)
            grid_val_track.append(grid_vals)
            grid_sample_track.append(grid_samples)

            # Update filter based on estimated speed: compute/load SICER VAST filters for tracked rank
            # Only update if speed change is significant compared to currently used filter speed
            speed_diff = abs(float(est_speed_track[fno]) - current_filter_speed)
            if (params['update_filter'] and params['apply_filter'] and (fno > 0)
                and (speed_diff >= params['update_filter_speed_diff'])):
                base_full_rirs = None
                if not adaptive_base:
                    # When not adaptive, we can reuse full GT RIRs for the current base
                    base_full_rirs = _load_gt_rirs_speed_cached(paths, params, int(round(old_speed_base)), rir_cache)
                current_filter = _update_filter_for_speed(
                    np.round(est_speed, 1), float(old_speed_base),
                    paths, params, adaptive_mode=adaptive_base,
                    adaptive_old_old_speed=None,
                    base_rirs_opt=base_full_rirs,
                    verbose=verbose
                )
                # Update cached filter FFT for next frames
                current_filter = _mask_filter_rows(current_filter)
                q_fft_curr = rfft(current_filter, n=n_fft_lsp, axis=1)
                current_filter_speed = float(est_speed_track[fno])  # Update current filter speed
                if verbose:
                    print(f"  • Updated LSP filter for speed {est_speed_track[fno]:.2f} m/s")
        else:
            # Baseline mode: no estimation; speed equals true speed
            est_speed_track[fno] = float(true_speed[fno])
            grid_val_track.append([])
            grid_sample_track.append([])

        filter_speed_track[fno] = np.round(float(current_filter_speed), 1)

        # Adaptive base switching
        if adaptive_base:
            base_obs_IRs = est_obs_IRs
            old_speed_base = np.round(float(est_speed), 1)

    results = {
        'est_speed_track': est_speed_track,
        'true_speed': true_speed,
        'obs_lsp_indices': obs_lsp_idx,
        'input_signal_power': input_signal_frame_power,
        'speed_change_frames': params['speed_change_frames'],
        'input_signal_array': input_signal,
        'grid_val_f0': grid_val_track[0] if len(grid_val_track) > 0 else [],
        'grid_sample_f0': grid_sample_track[0] if len(grid_sample_track) > 0 else [],
        # Variable-length per-frame lists saved as object arrays for NPZ
        'grid_val_track': np.array(grid_val_track[1:], dtype=object) if len(grid_val_track) > 1 else np.array([], dtype=object),
        'grid_sample_track': np.array(grid_sample_track[1:], dtype=object) if len(grid_sample_track) > 1 else np.array([], dtype=object),
        'input_label': input_label,
        'AC_performance_track': AC_performance_track,
        'nSDP_performance_track': nSDP_performance_track,
    }

    if save_results:
        # Nested, descriptive directory scheme similar to process_positions.save_metrics
        from datetime import datetime
        setup_str = str(params.get('array_setup', 'setup'))
        rt60_str = f"{float(params.get('rt60', 0.0))}"
        estimator = str(params.get('estimator', 'newton')).lower()
        start_speed = int(params.get('speed_change_start_speed', 333))
        speed_step = int(params.get('speed_change', 0))
        filters_updated = bool(params.get('update_filter', False))
        filters_applied = bool(params.get('apply_filter', False))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Active LSP tag for results naming
        all_use = np.array(params.get('use_lsp', np.arange(L)), dtype=int)
        active_sorted = sorted([int(x) for x in active_lsp_indices.tolist()])
        if len(active_sorted) == len(all_use) and np.all(np.sort(all_use) == np.array(active_sorted)):
            active_tag = 'active_lsp_all'
        else:
            active_tag = 'active_lsp_' + ('_'.join(str(i) for i in active_sorted))

        # Observation mic indices tag (by zone)
        obs_pieces = []
        for z in ['BZ', 'DZ']:
            if z in obs_mic_indices:
                idxs = np.asarray(obs_mic_indices[z]).astype(int).tolist()
                idxs_sorted = sorted(set(int(x) for x in idxs))
                obs_pieces.append(f"{z.lower()}_" + ("_".join(str(i) for i in idxs_sorted) if len(idxs_sorted) > 0 else "none"))
        other_zones = sorted([z for z in obs_mic_indices.keys() if z not in ('BZ','DZ')])
        for z in other_zones:
            idxs = np.asarray(obs_mic_indices[z]).astype(int).tolist()
            idxs_sorted = sorted(set(int(x) for x in idxs))
            obs_pieces.append(f"{z.lower()}_" + ("_".join(str(i) for i in idxs_sorted) if len(idxs_sorted) > 0 else "none"))
        obs_tag = 'obs_' + ('__'.join(obs_pieces) if len(obs_pieces) > 0 else 'none')



        window_descriptor = "" # to avoid issues if adaptive window is off
        if adaptive_window:
            window_descriptor = f"_ww{params['search_window_width']}_ml{params['search_middle_lookback']}"

        # filter_descriptor = "no_filter"
        if filters_updated:
            filter_descriptor = f"filters_updated_true_sd{params['update_filter_speed_diff']}"
        else:
            filter_descriptor = "filters_updated_false"

        # Build result directory with optional experiment_id component before timestamp
        root_dir = (
            paths['results_path']
            / "speed_tracking"
            / f"{setup_str}"
            / f"RT60_{rt60_str}"
            / f"input_{input_label}"
            / f"start_{start_speed}_step_{speed_step}_dt_{params['speed_change_time']}"
            / active_tag
            / obs_tag
        )

        if baseline_mode:
            base_dir_prefix = root_dir / f"{baseline_method.lower()}_baseline_r_{params['track_vast_rank']}"
        else:
            base_dir_prefix = (
                root_dir
                / (f"filters_applied_r_{params['track_vast_rank']}" if filters_applied else "filters_applied_false")
                / filter_descriptor
                / f"adaptive_base_{adaptive_base}"
                / f"adaptive_window_{adaptive_window}{window_descriptor}"
                / f"estimator_{estimator}"
            )

        exp_id = params.get('experiment_id', None)
        if exp_id:
            base_dir = base_dir_prefix / f"exp_{exp_id}" / timestamp
        else:
            base_dir = base_dir_prefix / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)

        out_path = base_dir.joinpath("results.npz")
        np.savez_compressed(str(out_path), **results)

        # # Save microphone signals separately for post-processing (names indicate obs-only mode)
        # bz_path = None
        # dz_path = None
        # if 'BZ' in mic_signals:
        #     bz_file = 'bz_mic_signals_obs_only.npz' if compute_obs_only else 'bz_mic_signals.npz'
        #     bz_path = base_dir.joinpath(bz_file)
        #     np.savez_compressed(str(bz_path), BZ=mic_signals['BZ'], indices=np.asarray(mic_indices_compute['BZ']))
        # if 'DZ' in mic_signals:
        #     dz_file = 'dz_mic_signals_obs_only.npz' if compute_obs_only else 'dz_mic_signals.npz'
        #     dz_path = base_dir.joinpath(dz_file)
        #     np.savez_compressed(str(dz_path), DZ=mic_signals['DZ'], indices=np.asarray(mic_indices_compute['DZ']))

        # # Save desired BZ signals if computed
        # desired_bz_path = None
        # if desired_bz_signals is not None:
        #     des_file = 'desired_bz_signals_obs_only.npz' if compute_obs_only else 'desired_bz_signals.npz'
        #     desired_bz_path = base_dir.joinpath(des_file)
        #     np.savez_compressed(str(desired_bz_path), desired_BZ=desired_bz_signals,
        #                         indices=np.asarray(mic_indices_compute['BZ']), desired_lsp_idx=int(desired_lsp_idx))

        # # Save small references to full-signal files in meta
        # results_paths = {'bz_signals_path': str(bz_path) if bz_path is not None else None,
        #          'dz_signals_path': str(dz_path) if dz_path is not None else None,
        #          'desired_bz_path': str(desired_bz_path) if desired_bz_path is not None else None}

        # Save params + paths JSON metadata
        def _to_jsonable(obj):
            if isinstance(obj, dict):
                return {str(k): _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            try:
                from pathlib import Path
                if isinstance(obj, Path):
                    return str(obj)
            except Exception:
                pass
            return obj

        json_path = base_dir.joinpath("meta.json")
        with open(json_path, 'w') as jf:
            meta = {
                'params': _to_jsonable(params),
                'params_meta': _to_jsonable(params.get('params_meta', {})),
                'experiment_id': _to_jsonable(params.get('experiment_id', None)),
                'timestamp': timestamp,
                'paths': _to_jsonable(paths),
                # 'signals': results_paths,
                'compute_obs_only': compute_obs_only,
            }
            json.dump(meta, jf, indent=2)

    if save_plots:  # Save plots alongside the results files
        _plot_speed_tracking({**results,**params}, base_dir, verbose=verbose)
        _plot_grid_search_loss({**results,**params}, base_dir, verbose=verbose)

    return results



def _plot_speed_tracking(results: Dict, save_path: Optional[str]=None, verbose: bool=False) -> None:
    """
    Plot estimated speed, true speed, and performance metrics over frames for the current tracking run.
    """
    est = results['est_speed_track']
    true = results['true_speed']
    ac_perf = results.get('AC_performance_track', None)
    nsdp_perf = results.get('nSDP_performance_track', None)
    # input_signal_power = results.get('input_signal_power', None)
    input_signal = results.get('input_signal_array', None)
    speed_change_frames = results.get('speed_change_frames', [])
    n_frames = est.shape[0]
    frame_size = results['frame_size']


    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    t = np.arange(n_frames)

    # Estimated speed per frame
    axes[0].plot(t, true, lw=2, label='True Speed', linestyle='-', marker='x')
    axes[0].plot(t, est, lw=2, label='Estimated Speed', marker='o', linestyle='--', markerfacecolor='none')
    for scf in speed_change_frames:
        axes[0].axvline(scf, color='red', linestyle=':', alpha=0.7)
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].set_xlabel('Frame')
    axes[0].grid(True, ls='--', alpha=0.4)
    axes[0].legend()

    # AC Performance
    if ac_perf is not None:
        axes[1].plot(t, ac_perf, lw=2, color='green')
        axes[1].set_ylabel('AC [dB]')
        axes[1].set_xlabel('Frame')
        axes[1].grid(True, ls='--', alpha=0.4)
    else:
        axes[1].set_visible(False)

    # nSDP Performance
    if nsdp_perf is not None:
        axes[2].plot(t, nsdp_perf, lw=2, color='purple')
        axes[2].set_ylabel('nSDP [dB]')
        axes[2].set_xlabel('Frame')
        axes[2].grid(True, ls='--', alpha=0.4)
    else:
        axes[2].set_visible(False)

    if input_signal is not None:
        samples = np.arange(len(input_signal))
        frame_indices = np.arange(frame_size, len(input_signal)+frame_size, frame_size)
        # axes[1].plot(t, 10*np.log10(input_signal_power), lw=2, color='orange')
        axes[-1].plot(samples, input_signal, lw=2, color='orange')
        axes[-1].set_ylabel('amplitude')
        axes[-1].set_xlabel('Sample')
        axes[-1].vlines(frame_indices, ymin=min(input_signal), ymax=max(input_signal), color='gray', linestyle=':', alpha=0.5)
        # Annotate frames
        for fi in frame_indices:
            # - 20 samples offset for visibility
            x_offset = int(frame_size/1.2)
            axes[-1].text(fi-x_offset, max(input_signal)-0.1*(max(input_signal)-min(input_signal)),
                         f'F{int(fi//frame_size)-1}', rotation=0, verticalalignment='bottom', fontsize=7, color='gray')

        axes[-1].grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.mkdir(parents=True, exist_ok=True)
        out_name = out_path.joinpath(f'speed_tracking_plot.png')
        plt.savefig(out_name, dpi=300)
        if verbose:
            print(f"Plot saved as '{out_name}'")
    else:
        plt.savefig('speed_tracking_plot.png', dpi=300)
        if verbose:
            print("Plot saved as 'speed_tracking_plot.png'")

    # plt.show()

def _plot_grid_search_loss(results: Dict, save_path: Optional[str]=None, verbose: bool=False) -> None:
    """
    Plot grid search loss curves per frame for the current tracking run, grouped by true speed. Each frame's curve is normalized and plotted with low alpha for visibility. Vertical lines indicate true speeds. Titles indicate true speed and frame indices for each group.
    """
    grid_vals = results['grid_val_track']
    grid_samples = results['grid_sample_track']
    grid_vals_f0 = results['grid_val_f0']
    grid_samples_f0 = results.get('grid_sample_f0')
    if len(grid_vals_f0) == 0:
        print("No grid search data to plot.")
        return
    true_speed = results['true_speed']
    unique_speeds = np.unique(true_speed)
    unique_speeds.sort()
    n_speeds = unique_speeds.shape[0]
    n_frames = len(grid_vals)
    if n_speeds < 2:
        height = 4
        color = 'k'
        alpha = 0.3
    else:
        color = None
        height = 3 * n_speeds
    fig, axes = plt.subplots(n_speeds, 1, figsize=(10, height), sharex=True)
    for idx, speed in enumerate(unique_speeds):
        speed_indices = np.where(true_speed == speed)[0]
        ax_speed = axes if n_speeds == 1 else axes[idx]
        if n_speeds >= 2:
            color = None
            alpha = np.max([0.3, 1.0 / len(speed_indices)])
        for si in speed_indices:
            if si == 0 and grid_vals_f0 is not None and grid_samples_f0 is not None:
                samples = grid_samples_f0
                vals = grid_vals_f0
            else:
                idx = si - 1 if grid_vals_f0 is not None else si
                samples = grid_samples[idx]
                vals = grid_vals[idx]
            ax_speed.plot(samples, vals/np.max(vals), marker=None, alpha=alpha, c=color, label=f'Frame {si}' if n_speeds >= 2 else None)
        ax_speed.axvline(speed, color='red', linestyle='--', label='True Speed', alpha=0.3)
        ax_speed.set_title(f'Loss per frame grid search (True Speed: {speed} m/s). Frames: [{speed_indices.min()}, {speed_indices.max()}]')
        ax_speed.set_ylabel('Normalized Cost')
        ax_speed.set_xlabel('Speed (m/s)')
        ax_speed.grid(True, ls='--', alpha=0.4)
        if n_speeds >= 2:
            ax_speed.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.mkdir(parents=True, exist_ok=True)
        out_name = out_path.joinpath(f'grid_search_results.png')
        plt.savefig(out_name, dpi=300)
        if verbose:
            print(f"Plot saved as '{out_name}'")
    else:
        plt.savefig('grid_search_results.png', dpi=300)
        if verbose:
            print("Plot saved as 'grid_search_results.png'")
    # plt.show()


def _update_filter_for_speed(new_speed: float, old_speed_base: float, paths: Dict,
                             params: Dict, adaptive_mode: bool = False,
                             adaptive_old_old_speed: Optional[float] = None,
                             base_rirs_opt: Optional[Dict[str, np.ndarray]] = None,
                             verbose: bool = False) -> np.ndarray:
    """Update control filter for a new speed by loading or computing SICER-based VAST filter.

    - Checks for existing SICER filters for (old→new) and mode.
    - If missing, ensures corrected full RIRs saved, computes covariances and VAST filter for tracked rank.
    - Returns filter of shape (L, J).
    """
    J = int(params['J'])
    L = len(params['use_lsp'])
    rt60 = params['rt60']
    array_setup = params['array_setup']
    rank = int(params['track_vast_rank'])
    mode_tag = 'adaptive' if adaptive_mode else 'normal'
    if not adaptive_mode:
        old_speed_base = int(round(old_speed_base))
    if new_speed == int(new_speed):
        new_speed = int(new_speed)

    filters_dir = paths['filter_files_path'].joinpath(f"RT60_{rt60}", 'SICER_VAST_filters')
    filters_dir.mkdir(parents=True, exist_ok=True)


     # 1) Try offline multi-rank filters first (no mode tag)
    if not adaptive_mode:
        try:
            filters_array, rank_list = load_sicer_filters_for_pair(new_speed, old_speed_base, paths, params)  # for ranks info
            if rank not in rank_list:
                    raise ValueError(f"Requested VAST rank {rank} not found in off-line SICER for speed {old_speed_base} -> {new_speed} (available: {rank_list})")
            r = rank_list.index(rank)
            return filters_array[r]  # (L, J)
        except FileNotFoundError:  # The offline multi-rank file may not exist
            pass
        except Exception:
            print("Unexpected error:", sys.exc_info()[0])
            raise
    # 2) Check for mode-tagged single-rank file
    if adaptive_mode:
        online_name = f"SICER_VAST_filters_init_speed_{old_speed_base}_interp_to_{new_speed}_J_{J}_{mode_tag}_rank_{rank}.npz"
    else:
        online_name = f"SICER_VAST_filters_init_speed_{old_speed_base}_interp_to_{new_speed}_J_{J}_rank_{rank}.npz"
    online_file = filters_dir.joinpath(online_name)
    if online_file.exists():
        data = np.load(str(online_file))
        if 'filter' in data:
            return data['filter']
        elif 'filters' in data:
            filters_array = data['filters']
            return filters_array[0]

    # 3) Ensure corrected full RIRs exist and/or generate them (save with mode tag)
    corrected_dir = paths['data_path'].joinpath(f"simulated_RIRs/SICER_corrected/{array_setup}_RT60_{rt60}")
    corrected_dir.mkdir(parents=True, exist_ok=True)
    if adaptive_mode:
        corrected_mat = corrected_dir.joinpath(f"init_{old_speed_base}_to_{new_speed}_{mode_tag}.mat")
    else:
        corrected_mat = corrected_dir.joinpath(f"init_{old_speed_base}_to_{new_speed}.mat")
    if not corrected_mat.exists():
        if verbose:
            print(f"Updating filters: computing corrected RIRs for {old_speed_base} to {new_speed} ({mode_tag})...")
        # Load full baseline IRs and apply SICER to all mics/LSP
        if not adaptive_mode:
            base_rirs = base_rirs_opt if base_rirs_opt is not None else load_gt_rirs_speed(paths, params, int(old_speed_base))
        else:
            base_mat = corrected_dir.joinpath(f"init_{adaptive_old_old_speed}_to_{old_speed_base}_{mode_tag}.mat")
            # For adaptive, we must load SICER corrected RIRs from previous step
            data = sio.loadmat(str(base_mat))
            rirs_base = data['RIRs_corrected']
            base_rirs = {'BZ': np.asarray(rirs_base['BZ'][0, 0]),
                          'DZ': np.asarray(rirs_base['DZ'][0, 0])}
            del data, rirs_base  # Free memory


        corrected = _apply_sicer_to_full_rirs(base_rirs, float(old_speed_base), float(new_speed), params)
        sio.savemat(str(corrected_mat), {
            "RIRs_corrected": {
                'BZ': corrected.get('BZ'),
                'DZ': corrected.get('DZ'),
            },
            "old_speed": float(old_speed_base),
            "new_speed": float(new_speed),
            "mode": mode_tag,
            "K": int(params['K']),
        })
        del base_rirs, corrected  # Free memory

    # 4) Compute covariances for corrected RIRs (mode-tagged names)
    if verbose:
        print(f"Updating filters: computing covariances for {old_speed_base} to {new_speed} ({mode_tag})...")
    covar_dir = paths['covariance_path'].joinpath('SICER_filters')
    covar_dir.mkdir(parents=True, exist_ok=True)
    covar_name = (
        f"{array_setup}_RT60_{rt60}_init_speed_{int(old_speed_base):d}_interp_to_{new_speed}_J_{J}_K_{int(params['K'])}_SICER_{mode_tag}"
    )
    str_save_R_B = covar_dir.joinpath(f'R_B_{covar_name}.npz')
    str_save_R_D = covar_dir.joinpath(f'R_D_{covar_name}.npz')
    str_save_r_B = covar_dir.joinpath(f'cross_r_B_{covar_name}.npz')
    if not (str_save_R_B.exists() and str_save_R_D.exists() and str_save_r_B.exists()):
        # Load corrected RIRs once
        data = sio.loadmat(str(corrected_mat))
        rirs_corr = data['RIRs_corrected']
        imp_resp_bz = np.asarray(rirs_corr['BZ'][0, 0])
        imp_resp_dz = np.asarray(rirs_corr['DZ'][0, 0])

        def _compute_RB_rB():
            # Compute and save R_B and r_B if missing
            if str_save_R_B.exists() and str_save_r_B.exists():
                return
            G_B = get_zone_convolution_mat(imp_resp_bz, int(J))
            p_T = G_B[:, int(params['ref_source']) * int(J)]
            p_T = p_T.reshape((imp_resp_bz.shape[0], imp_resp_bz.shape[-1] + int(J) - 1))
            d_B = np.zeros_like(p_T)
            md = int(params['model_delay'])
            # Note: assumes md > 0 as in original helper; keep behavior consistent
            d_B[:, md:] = p_T[:, :-md] if md > 0 else p_T
            d_B = d_B.flatten()

            if not str_save_R_B.exists():
                R_B_loc = G_B.T @ G_B
                np.savez_compressed(str_save_R_B, R_B=R_B_loc)
                del R_B_loc
            if not str_save_r_B.exists():
                r_B_loc = G_B.T @ d_B
                np.savez_compressed(str_save_r_B, r_B=r_B_loc)
                del r_B_loc
            del G_B, d_B, p_T

        def _compute_RD():
            # Compute and save R_D if missing
            if str_save_R_D.exists():
                return
            G_D = get_zone_convolution_mat(imp_resp_dz, int(J))
            R_D_loc = G_D.T @ G_D
            np.savez_compressed(str_save_R_D, R_D=R_D_loc)
            del G_D, R_D_loc

        # Run BZ and DZ covariance builds concurrently
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(_compute_RB_rB)
            f2 = ex.submit(_compute_RD)
            # Wait for both and surface any exceptions
            _ = f1.result()
            _ = f2.result()

        del data, rirs_corr, imp_resp_bz, imp_resp_dz  # Free memory
    # 5) Compute VAST filter for tracked rank and save
    R_B = np.load(str_save_R_B)['R_B']
    R_D = np.load(str_save_R_D)['R_D']
    r_B = np.load(str_save_r_B)['r_B']
    if verbose:
        print(f"Updating filters: computing VAST filter for rank {rank}...")
    eig_vec, eig_val = diagonalize_matrices(R_B, R_D, descend=True)
    mu = params['mu']
    q_vast = fit_vast_closed_form(rank, mu, r_B, J, L, eig_vec, eig_val, mat_out=True)

    np.savez_compressed(str(online_file), filter=q_vast,
                        new_speed=int(round(new_speed)), old_speed=int(old_speed_base),
                        mode=mode_tag, rank=rank)
    return q_vast


def _compute_frame_params_inplace(params: Dict) -> None:
    params['frame_size'] = int(params['frame_duration'] * params['fs'])
    params['frame_count'] = int(params['input_duration'] * params['fs']) // params['frame_size']
    params['speed_change_times'] = np.arange(params['speed_change_time'], params['input_duration'],
                                                 params['speed_change_time'],
                                              )
    params['speed_change_frames'] = (params['speed_change_times'] * params['fs']) // params['frame_size']




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate sound-speed tracking algorithm for robust SZC with VAST filters (GT or SICER) across sound speeds and ranks according to a config file and optional overrides.')
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to config module (e.g., exp1_SICER/config_RT60_0.1.py)')
    parser.add_argument('--apply-filter', dest='apply_filter', action='store_true',
                        help='Whether to apply the control filter during tracking. Overrides config setting.')
    parser.add_argument('--update-filter', dest='update_filter', action='store_true',
                        help='Whether to update the control filter during tracking. Overrides config setting.')
    parser.add_argument('--estimator', type=str, default='newton',
                        help='Estimator to use for speed tracking: "newton" or "bisect". Overrides config setting.')
    parser.add_argument('--obs-only', dest='obs_only', action='store_true',
                        help='Compute only observation microphone signals (faster).')
    parser.add_argument('--gt-baseline', dest='gt_baseline', action='store_true',
                        help='Compute baseline performance using precomputed GT VAST filters (no tracking).')
    parser.add_argument('--sicer-baseline', dest='sicer_baseline', action='store_true',
                        help='Compute baseline performance using precomputed SICER VAST filters (no tracking).')
    parser.add_argument('--active-lsp', dest='active_lsp', type=str, default=None,
                        help='Comma-separated list of active loudspeaker indices (0-based), e.g., "7" or "7,8". Defaults to all in use_lsp.')
    parser.add_argument('--params-file', dest='params_file', type=Path, default=None,
                        help='Path to a JSON file with parameter overrides. Keys in this file override the base config. CLI flags still take precedence.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        default=False,
                        help='Enable verbose output during tracking.')
    parser.add_argument('--quiet', dest='quiet', action='store_true',
                        default=False,
                        help='Reduce console output (disable progress bar and final completion print).')
    args = parser.parse_args()

    cfg = load_config_module(args.config)
    paths = get_shared_paths(config_env=CONFIG_ENV)
    params = cfg.shared_params.copy()

    # Apply JSON overrides before CLI flags so CLI wins in precedence
    if args.params_file is not None:
        try:
            with open(args.params_file, 'r') as pf:
                overrides = json.load(pf)
            if isinstance(overrides, dict):
                # Shallow update; nested dicts (e.g., grid_search) should be provided complete if overriding
                params.update(overrides)
            else:
                print(f"Warning: Ignoring params file {args.params_file} (not a dict)")
        except Exception as e:
            print(f"Warning: Failed to load params file {args.params_file}: {e}")

    if args.apply_filter:
        params['apply_filter'] = True

    if params['apply_filter'] is True:
        # Only consider updating filter if applying it
        if args.update_filter:
            params['update_filter'] = True
        else:
            params['update_filter'] = False
    params['estimator'] = args.estimator.lower()
    # Speed-up option: compute only observation microphones
    if args.obs_only:
        params['compute_obs_only'] = True

    # Active loudspeakers from CLI
    if args.active_lsp:
        try:
            parsed = [int(s) for s in args.active_lsp.split(',') if s.strip()!='']
            params['active_lsp_indices'] = np.asarray(parsed, dtype=int)
        except Exception:
            print(f"Warning: could not parse --active-lsp='{args.active_lsp}', using defaults.")


    # Run tracking (optionally in GT-baseline mode)
    if args.gt_baseline and args.sicer_baseline:
        raise ValueError("Cannot use both --gt-baseline and --sicer-baseline flags simultaneously.")
    if args.gt_baseline:
        params['gt_baseline'] = True
        params['apply_filter'] = True
        params['update_filter'] = False
    if args.sicer_baseline:
        params['sicer_baseline'] = True
        params['apply_filter'] = True
        params['update_filter'] = False
    if args.quiet:
        params['quiet'] = True
    results = run_speed_tracking(paths, params, save_plots=True, save_results=True, verbose=args.verbose)
    if not bool(params.get('quiet', False)):
        print('Tracking complete. Saved results and displayed plot.')


