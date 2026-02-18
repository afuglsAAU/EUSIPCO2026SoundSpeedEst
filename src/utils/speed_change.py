import numpy as np
from scipy.signal import firwin, convolve
from numpy.typing import ArrayLike
"""
Implementation of Sinc the Interpolation–Compression/Expansion Resampling (SICER) framework for sound speed perturbation correction, as described in [1].

Based on MATLAB code provided by the authors of [1], with adjustments for Python conventions and batch processing. The core idea is to use sinc interpolation to resample the impulse responses according to the ratio of old and new sound speeds, with an optional low-pass FIR prefilter for the speed-up case to mitigate aliasing.

Thank you to the authors of [1] for sharing their code and insights, which greatly facilitated this implementation.

[1] S. S. Bhattacharjee, J. R. Jensen, and M. G. Christensen, “Sound Speed Perturbation Robust Audio: Impulse Response Correction and Sound Zone Control,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 2008–2020, 2025, doi: 10.1109/TASLPRO.2025.3570949.


"""

def _sinc_interpolate(old_rir: np.ndarray, n: int, compression_factor: float, extra_sample_points: int = 500,
                      prefilter_taps: np.ndarray | None = None) -> np.ndarray:
    """Core sinc-based resampling used by speed_up / speed_down.

    Parameters
    ----------
    old_rir : 1D array of length n
        Original impulse response.
    n : int
        Original IR length.
    compression_factor : float
        Old_speed / New_speed. <1 => compress (speed up), >1 => expand (speed down).
    extra_sample_points : int
        Boundary padding to improve sinc summation accuracy.
    prefilter_taps : Optional FIR taps
        If provided, apply linear-phase FIR prefilter before interpolation.

    Returns
    -------
    1D array of length n
        The resampled impulse response.
    """
    if old_rir.ndim != 1:
        old_rir = old_rir.reshape(-1)

    nT = np.arange(n)
    tt = np.arange(-(n + extra_sample_points), (n + extra_sample_points) + 1)
    nT_compressed = nT * compression_factor

    # ndgrid equivalent: Ts1, T1 shapes (len(tt), len(nT_compressed))
    Ts1, T1 = np.meshgrid(tt, nT_compressed, indexing="ij")
    sinc_mat = np.sinc((Ts1 - T1) / compression_factor)

    # Optional LPF prefilter
    if prefilter_taps is not None:
        # Linear phase FIR group delay is (num_taps - 1) / 2
        lpf_delay = (len(prefilter_taps) - 1) // 2
        ir_lowpass_filtered = np.convolve(prefilter_taps, old_rir, mode="full")
        # Slice to recover original length n aligned by group delay
        ir_segment = ir_lowpass_filtered[lpf_delay:lpf_delay + n]
        ir_ct = sinc_mat @ ir_segment
    else:
        ir_ct = sinc_mat @ old_rir

    L = ir_ct.shape[0]
    start = n + extra_sample_points
    end_excl = L - (extra_sample_points + 1)  # python end exclusive
    return (1.0 / compression_factor) * ir_ct[start:end_excl]


def _build_sinc_matrix(n: int, compression_factor: float, extra_sample_points: int = 500):
    """Precompute sinc kernel matrix and slice indices for batch interpolation.

    Returns
    -------
    sinc_mat : ndarray, shape (2*n + 2*extra_sample_points + 1, n)
        Sinc kernel accumulation matrix.
    start : int
        Start index for output slice.
    end_excl : int
        End index (exclusive) for output slice.
    """
    nT = np.arange(n)
    tt = np.arange(-(n + extra_sample_points), (n + extra_sample_points) + 1)
    nT_compressed = nT * compression_factor
    Ts1, T1 = np.meshgrid(tt, nT_compressed, indexing="ij")
    sinc_mat = np.sinc((Ts1 - T1) / compression_factor)

    L = sinc_mat.shape[0]
    start = n + extra_sample_points
    end_excl = L - (extra_sample_points + 1)
    return sinc_mat, start, end_excl


def _prefilter_batch(segments: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply linear-phase FIR prefilter to each column in segments using full convolution.

    segments: shape (n, count)
    taps: shape (num_taps,)
    Returns shape (n, count) aligned by group delay.
    """
    # Use 2D convolution: (n, count) * (num_taps, 1) -> (n + num_taps - 1, count)
    full = convolve(segments, taps[:, None], mode="full", method="auto")
    lpf_delay = (len(taps) - 1) // 2
    n = segments.shape[0]
    return full[lpf_delay:lpf_delay + n, :]


def speed_up_batch(segments: np.ndarray, n: int, old_speed: float, new_speed: float,
                   extra_sample_points: int = 500, fir_order: int = 100) -> np.ndarray:
    """Batch version of speed_up for multiple segments.

    segments: 2D array of shape (n, count), each column an IR.
    Returns array of shape (n, count).
    """
    compression_factor = float(old_speed) / float(new_speed)
    if not (0.0 < compression_factor < 1.0):
        raise ValueError("For speed_up, old_speed/new_speed must be in (0,1).")
    taps = firwin(fir_order + 1, cutoff=compression_factor, window="hamming")
    sinc_mat, start, end_excl = _build_sinc_matrix(n, compression_factor, extra_sample_points)
    # Prefilter all columns, then apply sinc accumulation in one shot
    ir_segment = _prefilter_batch(segments, taps)
    ir_ct = sinc_mat @ ir_segment
    return (1.0 / compression_factor) * ir_ct[start:end_excl, :]


def speed_down_batch(segments: np.ndarray, n: int, old_speed: float, new_speed: float,
                     extra_sample_points: int = 500) -> np.ndarray:
    """Batch version of speed_down for multiple segments (no prefilter)."""
    compression_factor = float(old_speed) / float(new_speed)
    sinc_mat, start, end_excl = _build_sinc_matrix(n, compression_factor, extra_sample_points)
    ir_ct = sinc_mat @ segments
    return (1.0 / compression_factor) * ir_ct[start:end_excl, :]


def speed_up(old_rir: ArrayLike, n: int, old_speed: float, new_speed: float) -> np.ndarray:
    """Speed up the IR using sinc interpolation with LPF prefilter.
    """
    old_rir = np.asarray(old_rir, dtype=float)
    compression_factor = float(old_speed) / float(new_speed)
    # FIR order 100 -> 101 taps; cutoff in (0, 1) for valid firwin
    if not (0.0 < compression_factor < 1.0):
        raise ValueError("For speed_up, old_speed/new_speed must be in (0,1).")
    order = 100
    taps = firwin(order + 1, cutoff=compression_factor, window="hamming")
    return _sinc_interpolate(old_rir, n, compression_factor, prefilter_taps=taps)


def speed_down(old_rir: ArrayLike, n: int, old_speed: float, new_speed: float) -> np.ndarray:
    """Speed down the IR using sinc interpolation (no prefilter).
    """
    old_rir = np.asarray(old_rir, dtype=float)
    compression_factor = float(old_speed) / float(new_speed)
    # No LPF; compression_factor can be >= 1
    return _sinc_interpolate(old_rir, n, compression_factor, prefilter_taps=None)
