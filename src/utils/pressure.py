import numpy as np
from scipy.fft import rfft, irfft, next_fast_len


def compute_lsp_signal_frame(q, input_frame, params, prev_frame, q_fft=None, S_fft=None, n_fft=None):
    """
    Compute the loudspeaker signal frame by filtering the input signal with the control filters.

    Args:
        q (np.ndarray): Control filter coefficients of shape (L, J).
        input_frame (np.ndarray): Current frame of input signals.
        params (dict): Dictionary containing various parameters for the simulation.
        prev_frame (np.ndarray): Previous frame of loudspeaker signals.
        q_fft (np.ndarray, optional): Precomputed FFT of filters with shape (L, F). If provided, skips FFT of q.
        S_fft (np.ndarray, optional): Precomputed FFT of input frame with shape (F,). If provided, skips FFT of input_frame.
        n_fft (int, optional): FFT length to use. If None, computed as next_fast_len(frame_size + J - 1).

    Returns:
        tuple:
            curr_frame (np.ndarray): Current frame of loudspeaker signals.
            prev_frame (np.ndarray): Updated previous frame of loudspeaker signals.
    """
    frame_size, J = params['frame_size'], params['J']
    L = prev_frame.shape[0]

    # FFT-based convolution for all loudspeakers at once
    out_len = frame_size + J - 1
    if n_fft is None:
        n_fft = next_fast_len(out_len)

    # Compute input FFT once
    if S_fft is None:
        S_fft = rfft(input_frame, n=n_fft)
    # Compute filter FFTs for all loudspeakers (q assumed (L, J))
    if q_fft is None:
        Q_fft = rfft(q, n=n_fft, axis=1)  # (L, F)
    else:
        Q_fft = q_fft

    # Multiply in frequency domain and transform back
    LSP_fft = Q_fft * S_fft[np.newaxis, :]  # (L, F)
    temp_sig = irfft(LSP_fft, n=n_fft, axis=1)[:, :out_len]

    # Overlap-add computation
    curr_frame = temp_sig[:, :frame_size] + prev_frame
    prev_frame[:, :J - 1] = temp_sig[:, frame_size:]

    return curr_frame, prev_frame


def compute_pressure_from_ffts(IR_fft, Q_fft_rank, S_fft, n_fft, out_len):
    """Compute microphone pressures using cached FFTs.

    IR_fft: (M, L, F)
    Q_fft_rank: (L, F)
    S_fft: (F,)
    Returns: (M, out_len)
    """
    sum_m_l = np.einsum('mlf,lf->mf', IR_fft, Q_fft_rank)  # (M, F)
    P_fft = sum_m_l * S_fft  # (M, F)
    p_full = irfft(P_fft, n=n_fft, axis=1)
    return p_full[:, :out_len]


def compute_pressure_with_input_fft_cached(IR_fft, q, S_fft, n_fft, out_len):
    """Compute pressures via FFT using cached IR FFTs and input FFT.

    IR_fft shape: (M, L, F)
    q shape: (L, J)
    S_fft shape: (F,)
    Returns: (M, out_len)
    """
    Q_fft = rfft(q, n=n_fft, axis=1)  # (L, F)
    return compute_pressure_from_ffts(IR_fft, Q_fft, S_fft, n_fft, out_len)


def compute_mic_frame_from_lsp_fft(IR_fft, lsp_frame, n_fft, out_len, lsp_fft=None):
    """Compute microphone frame (sum over loudspeakers) from lsp frame via FFT.

    IR_fft: (M, L, F)
    lsp_frame: (L, T)
    lsp_fft: optional precomputed FFT of lsp frame, shape (L, F)
    Returns: (M, out_len)
    """
    LSP_fft = lsp_fft if lsp_fft is not None else rfft(lsp_frame, n=n_fft, axis=1)  # (L, F)
    sum_m_l = np.einsum('mlf,lf->mf', IR_fft, LSP_fft)  # (M, F)
    mic_full = irfft(sum_m_l, n=n_fft, axis=1)
    return mic_full[:, :out_len]


def compute_mic_signal_frame_fft_overlap(IR_fft, lsp_frame, prev_frame, frame_size, K, n_fft=None, lsp_fft=None):
    """FFT-based mic signal frame with overlap-add update.

    IR_fft: (M, L, F)
    lsp_frame: (L, frame_size)
    lsp_fft: optional precomputed FFT of lsp frame, shape (L, F)
    prev_frame: (M, frame_size) previous overlap buffer (only first K-1 used)
    frame_size: current frame size
    K: IR length
    n_fft: optional FFT length; if None, uses next_fast_len(frame_size + K - 1)

    Returns: (curr_frame, updated_prev_frame)
    """
    if n_fft is None:
        if lsp_fft is not None:
            # Infer n_fft from FFT size (F = n_fft//2 + 1)
            F = lsp_fft.shape[1]
            n_fft = (F - 1) * 2
        else:
            n_fft = next_fast_len(frame_size + K - 1)
    out_len = frame_size + K - 1
    mic_full = compute_mic_frame_from_lsp_fft(IR_fft, lsp_frame, n_fft, out_len, lsp_fft=lsp_fft)
    curr = mic_full[:, :frame_size] + prev_frame
    # Update prev buffer (only first K-1 samples are used next time)
    new_prev = prev_frame.copy()
    new_prev[:, :K-1] = mic_full[:, frame_size:]
    return curr, new_prev
