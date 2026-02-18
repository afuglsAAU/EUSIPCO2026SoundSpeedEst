from pathlib import Path
import numpy as np
import scipy.io as sio
import os

def audioread(signal, fs=16000):
    """
    Load an audio signal from a WAV file.

    Args:
        signal (str): Path to the audio signal file.
        fs (int, optional): Expected sampling frequency of the input signal. Default is 16000.

    Returns:
        tuple:
            s (np.ndarray): Audio samples as a numpy array (float if int input).
            FS (int): Sampling frequency of the input signal.

    Raises:
        ValueError: If there is a mismatch in the expected and actual sample rate, or if the file format is not supported.
        FileNotFoundError: If the file is not found.
        WindowsError: If the absolute path length exceeds the Windows maximum path length.
    """
    try:
        FS, s = sio.wavfile.read(signal)
        if FS != fs:
            raise ValueError(f"Mismatch in expected sample rate ({fs}Hz) and input sample rate ({FS}Hz). Need to resample input {signal} to {fs} Hz")

    except ValueError as value_error:
        if str(value_error) == "File format b'NIST' not understood. Only 'RIFF' and 'RIFX' supported.":
            raise ValueError(f"Probably reading a TIMIT style wav file: {signal}") from value_error
        raise value_error

    except FileNotFoundError as file_error:
        path_length = len(str(Path(signal).resolve()))
        if (path_length >= 260) and (os.name == 'nt'):
            raise WindowsError(f'Absolute path length of {path_length} exceeds Windows maximum path length of 260.') from file_error
        raise file_error

    if s.dtype == np.int16:
        max_nb = float(2 ** (16 - 1))
        s = s / max_nb  # Convert to float
    elif s.dtype == np.int32:
        max_nb = float(2 ** (32 - 1))
        s = s / max_nb  # Convert to float

    return s, FS

def audiowrite(name, fs, data, as_int=False, nb=16, allowed_clip=0.001):
    """
    Write an audio signal to a WAV file.

    Args:
        name (str): Path to save the audio file.
        fs (int): Sampling frequency.
        data (np.ndarray): Audio data to be saved.
        as_int (bool, optional): Whether to save the data as integers. Default is False.
        nb (int, optional): Bit depth for saving the data. Default is 16.
        allowed_clip (float, optional): Allowed fraction of clipped samples. Default is 0.001.

    Raises:
        ValueError: If too many samples are clipped or if an unknown bit depth is specified.

    Returns:
        None
    """
    if as_int:
        max_nb = 2**(nb - 1)
        if np.mean(np.abs(data) > 1) > allowed_clip:
            raise ValueError(f'Too many samples were clipped. Fraction of clipped ({np.mean(np.abs(data) > 1):.4f}) > allowed fraction ({allowed_clip:.4f}). Suggest decreasing input speech level. Affected filename: \n{name}')
            # name = name[:-4] + "_intClip.wav"

        if np.abs(data).max() > 1:
            data = data.clip(-1, 1)
        if nb == 16:
            if data.dtype != np.int16:
                sio.wavfile.write(name, fs, (data*max_nb).astype(np.int16))
            else:
                sio.wavfile.write(name, fs, data)
        elif nb == 32:
            if data.dtype != np.int32:
                sio.wavfile.write(name, fs, (data*max_nb).astype(np.int32))
            else:
                sio.wavfile.write(name, fs, data)
        else:
            raise ValueError(f"Unknown bit depth: {nb}. Implemented bit depths include: 16, 32.")
    else:
        sio.wavfile.write(name, fs, data)
