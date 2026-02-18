import numpy as np
import scipy.linalg as sla
import os
import gc
from pathlib import Path

def get_zone_convolution_mat(imp_resp, J):
    """
    Create the correlation/convolution matrix G for sound zone RIRs.

    Args:
        imp_resp (np.ndarray): Matrix containing RIRs for the sound zone of size M x L x K.
        J (int): Length of each control filter.

    Returns:
        np.ndarray: convolution matrix, shape (M*(K+J-1), L*J).
    """
    M, L, K = imp_resp.shape
    G = np.zeros((M * (K + J - 1), L * J))
    for m in range(M):
        imp_resp_mic = imp_resp[m]
        for ll in range(L):
            G[m * (K + J - 1):(m + 1) * (K + J - 1), ll * J:(ll + 1) * J] = sla.convolution_matrix(imp_resp_mic[ll], J)
    return G

def compute_covariance_matrices(imp_resp_bz, imp_resp_dz, J,
                                ref_source, model_delay,
                                covariance_path, save_name, save_name_DZ, verbose=True, **kwargs):
    """
    Compute and save/load the covariance matrices for bright and dark zones.

    Args:
        imp_resp_bz (np.ndarray): Impulse responses for the bright zone. Shape (M_b, L, K).
        imp_resp_dz (np.ndarray): Impulse responses for the dark zone. Shape (M_d, L, K).
        J (int): Length of each control filter.
        ref_source (int): Desired reference loudspeaker number (indexed from 0).
        model_delay (int): Delay in samples for the bright zone.
        covariance_path (str): Path to save/load covariance matrices.
        pos (tuple): Position configuration tuple (room, position, orientation, tilt).
        verbose (bool, optional): Whether to print loading messages.

    Returns:
        tuple:
            R_B (np.ndarray): Covariance matrix for the bright zone.
            r_B (np.ndarray): Cross-correlation vector for the bright zone.
            R_D (np.ndarray): Covariance matrix for the dark zone.
    """
    Path(covariance_path).mkdir(parents=True, exist_ok=True)  # Ensure directory exists for saving matrices
    str_save_R_B = Path(covariance_path) / f'R_B_{save_name}.npz'
    str_save_R_D = Path(covariance_path) / f'R_D_{save_name_DZ}.npz'
    str_save_r_B = Path(covariance_path) / f'cross_r_B_{save_name}.npz'

    if str_save_R_B.is_file() and str_save_r_B.is_file():
        if verbose:
            print('Load R_B, r_B')
        R_B = np.load(str_save_R_B)
        r_B = np.load(str_save_r_B)
    else:
        G_B = get_zone_convolution_mat(imp_resp_bz, J)
        p_T = G_B[:, ref_source * J]  # Desired loudspeaker indexed from 0

        # Add causality delay to the desired loudspeaker signal for each bright zone microphone
        p_T = p_T.reshape((imp_resp_bz.shape[0], imp_resp_bz.shape[-1] + J - 1))
        d_B = np.zeros_like(p_T)
        d_B[:, model_delay:] = p_T[:, :-model_delay]
        d_B = d_B.flatten()  # Flatten to 1D array again, (row-major order)

        # For-loop version of causality delay (commented out for performance, but kept for clarity)
        # d_B = np.zeros_like(p_T)
        # for m in range(M_b):
        #     p_m = p_T[m*(trim_IR_length+J-1):(m+1)*(trim_IR_length+J-1)]
        #     d_B[m*(trim_IR_length+J-1):(m+1)*(trim_IR_length+J-1)] = np.concatenate((np.zeros(model_delay), p_m[:-model_delay]))

        if str_save_R_B.is_file():
            R_B = np.load(str_save_R_B)
        else:
            R_B = G_B.T @ G_B
            np.savez_compressed(str_save_R_B, R_B=R_B)

        if str_save_r_B.is_file():
            r_B = np.load(str_save_r_B)
        else:
            r_B = G_B.T @ d_B
            np.savez_compressed(str_save_r_B, r_B=r_B)

        del G_B, d_B, p_T
        gc.collect()  # Force garbage collection

    if str_save_R_D.is_file():
        if verbose:
            print('Load R_D')
        R_D = np.load(str_save_R_D)
    else:
        G_D = get_zone_convolution_mat(imp_resp_dz, J)
        R_D = G_D.T @ G_D
        np.savez_compressed(str_save_R_D, R_D=R_D)
        del G_D
        gc.collect()  # Force garbage collection

    return R_B, r_B, R_D

def diagonalize_matrices(A, B, descend=True):
    """
    Perform joint diagonalization of matrices A and B using generalized eigenvalue decomposition.

    Args:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.
        descend (bool, optional): Whether to sort eigenvalues in descending order.

    Returns:
        tuple:
            eig_vec (np.ndarray): Eigenvectors.
            eig_val (np.ndarray): Eigenvalues (sorted if descend=True).
    """
    eig_val, eig_vec = sla.eigh(A, B, overwrite_a=True, overwrite_b=True)
    if descend:
        # sorting the eigenvalues in descending order
        idx = eig_val.argsort()[::-1]
        return eig_vec[:, idx], eig_val[idx]
    return eig_vec, eig_val
