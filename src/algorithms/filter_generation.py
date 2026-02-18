import numpy as np
import scipy.linalg as sla

def fit_vast_closed_form(vast_rank, mu, r_B, J, L, eig_vec, eig_val_vec, mat_out=False):
    """
    Compute the VAST filter using a closed-form solution.

    Args:
        vast_rank (int): Rank of the VAST filter.
        mu (float): Trade-off parameter.
        r_B (np.ndarray): Cross-correlation vector.
        J (int): Length of each control filter.
        L (int): Number of loudspeakers.
        eig_vec (np.ndarray): Eigenvectors.
        eig_val_vec (np.ndarray): Eigenvalues.

    Returns:
        np.ndarray: VAST filter coefficients.
    """
    # Vectorized version of the VAST filter
    # q_vast = np.zeros(J * L)
    # for v in range(vast_rank):
        # q_vast += (1 / (mu + eig_val_vec[v])) * (eig_vec[:, v].T @ r_B) * eig_vec[:, v]
    weights = 1 / (mu + eig_val_vec[:vast_rank])
    q_vast = eig_vec[:, :vast_rank] @ (weights * (eig_vec[:, :vast_rank].T @ r_B))
    if mat_out:
        # Reshape to matrix of size (L, J)
        q_vast = np.reshape(q_vast, (L, J))
    return q_vast


def wls_filter_calc(R_B, R_D, r_B, mu, reg_param=0.0, mat_out=True, L=None, J=None):
    """
    Weighted Least Squares (WLS) filter calculation.
    """
    # Scaling of the regularization factor by the norm of the R_B covariance matrix
    # to improve regularization.
    rg_w = sla.norm(R_B, ord=2) # Spectral norm (largest singular value), standard in Matlab
    reg_param = reg_param*rg_w
    A = (1-mu)*R_B + mu*R_D + reg_param * np.eye(R_B.shape[0])
    b = (1-mu)*r_B
    q_wls = np.linalg.solve(A , b)
    # Ainv = np.linalg.inv(A)  # Invert A in-place to save memory
    # b = (1-mu)*r_B
    # q_wls = Ainv @ b
    if mat_out:
        # Reshape to matrix of size (L, J) if L and J are provided
        assert L is not None and J is not None, "L and J must be provided for matrix output."
        # Reshape to matrix of size (L, J)
        q_wls = np.reshape(q_wls, (L, J))

    return q_wls

