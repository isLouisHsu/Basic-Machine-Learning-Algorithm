import numpy as np

def eig(A1, A2):
    """
    Params:
        A1, A2: {ndarray(n, n)}
    Returns:
        eigval: {ndarray(n)}
        eigvec: {ndarray(n, n)}
    Notes:
        A1 \alpha = \lambda A2 \alpha
    """
    s, u = np.linalg.eigh(A2 + np.diag(np.ones(A2.shape[0]))*1e-3)
    s_sqrt_inv = np.linalg.inv(np.diag(np.sqrt(s)))

    A = s_sqrt_inv.dot(u.T).dot(A1).dot(u).dot(s_sqrt_inv)
    eigval, P = np.linalg.eigh(A)
    eigvec = u.dot(s_sqrt_inv).dot(P)
    
    return eigval, eigvec
