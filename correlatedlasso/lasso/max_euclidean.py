"""Create Pi_T and \rho_T as found in the paper."""

import numpy as np
# typical optimal value of lambda is 
# σ sqrt( n log(p) )
# \rho_T * σ sqrt( n log(p) )

# X_T (X_⊤^T X_T)† X_⊤^T

def orthogonal_projector(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Generate Pi_T given X.
    
    Args:
        X: [N x P] design matrix
    """
    X_T = X[:, T]
    return X_T @ np.linalg.pinv(X_T.T @ X_T) @ X_T.T

def maximal_euclidean(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Generate rho_T given X.
    
    Args:
        X: [N x P] design matrix
    """
    N = X.shape[0]
    Pi_T = orthogonal_projector(X, T)

    norms = np.linalg.norm((np.eye(N) - Pi_T) @ X, axis=0)
    return N ** (-1/2) * np.max(norms)

if __name__ == "__main__":
    X = np.arange(15).reshape((5, 3))
    print(X)
    T = np.array([0, 2])
    Pi_T = orthogonal_projector(X, T)
    print(Pi_T.shape)

    rho_T = maximal_euclidean(X, T)
    print(rho_T)