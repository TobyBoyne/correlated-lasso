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


def universal(sigma, N, P):
    """Generate the universal regularisation parameter, 
    σ sqrt( n log(p) )"""
    return sigma * np.sqrt(2 * N * np.log(P))

def adjusted_universal(sigma, N, P, rho_T: np.ndarray):
    """Generate the adjusted universal regularisation parameter, 
    \rho_T * σ sqrt( n log(p) )"""
    delta = 1.0
    return rho_T * 2 / np.sqrt(N) * universal(sigma, N, P / delta)

def average_rho(beta_star, cov, sigma=1.0, N=50, seed=42, runs=10):
    """Find the optimal value of lambda for a given beta_star
    
    Args:
        runs: Number of times to run the experiment"""
    P = beta_star.shape[0]
    N = 50

    rho_Ts = np.zeros((runs,))
    rng = np.random.default_rng(seed)

    for i in range(runs):
        # generate X
        X = rng.multivariate_normal(mean=np.zeros((P,)), cov=cov, size=(N,))

        # compute rho_T
        T = np.argwhere(beta_star).flatten()
        rho_T = maximal_euclidean(X, T)

        rho_Ts[i] = rho_T
    
    return rho_Ts.mean(), rho_Ts.std()

    

if __name__ == "__main__":
    X = np.arange(15).reshape((5, 3))
    print(X)
    T = np.array([0, 2])
    Pi_T = orthogonal_projector(X, T)
    print(Pi_T.shape)

    rho_T = maximal_euclidean(X, T)
    print(rho_T)