import numpy as np

def generate_sparse_params(P, s, seed=42):
    """Generate a sparse set of parameters.

    For a model with P parameters, set s of them to be non-zero. 
    Non-zero parameters are drawn from a standard normal.
    
    Args:
        P: Number of parameters.
        s: The number of non-zero coefficients, |J*|.
        seed: Random seed for coefficient generation.
    """
    rng = np.random.default_rng(seed)

    beta_star = np.zeros((P,))
    nonzero_idxs = rng.choice(P, size=(s,), replace=False)
    beta_star[nonzero_idxs] = rng.standard_normal((s,))
    return beta_star