"""A set of functions to generate correlated data."""

import numpy as np

def uncorrelated(P, rho=0):
    return np.eye(P)

def all_correlated(P, rho):
    return (1 - rho) * np.eye(P) + rho

def pairwise_correlated(P, rho):
    """Generate a correlation matrix with the first two parameters correlated."""
    C = np.eye(P)
    # rng = np.arange(P-1)
    # symmetric
    # C[rng, rng+1] = rho
    # C[rng+1, rng] = rho
    C[0, 1] = rho
    C[1, 0] = rho
    return C

if __name__ == "__main__":
    print(uncorrelated(5, 0.5))
    print(all_correlated(5, 0.5))
    print(pairwise_correlated(5, 0.5))