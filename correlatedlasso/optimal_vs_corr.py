import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from lasso.optimal_lambda import optimal_lambda
from data.sparse import generate_sparse_params
import data.correlated as corr

def optimal_lambda_against_correlation(
    rhos: np.ndarray,
    beta_star: np.ndarray,
    cov_func: Callable[[float], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the optimal lambda for a given correlation function
    
    Args:
        rhos: Array of correlation parameters
        beta_star: True parameter vector
        cov_func: Function that returns a covariance matrix given a correlation
            parameter
            
    Returns:
        means: Mean optimal lambda for each rho
        stds: Standard deviation of optimal lambda for each rho
    """
    means = np.zeros_like(rhos)
    stds = np.zeros_like(rhos)

    for i, rho in enumerate(rhos):
        cov = cov_func(rho)
        mean, std = optimal_lambda(
            beta_star, cov, 
            sigma=0.1, N=100, runs=100    
        )
        means[i] = mean
        stds[i] = std
    
    return means, stds

def plot_optimal_lambda_against_correlation(
    ax: plt.Axes,
    rhos: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    colour: str = "C0",
) -> plt.Line2D:
    """Plot the optimal lambda against the correlation parameter"""
    l, *_ = ax.plot(rhos, means, label="Mean", color=colour)
    opt_lower = means - 1.96 * stds
    opt_upper = means + 1.96 * stds
    ax.plot(rhos, opt_lower, "--", color=colour)
    ax.plot(rhos, opt_upper, "--", color=colour)
    ax.fill_between(rhos, opt_lower, opt_upper, color=colour, alpha=0.3)
    return l



if __name__ == "__main__":
    P = 100
    beta_star = generate_sparse_params(P, 5)
    beta_star[0] = 1.0
    beta_star[1] = -1.0
    rhos = np.linspace(0.1, 0.9, num=10)

    fig, ax = plt.subplots()
    ax: plt.Axes

    covariance_functions = (
        corr.all_correlated,
        corr.pairwise_correlated,
    )
    colours = ("C0", "C1")

    for cov_func, colour in zip(covariance_functions, colours):
        means, stds = optimal_lambda_against_correlation(
            rhos, beta_star, cov_func=lambda rho: cov_func(P, rho)
        )
        l = plot_optimal_lambda_against_correlation(ax, rhos, means, stds, colour)
        l.set_label(cov_func.__name__)

    ax.set_xlabel(r"Correlation parameter, $\rho$")
    ax.set_ylabel(r"Optimal regularisation, $\lambda_\text{opt}$")
    ax.set_title(r"Performance of LASSO for different $\rho$ values")
    ax.legend()
    plt.show()