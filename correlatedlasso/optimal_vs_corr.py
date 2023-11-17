import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Callable

from lasso.optimal_lambda import optimal_lambda, lambda_path
from data.sparse import generate_sparse_params
import data.correlated as corr
from lasso.max_euclidean import maximal_euclidean  

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

def plot_lambda_path(
    ax: plt.Axes,
    alphas: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    colour: str = "C0",
) -> plt.Line2D:
    l, *_ = ax.loglog(alphas, means, label="Mean", color=colour, linewidth=2)
    # opt_lower = means - 0.5 * stds
    # opt_upper = means + 0.5 * stds

    # ax.plot(alphas, opt_lower, "--", color=colour)
    # ax.plot(alphas, opt_upper, "--", color=colour)
    # ax.fill_between(alphas, opt_lower, opt_upper, color=colour, alpha=0.3)

    return l



if __name__ == "__main__":
    P = 100
    beta_star = generate_sparse_params(P, 5)
    beta_star[0] = 1.0
    beta_star[1] = -1.0
    rhos = np.linspace(0.1, 0.9, num=10)

    fig, ax = plt.subplots()
    ax: plt.Axes
    cm = mpl.colormaps["viridis"]

    covariance_functions = (
        corr.all_correlated,
        corr.pairwise_correlated,
    )
    colours = (cm(0.0), cm(0.5))

    for cov_func, colour in zip(covariance_functions, colours):
        means, stds = optimal_lambda_against_correlation(
            rhos, beta_star, cov_func=lambda rho: cov_func(P, rho)
        )
        l = plot_optimal_lambda_against_correlation(ax, rhos, means, stds, colour)
        l.set_label(cov_func.__name__)

        # hacky save to use elsewhere
        if colour == colours[0]:
            np.savez("figs/all_correlated_vs_corr.npz", rhos=rhos, means=means, stds=stds)
            

    ax.set_xlabel(r"Correlation parameter, $r$")
    ax.set_ylabel(r"Optimal regularisation, $\lambda_\text{opt}$")
    ax.set_title(r"Performance of LASSO for different $r$ values")
    ax.legend()
    fig.savefig("figs/optimal_lambda_against_correlation.png", bbox_inches="tight")

    # ---

    fig, ax = plt.subplots()
    ax: plt.Axes

    alphas = np.geomspace(0.001, 0.5)
    rhos = np.linspace(0.1, 0.9, num=5)
    for rho in rhos:
        cov = corr.pairwise_correlated(P, rho)
        means, stds = lambda_path(beta_star, cov, alphas, sigma=0.1, N=100, runs=100)
        l = plot_lambda_path(ax, alphas[::-1], means, stds, colour=cm(rho))
        l.set_label(fr"$r={rho:.1f}$")

        # mins = alphas[::-1][np.argmin(means)]
        # ax.scatter(mins, np.min(means), marker="x", color="tab:orange", 
        #            s=100, zorder=10)
    ax.set_xlabel(r"Regularisation parameter, $\lambda$")
    ax.set_ylabel(r"Fit loss, $\| X( \hat \beta - \beta^* )\|_2$")
    ax.set_title(r"Performance of LASSO for different $r$ values")
    ax.legend()
    fig.savefig("figs/lambda_path.png", bbox_inches="tight")

    plt.show()