"""Plot the adjusted optimal lambda against the correlation parameter"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from data.sparse import generate_sparse_params
from lasso.max_euclidean import maximal_euclidean, average_rho, universal, adjusted_universal
from data.correlated import all_correlated

def plot_lambda_adjusted(
    ax: plt.Axes,
    rhos: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    colour: str = "C0",
) -> plt.Line2D:
    m = adjusted_universal(1.0, N, P, means)
    lower = adjusted_universal(1.0, N, P, means - 1.96 * stds)
    upper = adjusted_universal(1.0, N, P, means + 1.96 * stds)

    m /= m[0]
    lower /= lower[0]
    upper /= upper[0]

    l, *_ = ax.plot(rhos, m, label="Adjusted Optimal $\lambda$", color=colour, linewidth=2)

    ax.plot(rhos, lower, "--", color=colour)
    ax.plot(rhos, upper, "--", color=colour)
    ax.fill_between(rhos, lower, upper, color=colour, alpha=0.3)

    return l

def plot_experimental_optimal_lambda_against_correlation(
    ax: plt.Axes,
    rhos: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    colour: str = "C0",
) -> plt.Line2D:
    """Plot the optimal lambda against the correlation parameter"""
    l, *_ = ax.plot(rhos, means, label="Experimental Optimal $\lambda$", color=colour)
    opt_lower = means - 1.0 * stds
    opt_upper = means + 1.0 * stds
    ax.plot(rhos, opt_lower, "--", color=colour)
    ax.plot(rhos, opt_upper, "--", color=colour)
    ax.fill_between(rhos, opt_lower, opt_upper, color=colour, alpha=0.1)
    return l


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax: plt.Axes

    P = 100
    N = 100
    beta_star = generate_sparse_params(P, 5)
    beta_star[0] = 1.0
    beta_star[1] = -1.0

    rhos = np.linspace(0.0, 0.9, num=10)
    rng = np.random.default_rng(seed=42)
    
    means = np.zeros_like(rhos)
    stds = np.zeros_like(rhos)
    for i, rho in enumerate(rhos):
        mean, std = average_rho(beta_star, all_correlated(P, rho), N=N, runs=50)
        means[i] = mean
        stds[i] = std

    # plot_rho_T(ax, rhos, means, stds)
    # normalise
    plot_lambda_adjusted(ax, rhos, means, stds, colour="tab:orange")
    ax.hlines(1, rhos[0], rhos[-1], color="grey", linestyles=["--"])
    
    # load in data of all_correlated experimentally
    data = np.load(r"figs\all_correlated_vs_corr.npz")

    exp_means = data["means"]
    exp_stds = data["stds"]
    exp_rhos = data["rhos"]
    cm = mpl.colormaps["viridis"]
    plot_experimental_optimal_lambda_against_correlation(
        ax, exp_rhos, exp_means / exp_means[0], exp_stds / exp_means[0],
        colour=cm(0.0)
    )
    
    ax.set_title("*Normalised* adjusted optimal lambda against correlation")
    ax.set_ylabel(r"Adjusted regularisation, $\lambda$")
    ax.set_xlabel(r"Correlation parameter, $r$")
    ax.legend()
    fig.savefig("figs/adjusted_vs_experimental.png", bbox_inches="tight")
    plt.show()