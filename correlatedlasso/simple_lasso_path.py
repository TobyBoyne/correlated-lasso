"""Compare LASSO performance for different regularisation values"""

from sklearn.linear_model import lasso_path
import numpy as np
import matplotlib.pyplot as plt

from data.sparse import generate_sparse_params

P = 10
N = 500

def run_lasso(beta_star, rho=0.0, seed=42):
    rng = np.random.default_rng(seed)
    cov = (1 - rho) * np.eye(P) + rho # correlated
    X = rng.multivariate_normal(mean=np.zeros((P,)), cov=cov, size=(N,))
    # X is [N x P]

    # define our generating model

    # generate obs
    sigma = 0.5
    noise = sigma * rng.standard_normal((N,))
    y = X @ beta_star + noise

    alphas, beta_est, _ = lasso_path(X, y, alphas=np.geomspace(0.001, 1.0))
    # beta_est stores the path taken by the coordinate descent
    # beta_est is [P x num_its]

    loss = np.linalg.norm(X @ (beta_est - beta_star[:, None]), axis=0)
    return alphas, loss
    

if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.set_xlabel("Regularisation weight, $\lambda$")
    ax.set_ylabel("Parameter difference, $\| \hat \\beta - \\beta^* \|_2$")
    
    rhos = np.linspace(0.1, 0.9, num=5)
    rhos = [0.1]
    # make sparse beta
    # beta is [P]
    num_nonzeros = 2
    beta_star = generate_sparse_params(P, num_nonzeros)

    for rho in rhos:
        for seed in range(1):
            alphas, norms = run_lasso(beta_star, rho, seed)
            ax.loglog(alphas, norms, label=rf"$\rho={rho:.1f}$")

    ax.legend()
    plt.show()