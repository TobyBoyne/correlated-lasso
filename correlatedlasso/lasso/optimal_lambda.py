from sklearn.linear_model import lasso_path
import numpy as np


def optimal_lambda(beta_star, cov, sigma=1.0, N=50, seed=42, runs=10):
    """Find the optimal value of lambda for a given beta_star
    
    Args:
        runs: Number of times to run the experiment"""
    P = beta_star.shape[0]
    N = 50

    lambda_stars = np.zeros((runs,))
    rng = np.random.default_rng(seed)

    for i in range(runs):
        # generate X
        X = rng.multivariate_normal(mean=np.zeros((P,)), cov=cov, size=(N,))

        # generate obs
        noise = sigma * rng.standard_normal((N,))
        y = X @ beta_star + noise

        # compute lasso path and loss
        alphas, beta_est, _ = lasso_path(X, y, alphas=np.geomspace(0.001, 1.0))
        loss = np.linalg.norm(X @ (beta_est - beta_star[:, None]), axis=0)
        

        # find argmin loss
        lambda_star = alphas[np.argmin(loss)]
        lambda_stars[i] = lambda_star

    return lambda_stars.mean(), lambda_stars.std()


def lambda_path(beta_star, cov, alphas, sigma=1.0, N=50, seed=42, runs=10):
    """Calculate the loss for the path of lambda values
    
    Args:
        runs: Number of times to run the experiment"""
    P = beta_star.shape[0]
    N = 50
    N_alphas = 50

    losses = np.zeros((runs, N_alphas))
    rng = np.random.default_rng(seed)

    for i in range(runs):
        # generate X
        X = rng.multivariate_normal(mean=np.zeros((P,)), cov=cov, size=(N,))

        # generate obs
        noise = sigma * rng.standard_normal((N,))
        y = X @ beta_star + noise

        # compute lasso path and loss
        t, beta_est, _ = lasso_path(X, y, alphas=alphas)
        loss = np.linalg.norm(X @ (beta_est - beta_star[:, None]), axis=0)
        losses[i, :] = loss

    return losses.mean(axis=0), losses.std(axis=0)
