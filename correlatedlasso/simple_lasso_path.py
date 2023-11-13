"""Compare LASSO performance for different regularisation values"""

from sklearn.linear_model import lasso_path
import numpy as np
import matplotlib.pyplot as plt

from data.sparse import generate_sparse_params

P = 10
N = 500
rng = np.random.default_rng(seed=42)

cov = np.eye(P) # uncorrelated
X = np.random.multivariate_normal(mean=np.zeros((P,)), cov=cov, size=(N,))
# X is [N x P]

# define our generating model
# make sparse beta
num_nonzeros = 2
beta_star = generate_sparse_params(P, num_nonzeros)
# beta is [P]

# generate obs
sigma = 0.05
noise = sigma * rng.standard_normal((N,))
y = X @ beta_star + noise

alphas, beta_est, _ = lasso_path(X, y)
# beta_est stores the path taken by the coordinate descent
# beta_est is [P x num_its]
print(alphas)
print(beta_est.shape)
print(beta_est[:, -1])
print(beta_star)

plt.semilogx(alphas, np.linalg.norm(beta_est - beta_star[:, None], axis=0))
plt.xlabel("Regularisation weight, $\lambda$")
plt.ylabel("Parameter difference, $\| \hat \\beta - \\beta^* \|_2$")
plt.show()