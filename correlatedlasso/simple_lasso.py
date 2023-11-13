from sklearn.linear_model import Lasso
import numpy as np

P = 10
N = 500
rng = np.random.default_rng(seed=42)

cov = np.eye(P) # uncorrelated
X = np.random.multivariate_normal(mean=np.zeros((P,)), cov=cov, size=(N,))
# X is [N x P]

# define our generating model
# make sparse beta
beta_star = np.zeros((P,))
num_nonzeros = 2
nonzero_idxs = rng.choice(P, size=(num_nonzeros,), replace=False)
beta_star[nonzero_idxs] = 10 * rng.standard_normal((num_nonzeros,))
# beta is [P]

# generate obs
sigma = 0
noise = sigma * rng.standard_normal((N,))
y = X @ beta_star + noise

lasso = Lasso(alpha=1.0)
lasso.fit(X, y)

print(beta_star)
print(lasso.coef_)