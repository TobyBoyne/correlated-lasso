from sklearn.linear_model import Lasso
import numpy as np

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

lasso = Lasso(alpha=0.01)
lasso.fit(X, y)

print(beta_star)
print(lasso.coef_)