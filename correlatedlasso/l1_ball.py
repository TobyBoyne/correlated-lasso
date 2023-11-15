"""Diagram to show the LASSO interplay of l1 norm and OLS soln"""

import matplotlib.pyplot as plt
import numpy as np 

fig, ax = plt.subplots()

x, y = np.meshgrid(np.linspace(-1, 1, num=50), np.linspace(-1, 1, num=50))
l1 = np.abs(x) + np.abs(y)

beta_star = np.array([0.5, 1.0])

X = np.random.multivariate_normal(mean=np.zeros((2,)), cov=np.eye(2), size=(100,))

beta_grid = np.stack((x, y), axis=-1)[:, :, :, None]
print(X.shape, beta_grid.shape)
print((X @ beta_grid).shape)
loss = np.linalg.norm(np.squeeze(X @ (beta_grid - beta_star[None, :, None])), axis=-1)

ax: plt.Axes
ax.set_aspect("equal")
ax.contour(x, y, l1, levels=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ax.contour(x, y, loss)
ax.set_xlabel(r"$\beta_1$")
ax.set_ylabel(r"$\beta_2$")
plt.show()