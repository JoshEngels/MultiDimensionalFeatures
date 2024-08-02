# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)

dim = 10
c_matrix = np.random.rand(4, dim)

def f(a, b, epsilon_scale = 0.1):
    return [c_matrix[0, i] * np.sin(a) + c_matrix[1, i] * b + c_matrix[2, i] * (a == 4) + c_matrix[3, i] * (b == 1) + np.random.normal(0, 1) * epsilon_scale for i in range(dim)]


# %%


a_range = range(10)
b_range = range(10)

def plot_flattened_values(flattened_values):

    pca = PCA(n_components=3)

    pca.fit(flattened_values)

    pca_values = pca.transform(flattened_values)

    # Divide each column by its max
    pca_values = pca_values / pca_values.max(axis=0)

    pca_values = pca_values.reshape([len(a_range), len(b_range), 3])

    plt.imshow(pca_values)
    plt.ylabel("$\\alpha$")
    plt.xlabel("$\\beta$")

# 

a_values = np.array([[a for _ in b_range] for a in a_range])
b_values = np.array([[b for b in b_range] for _ in a_range])

func_values = np.array([[f(a, b) for b in b_range] for a in a_range])

flattened_values = func_values.reshape([len(a_range) * len(b_range), dim])
flattened_a = a_values.reshape([len(a_range) * len(b_range)])
flattened_b = b_values.reshape([len(a_range) * len(b_range)])

plot_flattened_values(flattened_values)


# %%

a_4 = flattened_a == 4
b_1 = flattened_b == 1
ones = np.ones_like(flattened_a)

explainers = np.stack([a_4, b_1, ones], axis=1)

linear_regression = np.linalg.lstsq(explainers, flattened_values, rcond=0.001)[0]

predicted_values = explainers @ linear_regression

residual = flattened_values - predicted_values

plot_flattened_values(residual)

# %%

explainers = np.stack([a_4, b_1, ones, np.sin(flattened_a)], axis=1)

linear_regression = np.linalg.lstsq(explainers, flattened_values, rcond=0.001)[0]

residual = flattened_values - explainers @ linear_regression

plot_flattened_values(residual)



# %%


explainers = np.stack([a_4, b_1, ones, np.sin(flattened_a), flattened_b], axis=1)

linear_regression = np.linalg.lstsq(explainers, flattened_values, rcond=0.001)[0]

residual = flattened_values - explainers @ linear_regression

plot_flattened_values(residual)

# %%

r_squared = 1 - np.sum(residual ** 2) / np.sum((flattened_values - flattened_values.mean(axis=0)) ** 2)

print(residual)

# %%
