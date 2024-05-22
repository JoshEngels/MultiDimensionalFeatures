import csv
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import os

plt.rcParams["text.usetex"] = True

rows = []
with open("results.csv", "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        rows.append(row)
rows = rows[1:]  # remove column labels
a, b, out_c, _, _, _ = zip(*rows)
a, b, out_c = [np.array([int(x) for x in arr]) for arr in (a, b, out_c)]

n_layers = 32
start_token = 12
n_tokens = 5
n_pca_dims = 20
mod = 7

a = a % mod
b = b % mod
c = (a + b) % mod
out_c = out_c % mod


order = np.argsort(mod * a + b)
a, b, c, out_c = [arr[order] for arr in (a, b, c, out_c)]
c_order = np.argsort(c)


def deconstruct(layer, n_feature_groups):
    token = 4
    activations = torch.load(
        "pca_components/layer"
        + str(layer)
        + "_token"
        + str(start_token + token)
        + "_pca"
        + str(n_pca_dims)
        + ".pt"
    )
    flat_activations = activations[order, :]  # problem, pca
    activations = flat_activations.reshape([mod, mod, n_pca_dims])

    def bias():
        return [np.ones([mod * mod])], "original"

    def oha():
        features = []
        for i in range(7):
            features.append(i == a)
        return features, "one hot $\\alpha$"

    def ohb():
        features = []
        for i in range(7):
            features.append(i == b)
        return features, "one hot $\\beta$"

    def ohab():
        features = []
        for i in range(7):
            features.append(i == a)
            features.append(i == b)
        return features, "one hot $\\alpha$, $\\beta$"

    def ohc():
        features = []
        for i in range(7):
            features.append(i == c)
        return features, "one hot $\\gamma$"

    def camb():
        features = []
        features.append(np.cos(2 * np.pi * (a - b) / 7))
        features.append(np.sin(2 * np.pi * (a - b) / 7))
        return features, "circle $\\alpha-\\beta$"

    def cc():
        features = []
        features.append(np.cos(2 * np.pi * (a + b) / 7))
        features.append(np.sin(2 * np.pi * (a + b) / 7))
        return features, "circle $\\gamma$"

    def tomorrow_a():
        features = []
        for i in range(7):
            features.append(
                (i == a).astype(np.float64) * (b == 1).astype(np.float64)
            )  # 98.1
        return features, "$\\alpha$ for tmr"

    if token == 3:
        features = {
            17: [bias(), oha(), ohb()],
            18: [bias(), oha(), ohb()],
            19: [bias(), oha(), ohb()],
            20: [bias(), oha(), ohb()],
            21: [bias(), oha(), ohb()],
            22: [bias(), oha(), ohb()],
            23: [bias(), oha(), ohb()],
            24: [bias(), oha(), ohb()],
            25: [bias(), oha(), ohb()],
            26: [bias(), oha(), ohb()],
            27: [bias(), oha(), ohb()],
            28: [bias(), oha(), ohb()],
            29: [bias(), oha(), ohb()],
        }[layer]

    if token == 4:
        features = {
            17: [bias(), ohab()],
            18: [bias(), ohab(), tomorrow_a(), camb()],
            19: [bias(), ohab(), tomorrow_a(), cc(), camb()],
            20: [bias(), ohab(), tomorrow_a(), cc(), camb()],
            21: [bias(), ohab(), cc(), camb()],
            22: [bias(), ohab(), cc(), camb(), tomorrow_a()],
            23: [bias(), ohab(), cc(), camb(), tomorrow_a()],
            24: [bias(), ohab(), cc(), camb(), tomorrow_a()],
            25: [bias(), ohab(), cc(), camb()],
            26: [bias(), ohab(), cc(), camb()],
            27: [bias(), ohab(), cc(), camb()],
            28: [bias(), ohab(), cc()],
            29: [bias(), ohab(), cc()],
        }[layer]

    feature_labels = [feature[1] for feature in features]
    features = [feature[0] for feature in features]

    max_feature_groups = len(features) - 1
    features = sum(features[: n_feature_groups + 1], [])
    added_feature_string = feature_labels[n_feature_groups]

    features = np.stack(
        [feature.astype(np.float32) for feature in features], axis=1
    )  # problem, feature
    XTX = features.T.dot(features / (mod**2))
    D, U = np.linalg.eigh(XTX)
    D = np.where(D > 1e-5, 1 / D, 0)
    XTX_inv = U.dot(np.diag(D)).dot(U.T)
    betas = np.matmul(
        XTX_inv, np.matmul((features / (mod**2)).T, flat_activations)
    )  # feature, pca
    predicted_activations = np.matmul(features, betas)  # problem, pca
    error = np.mean((flat_activations - predicted_activations) ** 2)  # dimensionless
    total_variance = np.mean(flat_activations**2) + 1e-10  # dimensionless
    r2 = 1 - (error / total_variance)

    print(n_feature_groups, max_feature_groups, features.shape, r2)
    return (
        flat_activations,
        predicted_activations,
        r2,
        max_feature_groups,
        added_feature_string,
    )


grids = []
grid_labels = []
r2_labels = []
for layer in range(17, 30):
    grids.append([])
    grid_labels.append([])
    r2_labels.append([])
    max_feature_groups = 100
    n_feature_groups = 0
    while True:
        (
            flat_activations,
            predicted_activations,
            r2,
            max_feature_groups,
            added_feature_string,
        ) = deconstruct(layer, n_feature_groups)

        residuals = flat_activations - predicted_activations
        U, S, Vh = np.linalg.svd(residuals)
        projected_residuals = U[:, :3] * S[:3]
        dot_colors = [(i / mod, j / mod, 0) for i in range(mod) for j in range(mod)]
        colors = (projected_residuals - np.min(projected_residuals, axis=0)) / (
            np.max(projected_residuals, axis=0) - np.min(projected_residuals, axis=0)
        )
        grids[-1].append(colors.reshape((mod, mod, 3)))
        grid_labels[-1].append(added_feature_string)
        r2_labels[-1].append("r2: {r2:.1f}\\%".format(r2=r2 * 100))

        if n_feature_groups == max_feature_groups:
            break
        n_feature_groups += 1


vertical_space = 3
first_label_height = 1
second_label_height = 2.4
original_grids = grids
max_grid_len = max([len(row) for row in grids])
grids = [
    row + [np.ones([mod, mod, 3]) for i in range(max_grid_len - len(row))]
    for row in grids
]
grids = [
    [
        np.concatenate(
            [
                np.concatenate([grid, np.ones([mod, 1, 3])], axis=1),
                np.ones([vertical_space, mod + 1, 3]),
            ],
            axis=0,
        )
        for grid in row
    ]
    for row in grids
]
grids = np.concatenate([np.concatenate(row, axis=0) for row in grids], axis=1)[
    :, :-1, :
]

pixelsize = 20
grids = grids.repeat(pixelsize, axis=0).repeat(pixelsize, axis=1)

fig, ax = plt.subplots()
ax.imshow(grids, interpolation="nearest")
for i, row in enumerate(original_grids):
    for j, grid in enumerate(row):
        if grid_labels[i][j] in ("one hot $\\alpha$+$\\beta$", "$\\alpha+1=\\beta$"):
            size = 3
        elif (
            "one hot" in grid_labels[i][j]
            or grid_labels[i][j] == "circle $\\alpha-\\beta$"
        ):
            size = 4
        else:
            size = 5
        ax.text(
            pixelsize * ((mod + 1) * i + mod / 2),
            pixelsize * ((mod + vertical_space) * j + mod + first_label_height),
            grid_labels[i][j],
            horizontalalignment="center",
            size=size,
        )
        ax.text(
            pixelsize * ((mod + 1) * i + mod / 2),
            pixelsize * ((mod + vertical_space) * j + mod + second_label_height),
            r2_labels[i][j],
            horizontalalignment="center",
            size=5,
        )
    ax.text(
        pixelsize * ((mod + 1) * i + mod / 2),
        pixelsize * (-1),
        "layer " + str(i + 17),
        horizontalalignment="center",
        size=6,
    )
plt.axis("off")
plt.savefig("feature_deconstruction_day_of_the_week.pdf", bbox_inches="tight")
plt.close()
