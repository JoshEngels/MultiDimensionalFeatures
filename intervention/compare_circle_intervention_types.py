# %%

from task import get_acts_pca, get_acts

from days_of_week_task import DaysOfWeekTask
from months_of_year_task import MonthsOfYearTask
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import dill as pickle
import pandas as pd
import os
import einops
from sklearn.decomposition import PCA
from circle_finding_utils import (
    find_c_circle,
    get_logit_diffs_from_subspace_formula_resid_intervention,
)
import argparse
from utils import is_notebook

probe_on_cos = True
probe_on_sin = True
probe_on_centered_linear = False

device = "cuda:0"
circle_letter = "a"
day_month_choice = "day"
model_name = "mistral"
use_inverse_regression_probe = False
intervention_pca_k = 5

torch.set_grad_enabled(False)


if not probe_on_cos and not probe_on_sin and not probe_on_centered_linear:
    raise ValueError("Must probe on at least one component.")

probe_file_extension = f"{'_cos' if probe_on_cos else ''}{'_sin' if probe_on_sin else ''}{'_centered_linear' if probe_on_centered_linear else ''}"
probe_file_extension = probe_file_extension[1:]

probe_dimension = sum([probe_on_cos, probe_on_sin, probe_on_centered_linear])

probe_index_to_description = []
if probe_on_cos:
    probe_index_to_description.append("cos")
if probe_on_sin:
    probe_index_to_description.append("sin")
if probe_on_centered_linear:
    probe_index_to_description.append("centered_linear")

task = DaysOfWeekTask(device, model_name=model_name)


token = task.a_token

# Index into problem.info to learn a circle on
circle_info_index = 0

# %%

mistral_pcas = pickle.load(open("../sae_multid_feature_discovery/fit_pca_days.pkl", "rb")).components_[1:3, :]

# %%

# Get original probe data

original_probe = torch.load(f"{task.prefix}/circle_probes_{circle_letter}/{probe_file_extension}_layer_8_token_{token}_pca_5.pt")
original_probe_data = []

for layer in [6, 7, 8, 9, 10]:

    (
        logit_diffs_before,
        logit_diffs_after,
        logit_diffs_replace_pca,
        logit_diffs_replace_all,
        logit_diffs_average_ablate,
        logit_diffs_zero_circle,
        logit_diffs_zero_everything_but_circle,
    ) = get_logit_diffs_from_subspace_formula_resid_intervention(
        task,
        probe_projection_qr=(original_probe["probe_q"], original_probe["probe_r"]),
        pca_k_project=5,
        layer=layer,
        token=token,
        target_to_embedding=original_probe["target_to_embedding"],
        letter_to_intervene_on=circle_letter,
    )

    average_before = np.mean(logit_diffs_before)
    average_after = np.mean(logit_diffs_after)
    average_replace_pca = np.mean(logit_diffs_replace_pca)
    average_replace_all = np.mean(logit_diffs_replace_all)
    average_average_ablate = np.mean(logit_diffs_average_ablate)
    average_zero_circle = np.mean(logit_diffs_zero_circle)
    average_zero_everything_but_circle = np.mean(logit_diffs_zero_everything_but_circle)

    original_probe_data.append((layer, average_before, average_after, average_replace_pca, average_replace_all, average_average_ablate, average_zero_circle, average_zero_everything_but_circle))
    original_probe_data.append((layer, logit_diffs_before, logit_diffs_after, logit_diffs_replace_pca, logit_diffs_replace_all, logit_diffs_average_ablate, logit_diffs_zero_circle, logit_diffs_zero_everything_but_circle))

# %%

# Get Mistral data

oned_targets = torch.tensor(
    [problem.info[circle_info_index] for problem in task.generate_problems()]
)

p = len(task.allowable_tokens)
k = 1
w = 2 * np.pi * k / p

multid_targets = torch.zeros((len(oned_targets), probe_dimension))
target_to_embedding = torch.zeros((p, probe_dimension))

current_probe_dimension = 0
if probe_on_cos:
    multid_targets[:, current_probe_dimension] = torch.cos(w * oned_targets)
    target_to_embedding[:, current_probe_dimension] = torch.cos(
        w * torch.arange(p)
    )
    current_probe_dimension += 1
if probe_on_sin:
    multid_targets[:, current_probe_dimension] = torch.sin(w * oned_targets)
    target_to_embedding[:, current_probe_dimension] = torch.sin(
        w * torch.arange(p)
    )
    current_probe_dimension += 1
if probe_on_centered_linear:
    multid_targets[:, current_probe_dimension] = oned_targets - (p - 1) / 2
    target_to_embedding[:, current_probe_dimension] = (
        torch.arange(p) - (p - 1) / 2
    )
    current_probe_dimension += 1

assert current_probe_dimension == probe_dimension

multid_targets_train = multid_targets
acts_train = get_acts(task, layer_fetch=8, token_fetch=token)
acts_train -= acts_train.mean(dim=0)

projections = (acts_train @ mistral_pcas.T).float()

least_squares_sol = torch.linalg.lstsq(
    projections, multid_targets_train
).solution

probe_q, probe_r = torch.linalg.qr(least_squares_sol)

predictions = projections @ least_squares_sol


# %%


mistral_data = []

for layer in [6, 7, 8, 9, 10]:

    (
        logit_diffs_before,
        logit_diffs_after,
        logit_diffs_replace_pca,
        logit_diffs_replace_all,
        logit_diffs_average_ablate,
        logit_diffs_zero_circle,
        logit_diffs_zero_everything_but_circle,
    ) = get_logit_diffs_from_subspace_formula_resid_intervention(
        task,
        probe_projection_qr=(probe_q, probe_r),
        pca_k_project=intervention_pca_k,
        layer=layer,
        token=token,
        target_to_embedding=target_to_embedding,
        letter_to_intervene_on=circle_letter,
        undo_matrix=torch.tensor(mistral_pcas).T.float(),
    )

    average_before = np.mean(logit_diffs_before)
    average_after = np.mean(logit_diffs_after)
    average_replace_pca = np.mean(logit_diffs_replace_pca)
    average_replace_all = np.mean(logit_diffs_replace_all)
    average_average_ablate = np.mean(logit_diffs_average_ablate)
    average_zero_circle = np.mean(logit_diffs_zero_circle)
    average_zero_everything_but_circle = np.mean(logit_diffs_zero_everything_but_circle)

    mistral_data.append((layer, average_before, average_after, average_replace_pca, average_replace_all, average_average_ablate, average_zero_circle, average_zero_everything_but_circle))
    mistral_data.append((layer, logit_diffs_before, logit_diffs_after, logit_diffs_replace_pca, logit_diffs_replace_all, logit_diffs_average_ablate, logit_diffs_zero_circle, logit_diffs_zero_everything_but_circle))

# %%



original_probe_varying_layer_data = []

for layer in [6, 7, 8, 9, 10]:

    original_probe = torch.load(f"{task.prefix}/circle_probes_{circle_letter}/{probe_file_extension}_layer_{layer}_token_{token}_pca_5.pt")

    (
        logit_diffs_before,
        logit_diffs_after,
        logit_diffs_replace_pca,
        logit_diffs_replace_all,
        logit_diffs_average_ablate,
        logit_diffs_zero_circle,
        logit_diffs_zero_everything_but_circle,
    ) = get_logit_diffs_from_subspace_formula_resid_intervention(
        task,
        probe_projection_qr=(original_probe["probe_q"], original_probe["probe_r"]),
        pca_k_project=5,
        layer=layer,
        token=token,
        target_to_embedding=original_probe["target_to_embedding"],
        letter_to_intervene_on=circle_letter,
    )

    average_before = np.mean(logit_diffs_before)
    average_after = np.mean(logit_diffs_after)
    average_replace_pca = np.mean(logit_diffs_replace_pca)
    average_replace_all = np.mean(logit_diffs_replace_all)
    average_average_ablate = np.mean(logit_diffs_average_ablate)
    average_zero_circle = np.mean(logit_diffs_zero_circle)
    average_zero_everything_but_circle = np.mean(logit_diffs_zero_everything_but_circle)

    original_probe_varying_layer_data.append((layer, average_before, average_after, average_replace_pca, average_replace_all, average_average_ablate, average_zero_circle, average_zero_everything_but_circle))
    original_probe_varying_layer_data.append((layer, logit_diffs_before, logit_diffs_after, logit_diffs_replace_pca, logit_diffs_replace_all, logit_diffs_average_ablate, logit_diffs_zero_circle, logit_diffs_zero_everything_but_circle))

# %%

# Save data

pickle.dump(original_probe_data, open("figs/original_probe_data.pkl", "wb"))
pickle.dump(mistral_data, open("figs/mistral_data.pkl", "wb"))
pickle.dump(original_probe_varying_layer_data, open("figs/original_probe_varying_layer_data.pkl", "wb"))

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

x = [6, 7, 8, 9, 10]

# Get means
average_after_original_probe = [x[2] for x in original_probe_data[::2]]
average_after_mistral = [x[2] for x in mistral_data[::2]]
average_after_original_probe_varying_layer = [x[2] for x in original_probe_varying_layer_data[::2]]

print(average_after_original_probe[0])
print(average_after_mistral[0])
print(average_after_original_probe_varying_layer[0])

import scipy
def mean_confidence_interval(data, confidence=0.96):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h
# Get confidence intervals
original_probe_means = []
original_probe_lower = []
original_probe_upper = []
for data in original_probe_data[1::2]:
    mean, lower, upper = mean_confidence_interval(data[2])
    original_probe_means.append(mean)
    original_probe_lower.append(lower)
    original_probe_upper.append(upper)

mistral_means = []
mistral_lower = []
mistral_upper = []
for data in mistral_data[1::2]:
    mean, lower, upper = mean_confidence_interval(data[2])
    mistral_means.append(mean)
    mistral_lower.append(lower)
    mistral_upper.append(upper)

varying_layer_means = []
varying_layer_lower = []
varying_layer_upper = []
for data in original_probe_varying_layer_data[1::2]:
    mean, lower, upper = mean_confidence_interval(data[2])
    varying_layer_means.append(mean)
    varying_layer_lower.append(lower)
    varying_layer_upper.append(upper)

ax.plot(x, original_probe_means, label="Intervene with Layer 8 Probe", marker="o")
ax.fill_between(x,
                original_probe_lower,
                original_probe_upper,
                alpha=0.3)

ax.plot(x, mistral_means, label="Intervene with SAE Subspace", marker="o")
ax.fill_between(x,
                mistral_lower,
                mistral_upper,
                alpha=0.3)

ax.plot(x, varying_layer_means, label="Intervene with Probe", marker="o")
ax.fill_between(x,
                varying_layer_lower,
                varying_layer_upper,
                alpha=0.3)

ax.set_xlabel("Layer")
ax.set_xticks(x)
ax.set_ylabel("Average logit difference after intervention")
ax.legend()
plt.savefig("figs/circle_intervention_comparison.pdf", bbox_inches="tight")

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Map each target value to a consistent color based on its position in the circle
cmap = plt.get_cmap("tab10")

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
added_labels = set()
for i in range(len(projections)):
    if int(oned_targets[i]) not in added_labels:
        added_labels.add(int(oned_targets[i]))
        plt.plot(projections[i, 0], projections[i, 1], ".", color=cmap(int(oned_targets[i])), markersize=10, label=days_of_week[int(oned_targets[i])])
    else:
        plt.plot(projections[i, 0], projections[i, 1], ".", color=cmap(int(oned_targets[i])), markersize=10)

# Sort legend by days of the week
handles, labels = ax.get_legend_handles_labels()
order = np.argsort([days_of_week.index(label) for label in labels])
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper left", bbox_to_anchor=(-0.1, 1.2), ncol=4)

ax.set_xlabel("Projection onto second SAE PCA component")
ax.set_ylabel("Projection onto third SAE PCA component")
plt.savefig("figs/circle_sae_projections.pdf", bbox_inches="tight")

# %%
