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

if not is_notebook():
    parser = argparse.ArgumentParser(description="Nonlinear LLMs")
    parser.add_argument(
        "problem_type",
        type=str,
        choices=["day", "month"],
        help='Choose "day" or "month"',
    )
    parser.add_argument(
        "intervene_on",
        type=str,
        choices=["a", "c"],
        help='Choose "a" or "c" for intervention',
    )
    parser.add_argument(
        "model",
        type=str,
        choices=["llama", "mistral"],
        help="Choose 'llama' or 'mistral' model",
    )
    parser.add_argument("--device", type=int, default=4, help="CUDA device number")
    parser.add_argument(
        "--use_inverse_regression_probe",
        action="store_true",
        help="Use inverse regression probe to find circle",
    )
    parser.add_argument(
        "--intervention_pca_k",
        type=int,
        help="Number of PCA components for intervention. Default 5 for learning probe, 20 for inverse regression probe.",
    )
    parser.add_argument(
        "--repeat_probing",
        action="store_true",
        help="Repeat probing until probes have low accuracy.",
    )
    parser.add_argument(
        "--probe_on_cos",
        action="store_true",
        help="Probe on cosine component of circle.",
    )
    parser.add_argument(
        "--probe_on_sin",
        action="store_true",
        help="Probe on sine component of circle.",
    )
    parser.add_argument(
        "--probe_on_centered_linear",
        action="store_true",
        help="Probe on linear representation with center of 0.",
    )
    args = parser.parse_args()
    device = f"cuda:{args.device}"
    day_month_choice = args.problem_type
    circle_letter = args.intervene_on
    model_name = args.model
    use_inverse_regression_probe = args.use_inverse_regression_probe
    intervention_pca_k = args.intervention_pca_k

    repeat_probing = args.repeat_probing
    probe_on_cos = args.probe_on_cos
    probe_on_sin = args.probe_on_sin
    probe_on_centered_linear = args.probe_on_centered_linear

else:
    # Modify this when running manually through a notebook

    repeat_probing = False
    probe_on_cos = True
    probe_on_sin = True
    probe_on_centered_linear = False

    # device = "cuda:3"
    # circle_letter = "a"
    # day_month_choice = "day"
    # model_name = "mistral"
    # use_inverse_regression_probe = False
    # intervention_pca_k = 5

    device = "cuda:4"
    circle_letter = "c"
    day_month_choice = "day"
    model_name = "mistral"
    use_inverse_regression_probe = True
    intervention_pca_k = 20

torch.set_grad_enabled(False)

# %%


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

# %%

if day_month_choice == "day":
    task = DaysOfWeekTask(device, model_name=model_name)
else:
    task = MonthsOfYearTask(device, model_name=model_name)

# %%

if intervention_pca_k is None:
    intervention_pca_k = 20 if use_inverse_regression_probe else 5

if circle_letter == "c":
    # Token to learn a circle and intervene on
    token = task.before_c_token

    # Index into problem.info to learn a circle on
    circle_info_index = 2

    # Layers to analyze interventions with (for experiments that use muliple layers)
    layers_to_analyze = range(1, 33)
elif circle_letter == "a":
    # Token to learn a circle and intervene on
    token = task.a_token

    # Index into problem.info to learn a circle on
    circle_info_index = 0

    # Layers to analyze interventions with (for experiments that use muliple layers)
    layers_to_analyze = range(33)
else:
    raise ValueError("Invalid circle letter")

problems = task.generate_problems()

# %%

# Train circle probes on every layer
# Can be used later for intervention

probe_projections = {}
target_to_embeddings = {}

os.makedirs(f"{task.prefix}/circle_probes_{circle_letter}", exist_ok=True)

all_maes = []
all_r_squareds = []

for layer in list(layers_to_analyze):
    maes = []
    results = []
    r_squareds = []
    for pca_k in [intervention_pca_k]:
        acts, pca_obj = get_acts_pca(
            task, layer, token, pca_k=pca_k, normalize_rms=False
        )
        acts = torch.tensor(acts).float()
        # print(pca_obj.explained_variance_ratio_)
        # print(np.sum(pca_obj.explained_variance_ratio_))

        oned_targets = torch.tensor(
            [problem.info[circle_info_index] for problem in problems]
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
        acts_train = acts

        least_squares_sol = torch.linalg.lstsq(
            acts_train, multid_targets_train
        ).solution
        predictions = acts @ least_squares_sol

        probe_q, probe_r = torch.linalg.qr(least_squares_sol)

        r_squares_per_dim = []
        for dim in range(probe_dimension):
            dim_mean = multid_targets_train[:, dim].mean()
            dim_values = multid_targets_train[:, dim]
            dim_predictions = predictions[:, dim]

            r_squared = (
                1
                - ((dim_values - dim_predictions) ** 2).mean()
                / ((dim_values - dim_mean) ** 2).mean()
            )
            r_squares_per_dim.append(r_squared.item())

        average_r_squared = np.mean(r_squares_per_dim)
        r_squareds.append(r_squares_per_dim)

        results.append([np.sum(pca_obj.explained_variance_ratio_), average_r_squared])

        probe_projections[(layer, pca_k)] = (probe_q, probe_r)
        target_to_embeddings[(layer, pca_k)] = target_to_embedding

        # Save circular probe
        torch.save(
            {
                "layer": layer,
                "token": token,
                "pca_k": pca_k,
                "probe_q": probe_q,
                "probe_r": probe_r,
                "target_to_embedding": target_to_embedding,
            },
            f"{task.prefix}/circle_probes_{circle_letter}/{probe_file_extension}_layer_{layer}_token_{token}_pca_{pca_k}.pt",
        )

        mae = (predictions - multid_targets_train).abs().mean()
        maes.append(mae)
        if pca_k == intervention_pca_k:
            for end_d in range(1, probe_dimension):
                plt.title(
                    f"Layer {layer}, token {token}, MAE {mae:.2f}, R^2s {r_squares_per_dim}"
                )
                plt.plot(predictions[:, end_d - 1], predictions[:, end_d], "o")
                plt.plot(multid_targets[:, end_d - 1], multid_targets[:, end_d], "o")
                for i, problem in enumerate(problems):
                    plt.text(
                        predictions[i, end_d - 1],
                        predictions[i, end_d],
                        problem.info[circle_info_index],
                    )
                plt.show()

            to_pca_again = acts_train - acts_train @ probe_q @ probe_q.T
            pca = PCA(n_components=2).fit_transform(to_pca_again.numpy())
            plt.plot(pca[:, 0], pca[:, 1], "o")
            for i in range(len(problems)):
                plt.text(pca[i, 0], pca[i, 1], problems[i].info[circle_info_index])
            plt.show()

    all_maes.append(maes)
    all_r_squareds.append(r_squareds)

plt.close()

# %%

for i, description in enumerate(probe_index_to_description):
    plt.plot([val[0][i] for val in all_r_squareds], label=f"R^2 {description}")

plt.plot(
    [(val[0][0] + val[0][1]) / 2 for val in all_r_squareds],
    label="Average",
)
plt.xlabel("Layer")
plt.ylabel("R^2")
plt.legend()
plt.ylim(0, 1)
plt.show()


plt.plot([val[0] for val in all_maes])
plt.xlabel("Layer")
plt.ylabel("MAE of circle fit")
plt.show()

plt.close()

# %%

if use_inverse_regression_probe:
    assert circle_letter == "c"

    new_probe_projections = {}
    for layer in layers_to_analyze:
        acts_pca, pca_obj = get_acts_pca(task, layer, token, pca_k=intervention_pca_k)
        acts_pca = torch.tensor(acts_pca).float()
        circle_projection = find_c_circle(task, acts_pca)
        results = acts_pca @ circle_projection
        plt.scatter(results[:, 0], results[:, 1])
        for i, problem in enumerate(problems):
            plt.text(results[i, 0], results[i, 1], problem.info[2])
        plt.title(f"Layer {layer}")
        plt.show()
        circle_projection_q, circle_projection_r = torch.linalg.qr(circle_projection)
        new_probe_projections[(layer, intervention_pca_k)] = (
            circle_projection_q,
            circle_projection_r,
        )

        plt.close()

    probe_projections = new_probe_projections

# %%

if use_inverse_regression_probe:
    layer_averages_file = f"figs/{task.name}/rotation_probing/{circle_letter}_{probe_file_extension}_all_layers_token_{token}_mean_logit_diffs_regression_circle.pkl"
else:
    layer_averages_file = f"figs/{task.name}/rotation_probing/{circle_letter}_{probe_file_extension}_all_layers_token_{token}_mean_logit_diffs.pkl"

# %%
if os.path.exists(layer_averages_file):
    layer_averages = pickle.load(open(layer_averages_file, "rb"))
else:
    layer_averages = []
    os.makedirs(f"figs/{task.name}/rotation_probing", exist_ok=True)
# %%

for layer in layers_to_analyze:
    already = False
    for already_computed in layer_averages:
        if already_computed[0] == layer and already_computed[1] == intervention_pca_k:
            already = True
    if already:
        print("Skipping layer", layer)
        continue

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
        probe_projection_qr=probe_projections[((layer, intervention_pca_k))],
        pca_k_project=intervention_pca_k,
        layer=layer,
        token=token,
        target_to_embedding=target_to_embeddings[(layer, intervention_pca_k)],
        letter_to_intervene_on=circle_letter,
    )

    average_before = np.mean(logit_diffs_before)
    average_after = np.mean(logit_diffs_after)
    average_replace_pca = np.mean(logit_diffs_replace_pca)
    average_replace_all = np.mean(logit_diffs_replace_all)
    average_average_ablate = np.mean(logit_diffs_average_ablate)
    average_zero_circle = np.mean(logit_diffs_zero_circle)
    average_zero_everything_but_circle = np.mean(logit_diffs_zero_everything_but_circle)

    layer_averages.append(
        (
            layer,
            intervention_pca_k,
            average_before,
            average_after,
            average_replace_pca,
            average_replace_all,
            average_average_ablate,
            average_zero_circle,
            average_zero_everything_but_circle,
        )
    )

    layer_averages.append(
        (
            layer,
            intervention_pca_k,
            logit_diffs_before,
            logit_diffs_after,
            logit_diffs_replace_pca,
            logit_diffs_replace_all,
            logit_diffs_average_ablate,
            logit_diffs_zero_circle,
            logit_diffs_zero_everything_but_circle,
        )
    )

    pickle.dump(
        layer_averages,
        open(layer_averages_file, "wb"),
    )


# %%


layer_averages = pickle.load(open(layer_averages_file, "rb"))
layer_averages = layer_averages[::2]

# %%

layer_averages.sort()

layers = []

average_replace_pca = []
average_replace_circle = []
average_no_replace = []
average_replace_all = []
average_average_ablate = []
average_zero_circle = []
average_zero_everything_but_circle = []

for t in layer_averages:
    if t[1] != intervention_pca_k:
        continue
    layers.append(t[0])
    average_no_replace.append(t[2])
    average_replace_circle.append(t[3])
    average_replace_pca.append(t[4])
    average_replace_all.append(t[5])
    average_average_ablate.append(t[6])
    average_zero_circle.append(t[7])
    average_zero_everything_but_circle.append(t[8])


plt.plot(layers, average_no_replace, label="Before")
plt.plot(layers, average_replace_all, label="After w/ Patch All")
plt.plot(layers, average_replace_circle, label="After w/ Patch Circle")
plt.plot(layers, average_replace_pca, label="After w/ Replace PCA")
plt.plot(layers, average_average_ablate, label="After w/ Average Ablate")
plt.plot(layers, average_zero_circle, label="After w/ Zero Out Circle")
plt.plot(
    layers,
    average_zero_everything_but_circle,
    label="After w/ Zero Out Everything But Circle",
)
plt.xlabel("Layer")
plt.ylabel("Average logit diff")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=2)
plt.tight_layout()
fig = plt.gcf()
plt.show()
fig.savefig(
    f"figs/{task.name}/rotation_probing/{circle_letter}_{probe_file_extension}_circle_token_{token}_logit_diffs_{intervention_pca_k}.pdf",
    bbox_inches="tight",
)

plt.close()

# TODO: Josh: Continue working in this direction, is this enough to prove sufficiency?


# %%

exit(0)

# -----------------------------------------------------------------------------------
# -------- Only run this section if not using regression to find circle in c --------
# -----------------------------------------------------------------------------------

# This section tries out a bunch of different pca intervention and circle dimensions
# (we zero ablate all pca interventions dims and replace with the (lower dim) circle projection)

if not use_inverse_regression_probe:
    layer = 4 if model_name == "llama" else 17
    output_file = f"figs/{task.name}/rotation_probing/{circle_letter}_layer_{layer}_token_{token}_mean_logit_diffs.pkl"

# %%

if not use_inverse_regression_probe:
    results = {}
    for pca_k_circle in range(2, 5):
        for pca_k_project in range(pca_k_circle, 15):
            (
                logit_diffs_before,
                logit_diffs_after,
                logit_diffs_replace_pca,
                logit_diffs_replace_all,
                logit_diffs_zero_pca,
            ) = get_logit_diffs_from_circular_resid_intervention(
                task=task,
                circle_projection_qr=probe_projections[(layer, pca_k_circle)],
                pca_k_project=pca_k_project,
                layer=layer,
                token=token,
            )

            average_before = np.mean(logit_diffs_before)
            average_after = np.mean(logit_diffs_after)
            average_replace_pca = np.mean(logit_diffs_replace_pca)
            average_replace_all = np.mean(logit_diffs_replace_all)
            average_zero_pca = np.mean(logit_diffs_zero_pca)

            results[(pca_k_project, pca_k_circle)] = (
                average_before,
                average_after,
                average_replace_pca,
                average_replace_all,
                average_zero_pca,
            )


# %%

if not use_inverse_regression_probe:
    os.makedirs(f"figs/{task.name}/rotation_probing", exist_ok=True)
    pickle.dump(results, open(output_file, "wb"))

# %%

if not use_inverse_regression_probe:
    results = pickle.load(open(output_file, "rb"))
    df = pd.DataFrame(
        columns=[
            "pca_k_project",
            "pca_k_circle",
            "average_before",
            "average_after",
            "average_replace_pca",
            "average_replace_all",
        ]
    )
    for i, (
        (pca_k_project, pca_k_circle),
        (
            average_before,
            average_after,
            average_replace_pca,
            average_replace_all,
            average_zero_out_pca,
        ),
    ) in enumerate(results.items()):
        df.loc[i] = {
            "pca_k_project": pca_k_project,
            "pca_k_circle": pca_k_circle,
            "average_before": average_before,
            "average_after": average_after,
            "average_replace_pca": average_replace_pca,
            "average_replace_all": average_replace_all,
            "average_zero_out_pca": average_zero_out_pca,
        }

    plt.scatter(
        df["pca_k_project"], df["pca_k_circle"], c=df["average_after"], cmap="seismic"
    )
    plt.colorbar()
    fig = plt.gcf()
    plt.show()
    fig.savefig(
        f"figs/{task.name}/rotation_probing/{circle_letter}_layer_{layer}_token_{token}_varying_pca_dim_interventions.png"
    )

plt.close()

# -----------------------------------------------------------------------------------
# ---------------------------- End section ------------------------------------------
# -----------------------------------------------------------------------------------
