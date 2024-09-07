import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from task import get_acts, get_acts_pca
from functools import partial
from tqdm import tqdm


def do_regression(task, explanatory_vecs, target, verbose=True, plot_index=0):
    stacked_explanatory_vecs = torch.stack(explanatory_vecs, dim=1).float()

    least_squares_sol = (
        torch.linalg.pinv(
            stacked_explanatory_vecs.T @ stacked_explanatory_vecs, atol=1e-2
        )
        @ stacked_explanatory_vecs.T
        @ target
    )
    predictions = stacked_explanatory_vecs @ least_squares_sol

    predictions = stacked_explanatory_vecs @ least_squares_sol

    average_predictions = target.mean(axis=0)

    r_squared = (
        1
        - ((target - predictions) ** 2).sum()
        / ((target - average_predictions) ** 2).sum()
    )

    residuals = target - predictions

    pca_of_residuals = PCA(n_components=8).fit(residuals)
    projected_residuals = pca_of_residuals.transform(residuals)

    if verbose:
        print(r_squared)

        print(
            pca_of_residuals.explained_variance_ratio_,
            np.sum(pca_of_residuals.explained_variance_ratio_),
        )

        plt.scatter(projected_residuals[:, 0], projected_residuals[:, 1])

        for i, problem in enumerate(task.generate_problems()):
            plt.text(
                projected_residuals[i, 0],
                projected_residuals[i, 1],
                problem.info[plot_index],
                fontsize=12,
            )

        plt.show()

    return r_squared.item(), residuals, predictions, least_squares_sol


# Tries to explain the input acts with one hot a, one hot b, circle in a - b, and circle in c. Then returns the pseudoinverse of the learned
# projection into the circle in c.
def find_c_circle(task, acts):
    a = torch.tensor([problem.info[0] for problem in task.generate_problems()])
    b = torch.tensor([problem.info[1] for problem in task.generate_problems()])
    c = torch.tensor([problem.info[2] for problem in task.generate_problems()])

    circle_size = len(task.allowable_tokens)

    explanatory_vecs = [torch.ones_like(a)]
    explanatory_vecs += [a == i for i in range(8)]
    explanatory_vecs += [b == i for i in range(8)]
    explanatory_vecs += [torch.cos(2 * np.pi * (a - b) / circle_size)]
    explanatory_vecs += [torch.sin(2 * np.pi * (a - b) / circle_size)]

    r_squared_before, _, _, _ = do_regression(
        task, explanatory_vecs, acts, verbose=False, plot_index=2
    )

    explanatory_vecs += [torch.cos(2 * np.pi * (c) / circle_size)]
    explanatory_vecs += [torch.sin(2 * np.pi * (c) / circle_size)]

    r_squared_after, _, _, least_squares_sol = do_regression(
        task, explanatory_vecs, acts, verbose=False, plot_index=2
    )

    print(r_squared_before, r_squared_after)

    psuedo_inverse_least_squares_sol = torch.linalg.pinv(least_squares_sol, atol=1e-2)

    return psuedo_inverse_least_squares_sol[:, -2:]


# Does an intervention on a target subspace mapped to be probe_projection_qr.
# The target_to_embedding matrix maps the target we are intervening on (a or c) to a vector in this subspace.
def get_logit_diffs_from_subspace_formula_resid_intervention(
    task,
    probe_projection_qr,
    pca_k_project,
    layer,
    token,
    target_to_embedding,
    letter_to_intervene_on,  # a or c
):
    problems = task.generate_problems()
    model = task.get_model()
    pca_projection_matrix = get_acts_pca(task, layer, token, pca_k=pca_k_project)[
        1
    ].components_
    device = next(model.parameters()).device

    probe_q, probe_r = probe_projection_qr
    probe_q = probe_q.to(device)
    probe_r = probe_r.to(device)
    target_embedding_in_q_space = target_to_embedding.to(device) @ probe_r.inverse()

    pca_projection_matrix = torch.tensor(pca_projection_matrix).float().to(device).T

    all_pcas = (
        torch.tensor(
            get_acts_pca(task=task, layer=layer, token=token, pca_k=pca_k_project)[0]
        )
        .float()
        .to(device)
    )

    all_acts = (
        get_acts(task=task, layer_fetch=layer, token_fetch=token).float().to(device)
    )
    average_acts = all_acts.mean(dim=0)

    def get_corresponding_problem_id(original_problem, to_change_to):
        for i, problem in enumerate(problems):
            if letter_to_intervene_on == "a":
                if (
                    problem.info[0] == to_change_to
                    and problem.info[1] == original_problem.info[1]
                ):
                    return i
            elif letter_to_intervene_on == "c":
                if problem.info[2] == to_change_to and problem.info[1] == 3:
                    return i

    def patch_in_subspace_activations(existing_residual_component, hook, token, change):
        local_device = existing_residual_component.device

        local_pca_projection_matrix = pca_projection_matrix.to(local_device)

        local_probe_q = probe_q.to(local_device)

        local_rep_in_q_space = target_embedding_in_q_space[change].to(local_device)

        to_add = local_rep_in_q_space @ local_probe_q.T @ local_pca_projection_matrix.T

        local_average_acts = average_acts.to(local_device)

        to_subtract = (
            local_average_acts
            @ local_pca_projection_matrix
            @ local_probe_q
            @ local_probe_q.T
            @ local_pca_projection_matrix.T
        )

        existing_residual_component[:, token, :] = (
            local_average_acts + to_add - to_subtract
        )

        return existing_residual_component

    def patch_replace_pca(
        existing_residual_component, hook, token, original_problem, to_change_to
    ):
        to_patch_in = None
        i = get_corresponding_problem_id(original_problem, to_change_to)
        to_patch_in = all_pcas[i]

        restricted_pos_residual = existing_residual_component[:, token, :]
        local_pca_projection_matrix = pca_projection_matrix.to(
            restricted_pos_residual.device
        )

        restricted_pos_residual -= (
            restricted_pos_residual @ local_pca_projection_matrix
        ) @ local_pca_projection_matrix.T

        to_patch_in = to_patch_in.to(restricted_pos_residual.device)
        restricted_pos_residual += to_patch_in @ local_pca_projection_matrix.T

        existing_residual_component[:, token, :] = restricted_pos_residual

        return existing_residual_component

    def patch_replace_all(
        existing_residual_component, hook, token, original_problem, to_change_to
    ):
        i = get_corresponding_problem_id(original_problem, to_change_to)
        existing_residual_component[:, token, :] = all_acts[i]
        return existing_residual_component

    def average_ablate_pca(existing_residual_component, hook, token):
        local_device = existing_residual_component.device
        local_average_acts = average_acts.to(local_device)
        existing_residual_component[:, token, :] = local_average_acts
        return existing_residual_component

    def zero_ablate_everything_but_subspace(existing_residual_component, hook, token):
        restricted_pos_residual = existing_residual_component[:, token, :]
        local_device = restricted_pos_residual.device
        local_pca_projection_matrix = pca_projection_matrix.to(local_device)
        local_average_acts = average_acts.to(local_device)

        subtracted_average = restricted_pos_residual - local_average_acts

        local_probe_q = probe_q.to(local_device)

        subspace = (
            subtracted_average
            @ local_pca_projection_matrix
            @ local_probe_q
            @ local_probe_q.T
            @ local_pca_projection_matrix.T
        )

        existing_residual_component[:, token, :] = subspace + local_average_acts

        return existing_residual_component

    def zero_ablate_subspace(existing_residual_component, hook, token):
        restricted_pos_residual = existing_residual_component[:, token, :]
        local_device = restricted_pos_residual.device
        local_pca_projection_matrix = pca_projection_matrix.to(local_device)
        local_average_acts = average_acts.to(local_device)

        subtracted_average = restricted_pos_residual - local_average_acts

        local_probe_q = probe_q.to(local_device)

        subspace = (
            subtracted_average
            @ local_pca_projection_matrix
            @ local_probe_q
            @ local_probe_q.T
            @ local_pca_projection_matrix.T
        )

        restricted_pos_residual -= subspace

        existing_residual_component[:, token, :] = restricted_pos_residual

        return existing_residual_component

    def get_replace_pca_hook(layer, token, original_problem, to_change_to):
        if layer == 0:
            layer_name = f"blocks.{layer}.hook_resid_pre"
        else:
            layer_name = f"blocks.{layer - 1}.hook_resid_post"
        return (
            layer_name,
            partial(
                patch_replace_pca,
                token=token,
                original_problem=original_problem,
                to_change_to=to_change_to,
            ),
        )

    def get_replace_all_hook(layer, token, original_problem, to_change_to):
        if layer == 0:
            layer_name = f"blocks.{layer}.hook_resid_pre"
        else:
            layer_name = f"blocks.{layer - 1}.hook_resid_post"
        return (
            layer_name,
            partial(
                patch_replace_all,
                token=token,
                original_problem=original_problem,
                to_change_to=to_change_to,
            ),
        )

    def get_subspace_hook(layer, token, change):
        if layer == 0:
            layer_name = f"blocks.{layer}.hook_resid_pre"
        else:
            layer_name = f"blocks.{layer - 1}.hook_resid_post"
        return (
            layer_name,
            partial(patch_in_subspace_activations, token=token, change=change),
        )

    def get_average_ablate_hook(layer, token):
        if layer == 0:
            layer_name = f"blocks.{layer}.hook_resid_pre"
        else:
            layer_name = f"blocks.{layer - 1}.hook_resid_post"
        return (
            layer_name,
            partial(average_ablate_pca, token=token),
        )

    def get_zero_ablate_everything_but_subspace_hook(layer, token):
        if layer == 0:
            layer_name = f"blocks.{layer}.hook_resid_pre"
        else:
            layer_name = f"blocks.{layer - 1}.hook_resid_post"
        return (
            layer_name,
            partial(zero_ablate_everything_but_subspace, token=token),
        )

    def get_zero_ablate_subspace_hook(layer, token):
        if layer == 0:
            layer_name = f"blocks.{layer}.hook_resid_pre"
        else:
            layer_name = f"blocks.{layer - 1}.hook_resid_post"
        return (
            layer_name,
            partial(zero_ablate_subspace, token=token),
        )

    def get_token(task, i):
        return task.allowable_tokens[i]

    target_size = len(task.allowable_tokens)

    logit_diffs_before = []
    logit_diffs_after = []
    logit_diffs_replace_pca = []
    logit_diffs_replace_all = []
    logit_diffs_zero_pca = []
    logit_diffs_zero_subspace = []
    logit_diffs_zero_everything_but_subspace = []
    for to_change_to in tqdm(range(target_size)):
        for problem in problems:
            prompt = problem.prompt

            original_a, original_b, original_c = problem.info

            if letter_to_intervene_on == "a":
                new_a = to_change_to

                if new_a == original_a:
                    continue

                new_c = (original_b + new_a) % target_size

            elif letter_to_intervene_on == "c":
                new_c = to_change_to

                if new_c == original_c:
                    continue

            original_correct_day = get_token(task, original_c)
            new_correct_day = get_token(task, new_c)

            original_correct_logit = model.to_single_token(original_correct_day)
            new_correct_logit = model.to_single_token(new_correct_day)

            assert new_correct_logit != original_correct_logit

            logits = model(prompt)[0][-1]
            logit_diffs_before.append(
                logits[original_correct_logit] - logits[new_correct_logit]
            )

            hook = get_subspace_hook(layer, token, to_change_to)
            logits = model.run_with_hooks(prompt, fwd_hooks=[hook])[0][-1]
            logit_diffs_after.append(
                logits[original_correct_logit] - logits[new_correct_logit]
            )

            hook = get_replace_pca_hook(layer, token, problem, to_change_to)
            logits = model.run_with_hooks(prompt, fwd_hooks=[hook])[0][-1]
            logit_diffs_replace_pca.append(
                logits[original_correct_logit] - logits[new_correct_logit]
            )

            hook = get_replace_all_hook(layer, token, problem, to_change_to)
            logits = model.run_with_hooks(prompt, fwd_hooks=[hook])[0][-1]
            logit_diffs_replace_all.append(
                logits[original_correct_logit] - logits[new_correct_logit]
            )

            hook = get_average_ablate_hook(layer, token)
            logits = model.run_with_hooks(prompt, fwd_hooks=[hook])[0][-1]
            logit_diffs_zero_pca.append(
                logits[original_correct_logit] - logits[new_correct_logit]
            )

            hook = get_zero_ablate_subspace_hook(layer, token)
            logits = model.run_with_hooks(prompt, fwd_hooks=[hook])[0][-1]
            logit_diffs_zero_subspace.append(
                logits[original_correct_logit] - logits[new_correct_logit]
            )

            hook = get_zero_ablate_everything_but_subspace_hook(layer, token)
            logits = model.run_with_hooks(prompt, fwd_hooks=[hook])[0][-1]
            logit_diffs_zero_everything_but_subspace.append(
                logits[original_correct_logit] - logits[new_correct_logit]
            )

    logit_diffs_before = [diff.cpu() for diff in logit_diffs_before]
    logit_diffs_after = [diff.cpu() for diff in logit_diffs_after]
    logit_diffs_replace_pca = [diff.cpu() for diff in logit_diffs_replace_pca]
    logit_diffs_replace_all = [diff.cpu() for diff in logit_diffs_replace_all]
    logit_diffs_zero_pca = [diff.cpu() for diff in logit_diffs_zero_pca]
    logit_diffs_zero_subspace = [diff.cpu() for diff in logit_diffs_zero_subspace]
    logit_diffs_zero_everything_but_subspace = [
        diff.cpu() for diff in logit_diffs_zero_everything_but_subspace
    ]

    return (
        np.array(logit_diffs_before),
        np.array(logit_diffs_after),
        np.array(logit_diffs_replace_pca),
        np.array(logit_diffs_replace_all),
        np.array(logit_diffs_zero_pca),
        np.array(logit_diffs_zero_subspace),
        np.array(logit_diffs_zero_everything_but_subspace),
    )
