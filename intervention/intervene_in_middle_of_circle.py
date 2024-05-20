# %%


from task import get_acts_pca, get_acts

from days_of_week_task import DaysOfWeekTask
from months_of_year_task import MonthsOfYearTask
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.cm import tab20


def get_points(num_angles=100, radius_vals=np.arange(0, 2, 0.1)):
    angles = 2 * np.pi * np.arange(num_angles) / num_angles
    all_points = []
    for radius in radius_vals:
        for angle in angles:
            all_points.append((radius * np.cos(angle), radius * np.sin(angle)))
    return np.array(all_points), angles, radius_vals


torch.set_grad_enabled(False)

# %%


def plot_intervention_on_circle_in_a(task, layer, pca_k, b):
    circle_tokens_str = task.allowable_tokens
    circle_size = len(circle_tokens_str)
    duration = b
    token = task.a_token
    circle_letter = "a"

    def vary_wthin_circle(circle_letter, duration, layer, token, pca_k, all_points):
        model = task.get_model()

        circle_projection_qr = torch.load(
            f"{task.prefix}/circle_probes_{circle_letter}/layer_{layer}_token_{token}_pca_{pca_k}.pt"
        )

        for problem in task.generate_problems():
            if problem.info == (0, duration, duration % circle_size):
                problem_to_run_on = problem

        pca_projection_matrix = get_acts_pca(task, layer, token, pca_k=pca_k)[
            1
        ].components_
        pca_projection_matrix = torch.tensor(pca_projection_matrix).to(device).T.float()

        probe_q, probe_r = (
            circle_projection_qr["probe_q"],
            circle_projection_qr["probe_r"],
        )
        probe_q = probe_q.to(device)
        probe_r = probe_r.to(device)

        def circle_point_to_q_space(circle_point):
            circle_point = torch.tensor(circle_point).float().to(device)
            return circle_point @ probe_r.inverse()

        all_acts = (
            get_acts(task=task, layer_fetch=layer, token_fetch=token).float().to(device)
        )
        average_acts = all_acts.mean(dim=0)

        def patch_in_circle_activations(
            existing_residual_component, hook, circle_point
        ):
            local_device = existing_residual_component.device

            local_pca_projection_matrix = pca_projection_matrix.to(local_device)

            to_add = (
                circle_point_to_q_space(circle_point)
                @ probe_q.T
                @ local_pca_projection_matrix.T
            )

            to_subtract = (
                average_acts
                @ local_pca_projection_matrix
                @ probe_q
                @ probe_q.T
                @ local_pca_projection_matrix.T
            )

            existing_residual_component[:, token, :] = (
                average_acts + to_add - to_subtract
            )

            return existing_residual_component

        def get_circle_hook(layer, circle_point):
            if layer == 0:
                layer_name = f"blocks.{layer}.hook_resid_pre"
            else:
                layer_name = f"blocks.{layer - 1}.hook_resid_post"
            return (
                layer_name,
                partial(patch_in_circle_activations, circle_point=circle_point),
            )

        circle_tokens = [
            model.to_single_token(token_str) for token_str in circle_tokens_str
        ]

        all_logits = []
        for circle_point in tqdm(all_points):
            hook = get_circle_hook(layer, circle_point)
            logits = model.run_with_hooks(problem_to_run_on.prompt, fwd_hooks=[hook])[
                0
            ][-1]
            circle_logits = logits[circle_tokens]
            all_logits.append(circle_logits)
        stacked_logits = torch.stack(all_logits).cpu().numpy()

        return stacked_logits

    all_points, angles, radius_vals = get_points()

    filename = f"figs/{task.name}/varying_circle/logits_{layer}_{token}_{pca_k}_{duration}_{circle_letter}.npy"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not os.path.exists(filename) or len(np.load(filename)) != len(all_points):
        all_logits = vary_wthin_circle(
            circle_letter, duration, layer, token, pca_k, all_points
        )
        np.save(filename, all_logits)

    all_logits = np.load(filename)

    # ---------------- Make plots ---------------

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot 1
    correct_a_int = []
    for radius in radius_vals:
        for angle in angles:
            starting_a_int = int(np.round(angle / (2 * np.pi) * circle_size))
            starting_a_int = starting_a_int % circle_size
            ending_a_int = (starting_a_int + duration) % circle_size
            correct_a_int.append(ending_a_int)
    correct_a_int = np.array(correct_a_int)

    for i in range(circle_size):
        axs[0, 0].scatter(
            all_points[correct_a_int == i, 0],
            all_points[correct_a_int == i, 1],
            label=circle_tokens_str[i],
            color=tab20(i),
        )
    axs[0, 0].set_title(f"Ground Truth Correct {task_level_granularity}")
    axs[0, 0].legend()
    axs[0, 0].set_xlim(-2, 2)
    axs[0, 0].set_ylim(-2, 2)
    axs[0, 0].set_aspect("equal", adjustable="box")

    # Plot 2
    best_a = np.argmax(all_logits, axis=1)
    for i in range(circle_size):
        axs[0, 1].scatter(
            all_points[best_a == i, 0],
            all_points[best_a == i, 1],
            label=circle_tokens_str[i],
            color=tab20(i),
        )
    axs[0, 1].set_title(f"Highest Logit {task_level_granularity}")
    axs[0, 1].legend()
    axs[0, 1].set_xlim(-2, 2)
    axs[0, 1].set_ylim(-2, 2)
    axs[0, 1].set_aspect("equal", adjustable="box")

    # Plot 3
    second_best_a = np.argsort(all_logits, axis=1)[:, -2]
    logit_diffs = (
        all_logits[np.arange(len(all_points)), best_a]
        - all_logits[np.arange(len(all_points)), second_best_a]
    )
    axs[1, 0].set_title(f"Logit Difference")
    sc = axs[1, 0].scatter(
        all_points[:, 0], all_points[:, 1], c=logit_diffs, cmap="viridis"
    )
    axs[1, 0].set_xlim(-2, 2)
    axs[1, 0].set_ylim(-2, 2)
    axs[1, 0].set_aspect("equal", adjustable="box")

    plt.colorbar(sc)

    # Plot 3
    # np.random.seed(37)
    # random_projection = np.random.randn(circle_size, 3)
    # colors = all_logits @ random_projection
    # colors = (colors - np.min(colors, axis=0)) / np.ptp(colors, axis=0)
    # axs[1, 0].set_title(f"Random Projection of {task_level_granularity} Logits into RGB")
    # axs[1, 0].scatter(all_points[:, 0], all_points[:, 1], c=colors)
    # axs[1, 0].set_xlim(-2, 2)
    # axs[1, 0].set_ylim(-2, 2)
    # axs[1, 0].set_aspect("equal", adjustable="box")

    # Plot 4
    # second_best_a = np.argsort(all_logits, axis=1)[:, -2]
    # third_best_a = np.argsort(all_logits, axis=1)[:, -3]

    # colors = np.zeros((len(all_points), 3))
    # colors[:, 0] = best_a / circle_size
    # colors[:, 1] = second_best_a / circle_size
    # colors[:, 2] = third_best_a / circle_size

    # axs[1, 1].set_title(f"Top 3 {task_level_granularity} Into RGB")
    # axs[1, 1].scatter(all_points[:, 0], all_points[:, 1], c=colors)
    # axs[1, 1].set_xlim(-2, 2)
    # axs[1, 1].set_ylim(-2, 2)
    # axs[1, 1].set_aspect("equal", adjustable="box")

    # Plot 4
    # Plot magnitude of max logit
    max_logit = np.max(all_logits, axis=1)
    axs[1, 1].set_title(f"Magnitude of Max Logit")
    sc = axs[1, 1].scatter(
        all_points[:, 0], all_points[:, 1], c=max_logit, cmap="viridis"
    )
    axs[1, 1].set_xlim(-2, 2)
    axs[1, 1].set_ylim(-2, 2)
    axs[1, 1].set_aspect("equal", adjustable="box")

    plt.suptitle(
        f"Intervening in circle in {circle_letter}, predicting {duration} {task_level_granularity}s ahead\nLayer {layer}, Token {token}"
    )

    plt.tight_layout()

    plt.colorbar(sc)

    fig = plt.gcf()

    plt.show()

    os.makedirs(f"figs/{task.name}/varying_circle", exist_ok=True)
    fig.savefig(
        f"figs/{task.name}/varying_circle/{layer}_{token}_{pca_k}_{duration}_{circle_letter}.pdf"
    )


# %%

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only_paper_plots", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    device = args.device

    if args.only_paper_plots:
        task_level_granularity = "day"
        model_name = "mistral"
        task = DaysOfWeekTask(device, model_name=model_name)
        layer = 5
        bs = range(2, 6)
        pca_k = 5
        for b in bs:
            plot_intervention_on_circle_in_a(task, layer, pca_k, b)

    else:
        for model_name in ["llama", "mistral"]:
            for task_level_granularity in ["day", "month"]:
                for layer in range(8):
                    if task_level_granularity == "day":
                        bs = range(1, 8)
                    elif task_level_granularity == "month":
                        bs = range(1, 13)
                    for b in bs:
                        if task_level_granularity == "day":
                            task = DaysOfWeekTask(device, model_name=model_name)
                        elif task_level_granularity == "month":
                            task = MonthsOfYearTask(device, model_name=model_name)
                        else:
                            raise ValueError(f"Unknown {task_level_granularity}")
                        for pca_k in [5]:
                            plot_intervention_on_circle_in_a(task, layer, pca_k, b)


# %%
