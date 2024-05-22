from utils import BASE_DIR  # Need this import to set the huggingface cache directory
import os
import numpy as np
import torch
from tqdm import tqdm
import einops
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import transformer_lens.patching as patching
import pickle as pkl
from sklearn.cross_decomposition import PLSRegression
import pandas as pd


class Problem:
    def __init__(self, prompt, target, info):
        self.prompt = prompt
        self.target = target
        self.info = info

    def __str__(self):
        return f"Prompt: {self.prompt}, Target: {self.target}, Info: {self.info}"

    def __repr__(self):
        return str(self)


def generate_and_save_acts(
    task,
    names_filter,
    save_file_prefix,
    verbose=False,
    save_results_csv=True,
    save_best_logit=True,
):
    if save_file_prefix != "" and save_file_prefix[-1] != "_":
        save_file_prefix += "_"
    model = task.get_model()
    forward_batch_size = 2
    num_tokens_to_generate = task.num_tokens_in_answer
    all_problems = task.generate_problems()
    output_file = task.prefix + "results.csv"

    if save_results_csv:
        os.makedirs(task.prefix, exist_ok=True)
        model_best_addition = "" if not save_best_logit else ", best_token"
        with open(output_file, "w") as f:
            f.write(
                f"a, b, c, ground_truth, model_out, model_correct{model_best_addition}\n"
            )

    total = 0
    count_correct = 0

    with torch.inference_mode():
        for start in tqdm(list(range(0, len(all_problems), forward_batch_size))):
            actual_batch_size = min(forward_batch_size, len(all_problems) - start)
            results = [""] * actual_batch_size
            problem_batch = all_problems[start : start + actual_batch_size]
            current_prompts = [problem.prompt for problem in problem_batch]
            for _ in range(num_tokens_to_generate):
                tokens = model.to_tokens(current_prompts)
                if hasattr(task, "token_device"):
                    tokens = tokens.to(task.token_device)
                if verbose:
                    tokens_to_print = tokens if actual_batch_size == 1 else tokens[0]
                    print(model.to_str_tokens(tokens_to_print))
                logit_batch, activation_batch = model.run_with_cache(
                    tokens,
                    names_filter=(
                        names_filter
                        if names_filter != "heads"
                        else lambda x: "attn" in x
                    ),
                )
                print(activation_batch.keys())

                max_logit = torch.argmax(logit_batch, dim=-1)[:, -1]

                for i in range(actual_batch_size):
                    result = model.to_string(max_logit[i])
                    results[i] += result
                    current_prompts[i] += result

            for i in range(actual_batch_size):
                results[i] = results[i].strip()

                current_problem_index = start + i
                if names_filter == "heads":
                    tensors = activation_batch.stack_head_results()[:, i, :, :]
                else:
                    tensors = []
                    for key in activation_batch.keys():
                        tensor = activation_batch[key][i]
                        tensors.append(tensor.unsqueeze(0).cpu())
                    tensors = torch.cat(tensors)
                if verbose:
                    print(tensors.shape)
                torch.save(
                    tensors,
                    f"{task.prefix}{save_file_prefix}{current_problem_index}.pt",
                )

                if save_results_csv:
                    save_best_addition = ""
                    if save_best_logit:
                        last_logits = logit_batch[i, -1]
                        best_logit = float("-inf")
                        best_token = None
                        for token in task.allowable_tokens:
                            logit = last_logits[model.to_single_token(token)]
                            if logit > best_logit:
                                best_logit = logit
                                best_token = token
                        save_best_addition = f",{best_token}"

                    current_problem = all_problems[current_problem_index]

                    model_correct = results[i] == current_problem.target
                    a, b, c = current_problem.info
                    count_correct += model_correct
                    total += 1
                    if verbose:
                        print(
                            count_correct / total,
                            results[i],
                            current_problem,
                        )
                    with open(output_file, "a") as f:
                        f.write(
                            f"{a},{b},{c},{current_problem.target},{results[i]},{model_correct}{save_best_addition}\n"
                        )


def get_all_acts(
    task,
    verbose=False,
    force_regenerate=False,
    save_results_csv=True,
    names_filter=lambda x: "resid_post" in x or "hook_embed" in x,
    save_file_prefix="",
    save_best_logit=True,
):
    if save_file_prefix != "" and save_file_prefix[-1] != "_":
        save_file_prefix += "_"
    all_problems = task.generate_problems()
    all_problems_already_generated = True
    for i in range(len(all_problems)):
        if not os.path.exists(f"{task.prefix}{save_file_prefix}{i}.pt"):
            all_problems_already_generated = False
            break
    if not all_problems_already_generated or force_regenerate:
        generate_and_save_acts(
            task,
            verbose=verbose,
            save_results_csv=save_results_csv,
            names_filter=names_filter,
            save_file_prefix=save_file_prefix,
            save_best_logit=save_best_logit,
        )
        torch.cuda.empty_cache()

    all_acts = []
    for i in range(0, len(all_problems)):
        tensors = torch.load(
            f"{task.prefix}{save_file_prefix}{i}.pt", map_location="cpu"
        )
        all_acts.append(tensors)
        if len(all_acts) > 1:
            all_acts[-1] = all_acts[-1].to(all_acts[0].device)

    all_acts = einops.rearrange(all_acts, "n layers tokens dim -> n layers tokens dim")

    return all_acts


def get_acts(
    task,
    layer_fetch,
    token_fetch,
    normalize_rms=False,
    names_filter=lambda x: "resid_post" in x or "hook_embed" in x,
    save_file_prefix="",
    force_regenerate=False,
):
    if save_file_prefix != "" and save_file_prefix[-1] != "_":
        save_file_prefix += "_"
    file_name = (
        f"{task.prefix}{save_file_prefix}layer{layer_fetch}_token{token_fetch}.pt"
    )
    if not os.path.exists(file_name) or force_regenerate:
        print(file_name, "not exists")
        all_acts = get_all_acts(
            task, names_filter=names_filter, save_file_prefix=save_file_prefix
        )
        for layer in range(all_acts.shape[1]):
            for token in range(all_acts.shape[2]):
                file_name = (
                    f"{task.prefix}{save_file_prefix}layer{layer}_token{token}.pt"
                )
                torch.save(
                    all_acts[:, layer, token, :].detach().cpu().clone(), file_name
                )
    data = torch.load(file_name)
    if normalize_rms:
        eps = 1e-5
        scale = (data.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        data = data / scale
    return data


def get_acts_pca(
    task,
    layer,
    token,
    pca_k,
    normalize_rms=False,
    names_filter=lambda x: "resid_post" in x or "hook_embed" in x,
    save_file_prefix="",
):
    act_file_name = f"{task.prefix}pca/{save_file_prefix}/layer{layer}_token{token}_pca{pca_k}{'_normalize' if normalize_rms else ''}.pt"
    pca_pkl_file_name = f"{task.prefix}pca/{save_file_prefix}/layer{layer}_token{token}_pca{pca_k}{'_normalize' if normalize_rms else ''}.pkl"
    os.makedirs(f"{task.prefix}/pca/{save_file_prefix}", exist_ok=True)

    if not os.path.exists(act_file_name) or not os.path.exists(pca_pkl_file_name):
        acts = get_acts(
            task,
            layer,
            token,
            normalize_rms=normalize_rms,
            names_filter=names_filter,
            save_file_prefix=save_file_prefix,
        )
        pca_object = PCA(n_components=pca_k).fit(acts)
        pca_acts = pca_object.transform(acts)
        torch.save(pca_acts, act_file_name)
        pkl.dump(pca_object, open(pca_pkl_file_name, "wb"))
    return torch.load(act_file_name), pkl.load(open(pca_pkl_file_name, "rb"))


def get_acts_pls(task, layer, token, pls_k, normalize_rms=False):
    act_file_name = f"{task.prefix}/pls/layer{layer}_token{token}_pls{pls_k}{'_normalize' if normalize_rms else ''}.pt"
    pls_pkl_file_name = f"{task.prefix}/pls/layer{layer}_token{token}_pls{pls_k}{'_normalize' if normalize_rms else ''}.pkl"
    os.makedirs(f"{task.prefix}/pls", exist_ok=True)

    # if not os.path.exists(act_file_name) or not os.path.exists(pls_pkl_file_name):
    if True:
        acts = get_acts(task, layer, token, normalize_rms=normalize_rms)
        # uncarried_c = [problem.info[0] + problem.info[1] for problem in task.generate_problems()]
        tens_digit_c = [(problem.info[2] // 10) for problem in task.generate_problems()]
        Y = np.array(tens_digit_c)
        pls = PLSRegression(n_components=pls_k)
        pls.fit(acts, Y)
        pls_acts = pls.transform(acts)
        torch.save(torch.tensor(pls_acts), act_file_name)
        pkl.dump(pls, open(pls_pkl_file_name, "wb"))

    return torch.load(act_file_name), pkl.load(open(pls_pkl_file_name, "rb"))


def _set_plotting_sizes():
    # Set plotting sizes
    SMALL_SIZE = 24
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 38

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc(
        "axes", labelsize=SMALL_SIZE
    )  # fontsize of the x and y labels for the small plots
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc(
        "figure", labelsize=BIGGER_SIZE
    )  # fontsize of the x and y labels for the big plots


def plot_pca(
    task,
    token_location,
    k,
    skip_k=0,
    include_embedding_layer=False,
    num_cols=6,
    plot_scaler=6,
    normalize_rms=False,
    layers=None,
    save_file_prefix="",
    legend_func=lambda x: x,
    plot_info_index=None,
    title=None,
):
    os.makedirs(f"figs/{task.name}/pca_plots", exist_ok=True)
    os.makedirs(f"figs/{task.name}/pca_plots_normalized", exist_ok=True)

    _set_plotting_sizes()
    sequential_color_map = plt.get_cmap("viridis")

    problems = task.generate_problems()

    if layers == None:
        layers = range(1, 33)
        if include_embedding_layer:
            layers = range(33)

    token_str = task.token_map[token_location]
    # token_str = ""

    num_rows = (len(layers) + num_cols - 1) // num_cols

    pca_results = [
        get_acts_pca(task, layer, token_location, k, normalize_rms=normalize_rms)
        for layer in layers
    ]

    effective_k = k - skip_k

    for color_by_name, color_by_index in task.how_to_color:
        if plot_info_index is not None and color_by_index != plot_info_index:
            continue

        if effective_k == 2:
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(plot_scaler * num_cols, plot_scaler * num_rows),
            )
            axes = axes.flatten()
        elif effective_k == 3:
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(plot_scaler * num_cols, plot_scaler * num_rows),
                subplot_kw={"projection": "3d"},
            )
            axes = axes.flatten()
        else:
            raise ValueError("effective_k must be 2 or 3")

        for layer_index, layer in enumerate(layers):
            dimensionally_reduced_acts, pca_object = pca_results[layer_index]
            percent_variance_explained = (
                np.sum(pca_object.explained_variance_ratio_[skip_k:]) * 100
            )

            dimensionally_reduced_acts = dimensionally_reduced_acts[:, skip_k:]

            ax = axes[layer_index]

            color_by_values = [problem.info[color_by_index] for problem in problems]

            # Check if we need to extract a specific integer from the color_by_values
            if "-" in color_by_name:
                assert isinstance(color_by_values[0], int)
                digit_index = int(color_by_name.split("-")[1])
                color_by_values = [
                    (x % (10 ** (digit_index + 1))) // (10**digit_index)
                    for x in color_by_values
                ]

            unique_color_values = np.unique(color_by_values)

            for index, value in enumerate(unique_color_values):
                indices_with_value = []
                for problem_index, problem_value in enumerate(color_by_values):
                    if problem_value == value:
                        indices_with_value.append(problem_index)

                indices_with_value = np.array(indices_with_value)

                if effective_k == 2:
                    ax.plot(
                        dimensionally_reduced_acts[indices_with_value, 0],
                        dimensionally_reduced_acts[indices_with_value, 1],
                        "o",
                        label=f"{legend_func(value)}",
                        color=sequential_color_map(index / len(unique_color_values)),
                    )
                elif effective_k == 3:
                    ax.scatter(
                        dimensionally_reduced_acts[indices_with_value, 0],
                        dimensionally_reduced_acts[indices_with_value, 1],
                        dimensionally_reduced_acts[indices_with_value, 2],
                        "o",
                        label=f"{legend_func(value)}",
                        color=sequential_color_map(index / len(unique_color_values)),
                    )

            if layer == 0:
                ax.set_title(
                    f"Embedding, {percent_variance_explained:.2f}% var explained"
                )
            else:
                ax.set_title(
                    f"L{layer}: {percent_variance_explained:.2f}% var explained"
                )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        if title is not None:
            plt.suptitle(title)
        else:
            plt.suptitle(
                f"PCA for token {token_location} ({token_str}) activations, colored by {color_by_name}{', normalized' if normalize_rms else ''}"
            )

        plt.tight_layout()

        if skip_k == 0:
            plt.savefig(
                f"figs/{task.name}/pca_plots{'_normalized' if normalize_rms else ''}/{save_file_prefix}token_{token_location}_{k}_{color_by_name}.pdf",
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"figs/{task.name}/pca_plots{'_normalized' if normalize_rms else ''}/{save_file_prefix}token_{token_location}_{k}_{color_by_name}_skip{skip_k}.pdf",
                bbox_inches="tight",
            )

        plt.close()


def create_and_save_combined_patching_results(
    task,
    keep_same_index,
    num_chars_in_answer_to_include,
    num_activation_patching_experiments_to_run,
    limit_per_example,
    layer_type,
    patching_sweep=None,
):
    model = task.get_model()

    os.makedirs(f"figs/{task.name}/patching", exist_ok=True)

    problems = task.generate_problems()

    possible_problem_pairs = []
    for i in range(len(problems)):
        added_for_i = 0
        for j in range(len(problems)):
            problem_1 = problems[i]
            problem_2 = problems[j]

            same_info_indices = []
            for k in range(len(problem_1.info)):
                if problem_1.info[k] == problem_2.info[k]:
                    same_info_indices.append(k)

            if len(same_info_indices) != 1 or same_info_indices[0] != keep_same_index:
                continue

            if (
                problem_1.target[:num_chars_in_answer_to_include]
                != problem_2.target[:num_chars_in_answer_to_include]
            ):
                continue

            if (
                problem_1.target[num_chars_in_answer_to_include]
                == problem_2.target[num_chars_in_answer_to_include]
            ):
                continue

            possible_problem_pairs.append((problem_1, problem_2))

            added_for_i += 1
            if added_for_i >= limit_per_example:
                break

    np.random.shuffle(possible_problem_pairs)
    possible_problem_pairs = possible_problem_pairs[
        :num_activation_patching_experiments_to_run
    ]

    # We will patch from clean to dirty to get it to be like clean
    all_patch_results = []
    with torch.no_grad():
        for clean_problem, corrupted_problem in possible_problem_pairs:
            clean_prompt = (
                clean_problem.prompt
                + clean_problem.target[:num_chars_in_answer_to_include]
            )
            corrupted_prompt = (
                corrupted_problem.prompt
                + corrupted_problem.target[:num_chars_in_answer_to_include]
            )
            clean_target = clean_problem.target[num_chars_in_answer_to_include:]
            corrupted_target = corrupted_problem.target[num_chars_in_answer_to_include:]

            clean_target_token = model.to_tokens(clean_target)[0][-1]
            corrupted_target_token = model.to_tokens(corrupted_target)[0][-1]
            print(
                model.to_str_tokens(clean_target_token),
                model.to_str_tokens(corrupted_target_token),
            )

            clean_logits, clean_cache = model.run_with_cache(clean_prompt)
            corrupted_logits, _ = model.run_with_cache(corrupted_prompt)

            last_clean_logits = clean_logits[0, -1]
            last_corrupted_logits = corrupted_logits[0, -1]

            clean_logit_diff = (
                last_clean_logits[clean_target_token]
                - last_clean_logits[corrupted_target_token]
            )
            corrupted_logit_diff = (
                last_corrupted_logits[clean_target_token]
                - last_corrupted_logits[corrupted_target_token]
            )

            def progress_to_clean_metric(new_corrupted_logits):
                new_corrupted_logit_diff = (
                    new_corrupted_logits[:, -1, clean_target_token]
                    - new_corrupted_logits[:, -1, corrupted_target_token]
                )
                return (new_corrupted_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )

            print(model.to_str_tokens(corrupted_prompt), num_chars_in_answer_to_include)

            if layer_type == "resid":
                result = patching.generic_activation_patch(
                    patch_setter=patching.layer_pos_patch_setter,
                    activation_name="resid_post",
                    index_axis_names=("layer", "pos"),
                    model=model,
                    corrupted_tokens=model.to_tokens(corrupted_prompt),
                    clean_cache=clean_cache,
                    patching_metric=progress_to_clean_metric,
                )

            elif layer_type == "attention":
                result = patching.get_act_patch_attn_out(
                    model=model,
                    corrupted_tokens=model.to_tokens(corrupted_prompt),
                    clean_cache=clean_cache,
                    patching_metric=progress_to_clean_metric,
                )

            elif layer_type == "mlp":
                result = patching.get_act_patch_mlp_out(
                    model=model,
                    corrupted_tokens=model.to_tokens(corrupted_prompt),
                    clean_cache=clean_cache,
                    patching_metric=progress_to_clean_metric,
                )
            elif layer_type == "attention_head":
                assert patching_sweep is not None

                cols = {"layer": [], "pos": [], "head": []}
                for layer in range(patching_sweep[1][0], patching_sweep[1][1]):
                    for head in range(model.cfg.n_heads):
                        cols["layer"].append(layer)
                        cols["pos"].append(patching_sweep[0])
                        cols["head"].append(head)

                patching_df = pd.DataFrame(cols)

                result = patching.get_act_patch_attn_head_out_by_pos(
                    model=model,
                    corrupted_tokens=model.to_tokens(corrupted_prompt),
                    clean_cache=clean_cache,
                    patching_metric=progress_to_clean_metric,
                    index_df=patching_df,
                    index_axis_names=None,
                )

                result = einops.rearrange(
                    result,
                    "(pos layer head) -> (pos layer) head",
                    pos=1,
                    head=model.cfg.n_heads,
                    layer=patching_sweep[1][1] - patching_sweep[1][0],
                )

                print(result.shape)
                print(result)

            all_patch_results.append(result.cpu().numpy())

    combined_patching = np.array(all_patch_results)
    np.save(
        f"figs/{task.name}/patching/{layer_type}/keep-same{keep_same_index}_chars-in-answer{num_chars_in_answer_to_include}_n{num_activation_patching_experiments_to_run}.npy",
        combined_patching,
    )


# keep_same_index is an index into info that we will keep the same during patching
# num_chars_in_answer_to_include is the number of chars to ensure are the same in both
# the clean and dirty versions of the target, and furthermore we ensure the next char is different
# This function randomly chooses num_activation_patching_experiments_to_run and averages
# the results
# TODO: For now this function assumes that keep_same_index != the index of the target
def activation_patching(
    task,
    keep_same_index,
    num_chars_in_answer_to_include,
    num_activation_patching_experiments_to_run,
    layer_type,
    limit_per_example=10,
    patching_sweep=None,
):
    combined_patching_prefix = f"figs/{task.name}/patching/{layer_type}/keep-same{keep_same_index}_chars-in-answer{num_chars_in_answer_to_include}_n{num_activation_patching_experiments_to_run}"
    combined_patching_file_npy = combined_patching_prefix + ".npy"
    os.makedirs(f"figs/{task.name}/patching/{layer_type}", exist_ok=True)
    if not os.path.exists(combined_patching_file_npy):
        create_and_save_combined_patching_results(
            task,
            keep_same_index,
            num_chars_in_answer_to_include,
            num_activation_patching_experiments_to_run,
            limit_per_example,
            layer_type,
            patching_sweep=patching_sweep,
        )
    combined_patching = np.load(combined_patching_file_npy)

    print(combined_patching.shape)
    average_patching_result = np.mean(combined_patching, axis=0)
    print(average_patching_result.shape)

    if layer_type == "attention_head":
        plt.imshow(
            average_patching_result,
            extent=(0, 32, patching_sweep[1][1], patching_sweep[1][0]),
            aspect="auto",
        )
    else:
        plt.imshow(
            average_patching_result,
            extent=(0, len(average_patching_result[0]), 33, 1),
            aspect="auto",
        )

    beginning = (
        "Residual stream"
        if layer_type == "resid"
        else (
            "Attention out"
            if layer_type == "attention"
            else "MLP out"
            if layer_type == "mlp"
            else "Attention head out"
        )
    )
    plt.title(
        f"{beginning} patching, same {['a', 'b', 'c'][keep_same_index]}, predict {task.prediction_names[num_chars_in_answer_to_include]}"
    )
    if layer_type == "attention_head":
        plt.xlabel("Head")
        plt.ylabel("Layer")
    else:
        plt.xlabel("Token Location")
        plt.ylabel("Layer")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(combined_patching_prefix + ".png", bbox_inches="tight")
    plt.close()
