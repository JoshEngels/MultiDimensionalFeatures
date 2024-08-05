"""
For a given layer and cluster, computes the PCAs of the reconstructed
activations of that cluster, and computes our
reducibility metrics, \epsilon-mixture and separability.
"""

import os
import time
import pickle
import json
import argparse

# hopefully this will help with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TRANSFORMERS_CACHE"] = "/om/user/ericjm/.cache/"

import einops
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sae_lens import SparseAutoencoderDictionary
# import transformer_lens
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from sklearn.decomposition import PCA
import plotly.subplots as sp
import plotly.graph_objects as go

from utils import BASE_DIR

def get_gpt2_sae(device, layer):

    if type(device) == int:
        device = f"cuda:{device}"

    GPT2_SMALL_RESIDUAL_SAES_REPO_ID = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = f"blocks.{layer}.hook_resid_pre"

    FILENAME = f"{hook_point}/cfg.json"
    path = hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    FILENAME = f"{hook_point}/sae_weights.safetensors"
    hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    FILENAME = f"{hook_point}/sparsity.safetensors"
    hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    folder_path = os.path.dirname(path)

    return SparseAutoencoderDictionary.load_from_pretrained(
            folder_path, device=device
        )[f"blocks.{layer}.hook_resid_pre"]

def get_cluster_activations(sparse_sae_activations, sae_neurons_in_cluster, decoder_vecs, threshold=0.0):
    current_token = None
    all_activations = []
    all_token_indices = []
    updated = False
    for sae_value, sae_index, token_index in tqdm(zip(
        sparse_sae_activations["sparse_sae_values"],
        sparse_sae_activations["sparse_sae_indices"],
        sparse_sae_activations["all_token_indices"],
    ), total = len(sparse_sae_activations["sparse_sae_values"]), disable=True):
        if current_token == None:
            current_token = token_index
            current_activations = np.zeros(768)
        if token_index != current_token:
            if updated:
                all_activations.append(current_activations)
                all_token_indices.append(token_index)
            updated = False
            current_token = token_index
            current_activations = np.zeros(768)
        if sae_index in sae_neurons_in_cluster and sae_value > threshold:
            updated = True
            current_activations += sae_value * decoder_vecs[sae_index]

    return np.stack(all_activations), all_token_indices

def get_pcas(args):

    ae = get_gpt2_sae(device="cpu", layer=args.layer)
    decoder_vecs = ae.W_dec.data.cpu().numpy()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # load up clusters
    with open(args.clusters_file, "rb") as f:
        clusters = pickle.load(f)
    cluster = clusters[args.cluster]

    if len(cluster) > args.size_limit:
        print(f"Cluster {args.cluster} has too many features ({len(cluster)}), exiting.")
        exit()

    sparse_sae_activations = np.load(args.activations_file)

    reconstructions, token_indices = get_cluster_activations(sparse_sae_activations, set(cluster), decoder_vecs, threshold=args.threshold)
    reconstructions, token_indices = reconstructions[:args.sample_limit], token_indices[:args.sample_limit]
    # token_strs = tokenizer.batch_decode(sparse_sae_activations['all_tokens'])

    pca = PCA(n_components=min(5, len(cluster)))
    reconstructions_pca = pca.fit_transform(reconstructions)
    return reconstructions_pca

def save_metrics_and_figures(reconstructions_pca, args):
    metrics = {}
    plt.figure(figsize=(8, reconstructions_pca.shape[1]*2))
    for pcai in range(reconstructions_pca.shape[1]-1):
        x = reconstructions_pca[:, pcai:pcai+2]
        # filter out points in x that are below the radius
        x = x[np.linalg.norm(x, axis=1) > args.radius]
        
        # convert to torch tensor
        x = torch.tensor(x, dtype=torch.float32)
        
        batch_size = x.shape[0]
        n = 2

        ### MIXTURE TESTING
        def get_concentration_probability(x, epsilon, temperature, a, b):
            x = torch.tensordot(x, a, dims=1) + b
            z = x / torch.sqrt(torch.mean(x**2))
            P = torch.mean(torch.sigmoid((epsilon - torch.abs(z)) / temperature))
            return P

        def get_parameters(x, epsilon=0.1):
            # Initialize the parameter x
            a = torch.randn(
                [n], requires_grad=True
            )  # Random initialization, requires_grad=True to track gradients
            b = torch.zeros(
                [], requires_grad=True
            )  # Random initialization, requires_grad=True to track gradients

            # Define hyperparameters
            learning_rate = 0.1
            num_iterations = 10000
            #    num_iterations = 100

            # Gradient Descent loop
            for i in range(num_iterations):
                temperature = 1 - i / num_iterations
                # Compute the function value and its gradient
                P = get_concentration_probability(x, epsilon, temperature, a, b)
                P.backward()  # Compute gradients
                with torch.no_grad():
                    # Update x using gradient descent
                    a += learning_rate * a.grad
                    b += learning_rate * b.grad

                # Manually zero the gradients after updating weights
                a.grad.zero_()
                b.grad.zero_()
            return a.detach().numpy(), b.detach().numpy(), P.item()

        ### SEPARABILITY TESTING
        def bin_(xy, n, bound):
            xy = torch.clip((xy + bound) / bound * n, 0, 2 * n - 0.1)
            ints = torch.floor(xy).type(torch.int64)
            dist = torch.zeros([(2 * n) ** 2])
            indices = ints[:, 0] * (2 * n) + ints[:, 1]
            dist.scatter_add_(0, indices, torch.ones(indices.shape, dtype=xy.dtype))
            dist = torch.reshape(dist, [2 * n, 2 * n])
            print(f"dist.shape: {dist.shape}")
            print(f"frac in bins: {dist.sum() / xy.numel()}")

            #    mods = xy % 1
            #    weightings = 1 - mods
            #    dist = torch.zeros([(2*n+2)**2])
            #    indices = ints[:,0]*(2*n+2) + ints[:,1]
            #    dist.scatter_add_(0, indices, weightings[:,0]*weightings[:,1])
            #    indices = (ints[:,0]+1)*(2*n+2) + ints[:,1]
            #    dist.scatter_add_(0, indices, (1-weightings[:,0])*weightings[:,1])
            #    indices = ints[:,0]*(2*n+2) + (ints[:,1]+1)
            #    dist.scatter_add_(0, indices, weightings[:,0]*(1-weightings[:,1]))
            #    indices = (ints[:,0]+1)*(2*n+2) + (ints[:,1]+1)
            #    dist.scatter_add_(0, indices, (1-weightings[:,0])*(1-weightings[:,1]))
            #    dist = torch.reshape(dist, [2*n+2, 2*n+2])
            return dist

        def mutual_info(xy, n, bound):
            dist = bin_(xy, n, bound)
            joint = dist / torch.sum(dist)
            marginal_x = torch.sum(joint, dim=1)
            marginal_y = torch.sum(joint, dim=0)
            product = marginal_x[:, None] * marginal_y[None, :]
            mutual_info = torch.sum(joint * torch.log((joint + 1e-4) / (product + 1e-4)))
            return mutual_info
            
        def optimize_mutual_info(xy, split, n, bound, angular_res):  # xy: n, d
            d_x = split
            d_y = xy.shape[1] - split
            assert d_x == 1 and d_y == 1

            mutual_infos = []
            for i in range(angular_res):
                angle = torch.tensor(2 * np.pi * i / angular_res)
                rotation = torch.cos(angle).type(xy.dtype) * torch.eye(
                    2, dtype=xy.dtype
                ) + torch.sin(angle).type(xy.dtype) * torch.tensor(
                    [[0, -1], [1, 0]], dtype=xy.dtype
                )
                minfo = mutual_info(torch.tensordot(xy, rotation, dims=1), n, bound)
                mutual_infos.append(minfo)

            max_ind = torch.argmin(torch.tensor(mutual_infos), dim=0)
            angle = 2 * np.pi * max_ind / angular_res
            rotation = torch.cos(angle).type(xy.dtype) * torch.eye(
                2, dtype=xy.dtype
            ) + torch.sin(angle).type(xy.dtype) * torch.tensor(
                [[0, -1], [1, 0]], dtype=xy.dtype
            )
            rot_xy = torch.tensordot(xy, rotation, dims=1)
            return mutual_infos, max_ind, rotation

        def normalize(x):
            x = x - torch.mean(x, dim=0)
            x = x / torch.mean(x**2)
            return x

        ax1 = plt.subplot(reconstructions_pca.shape[1]-1, 3, 3*pcai+1)
        ax2 = plt.subplot(reconstructions_pca.shape[1]-1, 3, 3*pcai+2)
        ax3 = plt.subplot(reconstructions_pca.shape[1]-1, 3, 3*pcai+3)
        axs = [ax1, ax2, ax3]

        # Mixture testing
        epsilon = 0.1
        a, b, P = get_parameters(x, epsilon)

        x_numpy = x.numpy()

        a_norm = np.sqrt(np.sum(a**2))
        normalized_a = a / a_norm

        proj_x = (np.tensordot(x_numpy, a, axes=1) + b) / a_norm
        eps = epsilon * np.sqrt(np.mean(proj_x**2))

        z = proj_x / np.sqrt(np.mean(proj_x**2))
        axs[1].hist(z, bins=100, color="k")
        axs[1].axvline(x=-epsilon, color="red", linestyle=(0, (5, 5)))
        axs[1].axvline(x=epsilon, color="red", linestyle=(0, (5, 5)))
        axs[1].set_xlabel("normalized $\\mathbf{v} \\cdot \\mathbf{f} + c$")
        axs[1].set_ylabel("count")
        # if dataset_num == 1:
        #     height = 65
        #     xpos = -3.8
        # if dataset_num == 2:
        #     height = 500
        #     xpos = 0.4
        # if dataset_num == 3:
        #     height = 150
        #     xpos = -1.1
        # if dataset_num == 4:
        #     height = 700
        #     xpos = 0.4
        # xpos = 0.5
        # height = 150
        axs[1].set_title("$M_\\epsilon(\\mathbf{f})=" + str(round(float(P), 4)) + "$", color='red')
        # axs[1].text(
        #     xpos,
        #     height,
        #     "$M_\\epsilon(\\mathbf{f})=" + str(round(float(P), 4)) + "$",
        #     color="red",
        # )
        axs[1].spines[["top", "left", "right"]].set_visible(False)
        axs[1].grid(axis="y")

        b_norm = b / a_norm
        b = normalized_a * b_norm

        axs[0].scatter(x_numpy[:, 0], x_numpy[:, 1], color="k", s=2)
        axs[0].axline(
            -b + eps * normalized_a[:2],
            slope=-a[0] / a[1],
            color="red",
            linestyle=(0, (5, 5)),
        )
        axs[0].axline(
            -b - eps * normalized_a[:2],
            slope=-a[0] / a[1],
            color="red",
            linestyle=(0, (5, 5)),
        )

        # axs[0].axis("equal")
        # axs[0].set_xlabel("representation dim 1")
        # axs[0].set_ylabel("representation dim 2")

        print("Mixture index: ", P)

        # Separability testing
        x = normalize(x)
        xy = x
        angular_res = 1000
        n = 20
        bound = 3
        mutual_infos, max_ind, net_transform = optimize_mutual_info(
            xy, 1, n, bound, angular_res
        )
        mutual_info = mutual_infos[max_ind]

        dist = bin_(xy, n, bound)

        # axs[0].scatter(xy[:,0], xy[:,1], color='k', s=2)
        inv_transform = np.linalg.inv(net_transform.numpy())
        # cross_size = [3, 4, 1.2, 5][dataset_num - 1]
        cross_size = 2
        dir_x = cross_size * inv_transform[0, :]
        dir_y = cross_size * inv_transform[1, :]
        axs[0].plot([-dir_x[0], dir_x[0]], [-dir_x[1], dir_x[1]], "g")
        axs[0].plot([-dir_y[0], dir_y[0]], [-dir_y[1], dir_y[1]], "g")

        print("Separability index: ", mutual_info)

        axs[0].axis("equal")
        axs[0].set_xlabel(f"PCA dim {pcai}")
        axs[0].set_ylabel(f"PCA dim {pcai+1}")

        axs[0].spines[["left", "bottom", "right", "top"]].set_visible(False)
        axs[0].tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        axs[0].tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

        axs[2].plot(
            2 * np.pi * np.arange(angular_res) / angular_res,
            [minfo.numpy() / np.log(2) for minfo in mutual_infos],
            "g",
        )
        offset = 0.0 if max_ind / angular_res % 0.25 > 0.1 else 0.25
        axs[2].plot(
            [
                2 * np.pi * (max_ind / angular_res % 0.25 + offset),
                2 * np.pi * (max_ind / angular_res % 0.25 + offset),
            ],
            [0, mutual_info / np.log(2)],
            "g--",
        )
        axs[2].set_title("$S(\\mathbf{f})=" + str(round(float(mutual_info / np.log(2)), 4)) + "$", color='green')
        # axs[2].text(
        #     2 * np.pi * (max_ind / angular_res % 0.25 + offset) + 0.1,
        #     max(0.05, mutual_info / np.log(2) - 0.3),
        #     "$S(\\mathbf{f})=" + str(round(float(mutual_info / np.log(2)), 4)) + "$",
        #     color="green",
        # )
        axs[2].set_xlabel("angle $\\theta$")
        axs[2].set_ylabel("Mutual info (bits)")

        axs[2].spines[["top"]].set_visible(False)
        axs[2].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
        axs[2].tick_params(axis="y", which="both", left=True, right=False, labelleft=True)
        axs[2].grid(axis="y")

        # set the x-spine (see below for more info on `set_position`)
        axs[2].spines["left"].set_position(("data", 0))
        axs[2].spines["right"].set_position(("data", 2 * np.pi))
        axs[2].spines["bottom"].set_position(("data", 0))

        axs[2].set_xlim((0, 2 * np.pi))
        axs[2].set_ylim((0, np.max([minfo.numpy() / np.log(2) for minfo in mutual_infos])))

        ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        axs[2].set_xticks(ticks)
        labels = ["0", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]
        axs[2].set_xticklabels(labels)

        metrics[f"{pcai}-{pcai+1}"] = {"mixture_index": P, "separability_index": (mutual_info / np.log(2)).item()}
    
    # save metrics and figures
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"gpt2-layer{args.layer}-cluster{args.cluster}-threshold{args.threshold}-radius{args.radius}.png"))
    plt.close()

    # save metrics to a file in args.save_dir as json
    with open(os.path.join(args.save_dir, f"gpt2-layer{args.layer}-cluster{args.cluster}-threshold{args.threshold}-radius{args.radius}-metrics.json"), "w") as f:
        json.dump(metrics, f)
    

def main(args):
    """
    args (argparse.Namespace): Command line arguments
    """
    os.makedirs(args.save_dir, exist_ok=True)
    reconstructions_pca = get_pcas(args)
    save_metrics_and_figures(reconstructions_pca, args)

    # for pcai in range(min(5, len(cluster))-1):
    #     x = reconstructions_pca[:, pcai:pcai+2]
    #     # convert to torch tensor
    #     x = torch.tensor(x, dtype=torch.float32)
        
    #     batch_size = x.shape[0]
    #     n = 2

    #     ### MIXTURE TESTING
    #     def get_concentration_probability(x, epsilon, temperature, a, b):
    #         x = torch.tensordot(x, a, dims=1) + b
    #         z = x / torch.sqrt(torch.mean(x**2))
    #         P = torch.mean(torch.sigmoid((epsilon - torch.abs(z)) / temperature))
    #         return P

    #     def get_parameters(x, epsilon=0.1):
    #         # Initialize the parameter x
    #         a = torch.randn(
    #             [n], requires_grad=True
    #         )  # Random initialization, requires_grad=True to track gradients
    #         b = torch.zeros(
    #             [], requires_grad=True
    #         )  # Random initialization, requires_grad=True to track gradients

    #         # Define hyperparameters
    #         learning_rate = 0.1
    #         num_iterations = 10000
    #         #    num_iterations = 100

    #         # Gradient Descent loop
    #         for i in range(num_iterations):
    #             temperature = 1 - i / num_iterations
    #             # Compute the function value and its gradient
    #             P = get_concentration_probability(x, epsilon, temperature, a, b)
    #             P.backward()  # Compute gradients
    #             with torch.no_grad():
    #                 # Update x using gradient descent
    #                 a += learning_rate * a.grad
    #                 b += learning_rate * b.grad

    #             # Manually zero the gradients after updating weights
    #             a.grad.zero_()
    #             b.grad.zero_()
    #         return a.detach().numpy(), b.detach().numpy(), P.item()

    #     ### SEPARABILITY TESTING
    #     def bin_(xy, n, bound):
    #         xy = torch.clip((xy + bound) / bound * n, 0, 2 * n - 0.1)
    #         ints = torch.floor(xy).type(torch.int64)
    #         dist = torch.zeros([(2 * n) ** 2])
    #         indices = ints[:, 0] * (2 * n) + ints[:, 1]
    #         dist.scatter_add_(0, indices, torch.ones(indices.shape, dtype=xy.dtype))
    #         dist = torch.reshape(dist, [2 * n, 2 * n])

    #         #    mods = xy % 1
    #         #    weightings = 1 - mods
    #         #    dist = torch.zeros([(2*n+2)**2])
    #         #    indices = ints[:,0]*(2*n+2) + ints[:,1]
    #         #    dist.scatter_add_(0, indices, weightings[:,0]*weightings[:,1])
    #         #    indices = (ints[:,0]+1)*(2*n+2) + ints[:,1]
    #         #    dist.scatter_add_(0, indices, (1-weightings[:,0])*weightings[:,1])
    #         #    indices = ints[:,0]*(2*n+2) + (ints[:,1]+1)
    #         #    dist.scatter_add_(0, indices, weightings[:,0]*(1-weightings[:,1]))
    #         #    indices = (ints[:,0]+1)*(2*n+2) + (ints[:,1]+1)
    #         #    dist.scatter_add_(0, indices, (1-weightings[:,0])*(1-weightings[:,1]))
    #         #    dist = torch.reshape(dist, [2*n+2, 2*n+2])
    #         return dist

    #     def mutual_info(xy, n, bound):
    #         dist = bin_(xy, n, bound)
    #         joint = dist / torch.sum(dist)
    #         marginal_x = torch.sum(joint, dim=1)
    #         marginal_y = torch.sum(joint, dim=0)
    #         product = marginal_x[:, None] * marginal_y[None, :]
    #         mutual_info = torch.sum(joint * torch.log((joint + 1e-4) / (product + 1e-4)))
    #         return mutual_info
            
    #     def optimize_mutual_info(xy, split, n, bound, angular_res):  # xy: n, d
    #         d_x = split
    #         d_y = xy.shape[1] - split
    #         assert d_x == 1 and d_y == 1

    #         mutual_infos = []
    #         for i in range(angular_res):
    #             angle = torch.tensor(2 * np.pi * i / angular_res)
    #             rotation = torch.cos(angle).type(xy.dtype) * torch.eye(
    #                 2, dtype=xy.dtype
    #             ) + torch.sin(angle).type(xy.dtype) * torch.tensor(
    #                 [[0, -1], [1, 0]], dtype=xy.dtype
    #             )
    #             minfo = mutual_info(torch.tensordot(xy, rotation, dims=1), n, bound)
    #             mutual_infos.append(minfo)

    #         max_ind = torch.argmin(torch.tensor(mutual_infos), dim=0)
    #         angle = 2 * np.pi * max_ind / angular_res
    #         rotation = torch.cos(angle).type(xy.dtype) * torch.eye(
    #             2, dtype=xy.dtype
    #         ) + torch.sin(angle).type(xy.dtype) * torch.tensor(
    #             [[0, -1], [1, 0]], dtype=xy.dtype
    #         )
    #         rot_xy = torch.tensordot(xy, rotation, dims=1)
    #         return mutual_infos, max_ind, rotation

    #     def normalize(x):
    #         x = x - torch.mean(x, dim=0)
    #         x = x / torch.mean(x**2)
    #         return x

    #     # Mixture testing
    #     epsilon = 0.01
    #     a, b, P = get_parameters(x, epsilon)

    #     x_numpy = x.numpy()

    #     a_norm = np.sqrt(np.sum(a**2))
    #     normalized_a = a / a_norm

    #     proj_x = (np.tensordot(x_numpy, a, axes=1) + b) / a_norm
    #     eps = epsilon * np.sqrt(np.mean(proj_x**2))

    #     z = proj_x / np.sqrt(np.mean(proj_x**2))

    #     b_norm = b / a_norm
    #     b = normalized_a * b_norm

    #     # Separability testing
    #     x = normalize(x)
    #     xy = x
    #     angular_res = 1000
    #     n = 20
    #     bound = 3
    #     mutual_infos, max_ind, net_transform = optimize_mutual_info(
    #         xy, 1, n, bound, angular_res
    #     )
    #     mutual_info = mutual_infos[max_ind]

    #     dist = bin_(xy, n, bound)

    #     # axs[0].scatter(xy[:,0], xy[:,1], color='k', s=2)
    #     # inv_transform = np.linalg.inv(net_transform.numpy())
    #     # cross_size = [3, 4, 1.2, 5][dataset_num - 1]
    #     # dir_x = cross_size * inv_transform[0, :]
    #     # dir_y = cross_size * inv_transform[1, :]
    #     # print("P type:", type(P))
    #     # print("mutual_info type:", type(mutual_info))

    #     metrics[f"{pcai}-{pcai+1}"] = {"mixture_index": P, "separability_index": (mutual_info / np.log(2)).item()}


    #     # fig.add_trace(go.Scatter(x=reconstructions_pca[:, pcai], y=reconstructions_pca[:, pcai+1],
    #     #                         mode='markers', marker=dict(color='darkblue', opacity=0.5), text=contexts_str,
    #     #                         hoverinfo='text',
    #     #                         hoverlabel=dict(bgcolor='white', font=dict(color='black'), bordercolor='black'),
    #     #                         showlegend=False),
    #     #             row=2, col=pcai+1)
    #     # fig.update_xaxes(title_text=f"PCA dim {pcai}", row=2, col=pcai+1)
    #     # fig.update_yaxes(title_text=f"PCA dim {pcai+1}", row=2, col=pcai+1)
    #     # fig.update_yaxes(title_standoff=0, row=2, col=pcai+1)
    
    # # save metrics to a file in args.save_dir as json
    # # save to  os.path.join(args.save_dir, f"gpt2-layer{args.layer}-cluster{args.cluster}-sep-metrics.json")
    # with open(os.path.join(args.save_dir, f"gpt2-layer{args.layer}-cluster{args.cluster}-sep-metrics.json"), "w") as f:
    #     json.dump(metrics, f)

    # # fig.update_layout(height=800, width=1200, title_text=f"Cluster rank {args.cluster} with {len(cluster)} features",
    # #                 font=dict(size=12))  # Adjust the main title font size here
    # #                 #   subplot_titles_font=dict(size=8))  # Adjust the subplot title font size here
    # # fig.update_annotations(font_size=13)

    # # fig.write_html(os.path.join(args.save_dir, f"gpt2-layer{args.layer}-cluster{args.cluster}.html"))
    # # fig.write_image(os.path.join(args.save_dir, f"gpt2-layer{args.layer}-cluster{args.cluster}.png"))
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create cluster figure")
    parser.add_argument("--layer", type=int, help="Layer of GPT-2", default=7)
    parser.add_argument("--clusters_file", type=str, help="File containing clusters", default="gpt-2_layer_7_clusters_spectral_n1000.pkl")
    parser.add_argument("--cluster", type=int, help="Cluster index to create plot of", default=138) # 138 is days, 251 is months
    parser.add_argument("--activations_file", type=str, help="File containing SAE activations (occurence_data)",
                        default="sae_activations_big_layer-7.npz")
    parser.add_argument("--size_limit", type=int, help="Size limit for the cluster", default=1000)
    parser.add_argument("--sample_limit", type=int, help="Max number of reconstructions in plot", default=20_000)
    parser.add_argument("--threshold", type=float, help="Threshold for activations", default=0.0)
    parser.add_argument("--radius", type=float, help="Exclude points in plane below this radius", default=0.0)
    parser.add_argument("--save_dir", type=str, help="Directory to save figures", default="metrics")
    args = parser.parse_args()

    main(args)
