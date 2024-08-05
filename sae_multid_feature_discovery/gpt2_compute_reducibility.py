"""
For a given layer and cluster, computes the PCAs of the reconstructed
activations of that cluster, and computes our
reducibility metrics, \epsilon-mixture and separability.
"""

import os
import einops
import pickle
import json
import argparse

# hopefully this will help with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ["TRANSFORMERS_CACHE"] = "/om/user/ericjm/.cache/"

import einops
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sae_lens import SAE
# import transformer_lens
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from sklearn.decomposition import PCA
import plotly.subplots as sp
import plotly.graph_objects as go

from utils import BASE_DIR

def get_gpt2_sae(device, layer):
    return SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=f"blocks.{layer}.hook_resid_pre",  # won't always be a hook point
        device=device
    )[0]


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
    decoder_vecs = ae.W_dec.data.detach().cpu().numpy()
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

def mutual_information(xy_hist, eps=1e-10):
    """Computes mutual information from a 2d histogram.
    Args:
        xy_hist (np.array): contains counts for each bin in 2d
        eps (float): small value to avoid log(0) and division by zero
    Returns:
        float: mutual information
    """
    joint = xy_hist / (np.sum(xy_hist) + eps)
    marginal_x = np.sum(joint, axis=1)
    marginal_y = np.sum(joint, axis=0)
    product = np.outer(marginal_x, marginal_y)
    mask = joint > 0
    mutual_info = np.sum(joint[mask] * np.log((joint[mask] + eps) / (product[mask] + eps)))
    return mutual_info

def rotate(xy, angle):
    """Rotates the cloud of points xy (2d) by `angle` radians.
    TODO: test this on simple examples
    """
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(xy, rotation)

def get_separability(xy, bins_per_dim, angles):
    """Computes minimum mutual information over rotations of `xy`
    by `angles`.
    Args:
        xy (np.array): shape (d, 2)
    Returns:
        (min_mutual_info, mutual_infos)
    """
    mutual_infos = []
    for angle in angles:
        rotated_xy = rotate(xy, angle)
        histogram, _, _ = np.histogram2d(rotated_xy[:, 0], rotated_xy[:, 1], bins=bins_per_dim)
        mutual_info = mutual_information(histogram)
        mutual_infos.append(mutual_info)
    return min(mutual_infos), np.array(mutual_infos)


def get_projection_loss(xy, epsilon, temperature, a, b):
    projections = einops.einsum(xy, a, "batch dim, dim -> batch") + b
    z = projections / torch.sqrt(torch.mean(projections**2))
    return torch.mean(torch.sigmoid((epsilon - torch.abs(z)) / temperature))

def get_mixture_subspace_params(xy, epsilon):
    xy = torch.tensor(xy, dtype=torch.float32)
    a = torch.randn([2], requires_grad=True)
    b = torch.zeros([], requires_grad=True)

    learning_rate = 0.1
    num_iterations = 10000

    # Gradient descent loop
    for i in range(num_iterations):
        temperature = 1 - i / num_iterations
        # Compute the function value and its gradient
        loss = get_projection_loss(xy, epsilon, temperature, a, b)
        loss.backward()  # Compute gradients
        with torch.no_grad():
            # Update x using gradient descent
            a += learning_rate * a.grad
            b += learning_rate * b.grad

        # Manually zero the gradients after updating weights
        a.grad.zero_()
        b.grad.zero_()

    return a.detach().numpy(), b.detach().numpy()

def save_metrics_and_figures(reconstructions_pca, args):
    """
    reconstructions_pca (np.array): PCA of the cluster reconstructions
    """
    metrics = {}
    plt.figure(figsize=(8, reconstructions_pca.shape[1]*2))
    for pcai in range(reconstructions_pca.shape[1]-1):
        xy = reconstructions_pca[:, pcai:pcai+2]

        # Trim points within a radius
        xy = xy[np.linalg.norm(xy, axis=1) > args.radius]

        ax1 = plt.subplot(reconstructions_pca.shape[1]-1, 3, 3*pcai+1)
        ax2 = plt.subplot(reconstructions_pca.shape[1]-1, 3, 3*pcai+2)
        ax3 = plt.subplot(reconstructions_pca.shape[1]-1, 3, 3*pcai+3)
        axs = [ax1, ax2, ax3]

        # ---------- Mixture testing ---------- 
        epsilon = args.epsilon
        a, b = get_mixture_subspace_params(xy, epsilon)

        a_norm = np.sqrt(np.sum(a**2))
        normalized_a = a 

        proj_x = (np.tensordot(xy, a, axes=1) + b) / a_norm
        eps = epsilon * np.sqrt(np.mean(proj_x**2))
        z = proj_x / np.sqrt(np.mean(proj_x**2))
        percent_within_epsilon = np.mean(np.abs(z) < epsilon)
        
        axs[1].hist(z, bins=100, color="k")
        axs[1].axvline(x=-epsilon, color="red", linestyle=(0, (5, 5)))
        axs[1].axvline(x=epsilon, color="red", linestyle=(0, (5, 5)))
        axs[1].set_xlabel("normalized $\\mathbf{v} \\cdot \\mathbf{f} + c$")
        axs[1].set_ylabel("count")
        axs[1].set_title("$M_\\epsilon(\\mathbf{f})=" + str(round(float(percent_within_epsilon), 4)) + "$", color='red')
        axs[1].spines[["top", "left", "right"]].set_visible(False)
        axs[1].grid(axis="y")

        b_norm = b / a_norm
        b = normalized_a * b_norm

        axs[0].scatter(xy[:, 0], xy[:, 1], color="k", s=2)
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
        print("Mixture index: ", percent_within_epsilon)

        # Separability testing
        bins_per_dim = 10
        angles = np.linspace(0, 2 * np.pi, 100)
        mutual_info, mutual_infos = get_separability(xy, bins_per_dim, angles)

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
            angles,
            mutual_infos / np.log(2),
            "g",
        )
        # offset = 0.0 if max_ind / angular_res % 0.25 > 0.1 else 0.25
        offset = 0
        min_ind = np.argmin(mutual_infos)
        axs[2].plot(
            [
                angles[min_ind] + offset,
                angles[min_ind] + offset,
            ],
            [0, mutual_info / np.log(2)],
            "g--",
        )
        axs[2].set_title("$S(\\mathbf{f})=" + str(round(float(mutual_info / np.log(2)), 4)) + "$", color='green')
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
        axs[2].set_ylim((0, np.max(mutual_infos / np.log(2))))

        ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        axs[2].set_xticks(ticks)
        labels = ["0", "$\\frac{\\pi}{2}$", "$\\pi$", "$\\frac{3\\pi}{2}$", "$2\\pi$"]
        axs[2].set_xticklabels(labels)

        metrics[f"{pcai}-{pcai+1}"] = {"mixture_index": percent_within_epsilon, "separability_index": (mutual_info / np.log(2)).item()}
    
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
    parser.add_argument("--epsilon", type=float, help="Epsilon for mixture index", default=0.1)
    parser.add_argument("--save_dir", type=str, help="Directory to save figures", default="metrics")
    args = parser.parse_args()

    main(args)
