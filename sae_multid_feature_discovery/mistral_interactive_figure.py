"""
Computes html and png figures for cluster of mistral SAE features.
"""
import os
import argparse
from collections import defaultdict
from itertools import islice

import pickle
import numpy as np
from tqdm.auto import tqdm
import plotly.subplots as sp
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import torch as t
from transformers import AutoTokenizer
from circuitsvis.tokens import colored_tokens

from saes.sparse_autoencoder import SparseAutoencoder

SAVE_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_similarity_matrix(layer):
    """Returns the angular similarity matrix beteen all SAE features."""
    if os.path.exists(os.path.join(SAVE_DIR, f"sae_layer{layer}_similarity.npy")):
        return np.load(os.path.join(SAVE_DIR, f"sae_layer{layer}_similarity.npy"))

    t.set_grad_enabled(False)
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    sae = SparseAutoencoder.load_from_pretrained(
        os.path.join(
            SCRIPT_DIR,
            "saes",
            f"Mistral-7B-v0.1_blocks.{args.layer}.hook_resid_pre_65536_final.pt",
        )
    ).to(device)

    Wdec_months = sae.W_dec  # (65536, 4096) -> (n_months_features, 4096)
    S = Wdec_months @ Wdec_months.T  # (n_months_features, n_months_features)
    S = t.clamp(S, -1, 1)
    S_ang = 1 - t.arccos(S) / t.pi
    S_ang = S_ang.detach().cpu().numpy()

    np.save(os.path.join(SAVE_DIR, f"sae_layer{layer}_similarity.npy"), S_ang)
    return S_ang


def get_cluster_activations(
    sparse_sae_activations,
    sae_neurons_in_cluster,
    decoder_vecs,
    sample_limit,
    max_indices=1e9,
):
    max_indices = int(max_indices)
    current_token = None
    all_activations = []
    all_token_indices = []
    updated = False
    for sae_value, sae_index, token_index in tqdm(
        islice(
            zip(
                sparse_sae_activations["sparse_sae_values"],
                sparse_sae_activations["sparse_sae_indices"],
                sparse_sae_activations["all_token_indices"],
            ),
            0,
            max_indices,
        ),
        total=max_indices,
        disable=True,
    ):
        if current_token == None:
            current_token = token_index
            current_activations = np.zeros(4096)
        if token_index != current_token:
            if updated:
                all_activations.append(current_activations)
                all_token_indices.append(token_index - 1)  # FIXED OFF-BY-ONE ERROR
                if len(all_activations) >= sample_limit:
                    break
            updated = False
            current_token = token_index
            current_activations = np.zeros(4096)
        if sae_index in sae_neurons_in_cluster:
            updated = True
            current_activations += sae_value * decoder_vecs[sae_index]

    return np.stack(all_activations), all_token_indices

def main(args):
    
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    # save the decoder vecs as a numpy array, so that in subsequent runs 
    # we won't have to load the model into GPU memory just to get the decoder
    if os.path.exists(f"sae_layer{args.layer}_decoder.npy"):
        decoder_vecs = np.load(f"sae_layer{args.layer}_decoder.npy")
    else:
        sae = SparseAutoencoder.load_from_pretrained(
            os.path.join(
                args.sae_path,
                f"Mistral-7B-v0.1_blocks.{args.layer}.hook_resid_pre_65536_final.pt",
            )
        ).to(device)
        decoder_vecs = sae.W_dec.data.cpu().numpy()
        np.save(
            os.path.join(args.sae_path, f"sae_layer{args.layer}_decoder.npy"), decoder_vecs
        )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    sparse_activations = np.load(args.activations_file)
    S_ang = get_similarity_matrix(args.layer)
    with open(args.clusters_file, "rb") as f:
        clusters = pickle.load(f)

    # clusters is a list of lists. many of these lists have only one element.
    assert args.cluster < len(clusters), "Cluster index out of bounds"

    cluster = clusters[args.cluster]
    cluster_sae_featureis = cluster

    if len(cluster) > args.size_limit:
        print(
            f"Cluster {args.cluster} has too many features ({len(cluster)}), exiting."
        )
        exit()

    reconstructions, token_indices = get_cluster_activations(
        sparse_activations, set(cluster_sae_featureis), decoder_vecs, args.sample_limit
    )
    token_strs = tokenizer.convert_ids_to_tokens(sparse_activations["all_tokens"])

    contexts = []
    for token_index in token_indices:
        contexts.append(
            token_strs[max(0, token_index - 10) : token_index + 1]
        )  # now using :token_index+1 since the off by one error is fixed in `get_cluster_activations`

    fig = sp.make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "Cosine similarity of decoder features",
            "",
            "SVD values of decoder features",
            "SVD values of mean-centered decoder features",
            *(f"PCA dims {pcai}, {pcai+1}" for pcai in range(min(5, len(cluster)) - 1)),
        ),
        horizontal_spacing=0.05,
    )

    cluster_features = decoder_vecs[cluster]
    cos_sims = cluster_features @ cluster_features.T
    np.fill_diagonal(cos_sims, np.nan)
    fig.add_trace(
        go.Heatmap(
            z=cos_sims,
            colorscale="Viridis",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Cosine similarity", x=0.25, y=0.83, len=0.4),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    u, s, vh = np.linalg.svd(cluster_features.T, full_matrices=False)
    fig.add_trace(
        go.Scatter(x=list(range(len(s))), y=s, mode="lines+markers", showlegend=False),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="Rank", row=1, col=3)
    fig.update_yaxes(title_text="Singular value", row=1, col=3)

    cluster_features_centered = cluster_features.T - cluster_features.T.mean(axis=0)
    u, s, vh = np.linalg.svd(cluster_features_centered, full_matrices=False)
    fig.add_trace(
        go.Scatter(x=list(range(len(s))), y=s, mode="lines+markers", showlegend=False),
        row=1,
        col=4,
    )
    fig.update_xaxes(title_text="Rank", row=1, col=4)
    fig.update_yaxes(title_text="Singular value", row=1, col=4)

    contexts_str = []
    for context in contexts:
        c = ""
        for i, token in enumerate(context):
            if i == len(context) - 1:
                c += "<b>" + token.replace("▁", " ") + "</b>"
            else:
                c += token
        contexts_str.append(c)

    # PCA scatter plots
    pca = PCA(n_components=min(5, len(cluster)))
    reconstructions_pca = pca.fit_transform(reconstructions)
    for pcai in range(min(5, len(cluster)) - 1):
        fig.add_trace(
            go.Scatter(
                x=reconstructions_pca[:, pcai],
                y=reconstructions_pca[:, pcai + 1],
                mode="markers",
                marker=dict(color="darkblue", opacity=0.5),
                text=contexts_str,
                hoverinfo="text",
                hoverlabel=dict(
                    bgcolor="white", font=dict(color="black"), bordercolor="black"
                ),
                showlegend=False,
            ),
            row=2,
            col=pcai + 1,
        )
        fig.update_xaxes(title_text=f"PCA dim {pcai}", row=2, col=pcai + 1)
        fig.update_yaxes(title_text=f"PCA dim {pcai+1}", row=2, col=pcai + 1)
        fig.update_yaxes(title_standoff=0, row=2, col=pcai + 1)

    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"Cluster rank {args.cluster} with {len(cluster)} features.",
        font=dict(size=12),
    )  # Adjust the main title font size here
    #   subplot_titles_font=dict(size=8))  # Adjust the subplot title font size here
    fig.update_annotations(font_size=13)

    fig.write_html(os.path.join(args.save_dir, f"mistral-7b-layer{args.layer}-cluster{args.cluster:04d}.html"))
    fig.write_image(os.path.join(args.save_dir, f"mistral-7b-layer{args.layer}-cluster{args.cluster:04d}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--cluster", type=int, default=61) # 61 is days, 832 is months
    parser.add_argument("--size_limit", type=int, help="Size limit for the cluster", default=1000)
    parser.add_argument("--sample_limit", type=int, help="Max number of reconstructions in plot", default=4_000)
    parser.add_argument("--activations_file", type=str, help="File containing SAE activations (occurence_data)",
                        default="sae_activations_big_layer-8.npz") # NOTE: current naming scheme for activations doesn't distinguish gpt-2 from mistral-7b, so be careful
    parser.add_argument("--sae_path", type=str, default="saes/mistral_saes",
                        help="Path to the directory containing the Mistral-7B SAE weights.")
    parser.add_argument("--save_dir", type=str, default="panes")
    args = parser.parse_args() 

    main(args)
