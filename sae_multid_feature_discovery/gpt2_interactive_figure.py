"""
For a given layer and cluster, creates an interactive plotly plot
of the cosine simlarities between the SAE decoder features in the 
cluster and also PCA projections of the reconstructed
activations with just the features in the cluster being allowed
to fire.
"""

import os
import time
import pickle
import argparse

# hopefully this will help with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ["TRANSFORMERS_CACHE"] = "/om/user/ericjm/.cache/"

import einops
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sae_lens import SAE
# import transformer_lens
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

def get_cluster_activations(sparse_sae_activations, sae_neurons_in_cluster, decoder_vecs):
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
        if sae_index in sae_neurons_in_cluster:
            updated = True
            current_activations += sae_value * decoder_vecs[sae_index]

    return np.stack(all_activations), all_token_indices

def main(args):
    """
    args (argparse.Namespace): Command line arguments
    """

    # SAE_CLUSTER_DIR = "/om/user/ericjm/results/saes/reconstructions-gpt2-1/clusters/"
    # SAE_HIDDENS_DIR = "/om/user/ericjm/results/saes/reconstructions-gpt2-2/hiddens/"
    # FIGURE_SAVE_DIR = os.path.join(
    #     "/om/user/ericjm/results/saes/reconstructions-gpt2-2/cluster_figures/",
    #     f"layer{args.layer}",
    #     f"nclusters{args.n_clusters:04d}",
    # )
    # FIGURE_SAVE_DIR_IMGS = os.path.join(
    #     "/om/user/ericjm/results/saes/reconstructions-gpt2-2/cluster_figures_imgs/",
    #     f"layer{args.layer}",
    #     f"nclusters{args.n_clusters:04d}",
    # )
    # os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)
    # os.makedirs(FIGURE_SAVE_DIR_IMGS, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    ae = get_gpt2_sae(device="cpu", layer=args.layer)
    decoder_vecs = ae.W_dec.data.cpu().numpy()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # load up clusters
    with open(args.clusters_file, "rb") as f:
        clusters = pickle.load(f)
    cluster = clusters[args.cluster]

    # with open(os.path.join(SAE_CLUSTER_DIR, f"clusters-layer{args.layer}nclusters{args.n_clusters}.pkl"), "rb") as f:
    #     clusters = pickle.load(f)
    # cluster = clusters[args.cluster]

    if len(cluster) > args.size_limit:
        print(f"Cluster {args.cluster} has too many features ({len(cluster)}), exiting.")
        exit()

    sparse_sae_activations = np.load(args.activations_file)

    reconstructions, token_indices = get_cluster_activations(sparse_sae_activations, set(cluster), decoder_vecs)
    reconstructions, token_indices = reconstructions[:args.sample_limit], token_indices[:args.sample_limit]
    token_strs = tokenizer.batch_decode(sparse_sae_activations['all_tokens'])

    contexts = []
    for token_index in token_indices:
        contexts.append(token_strs[max(0, token_index-10):token_index+3]) # thought it should be :token_index+1, but seems like there's an off-by-one error in Josh's script, so compensating here.

    fig = sp.make_subplots(rows=2, cols=4, subplot_titles=("Cosine similarity of decoder features",
                                                            "",
                                                            "SVD values of decoder features",
                                                            "SVD values of mean-centered decoder features",
                                                            *(f"PCA dims {pcai}, {pcai+1}" for pcai in range(min(5, len(cluster))-1))),
                                            horizontal_spacing=0.05)

    cluster_features = decoder_vecs[cluster]
    cos_sims = cluster_features @ cluster_features.T
    np.fill_diagonal(cos_sims, np.nan)
    fig.add_trace(go.Heatmap(z=cos_sims, colorscale='Viridis', zmin=-1, zmax=1,
                            colorbar=dict(title="Cosine similarity", x=0.25, y=0.83, len=0.4),
                            showlegend=False), row=1, col=1)

    u, s, vh = np.linalg.svd(cluster_features.T, full_matrices=False)
    fig.add_trace(go.Scatter(x=list(range(len(s))), y=s, mode='lines+markers', showlegend=False), row=1, col=3)
    fig.update_xaxes(title_text="Rank", row=1, col=3)
    fig.update_yaxes(title_text="Singular value", row=1, col=3)

    cluster_features_centered = cluster_features.T - cluster_features.T.mean(axis=0)
    u, s, vh = np.linalg.svd(cluster_features_centered, full_matrices=False)
    fig.add_trace(go.Scatter(x=list(range(len(s))), y=s, mode='lines+markers', showlegend=False), row=1, col=4)
    fig.update_xaxes(title_text="Rank", row=1, col=4)
    fig.update_yaxes(title_text="Singular value", row=1, col=4)

    contexts_str = []
    for context in contexts:
        c = ""
        for i, token in enumerate(context):
            if i == len(context)-4:
                c += "<b>" + token + "</b>"
            else:
                c += token
        contexts_str.append(c)

    # PCA scatter plots
    pca = PCA(n_components=min(5, len(cluster)))
    reconstructions_pca = pca.fit_transform(reconstructions)
    for pcai in range(min(5, len(cluster))-1):
        fig.add_trace(go.Scatter(x=reconstructions_pca[:, pcai], y=reconstructions_pca[:, pcai+1],
                                mode='markers', marker=dict(color='darkblue', opacity=0.5), text=contexts_str,
                                hoverinfo='text',
                                hoverlabel=dict(bgcolor='white', font=dict(color='black'), bordercolor='black'),
                                showlegend=False),
                    row=2, col=pcai+1)
        fig.update_xaxes(title_text=f"PCA dim {pcai}", row=2, col=pcai+1)
        fig.update_yaxes(title_text=f"PCA dim {pcai+1}", row=2, col=pcai+1)
        fig.update_yaxes(title_standoff=0, row=2, col=pcai+1)

    fig.update_layout(height=800, width=1200, title_text=f"Cluster rank {args.cluster} with {len(cluster)} features",
                    font=dict(size=12))  # Adjust the main title font size here
                    #   subplot_titles_font=dict(size=8))  # Adjust the subplot title font size here
    fig.update_annotations(font_size=13)

    fig.write_html(os.path.join(args.save_dir, f"gpt2-layer{args.layer}-cluster{args.cluster}.html"))
    fig.write_image(os.path.join(args.save_dir, f"gpt2-layer{args.layer}-cluster{args.cluster}.png"))
        
    pickle.dump((reconstructions_pca, contexts, pca.explained_variance_ratio_), open(f"data/gpt2-layer{args.layer}-cluster{args.cluster}.pkl", "wb"))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create cluster figure")
    parser.add_argument("--layer", type=int, help="Layer of GPT-2", default=7)
    parser.add_argument("--clusters_file", type=str, help="File containing clusters", default="gpt-2_layer_7_clusters_spectral_n1000.pkl")
    parser.add_argument("--cluster", type=int, help="Cluster index to create plot of", default=138) # 138 is days, 251 is months
    parser.add_argument("--activations_file", type=str, help="File containing SAE activations (occurence_data)",
                        default="sae_activations_big_layer-7.npz")
    parser.add_argument("--size_limit", type=int, help="Size limit for the cluster", default=1000)
    parser.add_argument("--sample_limit", type=int, help="Max number of reconstructions in plot", default=20_000)
    parser.add_argument("--save_dir", type=str, help="Directory to save figures", default="panes")
    args = parser.parse_args()

    main(args)
