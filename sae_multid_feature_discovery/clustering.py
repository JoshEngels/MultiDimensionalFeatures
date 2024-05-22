"""
Saves a list of list of indices of SAE features as a pickle file.
"""

from utils import get_sae, BASE_DIR
import torch
from tqdm import tqdm
import pickle
import argparse
from sklearn.cluster import SpectralClustering

torch.set_grad_enabled(False)

device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="mistral")
parser.add_argument("--layer", type=int, default=8)
parser.add_argument(
    "--method", type=str, choices=["graph", "spectral"], default="graph"
)
parser.add_argument("--include_only_first_k_sae_features", type=int)
args = parser.parse_args()


model_name = args.model_name
layer = args.layer
method = args.method

if model_name == "mistral":
    model_name = "mistral-7b"


sae = get_sae(device=device, model_name=model_name, layer=layer)
all_sae_features = sae.W_dec

if args.include_only_first_k_sae_features:
    all_sae_features = all_sae_features[: args.include_only_first_k_sae_features]


def spectral_cluster_sims(all_sims, n_clusters=1000):
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
    labels = sc.fit_predict(all_sims).tolist()
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(i)
    pickle.dump(
        clusters,
        open(f"{model_name}_layer_{layer}_clusters_spectral_n{n_clusters}.pkl", "wb"),
    )


def graph_cluster_sims(
    all_sims, top_k_for_graph=2, sim_cutoff=0.5, prune_clusters=True
):
    near_neighbors = torch.topk(all_sims, top_k_for_graph, dim=1)

    graph = [[] for _ in range(all_sims.shape[0])]
    sim_cutoff = 0.5
    for i in tqdm(range(all_sims.shape[0])):
        top_indices = near_neighbors.indices[i]
        top_sims = near_neighbors.values[i]
        top_indices = top_indices[top_sims > sim_cutoff]
        graph[i] = top_indices.tolist()

    # Add back edges
    for i in tqdm(range(all_sims.shape[0])):
        for j in graph[i]:
            if i not in graph[j]:
                graph[j].append(i)

    # Find connected components
    visited = [False] * all_sims.shape[0]
    components = []
    for i in range(all_sims.shape[0]):
        if visited[i]:
            continue
        component = []
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            stack.extend(graph[node])
        components.append(component)

    if prune_clusters:
        threshold = 1000
        components = [c for c in components if len(c) < threshold and len(c) > 1]

    with open(
        f"{model_name}_layer_{layer}_clusters_cutoff_{sim_cutoff}.pkl", "wb"
    ) as f:
        pickle.dump(components, f)


all_sims = all_sae_features @ all_sae_features.T

if method == "graph":
    graph_cluster_sims(all_sims)

else:
    all_sims = torch.clamp(all_sims, -1, 1)
    all_sims = 1 - torch.arccos(all_sims) / torch.pi
    all_sims = all_sims.detach().cpu().numpy()
    spectral_cluster_sims(all_sims)
