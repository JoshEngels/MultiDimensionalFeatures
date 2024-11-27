# %%

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np


# %%

clusters = pickle.load(open(f"mistral_layer_8_clusters_cutoff_0.5.pkl", "rb"))
clusters = [cluster for cluster in clusters if len(cluster) > 1]
print(clusters[61])
print(clusters[832])
days_of_week_cluster = clusters[61]
months_cluster = clusters[832]

# %%


bins = np.linspace(0, 100, 100)
means = []
cutoffs = list(np.logspace(start=-1, stop=0, num=20)) + [0.5]
cutoffs.sort()
# for cluster, name in [(days_of_week_cluster, "Days of Week"), (months_cluster, "Months")]:
for cluster, name in [(days_of_week_cluster, "Days of Week"), (months_cluster, "Months")]:
    all_jaccard_sims = []
    for sim_cutoff in cutoffs:
        all_jaccard_sims.append([])
        for top_k_for_graph in [2, 3, 4]:   
            clusters = pickle.load(open(f"mistral-7b_layer_8_clusters_cutoff_{sim_cutoff}_topk_{top_k_for_graph}.pkl", "rb"))
            plt.hist([len(c) for c in clusters], label=f"cutoff={sim_cutoff}, top_k={top_k_for_graph}", alpha=0.5, bins=bins)
            means.append(np.mean([len(c) for c in clusters]))

            # Find size of cluster containing days_of_week_cluster if such a cluster exists, or -1 otherwise
            # Find cluster containing any of the days_of_week_cluster neurons
            found = False
            best_jaccard_sim = 0
            for c in clusters:
                jaccard_sim = len(set(cluster) & set(c)) / len(set(cluster) | set(c))
                if jaccard_sim > best_jaccard_sim:
                    best_jaccard_sim = jaccard_sim
            all_jaccard_sims[-1].append(best_jaccard_sim)
    plt.figure(figsize=(8, 6))
    plt.imshow(all_jaccard_sims, aspect='auto')
    plt.colorbar(label='Max Jaccard Similarity With GT Cluster')
    plt.xlabel('Top-k for Graph (k=2,3,4)')
    plt.ylabel('Similarity Cutoff')
    plt.xticks([0,1,2], ['k=2', 'k=3', 'k=4'])
    plt.yticks(range(len(cutoffs)), [f"{cutoff:.2f}" for cutoff in cutoffs])
    plt.title(f'Cluster Stability Across Parameters for {name} Cluster')
    plt.show()


plt.legend()
plt.show()

# %%

from utils import get_mistral_sae

sae = get_mistral_sae(device="cpu", layer=8)
# %%

sims = sae.W_dec @ sae.W_dec.T

# %%
filter = clusters[61]
print(filter)
filtered_sims = sims[filter, :][:, filter]
# Set diagonal to nan
np.fill_diagonal(filtered_sims.detach().cpu().numpy(), np.nan)
print(filtered_sims)

plt.imshow(filtered_sims.detach().cpu().numpy())
plt.colorbar()
plt.show()
# %%

filter = clusters[832]
print(filter)
filtered_sims = sims[filter, :][:, filter]

# Set diagonal to nan
np.fill_diagonal(filtered_sims.detach().cpu().numpy(), np.nan)
# print(filtered_sims)



plt.imshow(filtered_sims.detach().cpu().numpy())
plt.colorbar()
plt.show()

# %%

# print(sims[20629, 23936])
local_sims = sims[[398, 53172, 31484, 35234, 54166, 52464, 20629, 23936], :][:, [398, 53172, 31484, 35234, 54166, 52464, 20629, 23936]]
np.fill_diagonal(local_sims.detach().cpu().numpy(), np.nan)
plt.imshow(local_sims.detach().cpu().numpy())
plt.colorbar()
plt.show()
# %%

print(local_sims[5].detach().cpu().numpy())