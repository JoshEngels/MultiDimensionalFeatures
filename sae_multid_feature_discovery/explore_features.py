#%%

import pickle
import matplotlib.pyplot as plt
import numpy as np

plot_is = [23, 28, 58, 32, 34, 93, 46, 13, 138, 96, 65, 2, 68, 7, 212, 157, 8, 101, 173, 209]

cluster_descriptions = [
    "Letter 'B'",
    "Letter 'M'",
    "Substring 'Re'",
    "Letter 'G'",
    "Letter 'H'",
    "Letter 'T'",
    "Letter 'D'",
    "Substring 'Al'",
    "Days of Week",
    "Token 'Such'",
    "Letter 'N'",
    "Token/Concept 'There'",
    "Token/Concept 'Trying'",
    "Concept 'Formerly'",
    "Years of 20th Century",
    "Years of 21st Century",
    "Substring 'Un'",
    "Letter 'A'",
    "Word 'So'",
    "Concept 'Person'"
]

# Create a large figure with 5x2 grid for first 10 clusters
fig_all = plt.figure(figsize=(30, 50)) 
gs = fig_all.add_gridspec(5, 2, hspace=0.4, wspace=0.8)

current_idx = 0
for rank, (plot_i, title) in enumerate(zip(plot_is, cluster_descriptions)):

    if current_idx >= 10:
        break

    row = current_idx // 2
    col = current_idx % 2
    
    reconstructions_pca, contexts, explained_variance_ratios = pickle.load(open(f"data/gpt2-layer7-cluster{plot_i}.pkl", "rb"))

    if len(explained_variance_ratios) < 4:
        continue

    current_idx += 1

    tokens = [context[-4] for context in contexts]
    next_tokens = [context[-3] for context in contexts]
    prior_tokens = [context[-5] for context in contexts]

    start_dim = 1
    def plot_token_scatter(ax, tokens, reconstructions_pca, start_dim, explained_variance_ratios, title_prefix, top_k=10, show_legend=False):
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
            
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        unique_tokens = [t[0] for t in top_tokens]

        colors = plt.cm.tab10(np.arange(len(unique_tokens)))
        token_to_color = dict(zip(unique_tokens, colors))

        for token in unique_tokens:
            mask = [t == token for t in tokens]
            ax.scatter(reconstructions_pca[mask, start_dim], 
                    reconstructions_pca[mask, start_dim+1],
                    s=1, 
                    c=[token_to_color[token]], 
                    label=token)

        other_mask = [t not in unique_tokens for t in tokens]
        ax.scatter(reconstructions_pca[other_mask, start_dim],
                reconstructions_pca[other_mask, start_dim+1],
                s=1,
                c='lightgrey',
                label='other')

        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, markerscale=10)
        
        ax.set_xlabel(f"PCA dim {start_dim + 1} ({explained_variance_ratios[start_dim]:.1%} var)", fontsize=15)
        ax.set_ylabel(f"PCA dim {start_dim+2} ({explained_variance_ratios[start_dim+1]:.1%} var)", fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])

    # Create subplot for this cluster
    subfig = fig_all.add_subfigure(gs[row, col])
    subfig.suptitle(f'Cluster {plot_i} (Rank {rank+1}), Rough Description: {title}', fontsize=25)
    
    # Create 2x3 grid within each subfigure with more vertical space between rows
    num_cols = min(3, len(explained_variance_ratios) - 1)
    axs = subfig.subplots(2, num_cols, gridspec_kw={'hspace': 0.4})

    # Add row titles with adjusted vertical positions
    subfig.text(0.5, 0.9, "Current token (top 10 tokens colored)", ha='center', va='center', fontsize=20)
    subfig.text(0.5, 0.46, "Next token (top 10 tokens colored)", ha='center', va='center', fontsize=20)

    # Plot for current tokens
    for i, start_d in enumerate(range(num_cols)):
        plot_token_scatter(axs[0,i], tokens, reconstructions_pca, start_d, explained_variance_ratios, "Current", 
                         show_legend=(i==num_cols-1))

    # Add horizontal line between rows, adjusted position
    subfig_supxaxis = subfig.add_subplot(111, frame_on=False)
    subfig_supxaxis.axhline(y=0.5, color='black', linestyle='-', linewidth=3, xmin=-0.2, xmax=1.2)
    subfig_supxaxis.set_xticks([])
    subfig_supxaxis.set_yticks([])

    # Plot for next tokens  
    for i, start_d in enumerate(range(num_cols)):
        plot_token_scatter(axs[1,i], next_tokens, reconstructions_pca, start_d, explained_variance_ratios, "Next",
                         show_legend=(i==num_cols-1))

plt.savefig("plots/all_clusters_grid.png", dpi=50, bbox_inches='tight')
plt.close()
# %%
