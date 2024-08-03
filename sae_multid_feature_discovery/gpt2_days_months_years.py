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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
# import plotly.subplots as sp
# import plotly.graph_objects as go


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

activations_file = "sae_activations_big_layer-7.npz"
cluster_file = "gpt-2_layer_7_clusters_spectral_n1000.pkl"

layer = 7
n_clusters = 1000
clusteri = 138
sample_limit = 20_000

ae = get_gpt2_sae(device="cpu", layer=layer)
decoder_vecs = ae.W_dec.data.cpu().numpy()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

with open(cluster_file, "rb") as f:
    clusters = pickle.load(f)
cluster_days = clusters[clusteri]

sparse_sae_activations = np.load(activations_file)

reconstructions_days, token_indices_days = get_cluster_activations(sparse_sae_activations, set(cluster_days), decoder_vecs)
reconstructions_days, token_indices_days = reconstructions_days[:sample_limit], token_indices_days[:sample_limit]
token_strs_days = tokenizer.batch_decode(sparse_sae_activations['all_tokens'])

contexts_days = []
for token_index_days in token_indices_days:
    contexts_days.append(token_strs_days[max(0, token_index_days-10):token_index_days]) # thought it should be :token_index+1, but seems like there's an off-by-one error in Josh's script, so compensating here.

layer = 7
n_clusters = 1000
clusteri = 251
sample_limit = 20_000

ae = get_gpt2_sae(device="cpu", layer=layer)
decoder_vecs = ae.W_dec.data.cpu().numpy()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

cluster_months = clusters[clusteri]

reconstructions_months, token_indices_months = get_cluster_activations(sparse_sae_activations, set(cluster_months), decoder_vecs)
reconstructions_months, token_indices_months = reconstructions_months[:sample_limit], token_indices_months[:sample_limit]
token_strs_months = tokenizer.batch_decode(sparse_sae_activations['all_tokens'])

contexts_months = []
for token_index_months in token_indices_months:
    contexts_months.append(token_strs_months[max(0, token_index_months-10):token_index_months]) # thought it should be :token_index+1, but seems like there's an off-by-one error in Josh's script, so compensating here.

layer = 7
n_clusters = 1000
clusteri = 212
sample_limit = 20_000

ae = get_gpt2_sae(device="cpu", layer=layer)
decoder_vecs = ae.W_dec.data.cpu().numpy()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

cluster_years = clusters[clusteri]

reconstructions_years, token_indices_years = get_cluster_activations(sparse_sae_activations, set(cluster_years), decoder_vecs)
reconstructions_years, token_indices_years = reconstructions_years[:sample_limit], token_indices_years[:sample_limit]
token_strs_years = tokenizer.batch_decode(sparse_sae_activations['all_tokens'])

contexts_years = []
for token_index_years in token_indices_years:
    contexts_years.append(token_strs_years[max(0, token_index_years-10):token_index_years]) # thought it should be :token_index+1, but seems like there's an off-by-one error in Josh's script, so compensating here.

import matplotlib.colorbar as cbar
days_of_week = {
    "monday": 0,
    "mondays": 0,
    "mon": 0,
    "tuesday": 1,
    "tuesdays": 1,
    "tues": 1,
    "wednesday": 2,
    "wednesdays": 2,
    "wed": 2,
    "thursday": 3,
    "thursdays": 3,
    "thurs": 3,
    "friday": 4,
    "fridays": 4,
    "fri": 4,
    "saturday": 5,
    "saturdays": 5,
    "sat": 5,
    "sunday": 6,
    "sundays": 6,
    "sun": 6
}

fig = plt.figure(figsize=(5.5, 2.0))
ax1 = plt.subplot(1, 3, 1)
# do PCA
pca = PCA(n_components=min(5, len(cluster_days)))
reconstructions_pca = pca.fit_transform(reconstructions_days)
colors = []
# colorwheel = plt.cm.hsv(np.linspace(0, 1-1/7, 7))
colorwheel = plt.cm.tab10(np.linspace(0, 1, 10))
n_greys = 0
for context in contexts_days:
    token = context[-1]
    if token.lower().strip() in days_of_week:
        color = colorwheel[days_of_week[token.lower().strip()]]
    else:
        color = "#BBB"
        n_greys += 1
    colors.append(color)
plt.scatter(reconstructions_pca[:, 1], reconstructions_pca[:, 2], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 2", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 3", fontsize=8, labelpad=-1)
plt.title("Days of the Week", fontsize=8)
print(n_greys / len(contexts_days))

# Create custom legend
legend_elements_1 = [
    Line2D([0], [0], marker='o', color='w', label='Monday', markerfacecolor=colorwheel[0], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Tuesday', markerfacecolor=colorwheel[1], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Wednesday', markerfacecolor=colorwheel[2], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Thursday', markerfacecolor=colorwheel[3], markersize=3)
]

legend_elements_2 = [
    Line2D([0], [0], marker='o', color='w', label='Friday', markerfacecolor=colorwheel[4], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Saturday', markerfacecolor=colorwheel[5], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Sunday', markerfacecolor=colorwheel[6], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='#BBB', markersize=3)
]

legend1 = ax1.legend(handles=legend_elements_1, loc='upper left', fontsize=4, frameon=False, labelspacing=0.2, handletextpad=0.1)
legend2 = ax1.legend(handles=legend_elements_2, loc='upper right', fontsize=4, frameon=False, labelspacing=0.2, handletextpad=0.1)
ax1.add_artist(legend1)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)


ax2 = plt.subplot(1, 3, 2)

months_of_year = {
    "january": 0,
    "jan": 0,
    "february": 1,
    "feb": 1,
    "march": 2,
    "mar": 2,
    "april": 3,
    "apr": 3,
    "may": 4,
    "june": 5,
    "jun": 5,
    "july": 6,
    "jul": 6,
    "august": 7,
    "aug": 7,
    "september": 8,
    "sep": 8,
    "october": 9,
    "oct": 9,
    "november": 10,
    "nov": 10,
    "december": 11,
    "dec": 11
}

pca = PCA(n_components=min(5, len(cluster_months)))
reconstructions_pca = pca.fit_transform(reconstructions_months)
colors = []
colorwheel = plt.cm.rainbow(np.linspace(0, 1-1/12, 12))
# colorwheel = plt.cm.tab20(np.linspace(0, 1, 20))
for context in contexts_months:
    token = context[-1]
    if token.lower().strip() in months_of_year:
        color = colorwheel[months_of_year[token.lower().strip()]]
    else:
        color = "#BBB"
    colors.append(color)
plt.scatter(reconstructions_pca[:, 1], reconstructions_pca[:, 2], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 2", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 3", fontsize=8, labelpad=-2)
plt.title("Months of the Year", fontsize=8)

# Create custom legend
legend_elements_1 = [
    Line2D([0], [0], marker='o', color='w', label='January', markerfacecolor=colorwheel[0], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='February', markerfacecolor=colorwheel[1], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='March', markerfacecolor=colorwheel[2], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='April', markerfacecolor=colorwheel[3], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='May', markerfacecolor=colorwheel[4], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='June', markerfacecolor=colorwheel[5], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='July', markerfacecolor=colorwheel[6], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='August', markerfacecolor=colorwheel[7], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='September', markerfacecolor=colorwheel[8], markersize=3)
]

legend_elements_2 = [
    Line2D([0], [0], marker='o', color='w', label='October', markerfacecolor=colorwheel[9], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='November', markerfacecolor=colorwheel[10], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='December', markerfacecolor=colorwheel[11], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='#BBB', markersize=3)
]

legend1 = ax2.legend(handles=legend_elements_1, loc='upper left', fontsize=4, frameon=False, labelspacing=0.2, handletextpad=0.1)
legend2 = ax2.legend(handles=legend_elements_2, loc='upper right', fontsize=4, frameon=False, labelspacing=0.2, handletextpad=0.1)
ax2.add_artist(legend1)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)



ax3 = plt.subplot(1, 3, 3)

pca = PCA(n_components=min(5, len(cluster_days)))
reconstructions_pca = pca.fit_transform(reconstructions_years)
colors = []
colorwheel = plt.cm.viridis(np.linspace(0, 1, 100))
for context in contexts_years:
    token = context[-1]
    if token.strip().isdigit():
        if 1900 <= int(token) <= 1999:
            color = colorwheel[int(token) % 100]
        else:
            color = "#BBB"
    colors.append(color)
plt.scatter(reconstructions_pca[:, 2], reconstructions_pca[:, 3], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 3", fontsize=7, labelpad=-2)
plt.ylabel("PCA axis 4", fontsize=7, labelpad=-2)
plt.title("Years of the 20th Century", fontsize=8)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

# x_line = 0.9 * ax1.get_position().x1 + 0.1 * ax2.get_position().x0
x_line = ax1.get_position().x1 - 0.02
fig.add_artist(Line2D([x_line, x_line], [0.1, 0.9], transform=fig.transFigure, color='grey', linewidth=0.5))
x_line = ax3.get_position().x0 - 0.01
fig.add_artist(Line2D([x_line, x_line], [0.1, 0.9], transform=fig.transFigure, color='grey', linewidth=0.5))


# Create the colorbar for the third subplot
norm = plt.Normalize(1900, 1999)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

# Add the colorbar to the figure
cbar_ax = fig.add_axes([0.855, 0.85, 0.12, 0.02])  # Adjust the position and size as needed
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=6)
# cbar.set_label('Year', fontsize=7)

# Set custom ticks at 1900 and 1999
cbar.set_ticks([1900, 1950, 1999])
cbar.set_ticklabels(['1900', '1950', '1999'])

# plt.tight_layout(pad=-0.5, w_pad=-0.5, h_pad=-0.5)
# Save figure with tight bounding box
plt.tight_layout(pad=0.7)
plt.savefig("gpt2nonlinears.pdf", bbox_inches='tight')

# Now make a larger plot showing PCA axis 1 & 2, 2 & 3, 3 & 4 where each row is a different representation type
# so it'll be a 3x3 plot

days_of_week = {
    "monday": 0,
    "mondays": 0,
    "mon": 0,
    "tuesday": 1,
    "tuesdays": 1,
    "tues": 1,
    "wednesday": 2,
    "wednesdays": 2,
    "wed": 2,
    "thursday": 3,
    "thursdays": 3,
    "thurs": 3,
    "friday": 4,
    "fridays": 4,
    "fri": 4,
    "saturday": 5,
    "saturdays": 5,
    "sat": 5,
    "sunday": 6,
    "sundays": 6,
    "sun": 6
}

months_of_year = {
    "january": 0,
    "jan": 0,
    "february": 1,
    "feb": 1,
    "march": 2,
    "mar": 2,
    "april": 3,
    "apr": 3,
    "may": 4,
    "june": 5,
    "jun": 5,
    "july": 6,
    "jul": 6,
    "august": 7,
    "aug": 7,
    "september": 8,
    "sep": 8,
    "october": 9,
    "oct": 9,
    "november": 10,
    "nov": 10,
    "december": 11,
    "dec": 11
}

fig = plt.figure(figsize=(5.5, 5.5))

#####################################
#              ROW 1
#####################################

pca = PCA(n_components=min(5, len(cluster_days)))
reconstructions_pca = pca.fit_transform(reconstructions_days)
colors = []
# colorwheel = plt.cm.hsv(np.linspace(0, 1-1/7, 7))
colorwheel = plt.cm.tab10(np.linspace(0, 1, 10))
for context in contexts_days:
    token = context[-1]
    if token.lower().strip() in days_of_week:
        color = colorwheel[days_of_week[token.lower().strip()]]
    else:
        color = "#BBB"
    colors.append(color)

ax11 = plt.subplot(3, 3, 1)

plt.scatter(reconstructions_pca[:, 0], reconstructions_pca[:, 1], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 1", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 2", fontsize=8, labelpad=-1)
plt.title("Days of the Week", fontsize=9)

ax11.spines['top'].set_visible(False)
ax11.spines['right'].set_visible(False)
ax11.spines['bottom'].set_visible(False)
ax11.spines['left'].set_visible(False)


ax12 = plt.subplot(3, 3, 2)
# do PCA

plt.scatter(reconstructions_pca[:, 1], reconstructions_pca[:, 2], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 2", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 3", fontsize=8, labelpad=-1)
# plt.title("Days of the Week", fontsize=9)

# Create custom legend
legend_elements_1 = [
    Line2D([0], [0], marker='o', color='w', label='Monday', markerfacecolor=colorwheel[0], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Tuesday', markerfacecolor=colorwheel[1], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Wednesday', markerfacecolor=colorwheel[2], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Thursday', markerfacecolor=colorwheel[3], markersize=3)
]

legend_elements_2 = [
    Line2D([0], [0], marker='o', color='w', label='Friday', markerfacecolor=colorwheel[4], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Saturday', markerfacecolor=colorwheel[5], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Sunday', markerfacecolor=colorwheel[6], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='#BBB', markersize=3)
]

legend1 = ax12.legend(handles=legend_elements_1, loc='upper left', fontsize=4, frameon=False, labelspacing=0.2, handletextpad=0.1)
legend2 = ax12.legend(handles=legend_elements_2, loc='upper right', fontsize=4, frameon=False, labelspacing=0.2, handletextpad=0.1)
ax12.add_artist(legend1)

ax12.spines['top'].set_visible(False)
ax12.spines['right'].set_visible(False)
ax12.spines['bottom'].set_visible(False)
ax12.spines['left'].set_visible(False)

ax13 = plt.subplot(3, 3, 3)

plt.scatter(reconstructions_pca[:, 2], reconstructions_pca[:, 3], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 3", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 4", fontsize=8, labelpad=-1)
# plt.title("Days of the Week", fontsize=8)

ax13.spines['top'].set_visible(False)
ax13.spines['right'].set_visible(False)
ax13.spines['bottom'].set_visible(False)
ax13.spines['left'].set_visible(False)


#####################################
#              ROW 2
#####################################

pca = PCA(n_components=min(5, len(cluster_months)))
reconstructions_pca = pca.fit_transform(reconstructions_months)
colors = []
colorwheel = plt.cm.rainbow(np.linspace(0, 1-1/12, 12))
for context in contexts_months:
    token = context[-1]
    if token.lower().strip() in months_of_year:
        color = colorwheel[months_of_year[token.lower().strip()]]
    else:
        color = "#BBB"
    colors.append(color)

ax21 = plt.subplot(3, 3, 4)

plt.scatter(reconstructions_pca[:, 0], reconstructions_pca[:, 1], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 1", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 2", fontsize=8, labelpad=-2)
plt.title("Months of the Year", fontsize=9)

ax21.spines['top'].set_visible(False)
ax21.spines['right'].set_visible(False)
ax21.spines['bottom'].set_visible(False)
ax21.spines['left'].set_visible(False)


ax22 = plt.subplot(3, 3, 5)

plt.scatter(reconstructions_pca[:, 1], reconstructions_pca[:, 2], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 2", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 3", fontsize=8, labelpad=-2)
# plt.title("Months of the Year", fontsize=8)


# Create custom legend
legend_elements_1 = [
    Line2D([0], [0], marker='o', color='w', label='January', markerfacecolor=colorwheel[0], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='February', markerfacecolor=colorwheel[1], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='March', markerfacecolor=colorwheel[2], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='April', markerfacecolor=colorwheel[3], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='May', markerfacecolor=colorwheel[4], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='June', markerfacecolor=colorwheel[5], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='July', markerfacecolor=colorwheel[6], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='August', markerfacecolor=colorwheel[7], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='September', markerfacecolor=colorwheel[8], markersize=3)
]

legend_elements_2 = [
    Line2D([0], [0], marker='o', color='w', label='October', markerfacecolor=colorwheel[9], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='November', markerfacecolor=colorwheel[10], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='December', markerfacecolor=colorwheel[11], markersize=3),
    Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='#BBB', markersize=3)
]

legend1 = ax22.legend(handles=legend_elements_1, loc='upper left', fontsize=4, frameon=False, labelspacing=0.2, handletextpad=0.1)
legend2 = ax22.legend(handles=legend_elements_2, loc='upper right', fontsize=4, frameon=False, labelspacing=0.2, handletextpad=0.1)
ax22.add_artist(legend1)

ax22.spines['top'].set_visible(False)
ax22.spines['right'].set_visible(False)
ax22.spines['bottom'].set_visible(False)
ax22.spines['left'].set_visible(False)


ax23 = plt.subplot(3, 3, 6)

plt.scatter(reconstructions_pca[:, 2], reconstructions_pca[:, 3], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 3", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 4", fontsize=8, labelpad=-2)
# plt.title("Months of the Year", fontsize=8)

ax23.spines['top'].set_visible(False)
ax23.spines['right'].set_visible(False)
ax23.spines['bottom'].set_visible(False)
ax23.spines['left'].set_visible(False)

#####################################
#              ROW 3
#####################################


pca = PCA(n_components=min(5, len(cluster_days)))
reconstructions_pca = pca.fit_transform(reconstructions_years)
colors = []
colorwheel = plt.cm.viridis(np.linspace(0, 1, 100))
for context in contexts_years:
    token = context[-1]
    if token.strip().isdigit():
        if 1900 <= int(token) <= 1999:
            color = colorwheel[int(token) % 100]
        else:
            color = "#BBB"
    colors.append(color)

ax31 = plt.subplot(3, 3, 7)

plt.scatter(reconstructions_pca[:, 0], reconstructions_pca[:, 1], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 1", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 2", fontsize=8, labelpad=-1)
plt.title("Years of the 20th Century", fontsize=9)

ax31.spines['top'].set_visible(False)
ax31.spines['right'].set_visible(False)
ax31.spines['bottom'].set_visible(False)
ax31.spines['left'].set_visible(False)

ax32 = plt.subplot(3, 3, 8)

plt.scatter(reconstructions_pca[:, 1], reconstructions_pca[:, 2], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 2", fontsize=8, labelpad=-2)
plt.ylabel("PCA axis 3", fontsize=8, labelpad=-1)
# plt.title("Years of the 20th Century", fontsize=8)

ax32.spines['top'].set_visible(False)
ax32.spines['right'].set_visible(False)
ax32.spines['bottom'].set_visible(False)
ax32.spines['left'].set_visible(False)

ax33 = plt.subplot(3, 3, 9)

plt.scatter(reconstructions_pca[:, 2], reconstructions_pca[:, 3], s=1, color=colors, alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.xlabel("PCA axis 3", fontsize=7, labelpad=-2)
plt.ylabel("PCA axis 4", fontsize=7, labelpad=-2)
# plt.title("Years of the 20th Century", fontsize=8)

ax33.spines['top'].set_visible(False)
ax33.spines['right'].set_visible(False)
ax33.spines['bottom'].set_visible(False)
ax33.spines['left'].set_visible(False)

# x_line = 0.9 * ax1.get_position().x1 + 0.1 * ax2.get_position().x0
# x_line = ax1.get_position().x1 - 0.02
# fig.add_artist(Line2D([x_line, x_line], [0.1, 0.9], transform=fig.transFigure, color='grey', linewidth=0.5))
# x_line = ax33.get_position().x0 - 0.01
# fig.add_artist(Line2D([x_line, x_line], [0.1, 0.9], transform=fig.transFigure, color='grey', linewidth=0.5))

y_line = ax11.get_position().y0 + 0.01
fig.add_artist(Line2D([0.02, 0.98], [y_line, y_line], transform=fig.transFigure, color='grey', linewidth=1.0))
y_line = ax31.get_position().y1
fig.add_artist(Line2D([0.02, 0.98], [y_line, y_line], transform=fig.transFigure, color='grey', linewidth=1.0))

# Create the colorbar for the third subplot
norm = plt.Normalize(1900, 1999)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

# Add the colorbar to the figure
cbar_ax = fig.add_axes([0.855, 0.27, 0.12, 0.007])  # Adjust the position and size as needed
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=6)
# cbar.set_label('Year', fontsize=7)

# Set custom ticks at 1900 and 1999
cbar.set_ticks([1900, 1950, 1999])
cbar.set_ticklabels(['1900', '1950', '1999'])

# plt.tight_layout(pad=-0.5, w_pad=-0.5, h_pad=-0.5)
# Save figure with tight bounding box
plt.tight_layout(pad=0.7, h_pad=1.0)
plt.savefig("gpt2nonlinears3projs.pdf", bbox_inches='tight')


