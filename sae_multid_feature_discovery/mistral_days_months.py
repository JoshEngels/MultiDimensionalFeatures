import os
import argparse
from collections import defaultdict
from itertools import islice

import pickle
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA

import torch as t
from transformers import AutoTokenizer

from saes.sparse_autoencoder import SparseAutoencoder
import dill as pickle

def get_cluster_activations(
    sparse_sae_activations,
    sae_neurons_in_cluster,
    decoder_vecs,
    sample_limit,
    max_indices=1e8,
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
        disable=False,
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

def main(clusters_file):

    # ------------------------------
    # Load the SAE, clusters, and tokenizer
    # ------------------------------

    from utils import get_mistral_sae

    sae = get_mistral_sae(device="cpu", layer=8)
    decoder_vecs = sae.W_dec.detach().cpu().numpy()
    # np.save("sae_layer8_decoder.npy", sae.W_dec.detach().cpu().numpy())
    # decoder_vecs = np.load("sae_layer8_decoder.npy")
    
    with open(clusters_file, "rb") as f:
        clusters = pickle.load(f)
    # clusters is a list of lists. some of these lists have only one element.
    clusters = [cluster for cluster in clusters if len(cluster) > 1]

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token


    # ------------------------------
    # Load the activations
    # ------------------------------

    sparse_activations = np.load(f"sae_activations_big_layer-8.npz")
    token_strs = tokenizer.convert_ids_to_tokens(sparse_activations["all_tokens"])

    # ------------------------------
    # Compute and plot the features
    # ------------------------------

    # Days of the week
    cluster = clusters[61]
    reconstructions_days, token_indices_days = get_cluster_activations(
        sparse_activations, set(cluster), decoder_vecs, sample_limit=4_000, max_indices=1e9
    )
    print("Days of the week:", cluster)

    # ## Months of the year
    cluster = clusters[832]
    reconstructions_months, token_indices_months = get_cluster_activations(
        sparse_activations, set(cluster), decoder_vecs, sample_limit=4_000, max_indices=1e9
    )
    print("Months of the year:", cluster)

    fig = plt.figure(figsize=(5.5, 2.6))

    days_of_week = {
        "monday": 0,
        "mondays": 0,
        "mon": 0,
        "tuesday": 1,
        "uesday": 1,
        "ue": 1,
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
        "aturday": 5,
        "saturdays": 5,
        "sat": 5,
        "sunday": 6,
        "sundays": 6,
        "sun": 6,
        "weekend": 7,
        "end": 7,
        "ends": 7,
        "weekends": 7,
    }

    ax1 = plt.subplot(1, 2, 1)
    # do PCA
    pca = PCA(n_components=min(5, len(clusters[61])))
    fit_pca = pca.fit(reconstructions_days)
    reconstructions_pca = fit_pca.transform(reconstructions_days)

    # Save fit_pca to a file using pickle
    with open("fit_pca_days.pkl", "wb") as f:
        pickle.dump(fit_pca, f)

    colors = []
    # colorwheel = plt.cm.hsv(np.linspace(0, 1-1/7, 7))
    colorwheel = plt.cm.tab10(np.linspace(0, 1, 10))
    colorwheel[7][0:3] = 0.0  # set weekend to black
    colorwheel[7][3] = 1.0
    n_greys = 0
    for tokeni in token_indices_days:
        token = token_strs[tokeni].replace("▁", "").replace("▁", "").lower().strip()
        if token in days_of_week:
            color = colorwheel[days_of_week[token]]
        else:
            color = "#BBB"
            n_greys += 1
        colors.append(color)
    plt.scatter(
        reconstructions_pca[:, 1], reconstructions_pca[:, 2], s=3, color=colors, alpha=0.6
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("PCA axis 2", fontsize=8, labelpad=-2)
    plt.ylabel("PCA axis 3", fontsize=8, labelpad=-1)
    plt.title("Days of the Week (Mistral 7B)", fontsize=8)
    print(n_greys / len(token_indices_days))

    # Create custom legend
    legend_elements_1 = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Monday",
            markerfacecolor=colorwheel[0],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Tuesday",
            markerfacecolor=colorwheel[1],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Wednesday",
            markerfacecolor=colorwheel[2],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Thursday",
            markerfacecolor=colorwheel[3],
            markersize=4,
        ),
    ]

    legend_elements_2 = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Friday",
            markerfacecolor=colorwheel[4],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Saturday",
            markerfacecolor=colorwheel[5],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Sunday",
            markerfacecolor=colorwheel[6],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="weekend",
            markerfacecolor="black",
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Other",
            markerfacecolor="#BBB",
            markersize=4,
        ),
    ]

    legend1 = ax1.legend(
        handles=legend_elements_1,
        loc="upper left",
        fontsize=5,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.1,
    )
    legend2 = ax1.legend(
        handles=legend_elements_2,
        loc="upper right",
        fontsize=5,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.1,
    )
    ax1.add_artist(legend1)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    ax2 = plt.subplot(1, 2, 2)


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
        "dec": 11,
        "winter": 12,
        "spring": 13,
        "summer": 14,
        "fall": 15,
        "autumn": 15,
    }

    pca = PCA(n_components=min(5, len(clusters[832])))
    reconstructions_pca = pca.fit_transform(reconstructions_months)
    colors = []
    colorwheel = np.concatenate(
        [
            plt.cm.rainbow(np.linspace(0, 1 - 1 / 12, 12)),
            plt.cm.winter([0]),
            plt.cm.spring([0]),
            plt.cm.summer([0]),
            plt.cm.autumn([0]),
        ]
    )

    # colorwheel = plt.cm.tab20(np.linspace(0, 1, 20))
    # for tokeni in token_indices_months:
    #     token = token_strs[tokeni].replace("▁", "").replace("▁", "").lower().strip()
    #     if token.lower().strip() in months_of_year:
    #         color = colorwheel[months_of_year[token.lower().strip()]]
    #     else:
    #         color = "#BBB"
    #     colors.append(color)
    # plt.scatter(reconstructions_pca[:, 1], reconstructions_pca[:, 2], s=1, color=colors, alpha=0.6)

    season_markers = ["winter", "spring", "summer", "fall", "autumn"]
    for i, tokeni in enumerate(token_indices_months):
        token = token_strs[tokeni].replace("▁", "").replace("▁", "").lower().strip()
        if token.lower().strip() in months_of_year:
            color = colorwheel[months_of_year[token.lower().strip()]]
            if token in season_markers:
                marker = "^"  # triangle marker for seasons
            else:
                marker = "o"  # default marker for months
        else:
            color = "#BBB"
            marker = "o"
        plt.scatter(
            reconstructions_pca[i, 1],
            reconstructions_pca[i, 2],
            s=3,
            color=color,
            alpha=0.6,
            marker=marker,
        )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("PCA axis 2", fontsize=8, labelpad=-2)
    plt.ylabel("PCA axis 3", fontsize=8, labelpad=-2)
    plt.title("Months of the Year (Mistral 7B)", fontsize=8)

    # Create custom legend
    legend_elements_1 = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="January",
            markerfacecolor=colorwheel[0],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="February",
            markerfacecolor=colorwheel[1],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="March",
            markerfacecolor=colorwheel[2],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="April",
            markerfacecolor=colorwheel[3],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="May",
            markerfacecolor=colorwheel[4],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="June",
            markerfacecolor=colorwheel[5],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="July",
            markerfacecolor=colorwheel[6],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="August",
            markerfacecolor=colorwheel[7],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="September",
            markerfacecolor=colorwheel[8],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="October",
            markerfacecolor=colorwheel[9],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="November",
            markerfacecolor=colorwheel[10],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="December",
            markerfacecolor=colorwheel[11],
            markersize=4,
        ),
    ]

    legend_elements_2 = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Winter",
            markerfacecolor=colorwheel[12],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Spring",
            markerfacecolor=colorwheel[13],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Summer",
            markerfacecolor=colorwheel[14],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Fall",
            markerfacecolor=colorwheel[15],
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Other",
            markerfacecolor="#BBB",
            markersize=4,
        ),
    ]

    legend1 = ax2.legend(
        handles=legend_elements_1,
        loc="upper left",
        fontsize=5,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.1,
    )
    legend2 = ax2.legend(
        handles=legend_elements_2,
        loc="upper right",
        fontsize=5,
        frameon=False,
        labelspacing=0.2,
        handletextpad=0.1,
    )
    ax2.add_artist(legend1)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    x_line = ax1.get_position().x1 + 0.02
    fig.add_artist(
        Line2D(
            [x_line, x_line],
            [0.05, 0.95],
            transform=fig.transFigure,
            color="grey",
            linewidth=0.5,
        )
    )

    plt.tight_layout(pad=0.7)
    plt.savefig("mistral7bnonlinears.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters_file", type=str, default="mistral_layer_8_clusters_cutoff_0.5.pkl",
        help="Path to the file containing the clusters of Mistral-7B SAE features.")
    args = parser.parse_args()

    main(args.clusters_file)
