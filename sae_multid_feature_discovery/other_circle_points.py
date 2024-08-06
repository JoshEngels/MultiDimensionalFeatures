# %%

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

# %%


tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1"
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    cache_dir="/media/MODELS",
)

# %%

hidden_states = []
texts = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "the weekend",
    "very late on Monday",
    "very late on Tuesday",
    "very late on Wednesday",
    "very late on Thursday",
    "very late on Friday",
    "very late on Saturday",
    "very late on Sunday",
    "very early on Monday",
    "very early on Tuesday",
    "very early on Wednesday",
    "very early on Thursday",
    "very early on Friday",
    "very early on Saturday",
    "very early on Sunday",
]

for day in texts:
    prompt = f"{day}"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model(**inputs, output_hidden_states=True)

    hidden_states.append(
        [outputs.hidden_states[layer][0][-1].cpu().numpy() for layer in range(len(outputs.hidden_states))]
    )


# %%

hidden_states = np.array(hidden_states)

skip_up_to = 0

# for layer in range(81):
# for layer in range(62, 63):
# for layer in range(8, 9):
for layer in range(30, 31):

    fig, ax1 = plt.subplots()

    layer_hidden_states = hidden_states[:, layer]

    pca = PCA(n_components=5)
    pca.fit(layer_hidden_states[:7])

    layer_hidden_states_pca = pca.transform(layer_hidden_states)

    percent_explained = pca.explained_variance_ratio_[:2].sum()

    dims_to_plot = [0, 1]

    ax1.scatter(layer_hidden_states_pca[skip_up_to:, dims_to_plot[0]], layer_hidden_states_pca[skip_up_to:, dims_to_plot[1]])
    text_labels = []
    for i, txt in enumerate(texts):
        if i >= skip_up_to:
            text = ax1.annotate(
                txt, (layer_hidden_states_pca[i, dims_to_plot[0]], layer_hidden_states_pca[i, dims_to_plot[1]])
            )
            text_labels.append(text)

    def change_position(i, delta_x, delta_y):
        text_labels[i].set_position((text_labels[i].get_position()[0] + delta_x, text_labels[i].get_position()[1] + delta_y))

    for i in range(len(texts)):
        change_position(i, 0.25, 0)


    # Thursday
    change_position(3, -1.7, 0.5)

    # very early on Sunday
    change_position(-1, -0.2, -.7)

    # Wednesday
    change_position(2, 0, -0.5)

    change_position(8, 0, -0.6)

    # very early on Tuesday
    change_position(16, -7, -0.5)

    change_position(15, 0, 0.3)

    # Very late on wednesday
    change_position(10, -1.9, -0.8)

    # Very late on friday
    change_position(12, 0, -0.5)

    # Very early on Friday
    change_position(19, -6, -0.7)

    # Remove outer axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Add center axis
    ax1.axhline(0, color='black', lw=0.5)
    ax1.axvline(0, color='black', lw=0.5)

    # Make plot square
    ax1.set_aspect('equal', adjustable='box')

    plt.title(f"Mistral Layer {layer} Projected into Weekday Plane")

    plt.show()

# %%


hidden_states = []
texts = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Fall",
    "Winter",
    "Spring",
    "Summer"
    # "mid January",
    # "mid February",
    # "mid March",
    # "mid April",
    # "mid May",
    # "mid June",
    # "mid July",
    # "mid August",
    # "mid September",
    # "mid October",
    # "mid November",
    # "mid December",
    # "late January",
    # "late February",
    # "late March",
    # "late April",
    # "late May",
    # "late June",
    # "late July",
    # "late August",
    # "late September",
    # "late October",
    # "late November",
    # "late December",
    # "early January",
    # "early February",
    # "early March",
    # "early April",
    # "early May",
    # "early June",
    # "early July",
    # "early August",
    # "early September",
    # "early October",
    # "early November",
    # "early December",
]

for month in texts:
    prompt = f"Approximately two months from {month}"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model(**inputs, output_hidden_states=True)

    hidden_states.append(
        [outputs.hidden_states[layer][0][-1].cpu().numpy() for layer in range(len(outputs.hidden_states))]
    )


# %%

hidden_states = np.array(hidden_states)

# for layer in range(81):
for layer in range(12, 13):

    layer_hidden_states = hidden_states[:, layer]

    pca = PCA(n_components=2)
    pca.fit(layer_hidden_states[:12])

    layer_hidden_states_pca = pca.transform(layer_hidden_states)

    percent_explained = pca.explained_variance_ratio_[:2].sum()

    plt.scatter(layer_hidden_states_pca[:, 0], layer_hidden_states_pca[:, 1])
    for i, txt in enumerate(texts):
        plt.annotate(
            txt, (layer_hidden_states_pca[i, 0], layer_hidden_states_pca[i, 1])
        )

    plt.title(f"Layer {layer} ({percent_explained:.2f})")

    plt.show()

# %%
