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
    # cache_dir="/media/MODELS",
)
# %%

standard_texts = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "the weekend"
]

experiments = []

extra_text = ""
flip_axis = True

experiment_1 = []
for day in standard_texts:
    experiment_1.append(f"{extra_text}{day}")
for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    for time in ["morning", "evening"]:
        experiment_1.append(f"{extra_text}{time} on {day}")
experiments.append(experiment_1)

experiment_2 = []
for day in standard_texts:
    experiment_2.append(f"{extra_text}{day}")
for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    for time in ["very early", "very late"]:
        experiment_2.append(f"{extra_text}{time} on {day}")
experiments.append(experiment_2)

for texts, name in zip(experiments, ["experiment_1", "experiment_2"]):

    fig, ax = plt.subplots(1, 1, figsize=(5.5*4/10, 1.8))  # Create a figure with two subplots side by side

    hidden_states = []
    for day in texts:
        prompt = f"{day}"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states.append(
            [outputs.hidden_states[layer][0][-1].cpu().numpy() for layer in range(len(outputs.hidden_states))]
        )

    hidden_states = np.array(hidden_states)

    for i in range(len(texts)):
        texts[i] = texts[i][len(extra_text):]

    special = range(7)
    skip_up_to = 0

    layer = 30  # Assuming we're only interested in layer 30 as in the original code

    layer_hidden_states = hidden_states[:, layer]

    pca = PCA(n_components=5)
    pca.fit(layer_hidden_states[:7])

    print(f"{name} explained variance ratio:", pca.explained_variance_ratio_)

    layer_hidden_states_pca = pca.transform(layer_hidden_states)[:, :2]

    ax.axis('equal')

    percent_explained = pca.explained_variance_ratio_[:2].sum()

    dims_to_plot = [0, 1]

    ax.scatter(layer_hidden_states_pca[skip_up_to:, dims_to_plot[0]], layer_hidden_states_pca[skip_up_to:, dims_to_plot[1]], s=1)
    ax.scatter(layer_hidden_states_pca[special, dims_to_plot[0]], layer_hidden_states_pca[special, dims_to_plot[1]], color='red', s=3)

    text_labels = []
    for i, txt in enumerate(texts):
        if i >= skip_up_to:
            if i in special:
                text = ax.annotate(
                    txt, (layer_hidden_states_pca[i, dims_to_plot[0]], layer_hidden_states_pca[i, dims_to_plot[1]]),
                    color='red',
                    fontsize=7
                )
            else:
                text = ax.annotate(
                    txt, (layer_hidden_states_pca[i, dims_to_plot[0]], layer_hidden_states_pca[i, dims_to_plot[1]]),
                    fontsize=5
                )
            text_labels.append(text)

    def change_position(i, delta_x, delta_y):
        text_labels[i].set_position((text_labels[i].get_position()[0] + delta_x, text_labels[i].get_position()[1] + delta_y))
        # pass

    for i in range(len(texts)):
        change_position(i, 0.25, 0)

    if "very early on Friday" in texts:
        change_position(texts.index("very early on Friday"), -0.3, -0.2)
    if "very late on Friday" in texts:
        change_position(texts.index("very late on Friday"), 0, 0.2)

    # Remove outer axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add center axis
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)

    # ax.set_title(f"Mistral Layer {layer} - {name}")

    if flip_axis:
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"mistral_weekdays_{name}.pdf", bbox_inches='tight')
    plt.show()
    plt.close()

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
    # "January",
    # "February",
    # "March",
    # "April",
    # "May",
    # "June",
    # "July",
    # "August",
    # "September",
    # "October",
    # "November",
    # "December",
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
