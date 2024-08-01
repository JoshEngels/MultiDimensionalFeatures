# %%

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

# %%

# tokenizer = AutoTokenizer.from_pretrained(
#     "NousResearch/Meta-Llama-3.1-8B", cache_dir="/media/MODELS"
# )

# model = AutoModelForCausalLM.from_pretrained(
#     "NousResearch/Meta-Llama-3.1-8B",
#     device_map="auto",
#     cache_dir="/media/MODELS",
# )


# tokenizer = AutoTokenizer.from_pretrained(
#     "NousResearch/Meta-Llama-3.1-70B", cache_dir="/media/MODELS"
# )

# model = AutoModelForCausalLM.from_pretrained(
#     "NousResearch/Meta-Llama-3.1-70B",
#     device_map="auto",
#     load_in_8bit = True,
#     cache_dir="/media/MODELS",
# )

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
    # "halfway between Monday and Tuesday",
    # "halfway between Tuesday and Wednesday",
    # "halfway between Wednesday and Thursday",
    # "halfway between Thursday and Friday",
    # "halfway between Friday and Saturday",
    # "halfway between Saturday and Sunday",
    # "halfway between Sunday and Monday",
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
for layer in range(31, 32):

    layer_hidden_states = hidden_states[:, layer]

    pca = PCA(n_components=5)
    pca.fit(layer_hidden_states[:7])

    layer_hidden_states_pca = pca.transform(layer_hidden_states)

    percent_explained = pca.explained_variance_ratio_[:2].sum()

    dims_to_plot = [0, 1]

    plt.scatter(layer_hidden_states_pca[skip_up_to:, dims_to_plot[0]], layer_hidden_states_pca[skip_up_to:, dims_to_plot[1]])
    for i, txt in enumerate(texts):
        if i >= skip_up_to:
            plt.annotate(
                txt, (layer_hidden_states_pca[i, dims_to_plot[0]], layer_hidden_states_pca[i, dims_to_plot[1]])
            )

    plt.title(f"Layer {layer} ({percent_explained:.2f})")

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
    "late January",
    "late February",
    "late March",
    "late April",
    "late May",
    "late June",
    "late July",
    "late August",
    "late September",
    "late October",
    "late November",
    "late December",
    "early January",
    "early February",
    "early March",
    "early April",
    "early May",
    "early June",
    "early July",
    "early August",
    "early September",
    "early October",
    "early November",
    "early December",
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
for layer in range(3, 4):

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
