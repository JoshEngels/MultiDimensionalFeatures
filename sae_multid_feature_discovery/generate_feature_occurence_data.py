# %%

import os
from utils import BASE_DIR


# hopefully this will help with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TRANSFORMERS_CACHE"] = f"{BASE_DIR}.cache/"

import einops
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sae_lens import SAE
import torch as t
import transformer_lens
from datasets import load_dataset
from utils import get_sae
import argparse

device = "cuda:0" if t.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default="mistral", choices=["mistral", "gpt-2"]
)
args = parser.parse_args()

if args.model_name == "mistral":
    # Mistral-7B hyperparameters
    model_name = "mistral-7b"
    batch_size = 16
    layers_to_evaluate = [8, 16, 24]
    num_devices = 2
    sae_hidden_size = 65536

else:
    # GPT hyperparameters
    model_name = "gpt-2"
    layers_to_evaluate = range(12)
    batch_size = 64
    num_devices = 1
    num_workers = 8
    sae_hidden_size = 24576

model = transformer_lens.HookedTransformer.from_pretrained(
    model_name, device=device, n_devices=num_devices
)

ctx_len = 256

num_sae_activations_to_save = 10**9

save_folder = f"{BASE_DIR}{model_name}"
os.makedirs(save_folder, exist_ok=True)

t.set_grad_enabled(False)

# %%


def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()

def get_gpt2_sae(device, layer):
    return SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=f"blocks.{layer}.hook_resid_pre",  # won't always be a hook point
        device=device
    )[0]

aes = [
    get_sae(device=device, model_name=model_name, layer=layer)
    for layer in layers_to_evaluate
]

# %%


def save_coocurring_sae_features(model, ae, output_file, data):
    def next_batch_activations():
        tokenized_batch = []
        while len(tokenized_batch) < batch_size:
            batch = next(data)
            tokens = model.to_tokens(batch)[0, :ctx_len]
            if len(tokens) < ctx_len:
                continue
            tokenized_batch.append(tokens)

        tokenized_batch = t.stack(tokenized_batch).to(device)

        layer_name = f"blocks.{layer}.hook_resid_pre"
        result = model.run_with_cache(tokenized_batch, names_filter=layer_name)
        activations = result[1][layer_name]
        activations = einops.rearrange(activations, "b ctx_len d -> (b ctx_len) d")
        return activations, tokenized_batch.flatten()

    sparse_sae_values = np.zeros(num_sae_activations_to_save, dtype=float)  # Length n
    sparse_sae_indices = np.zeros(
        num_sae_activations_to_save, dtype=np.int32
    )  # Length n
    sparse_sae_original_space_projections = np.zeros(
        num_sae_activations_to_save, dtype=np.int32
    )  # Length n
    all_token_indices = np.zeros(
        num_sae_activations_to_save, dtype=np.int32
    )  # Length n, indices into all_tokens
    all_tokens = []  # Length ctx_len * num_batches * batch_size - padding, tokens
    all_activation_norms = []  # Same length as all_tokens, hidden activation norms

    # Force memory allocation
    sparse_sae_values.fill(0)
    sparse_sae_indices.fill(0)
    sparse_sae_original_space_projections.fill(0)
    all_token_indices.fill(0)

    total_pairs_saved = 0
    current_token_offset = 0
    pbar = tqdm(total=num_sae_activations_to_save, position=0, leave=True)
    while total_pairs_saved < num_sae_activations_to_save:
        activations, tokens = next_batch_activations()

        activation_norms = (activations.shape[-1] ** 0.5) / activations.norm(dim=-1)

        assert len(tokens) == ctx_len * batch_size
        assert activation_norms.shape == tokens.shape

        all_tokens.append(np.array(tokens.cpu()))
        all_activation_norms.append(np.array(activation_norms.cpu()))

        activations = activations.to(device)
        forward_pass = ae.forward(activations)
        if isinstance(forward_pass, tuple):
            hidden_sae = forward_pass[1]
        else:
            hidden_sae = forward_pass.feature_acts

        nonzero_sae = hidden_sae.abs() > 1e-6
        nonzero_sae_values = hidden_sae[nonzero_sae]
        nonzero_sae_indices = nonzero_sae.nonzero(
            as_tuple=False
        )  # z * 2, (nonzero) indices into hidden_sae, [, :0] is token and [, :1] is sae feature, z is num non zeros

        num_new_pairs = nonzero_sae_indices.shape[0]
        actual_num_new_pairs = min(
            num_new_pairs, num_sae_activations_to_save - total_pairs_saved
        )

        sparse_sae_values[
            total_pairs_saved : total_pairs_saved + actual_num_new_pairs
        ] = np.array(nonzero_sae_values.cpu())[:actual_num_new_pairs]

        sparse_sae_indices[
            total_pairs_saved : total_pairs_saved + actual_num_new_pairs
        ] = np.array(nonzero_sae_indices[:, 1].cpu())[:actual_num_new_pairs]

        all_token_indices[
            total_pairs_saved : total_pairs_saved + actual_num_new_pairs
        ] = np.array((nonzero_sae_indices[:, 0] + current_token_offset).cpu())[
            :actual_num_new_pairs
        ]

        current_token_offset += len(tokens)

        pbar.set_description(
            f"Num tokens = {current_token_offset}, num SAE pairs saved = {total_pairs_saved / 1e6:.2f}M"
        )
        pbar.update(actual_num_new_pairs)

        total_pairs_saved += num_new_pairs

    all_tokens = np.concatenate(all_tokens)

    np.savez_compressed(
        output_file,
        sparse_sae_values=sparse_sae_values,
        sparse_sae_indices=sparse_sae_indices,
        all_token_indices=all_token_indices,
        all_tokens=all_tokens,
    )


for ae, layer in zip(aes, layers_to_evaluate):
    data = hf_dataset_to_generator("monology/pile-uncopyrighted")

    output_file = f"{save_folder}/sae_activations_big_layer-{layer}.npz"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_coocurring_sae_features(model, ae, output_file, data)
å
