from huggingface_hub import hf_hub_download
import os

BASE_DIR = "/data/scratch/jae/"

def get_gpt2_sae(device, layer):
    from sae_lens import SAE

    return SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=f"blocks.{layer}.hook_resid_pre",  # won't always be a hook point
        device=device
    )[0]


def get_mistral_sae(device, layer):
    from saes.sparse_autoencoder import SparseAutoencoder, LanguageModelSAERunnerConfig

    if type(device) == int:
        device = f"cuda:{device}"

    return (
        SparseAutoencoder(LanguageModelSAERunnerConfig())
        .load_from_pretrained(
            f"saes/mistral_saes/Mistral-7B-v0.1_blocks.{layer}.hook_resid_pre_65536_final.pt"
        )
        .to(device)
    )

# def get_mistral_sae(device, layer):
#     from sae_lens import SAE

#     return SAE.from_pretrained(
#         release="mistral-7b-res-wg",
#         sae_id=f"blocks.{layer}.hook_resid_pre",
#         device=device
#     )[0]



def get_sae(device, model_name, layer):
    if model_name == "gpt-2":
        return get_gpt2_sae(device, layer)
    elif model_name == "mistral-7b":
        return get_mistral_sae(device, layer)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
