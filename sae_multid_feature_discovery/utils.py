from huggingface_hub import hf_hub_download
import os

BASE_DIR = "/data/scratch/ANON/"


def get_gpt2_sae(device, layer):
    from sae_lens import SparseAutoencoderDictionary

    if type(device) == int:
        device = f"cuda:{device}"

    GPT2_SMALL_RESIDUAL_SAES_REPO_ID = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = f"blocks.{layer}.hook_resid_pre"

    FILENAME = f"{hook_point}/cfg.json"
    path = hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    FILENAME = f"{hook_point}/sae_weights.safetensors"
    hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    FILENAME = f"{hook_point}/sparsity.safetensors"
    hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    folder_path = os.path.dirname(path)

    return SparseAutoencoderDictionary.load_from_pretrained(folder_path, device=device)[
        f"blocks.{layer}.hook_resid_pre"
    ]


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


def get_sae(device, model_name, layer):
    if model_name == "gpt-2":
        return get_gpt2_sae(device, layer)
    elif model_name == "mistral-7b":
        return get_mistral_sae(device, layer)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
