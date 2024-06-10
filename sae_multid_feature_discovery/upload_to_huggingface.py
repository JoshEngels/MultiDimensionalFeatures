# %%


try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
except:
    pass

from utils import get_mistral_sae
import torch

torch.set_grad_enabled(False)
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE, SAEConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from huggingface_hub import HfApi
import einops
from functools import partial
import numpy as np

# %%


model = HookedTransformer.from_pretrained("mistral-7b", device="cuda:0")

# %%


def get_save_path(layer):
    return f"/home/jengels/MultiDimensionalFeatures/sae_multid_feature_discovery/saes/sae_lens_models/mistral-7b_layer_{layer}"


def upload(layer):
    original_sae = get_mistral_sae(device="cpu", layer=layer)
    w_enc = original_sae.W_enc
    b_enc = original_sae.b_enc
    w_dec = original_sae.W_dec
    b_dec = original_sae.b_dec

    test_data = torch.rand((1, 4096)).to(torch.float32)
    forward_1 = original_sae.forward(test_data)

    config = SAEConfig(
        d_in=w_enc.shape[0],
        d_sae=w_enc.shape[1],
        activation_fn_str="relu",
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        context_size=256,
        model_name="mistral-7b",
        hook_name=f"blocks.{layer}.hook_resid_pre",
        hook_layer=layer,
        hook_head_index=None,
        prepend_bos=False,
        dataset_path="monology/pile-uncopyrighted",
        normalize_activations=False,
        # misc
        dtype="float32",
        device="cpu",
        sae_lens_training_version=None,
    )

    sae = SAE(config)
    sae.load_state_dict(
        {
            "W_enc": w_enc,
            "b_enc": b_enc,
            "W_dec": w_dec,
            "b_dec": b_dec,
        }
    )

    norm_coeff = (test_data.shape[-1] ** 0.5) / test_data.norm(dim=-1, keepdim=True)
    forward_2 = sae.forward(test_data * norm_coeff)
    forward_2 /= norm_coeff

    print(forward_1)
    print(forward_2)

    sae.save_model(get_save_path(layer))

    api = HfApi()
    api.upload_folder(
        folder_path=get_save_path(layer),
        path_in_repo=f"mistral_7b_layer_{layer}",
        repo_id="JoshEngels/Mistral-7B-Residual-Stream-SAEs",
        repo_type="model",
        token="REDACTED",
    )


def validate(layer):

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="mistral-7b-res-wg",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=f"blocks.{layer}.hook_resid_pre",  # won't always be a hook point
        device="cuda:0",
    )

    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    valid_size = 400
    valid_dataset = list(
        load_dataset("monology/pile-uncopyrighted", streaming=True)["train"].take(
            valid_size
        )
    )

    context_size = 256

    batch_size = 4

    token_dataset = []
    for entry in tqdm(valid_dataset):
        tokens = model.to_tokens(entry["text"])
        if len(tokens[0]) < context_size:
            continue
        token_dataset.append(tokens[0, :context_size])

    token_dataset = torch.vstack(token_dataset)
    all_l0s = []
    all_variances_explained = []
    all_ce_percent_loss_recovered = []
    for batch_start in tqdm(
        range(0, len(token_dataset) // batch_size * batch_size, batch_size)
    ):

        # activation store can give us tokens.
        batch_tokens = token_dataset[batch_start : batch_start + batch_size]
        _, cache = model.run_with_cache(
            batch_tokens, prepend_bos=True, names_filter=sae.cfg.hook_name
        )

        # Skip the BOS token
        sae_in = cache[sae.cfg.hook_name]
        sae_in = sae_in[:, 1:, :]
        sae_in = einops.rearrange(sae_in, "b c d -> (b c) d")

        # Calculate normalization coefficient
        norm_coeff = (sae_in.shape[-1] ** 0.5) / sae_in.norm(dim=-1, keepdim=True)

        feature_acts = sae.encode(sae_in * norm_coeff)
        sae_out = sae.decode(feature_acts)
        sae_out /= norm_coeff

        all_l0s.append((feature_acts > 0).float().sum(-1).detach().cpu())

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance
        all_variances_explained.append(explained_variance.detach().cpu())

        # next we want to do a reconstruction test.
        def reconstr_hook(activation, hook, sae_out):
            sae_out = einops.rearrange(sae_out, "(b c) d -> b c d", c=255)
            activation[:, 1:, :] = sae_out
            return activation

        def zero_abl_hook(activation, hook):
            return torch.zeros_like(activation)

        orig = model(batch_tokens, return_type="loss").item()
        reconstr = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[
                (
                    sae.cfg.hook_name,
                    partial(reconstr_hook, sae_out=sae_out),
                )
            ],
            return_type="loss",
        ).item()
        zero_ablated = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(sae.cfg.hook_name, zero_abl_hook)],
        ).item()

        # This will be an average of averages, but it's okay because all
        # the averages are over the same number of tokens.
        all_ce_percent_loss_recovered.append(
            1 - (orig - reconstr) / (orig - zero_ablated)
        )

    l0 = torch.concatenate(all_l0s)
    print(l0.shape)
    print("average l0", l0.mean().item())
    plt.hist(l0.flatten().cpu().numpy())
    plt.xlabel("L0")
    plt.ylabel("Frequency")
    plt.title(f"L0 of Mistral-7B Layer {layer} SAE")
    plt.show()

    variances = torch.vstack(all_variances_explained)
    print("average variance explained", variances.mean().item())
    variances = einops.rearrange(variances, "n (b c) -> (n b) c", c=255)
    average_variance_by_context_position = variances.mean(0)
    plt.plot(average_variance_by_context_position)
    plt.xlabel("Context Position")
    plt.ylabel("Average Variance Explained")
    plt.title(
        f"Average Variance Explained by Context Position for Mistral-7B Layer {layer} SAE"
    )
    plt.show()

    ce_percent_loss_recovered = np.mean(all_ce_percent_loss_recovered)
    print("average cross-entropy percent loss recovered", ce_percent_loss_recovered)


for layer in [8, 16, 24]:
    # upload(layer)
    validate(layer)

# %%
