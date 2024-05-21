from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, cast
import torch
from pydantic import Field
import wandb
from typing import Set


@dataclass
class RunnerConfig(ABC):
    """
    The config that's shared across all runners.
    """

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-2l"
    hook_point: str = "blocks.{layer}.hook_mlp_out"
    hook_point_layer: int = 0
    hook_point_head_index: Optional[int] = None
    dataset_path: str = "NeelNanda/c4-tokenized-2b"
    activation_path: str = "activation_cache/test/"
    is_dataset_tokenized: bool = True
    context_size: int = 128
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_point_head_index}"
    )

    # SAE Parameters
    d_in: int = 512

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    total_training_tokens: int = 2_000_000
    store_batch_size: int = 32

    # Misc
    device: str | torch.device = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = f"activations/{self.dataset_path.replace('/', '_')}/{self.model_name.replace('/', '_')}/{self.hook_point}"
            if self.hook_point_head_index is not None:
                self.cached_activations_path += f"_{self.hook_point_head_index}"


@dataclass
class LanguageModelSAERunnerConfig(RunnerConfig):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # SAE Parameters
    expansion_factor: int = 4
    from_pretrained_path: Optional[str] = None
    d_sae: Optional[int] = None

    # Init parameters
    b_dec_init_method: str = "mean"
    init_tied_decoder: bool = True
    init_b_enc: float = 0.03

    # Training Parameters
    l1_coefficient: float = 1e-3
    lp_norm: float = 1
    weight_decay: float = 1e-3
    lr: float = 3e-4
    lr_end: float | None = None  # only used for cosine annealing, default is lr / 10
    lr_scheduler_name: str = (
        "constant"  # constant, cosineannealing, cosineannealingwarmrestarts
    )
    lr_warm_up_steps: int = 5000
    lr_decay_steps: int = 0
    train_batch_size: int = 4096
    n_restart_cycles: int = 0  # only used for cosineannealingwarmrestarts

    # Resampling protocol args
    # feature_sampling_window: int = 2000
    # dead_feature_window: int = 1000  # unless this window is larger feature sampling,
    resample_threshold: int = (
        1000  # number of steps without a feature firing to be considered dead
    )
    # steps_to_resample: frozenset[int] = {10_000, 25_000, 60_000, 100_000}
    # steps_to_resample: list[int] = [1_000, 3_000, 6_000, 12_000]
    steps_to_resample = {12_000, 30_000, 60_000, 90_000, 130_000}
    # steps_to_resample = {2500, 6_000, 12_000}
    resampling_method: str = "residual"

    # WANDB
    log_to_wandb: bool = True
    wandb_log_dir: str = "wandb"
    wandb_project: str = "mats_sae_training_language_model"
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 10

    # Misc
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"
    prepend_bos: bool = True
    verbose: bool = True
    skip_eval_loop: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.expansion_factor, list):
            self.d_sae = self.d_in * self.expansion_factor
        self.tokens_per_buffer = (
            self.train_batch_size * self.context_size * self.n_batches_in_buffer
        )

        if self.run_name is None:
            self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"

        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )
        if self.b_dec_init_method == "zeros":
            print(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        self.device = torch.device(self.device)

        if self.lr_end is None:
            self.lr_end = self.lr / 10

        unique_id = cast(
            Any, wandb
        ).util.generate_id()  # not sure why this type is erroring
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        if self.verbose:
            print(
                f"Run name: {self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"
            )


@dataclass
class CacheActivationsRunnerConfig(RunnerConfig):
    """
    Configuration for caching activations of an LLM.
    """

    # Activation caching stuff
    shuffle_every_n_buffers: int = 10
    n_shuffles_with_last_section: int = 10
    n_shuffles_in_entire_dir: int = 10
    n_shuffles_final: int = 100

    def __post_init__(self):
        super().__post_init__()
        if self.use_cached_activations:
            # this is a dummy property in this context; only here to avoid class compatibility headaches
            raise ValueError(
                "use_cached_activations should be False when running cache_activations_runner"
            )
