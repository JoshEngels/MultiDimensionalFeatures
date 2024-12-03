# %%

import os
from utils import setup_notebook, BASE_DIR

setup_notebook()

import numpy as np
import transformer_lens
import torch
from task import Problem, get_acts, plot_pca, get_all_acts, get_acts_pca
from task import activation_patching


device = "cuda:4"
#
# %%


days_of_week = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

day_intervals = [
    "Zero days",
    "One day",
    "Two days",
    "Three days",
    "Four days",
    "Five days",
    "Six days",
    "Seven days",
]


class DaysOfWeekTask:
    def __init__(self, device, model_name="mistral", n_devices=None):
        self.device = device

        self.model_name = model_name

        self.n_devices = n_devices

        # Tokens we expect as possible answers. Best of these can optionally be saved (as opposed to best logit overall)
        self.allowable_tokens = days_of_week

        self.prefix = f"{BASE_DIR}{model_name}_days_of_week/"
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)

        self.num_tokens_in_answer = 1

        self.prediction_names = ["day of week"]

        # Leave tokens until difference commented out since we never want to plot them
        if model_name == "mistral":
            self.token_map = {
                # 0: "<s>",
                # 1: "Let",
                # 2: "apostrophe",
                # 3: "s after apostrophe",
                # 4: "do",
                # 5: "some",
                # 6: "days (first)",
                # 7: "of",
                # 8: "the",
                # 9: "week",
                # 10: "math",
                # 11: "<Period>",
                12: "<Num duration days>",
                13: "days (second)",
                14: "from",
                15: "<Start day of week>",
                16: "is",
                17: "<Target day of week>",
            }
        else:
            self.token_map = {
                # 0: "<|begin_of_text|>",
                # 1: "Let",
                # 2: "apostrophe s",
                # 3: "do",
                # 4: "some",
                # 5: "days",
                # 6: "of",
                # 7: "the",
                # 8: "week",
                # 9: "math",
                # 10: "<Period>",
                11: "<Num duration days>",
                12: "days (second)",
                13: "from",
                14: "<Start day of week>",
                15: "is",
                16: "<Target day of week>",
            }

        self.b_token = 11 + (1 if model_name == "mistral" else 0)
        self.a_token = 14 + (1 if model_name == "mistral" else 0)
        self.before_c_token = 15 + (1 if model_name == "mistral" else 0)

        # (Friendly name, index into Problem.info)
        self.how_to_color = [
            ("target_day", 2),
            ("start_day", 0),
            ("duration_days", 1),
        ]

        # Used for figures folder
        self.name = f"{model_name}_days_of_week"

        self._lazy_model = None

    def _get_prompt(self, starting_day_int, num_days_int):
        starting_day_str = days_of_week[starting_day_int]
        num_days_str = day_intervals[num_days_int]
        prompt = f"Let's do some days of the week math. {num_days_str} from {starting_day_str} is"

        correct_answer_int = (starting_day_int + num_days_int) % 7
        correct_answer_str = days_of_week[correct_answer_int]

        # TODO: Should we distinguish between carrys and not in the correct answer?
        return prompt, correct_answer_str, correct_answer_int

    def generate_problems(self):
        np.random.seed(42)
        problems = []
        for starting_day in range(7):
            for num_days in range(1, 8):
                prompt, correct_answer_str, correct_answer_int = self._get_prompt(
                    starting_day_int=starting_day, num_days_int=num_days
                )
                problems.append(
                    Problem(
                        prompt,
                        correct_answer_str,
                        (starting_day, num_days, correct_answer_int),
                    )
                )

        np.random.shuffle(problems)
        return problems

    def get_model(self):
        if self.n_devices is None:
            self.n_devices = (
                min(2, max(1, torch.cuda.device_count()))
                if "llama" == self.model_name
                else 1
            )
        if self._lazy_model is None:
            if self.model_name == "mistral":
                self._lazy_model = transformer_lens.HookedTransformer.from_pretrained(
                    "mistral-7b", device=self.device, n_devices=self.n_devices
                )
            elif self.model_name == "llama":
                self._lazy_model = transformer_lens.HookedTransformer.from_pretrained(
                    # "NousResearch/Meta-Llama-3-8B",
                    "meta-llama/Meta-Llama-3-8B",
                    device=self.device,
                    n_devices=self.n_devices,
                )
        return self._lazy_model

    def important_tokens(self):
        important_tokens = [12, 13, 14, 15, 16]
        if self.model_name == "llama":
            for i in range(len(important_tokens)):
                important_tokens[i] -= 1
        return important_tokens


# %%

if __name__ == "__main__":
    task = DaysOfWeekTask(device, model_name="llama")
    # task = DaysOfWeekTask(device, model_name="mistral")


# %%

if __name__ == "__main__":
    # Force generation of PCA k = 20
    for layer in range(33):
        for token in task.important_tokens():
            _ = get_acts_pca(task, layer=layer, token=token, pca_k=20)


# %%

if __name__ == "__main__":
    do_pca = True
    if do_pca:
        for token_location in task.important_tokens():
            # for normalize in [True, False]:
            #     for k in [2,3]:
            for normalize in [False]:
                for k in [2]:
                    plot_pca(
                        task,
                        token_location=token_location,
                        k=k,
                        normalize_rms=normalize,
                        include_embedding_layer=True,
                    )
# %%

if __name__ == "__main__":
    for layer_type in ["mlp", "attention", "resid"]:
        # for layer_type in ["attention"]:
        for keep_same_index in [0, 1]:
            activation_patching(
                task,
                keep_same_index=keep_same_index,
                num_chars_in_answer_to_include=0,
                num_activation_patching_experiments_to_run=20,
                layer_type=layer_type,
            )


# %%
