# %%

import os
from pathlib import Path
from utils import setup_notebook, BASE_DIR

setup_notebook()

import numpy as np
import transformer_lens
from task import Problem, get_acts, plot_pca, get_all_acts, get_acts_pca
from task import activation_patching


device = "cuda:4"
#
# %%


months_of_year = [
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
]

numbers = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    # "Thirteen",
    # "Fourteen",
    # "Fifteen",
    # "Sixteen",
    # "Seventeen",
    # "Eighteen",
    # "Nineteen",
    # "Twenty",
    # "Twenty-one",
    # "Twenty-two",
    # "Twenty-three",
    # "Twenty-four"
]


class MonthsOfYearTask:
    def __init__(self, device, model_name="mistral", n_devices=None):
        self.device = device

        self.model_name = model_name

        self.n_devices = n_devices

        # Tokens we expect as possible answers. Best of these can optionally be saved (as opposed to best logit overall)
        self.allowable_tokens = months_of_year

        self.prefix = Path(BASE_DIR) / f"{model_name}_months_of_year"
        self.prefix.mkdir(parents=True, exist_ok=True)

        self.num_tokens_in_answer = 1

        self.prediction_names = ["day of week"]

        # Leave tokens until difference commented out since we never want to plot them
        if model_name == "llama":
            self.token_map = {
                # 0: "<|begin_of_text|>",
                # 1: "Let",
                # 2: "apostrophe s",
                # 3: "do",
                # 4: "some",
                # 5: "calendar",
                # 6: "math",
                # 7: "<Period>",
                8: "<Num duration months>",
                9: "months",
                10: "from",
                11: "<Start month>",
                12: "is",
                13: "<Target month>",
            }
        else:
            self.token_map = {
                # 0-9 is tokenization of EITHER normal start OR period space normal start if tokenization of month duration is two tokens
                10: "<Num duration months>",
                11: "months",
                12: "from",
                13: "<Start month>",
                14: "is",
                15: "<Target month>",
            }

        self.b_token = 8 + (2 if model_name == "mistral" else 0)
        self.a_token = 11 + (2 if model_name == "mistral" else 0)
        self.before_c_token = 12 + (2 if model_name == "mistral" else 0)

        # (Friendly name, index into Problem.info)
        self.how_to_color = [
            ("target_day", 2),
            ("start_day", 0),
            ("duration_days", 1),
        ]

        # Used for figures folder
        self.name = f"{model_name}_months_of_year"

        self._lazy_model = None

    def _get_prompt(self, starting_month_int, num_months_int):
        starting_month_str = months_of_year[starting_month_int]
        num_months_str = numbers[num_months_int]
        prompt = f"Let's do some calendar math. {num_months_str} months from {starting_month_str} is"

        # Aweful hack to make mistral tokenizations the same length
        if self.model_name == "mistral" and num_months_int not in [9, 11, 12]:
            prompt = ". " + prompt

        correct_answer_int = (starting_month_int + num_months_int) % 12
        correct_answer_str = months_of_year[correct_answer_int]

        return prompt, correct_answer_str, correct_answer_int

    def generate_problems(self):
        np.random.seed(42)
        problems = []
        for starting_month in range(12):
            for num_months in range(1, 13):
                prompt, correct_answer_str, correct_answer_int = self._get_prompt(
                    starting_month_int=starting_month, num_months_int=num_months
                )
                problems.append(
                    Problem(
                        prompt,
                        correct_answer_str,
                        (starting_month, num_months, correct_answer_int),
                    )
                )

        np.random.shuffle(problems)
        return problems

    def get_model(self):
        if self.n_devices is None:
            self.n_devices = 2 if "llama" == self.model_name else 1
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
        important_tokens = [10, 11, 12, 13, 14]
        if self.model_name == "llama":
            for i in range(len(important_tokens)):
                important_tokens[i] -= 2
        return important_tokens


# %%

if __name__ == "__main__":
    task = MonthsOfYearTask(device, model_name="llama")
    # task = MonthsOfYearTask(device, model_name="mistral")

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
