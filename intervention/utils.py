import os
import dill as pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "cache"

os.environ["TRANSFORMERS_CACHE"] = f"{(Path(BASE_DIR) / '.cache').absolute()}/"


def setup_notebook():
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.magic("load_ext autoreload")
        ipython.magic("autoreload 2")

    except:
        pass


def expand_descriptions(compressed_descriptions):
    expand_descriptions = []
    for start_index, end_index, description in compressed_descriptions:
        expand_descriptions.extend([description] * (end_index - start_index))
    return expand_descriptions


def get_lambdas_and_descriptions(layer, explanation_name, task, token):
    file_prefix = (
        f"figs/{task.name}/regression/token_{token}/layer_{layer}/{explanation_name}"
    )
    lambdas = pickle.load(open(f"{file_prefix}_lambdas.pkl", "rb"))
    descriptions = pickle.load(open(f"{file_prefix}_descriptions.pkl", "rb"))
    return lambdas, descriptions


def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
