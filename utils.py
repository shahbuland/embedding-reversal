from contextlib import contextmanager
from joblib import dump, load
import os
import torch

def load_or_compute(filename, compute_fn, root_dir="checkpointed_variables", always_compute=False):
    """
    This function allows for checkpointing variables in code so that
    they do not need to be recomputed. For example, computing
    embeddings with a neural network over a whole dataset is time consuming.
    Subsequent script runs should not recompute the embeddings if it's already
    been done once.

    :param filename: Name for the variable
    :param compute_fn: Function to compute the data if it's not already saved
    :param root_dir: Root directory to put any checkpointed variables in
    :param always_compute: Bypass any checks (useful if we wanna just reset file)
    """
    os.makedirs(root_dir, exist_ok=True)
    fp = os.path.join(root_dir, filename)
    try:
        assert not always_compute
        data = load(fp)
    except (FileNotFoundError, AssertionError):
        data = compute_fn()
        dump(data, fp)
    return data

@contextmanager
def skip_torch_init():
    """
    Speeds up model init by skipping default parameters. Only use when you are certain a checkpoint will be loaded.
    """
    original_linear_init = torch.nn.Linear.reset_parameters
    original_norm_init = torch.nn.LayerNorm.reset_parameters
    original_conv2d_init = torch.nn.Conv2d.reset_parameters

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    setattr(torch.nn.Conv2d, "reset_parameters", lambda self: None)

    try:
        yield
    finally:
        setattr(torch.nn.Linear, "reset_parameters", original_linear_init)
        setattr(torch.nn.LayerNorm, "reset_parameters", original_norm_init)
        setattr(torch.nn.Conv2d, "reset_parameters", original_conv2d_init)