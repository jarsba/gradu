import torch
import jax
import numpy as np
import numpy.random as npr


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    return rng


def get_seed(*data):
    """Get a unique seed for given data using numpy SeedSequence.
    Returns:
        int: The seed.
    """
    seed_sequence = npr.SeedSequence([abs(x.__hash__()) for x in data])
    return int(npr.default_rng(seed_sequence.spawn(1)[0]).integers(0, 2 ** 32, 1)[0])
