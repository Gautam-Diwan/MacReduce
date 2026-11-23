"""Isolate data sampling from runtime setup and evaluation random draws."""

from contextlib import contextmanager

import numpy as np


@contextmanager
def numpy_seed(seed):
    previous = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(previous)


def isolated_batches(iterator, seed):
    """Give a legacy NumPy-global iterator its own reproducible RNG stream."""

    state = np.random.RandomState(seed).get_state()
    while True:
        previous = np.random.get_state()
        np.random.set_state(state)
        try:
            batch = next(iterator)
            state = np.random.get_state()
        except StopIteration:
            return
        finally:
            np.random.set_state(previous)
        yield batch
