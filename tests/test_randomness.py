import numpy as np
import pytest

from macreduce.randomness import isolated_batches, numpy_seed


def test_evaluation_seed_repeats_samples_and_restores_training_rng():
    np.random.seed(123)
    expected_next = np.random.randint(100000)
    np.random.seed(123)
    with numpy_seed(0):
        first = np.random.permutation(100)
    with numpy_seed(0):
        second = np.random.permutation(100)
    np.testing.assert_array_equal(first, second)
    assert np.random.randint(100000) == expected_next
    with pytest.raises(RuntimeError), numpy_seed(99):
        raise RuntimeError("failed evaluation")


def test_batch_stream_survives_setup_evaluation_and_resume_random_draws():
    def batches():
        for _ in range(3):
            for value in np.random.permutation(5):
                yield value

    reference = list(isolated_batches(batches(), 7))
    np.random.seed(999)
    np.random.standard_normal(1000)  # backend setup
    stream = isolated_batches(batches(), 7)
    actual = []
    for value in stream:
        actual.append(value)
        np.random.standard_normal(10)  # evaluation or library setup between batches
    assert actual == reference
    resumed = isolated_batches(batches(), 7)
    for _ in range(7):
        next(resumed)
    assert list(resumed) == reference[7:]
