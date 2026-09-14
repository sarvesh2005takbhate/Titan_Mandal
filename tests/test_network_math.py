import numpy as np

from src.build_networks import distance_from_similarity


def test_distance_from_similarity_handles_positive_and_nonpositive_values():
    assert np.isclose(distance_from_similarity(1.0), 1.0)
    assert np.isclose(distance_from_similarity(0.5), 2.0)
    assert np.isinf(distance_from_similarity(0.0))
    assert np.isinf(distance_from_similarity(-0.3))
