from unittest import mock

import numpy as np
import pytest

from avae.base import dims_after_pooling
from avae.utils_learning import combine_accuracy_data


@pytest.mark.parametrize(
    "n_pools, expected", [(0, 64), (1, 32), (2, 16), (3, 8)]
)
def test_dims_after_pooling_ndim(n_pools, expected):
    """Test dimension calculation after 2x2 pooling op."""
    start = 64
    after_pool = dims_after_pooling(start, n_pools)
    assert after_pool == expected


def test_combine_accuracy_data_gathers_all_ranks():
    local_data = ([[1.0, 2.0]], ["a"], [[3.0, 4.0]], ["b"])
    remote_data = ([[5.0, 6.0]], ["c"], [[7.0, 8.0]], ["d"])

    def gather_object(_local, gathered, dst):
        assert dst == 0
        gathered[:] = [local_data, remote_data]

    with (
        mock.patch("avae.utils_learning.dist.is_available", return_value=True),
        mock.patch(
            "avae.utils_learning.dist.is_initialized", return_value=True
        ),
        mock.patch(
            "avae.utils_learning.dist.gather_object",
            side_effect=gather_object,
        ),
    ):
        combined = combine_accuracy_data(*local_data, True, 2)

    assert combined is not None
    z_train, y_train, z_val, y_val = combined
    np.testing.assert_array_equal(z_train, [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_array_equal(y_train, ["a", "c"])
    np.testing.assert_array_equal(z_val, [[3.0, 4.0], [7.0, 8.0]])
    np.testing.assert_array_equal(y_val, ["b", "d"])


def test_combine_accuracy_data_nonzero_rank_participates_in_gather():
    local_data = ([[1.0]], ["a"], [[2.0]], ["b"])

    with (
        mock.patch("avae.utils_learning.dist.is_available", return_value=True),
        mock.patch(
            "avae.utils_learning.dist.is_initialized", return_value=True
        ),
        mock.patch("avae.utils_learning.dist.gather_object") as gather_object,
    ):
        combined = combine_accuracy_data(*local_data, False, 2)

    assert combined is None
    gather_object.assert_called_once_with(local_data, None, dst=0)


def test_combine_accuracy_data_rejects_missing_rank():
    local_data = ([[1.0]], ["a"], [[2.0]], ["b"])

    with (
        mock.patch("avae.utils_learning.dist.is_available", return_value=True),
        mock.patch(
            "avae.utils_learning.dist.is_initialized", return_value=True
        ),
        mock.patch("avae.utils_learning.dist.gather_object"),
        pytest.raises(RuntimeError, match="was not gathered from every rank"),
    ):
        combine_accuracy_data(*local_data, True, 2)
