import os
import unittest

import pytest

from avae import config
from tests import testdata_mrc
from tests.test_train_eval_pipeline import helper_train_eval, pipeline_params

pytestmark = pytest.mark.integration


class TrainEvalIntegrationTest(unittest.TestCase):
    """Run the complete training, evaluation, and visualisation pipeline."""

    def test_cnn_mrc_artifacts(self):
        test_data = os.path.dirname(testdata_mrc.__file__)
        data_params = pipeline_params(test_data)
        data_params["vis_all"] = True
        data = config.load_config_params(local_args=data_params)
        data.vis_pose_class = "1b23,1dkg"

        (
            n_dir_train,
            n_plots_train,
            n_latent_train,
            n_states_train,
            n_plots_eval,
            n_latent_eval,
            n_states_eval,
        ) = helper_train_eval(data)

        self.assertEqual(n_dir_train, 3)
        self.assertEqual(n_plots_train, 34)
        self.assertEqual(n_latent_train, 1)
        self.assertEqual(n_states_train, 2)
        self.assertEqual(n_plots_eval, 56)
        self.assertEqual(n_latent_eval, 2)
        self.assertEqual(n_states_eval, 3)


if __name__ == "__main__":
    unittest.main()
