import os
import random
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
import pandas as pd
import torch

from avae import config, vis
from tests import testdata_mrc
from tests.test_train_eval_pipeline import helper_train_eval

torch.random.manual_seed(10)
random.seed(10)


class VisPipelineTest(unittest.TestCase):
    """Test pipeline with isolated visualisations."""

    def setUp(self) -> None:
        """Setup data and output directories."""
        self.test_data = os.path.dirname(testdata_mrc.__file__)

        self.data_params = {  # only specify if differs from default
            # data
            "datapath": self.test_data,
            "affinity": os.path.join(self.test_data, "affinity_fsc_10.csv"),
            "classes": os.path.join(self.test_data, "classes.csv"),
            "split": 10,
            # preprocess
            "rescale": 32,
            "gaussian_blur": True,
            "normalise": True,
            "shift_min": True,
            # model
            "epochs": 1,
            "batch": 25,
            "model": "cnn",
            "channels": 3,
            "depth": 4,
            "latent_dims": 8,
            "pose_dims": 3,
            "learning": 0.03,
            "beta": 1,
            "gamma": 1,
            "gpu_devices": "0",
            # vis
            "vis_all": False,
            "freq_all": 1,
            "vis_format": "png",
        }

        self.data = config.load_config_params(local_args=self.data_params)

    def test_pipeline_dispatches_visualisations(self):
        visualisations = (
            "vis_acc",
            "vis_rec",
            "vis_emb",
            "vis_dynamic",
            "vis_sim",
            "vis_int",
            "vis_dis",
            "vis_pos",
            "vis_his",
            "vis_aff",
            "vis_cyc",
        )
        for name in visualisations:
            setattr(self.data, name, True)
        self.data.vis_pose_class = "1b23"
        self.data.vis_z_n_int = "1,2"

        vis_calls = {
            "accuracy_plot": 1,
            "f1_plot": 1,
            "recon_plot": 2,
            "latent_space_similarity_plot": 2,
            "latent_embed_plot_tsne": 2,
            "dyn_latentembed_plot": 1,
            "latent_disentamglement_plot": 1,
            "pose_disentanglement_plot": 1,
            "pose_class_disentanglement_plot": 1,
            "latent_4enc_interpolate_plot": 1,
            "interpolations_plot": 1,
            "plot_cyc_variable": 2,
        }
        # Expected artifacts when these functions are not mocked:
        # accuracy/F1: 11 files; reconstruction: 8 files; similarity: 2 files;
        # static embeddings: 2 plots; dynamic embedding: 1 HTML;
        # latent disentanglement: 1 plot; pose disentanglement: 1 plot plus
        # one per requested class; interpolation: 1 plot; affinity: 1 plot;
        # distributions: 3 plots; cyclical variables: 2 plots.

        with ExitStack() as stack:
            patched_vis = {
                name: stack.enter_context(mock.patch(f"avae.vis.{name}"))
                for name in vis_calls
            }
            # data.py imports these functions directly, so patch the aliases
            # where they are looked up rather than their definitions in vis.py.
            plot_affinity = stack.enter_context(
                mock.patch("avae.data.plot_affinity_matrix")
            )
            plot_distribution = stack.enter_context(
                mock.patch("avae.data.plot_classes_distribution")
            )
            compute_accuracy = stack.enter_context(
                mock.patch(
                    "avae.utils_learning.accuracy",
                    return_value=(
                        1.0,
                        1.0,
                        1.0,
                        np.array([]),
                        np.array([]),
                    ),
                )
            )
            compute_tsne = stack.enter_context(
                mock.patch(
                    "avae.utils_learning.tsne_embedding",
                    side_effect=lambda xs, **kwargs: np.zeros((len(xs), 2)),
                )
            )

            helper_train_eval(
                self.data,
                eval=False,
                noplot=True,
                nolat=True,
                nostate=True,
            )

        for name, expected_calls in vis_calls.items():
            # Each enabled training visualisation is dispatched as expected.
            self.assertEqual(
                patched_vis[name].call_count,
                expected_calls,
                name,
            )
        # The affinity matrix is plotted once while loading training data.
        self.assertEqual(plot_affinity.call_count, 1)
        # Distribution plots are requested for train, validation, and test.
        self.assertEqual(plot_distribution.call_count, 3)
        # Accuracy is computed once and reused by both accuracy renderers.
        self.assertEqual(compute_accuracy.call_count, 1)
        # Separate latent and pose spaces each require one t-SNE calculation.
        self.assertEqual(compute_tsne.call_count, 2)
        # Both static plots receive precomputed coordinates.
        self.assertTrue(
            all(
                call.kwargs.get("embedding") is not None
                for call in patched_vis[
                    "latent_embed_plot_tsne"
                ].call_args_list
            )
        )
        # The dynamic plot reuses the precomputed latent coordinates.
        self.assertIsNotNone(
            patched_vis["dyn_latentembed_plot"].call_args.kwargs.get(
                "embedding"
            )
        )

    def test_reconstruction_csv_matches_grid_rows(self):
        images = torch.arange(32, dtype=torch.float32).reshape(4, 1, 2, 2, 2)
        selected_indices = np.array([3, 1, 0, 2])
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory(prefix="avae-recon-") as temp_dir:
            os.chdir(temp_dir)
            try:
                with mock.patch(
                    "numpy.random.choice", return_value=selected_indices
                ):
                    vis.recon_plot(
                        images,
                        images,
                        ["a", "b", "c", "d"],
                        data_dim=3,
                    )
                reconstruction_index = pd.read_csv("plots/recon_trn.csv")
            finally:
                os.chdir(original_cwd)

        # CSV rows preserve the sampled positions from the source batch.
        self.assertEqual(
            reconstruction_index["batch_index"].tolist(),
            selected_indices.tolist(),
        )
        # Each sampled position is paired with its corresponding class label.
        self.assertEqual(
            reconstruction_index["label"].tolist(),
            ["d", "b", "a", "c"],
        )

    def test_loss_plot_writes_both_views(self):
        original_cwd = os.getcwd()
        train_history = [np.ones(4), np.full(4, 0.5)]
        val_history = [np.ones(4), np.full(4, 0.75)]

        with tempfile.TemporaryDirectory(prefix="avae-loss-") as temp_dir:
            os.chdir(temp_dir)
            try:
                vis.loss_plot(
                    2,
                    np.ones(2),
                    np.ones(2),
                    train_history,
                    val_history,
                )
                plot_files = set(os.listdir("plots"))
            finally:
                os.chdir(original_cwd)

        # Loss rendering writes both component and total-loss views.
        self.assertEqual(plot_files, {"loss.png", "loss_total.png"})

    def test_accuracy_plot_handles_class_absent_from_validation(self):
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory(prefix="avae-accuracy-") as temp_dir:
            os.chdir(temp_dir)
            try:
                with self.assertLogs(level="WARNING") as captured_logs:
                    with np.errstate(divide="raise", invalid="raise"):
                        vis.accuracy_plot(
                            np.array(["a", "a", "b", "b"]),
                            np.array(["a", "a", "b", "b"]),
                            np.array(["a", "a"]),
                            np.array(["a", "a"]),
                        )
                normalised = pd.read_csv(
                    "plots/confusion_valid_norm.csv"
                ).to_numpy()
            finally:
                os.chdir(original_cwd)

        self.assertTrue(np.isfinite(normalised).all())
        np.testing.assert_array_equal(normalised[1], np.zeros(2))
        self.assertIn(
            "no samples for classes ['b']",
            " ".join(captured_logs.output),
        )
