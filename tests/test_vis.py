import os
import random
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

import mrcfile
import numpy as np
import pandas as pd
import torch

from avae import config, utils_vis, vis
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
        self.data.vis_class = "1b23"

        vis_calls = {
            "confusion_plot": 1,
            "f1_plot": 1,
            "recon_plot": 2,
            "latent_space_similarity_plot": 2,
            "latent_embed_plot_tsne": 2,
            "dyn_latentembed_plot": 1,
            "latent_disentamglement_plot": 1,
            "latent_class_disentanglement_plot": 1,
            "pose_disentanglement_plot": 1,
            "pose_class_disentanglement_plot": 1,
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
        self.assertEqual(
            patched_vis["confusion_plot"].call_args.kwargs["epoch"], 0
        )
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
                        images + 100,
                        ["a", "b", "c", "d"],
                        data_dim=3,
                    )
                reconstruction_index = pd.read_csv("plots/recon_trn.csv")
                grid_in = mrcfile.read("plots/recon_trn_in.mrc")
                grid_out = mrcfile.read("plots/recon_trn_out.mrc")
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
        # inputs and reconstructions are separate files with the same layout
        self.assertEqual(grid_in.shape, (8, 2, 2))
        self.assertEqual(grid_out.shape, grid_in.shape)
        for row, source in enumerate(selected_indices):
            cell = slice(row * 2, (row + 1) * 2)
            np.testing.assert_array_equal(grid_in[cell], images[source, 0])
            np.testing.assert_array_equal(
                grid_out[cell], images[source, 0] + 100
            )

    def tearDown(self) -> None:
        utils_vis.set_vis_dir(None)
        utils_vis.set_run_id("")

    def test_run_id_tags_dirs_and_clear_best_dirs_is_scoped(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="avae-runid-") as temp_dir:
            os.chdir(temp_dir)
            try:
                utils_vis.set_run_id("261027-0930")
                self.assertEqual(
                    utils_vis.vis_dir(5),
                    "plots/epochs_261027-0930/epoch_0005",
                )
                self.assertEqual(
                    utils_vis.vis_dir(12, best="loss"),
                    "plots/best_loss_epoch0012_261027-0930",
                )
                self.assertEqual(
                    utils_vis.vis_dir(12, best="acc"),
                    "plots/best_acc_epoch0012_261027-0930",
                )
                for d in (
                    "best_loss_epoch0012_261027-0930",
                    "best_acc_epoch0012_261027-0930",
                    "best_loss_epoch0012_250101-0000",
                    "best_epoch0012",
                ):
                    os.makedirs(os.path.join("plots", d))
                utils_vis.clear_best_dirs("loss")
                self.assertFalse(
                    os.path.exists("plots/best_loss_epoch0012_261027-0930")
                )
                self.assertTrue(
                    os.path.exists("plots/best_acc_epoch0012_261027-0930")
                )
                self.assertTrue(
                    os.path.exists("plots/best_loss_epoch0012_250101-0000")
                )
                self.assertTrue(os.path.exists("plots/best_epoch0012"))
                utils_vis.clear_best_dirs("acc")
                self.assertFalse(
                    os.path.exists("plots/best_acc_epoch0012_261027-0930")
                )
            finally:
                os.chdir(original_cwd)

    def test_plots_dir_separates_per_epoch_plots(self):
        original_cwd = os.getcwd()
        train_history = [np.ones(4), np.full(4, 0.5)]
        val_history = [np.ones(4), np.full(4, 0.75)]

        with tempfile.TemporaryDirectory(prefix="avae-epochs-") as temp_dir:
            os.chdir(temp_dir)
            try:
                utils_vis.set_vis_dir(3)
                vis.confusion_plot(
                    np.array(["a", "a", "b", "b"]),
                    np.array(["a", "a", "b", "b"]),
                    np.array(["a", "a", "b", "b"]),
                    np.array(["a", "a", "b", "b"]),
                    epoch=3,
                )
                vis.loss_plot(
                    2,
                    np.ones(2),
                    np.ones(2),
                    train_history,
                    val_history,
                )
                epoch_files = os.listdir("plots/epochs/epoch_0003")
                root_files = os.listdir("plots")
                epoch_dirs = os.listdir("plots/epochs")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(epoch_dirs, ["epoch_0003"])
        self.assertIn("confusion_val_norm.csv", epoch_files)
        self.assertNotIn("loss.png", epoch_files)
        self.assertIn("loss.png", root_files)
        self.assertIn("loss_total.png", root_files)
        self.assertNotIn("confusion_val_norm.csv", root_files)

    def test_vis_dir_epoch_and_best_paths(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="avae-dirs-") as temp_dir:
            os.chdir(temp_dir)
            try:
                utils_vis.set_vis_dir(3)
                epoch_path = utils_vis.plots_path("a.png")
                utils_vis.set_vis_dir(3, best="loss")
                best_path = utils_vis.plots_path("a.png")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(epoch_path, "plots/epochs/epoch_0003/a.png")
        self.assertEqual(best_path, "plots/best_loss_epoch0003/a.png")

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

    def _grid_files(self, size, **vis_dir_kwargs):
        """Save a 2x3 grid of volumes with side `size`; return the PNG grid
        array shape, the plots/ file list and the .mrc grid shape."""
        original_cwd = os.getcwd()
        images = np.random.rand(2, 3, size, size, size).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="avae-grid-") as temp_dir:
            os.chdir(temp_dir)
            try:
                utils_vis.set_vis_dir(**vis_dir_kwargs)
                with mock.patch(
                    "avae.vis.save_imshow_png", wraps=vis.save_imshow_png
                ) as spy:
                    vis._save_grid_figure(
                        2, 3, (size, size, size), images, "grid_trn"
                    )
                files = sorted(
                    os.path.relpath(os.path.join(r, f), temp_dir)
                    for r, _, fs in os.walk("plots")
                    for f in fs
                )
                with mrcfile.open("plots/grid_trn.mrc") as mrc:
                    mrc_shape = mrc.data.shape
                self.assertTrue(os.path.exists("plots/grid_trn.png"))
                grid = spy.call_args.args[1]
            finally:
                os.chdir(original_cwd)
        return grid.shape, files, mrc_shape

    def test_3d_grid_writes_mrc_and_cropped_png(self):
        size = 32
        grid_shape, files, mrc_shape = self._grid_files(
            size, epoch=3, copy_to=("plots",)
        )
        tile = size - 2 * vis.PNG_EDGE_CROP
        self.assertEqual(grid_shape, (2 * tile, 3 * tile))
        self.assertEqual(mrc_shape, (2 * size, 3 * size, size))
        self.assertEqual(
            files,
            [
                "plots/epochs/epoch_0003/grid_trn.mrc",
                "plots/epochs/epoch_0003/grid_trn.png",
                "plots/grid_trn.mrc",
                "plots/grid_trn.png",
            ],
        )

    def test_3d_grid_png_crop_skipped_for_tiny_volumes(self):
        size = 8
        grid_shape, _, mrc_shape = self._grid_files(size, epoch=None)
        self.assertEqual(grid_shape, (2 * size, 3 * size))
        self.assertEqual(mrc_shape, (2 * size, 3 * size, size))

    def test_mode_file_names_have_no_stray_underscores(self):
        original_cwd = os.getcwd()
        y = np.array(["a", "a", "b", "b"])
        z = np.random.rand(4, 3)
        with tempfile.TemporaryDirectory(prefix="avae-modes-") as temp_dir:
            os.chdir(temp_dir)
            try:
                for mode in ("val", "eval"):
                    vis.confusion_plot(y, y, y, y, mode=mode)
                    vis.f1_plot(y, y, y, y, mode=mode)
                for mode in ("", "trn", "eval"):
                    vis.latent_space_similarity_plot(z, y, mode=mode)
                files = {f for _, _, fs in os.walk("plots") for f in fs}
            finally:
                os.chdir(original_cwd)
        for name in (
            "confusion_trn_norm.csv",
            "confusion_val_norm.csv",
            "confusion_trn_eval_norm.csv",
            "confusion_eval_norm.csv",
            "f1_trn.csv",
            "f1_val.csv",
            "f1_trn_eval.csv",
            "f1_eval.csv",
            "f1.png",
            "f1_eval.png",
            "similarity_mean.png",
            "similarity_mean_trn.png",
            "similarity_mean_eval.png",
        ):
            self.assertIn(name, files)
        for name in files:
            stem = name.rsplit(".", 1)[0]
            self.assertFalse(stem.endswith("_") or "__" in stem, name)

    def test_confusion_plot_handles_class_absent_from_validation(self):
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory(prefix="avae-accuracy-") as temp_dir:
            os.chdir(temp_dir)
            try:
                with (
                    self.assertLogs(level="WARNING") as captured_logs,
                    mock.patch(
                        "avae.vis.plt.title", wraps=vis.plt.title
                    ) as plot_title,
                ):
                    with np.errstate(divide="raise", invalid="raise"):
                        vis.confusion_plot(
                            np.array(["a", "a", "b", "b"]),
                            np.array(["a", "a", "b", "b"]),
                            np.array(["a", "a"]),
                            np.array(["a", "a"]),
                        )
                normalised = pd.read_csv(
                    "plots/confusion_val_norm.csv"
                ).to_numpy()
            finally:
                os.chdir(original_cwd)

        self.assertTrue(np.isfinite(normalised).all())
        np.testing.assert_array_equal(normalised[1], np.zeros(2))
        self.assertIn(
            "no samples for classes ['b']",
            " ".join(captured_logs.output),
        )
        titles = [call.args[0] for call in plot_title.call_args_list]
        self.assertTrue(all("epoch 1:" in title for title in titles))
        self.assertFalse(
            any("e+" in title or "e-" in title for title in titles)
        )

    def _save_in_temp(self, per_epoch, **vis_dir_kwargs):
        """Save a figure in a temp dir; return the sorted plots/ file list."""
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="avae-copy-") as temp_dir:
            os.chdir(temp_dir)
            try:
                utils_vis.set_vis_dir(**vis_dir_kwargs)
                fig = vis.plt.figure()
                utils_vis.save_figure(
                    fig,
                    utils_vis.plots_path("a.png", per_epoch=per_epoch),
                    "png",
                )
                return sorted(
                    os.path.relpath(os.path.join(root, f), temp_dir)
                    for root, _, files in os.walk("plots")
                    for f in files
                )
            finally:
                os.chdir(original_cwd)

    def test_save_figure_copies_to_copy_to(self):
        found = self._save_in_temp(True, epoch=3, copy_to=("plots",))
        self.assertEqual(
            found,
            ["plots/a.png", "plots/epochs/epoch_0003/a.png"],
        )

    def test_non_per_epoch_file_not_copied(self):
        found = self._save_in_temp(
            False, epoch=3, copy_to=("plots", "elsewhere")
        )
        self.assertEqual(found, ["plots/a.png"])

    def test_no_copy_without_epoch_dir(self):
        found = self._save_in_temp(True, epoch=None)
        self.assertEqual(found, ["plots/a.png"])


class _RecordingDecoder:
    """Stand-in for vae.decoder that records its inputs."""

    def __init__(self, dsize):
        self.dsize = dsize
        self.calls = []

    def __call__(self, z, pose):
        self.calls.append((z.clone(), None if pose is None else pose.clone()))
        return torch.zeros(z.shape[0], *self.dsize)


class _FakeVae:
    def __init__(self, dsize):
        self.decoder = _RecordingDecoder(dsize)


class TraversalTest(unittest.TestCase):
    """Unit tests of the interpolation and disentanglement traversals."""

    dsize = (1, 4, 4)

    def setUp(self):
        rng = np.random.default_rng(0)
        self.lats = rng.normal(size=(20, 3))
        self.poses = rng.normal(size=(20, 2))
        self.labels = np.array(["a", "b"] * 10)
        self.vae = _FakeVae(self.dsize)
        patcher = mock.patch("avae.vis._save_grid_figure")
        self.save_grid = patcher.start()
        self.addCleanup(patcher.stop)

    def test_interpolate(self):
        out = utils_vis.interpolate([0.0, 10.0], [1.0, 20.0], 3)
        np.testing.assert_allclose(out, [[0, 10], [0.5, 15], [1, 20]])
        # leading axes broadcast: (2, D) -> (2, steps, D)
        out = utils_vis.interpolate(np.zeros((2, 2)), np.ones((2, 2)), 5)
        self.assertEqual(out.shape, (2, 5, 2))
        self.assertEqual(out.dtype, np.float32)

    def test_pose_disentanglement_steps(self):
        mean, std = self.poses.mean(axis=0), self.poses.std(axis=0)
        for n in (7, 15):
            vae = _FakeVae(self.dsize)
            vis.pose_disentanglement_plot(
                self.dsize,
                self.lats,
                self.poses,
                vae,
                "cpu",
                number_of_samples=n,
            )
            z, pose = vae.decoder.calls[0]
            pose = pose.numpy().reshape(2, n, 2)
            for d in range(2):
                np.testing.assert_allclose(
                    pose[d, :, d],
                    mean[d] + std[d] * np.linspace(-2, 2, n),
                    rtol=1e-4,
                    atol=1e-5,
                )
                other = 1 - d
                np.testing.assert_allclose(
                    pose[d, :, other], mean[other], rtol=1e-5
                )
            np.testing.assert_allclose(
                z.numpy(),
                np.tile(self.lats.mean(axis=0), (2 * n, 1)),
                rtol=1e-5,
            )

    def test_latent_disentanglement_steps(self):
        mean, std = self.lats.mean(axis=0), self.lats.std(axis=0)
        for n in (7, 15):
            vae = _FakeVae(self.dsize)
            vis.latent_disentamglement_plot(
                self.dsize,
                self.lats,
                vae,
                "cpu",
                poses=self.poses,
                number_of_samples=n,
            )
            z, _ = vae.decoder.calls[0]
            z = z.numpy().reshape(3, n, 3)
            for d in range(3):
                np.testing.assert_allclose(
                    z[d, :, d],
                    mean[d] + std[d] * np.linspace(-2, 2, n),
                    rtol=1e-4,
                    atol=1e-5,
                )
            self.assertEqual(self.save_grid.call_args[0][1], n)
        self.assertEqual(
            self.save_grid.call_args[0][4], "disentanglement-latent_avg_trn"
        )

    def test_pose_plots_use_default_sweep(self):
        vis.pose_disentanglement_plot(
            self.dsize, self.lats, self.poses, self.vae, "cpu"
        )
        _, pose = self.vae.decoder.calls[0]
        self.assertEqual(pose.shape[0], 2 * 15)
        self.assertEqual(
            self.save_grid.call_args[0][4], "disentanglement-pose_avg_trn"
        )

    def test_latent_class_plot_calls_averaged_plot(self):
        with mock.patch("avae.vis.latent_disentamglement_plot") as avg:
            vis.latent_class_disentanglement_plot(
                self.dsize,
                self.lats,
                self.labels,
                "a, b",
                self.vae,
                "cpu",
                poses=self.poses,
                mode="eval",
            )
        self.assertEqual(
            [c.kwargs["label"] for c in avg.call_args_list], ["a", "b"]
        )
        for call, cls in zip(avg.call_args_list, "ab"):
            mask = self.labels == cls
            np.testing.assert_array_equal(call.args[1], self.lats[mask])
            np.testing.assert_array_equal(
                call.kwargs["poses"], self.poses[mask]
            )
            self.assertEqual(call.kwargs["mode"], "eval")

    def test_pose_class_plot_calls_averaged_plot(self):
        with mock.patch("avae.vis.pose_disentanglement_plot") as avg:
            vis.pose_class_disentanglement_plot(
                self.dsize,
                self.lats,
                self.labels,
                "a,b",
                self.poses,
                self.vae,
                "cpu",
                number_of_samples=7,
            )
        self.assertEqual(
            [c.kwargs["label"] for c in avg.call_args_list], ["a", "b"]
        )
        for call, cls in zip(avg.call_args_list, "ab"):
            mask = self.labels == cls
            np.testing.assert_array_equal(call.args[1], self.lats[mask])
            np.testing.assert_array_equal(call.args[2], self.poses[mask])
            self.assertEqual(call.kwargs["number_of_samples"], 7)

    def test_pose_class_plot_file_names(self):
        vis.pose_class_disentanglement_plot(
            self.dsize,
            self.lats,
            self.labels,
            "a,b",
            self.poses,
            self.vae,
            "cpu",
        )
        self.assertEqual(
            [c[0][4] for c in self.save_grid.call_args_list],
            ["disentanglement-pose_a_trn", "disentanglement-pose_b_trn"],
        )

    def test_degenerate_class_warns(self):
        labels = np.array(["a"] + ["b"] * 19)
        with self.assertLogs(level="WARNING") as logs:
            vis.pose_class_disentanglement_plot(
                self.dsize, self.lats, labels, "a", self.poses, self.vae, "cpu"
            )
        self.assertIn("single sample", logs.output[0])
        poses = np.zeros_like(self.poses)
        with self.assertLogs(level="WARNING") as logs:
            vis.latent_class_disentanglement_plot(
                self.dsize,
                np.zeros_like(self.lats),
                self.labels,
                "a",
                self.vae,
                "cpu",
                poses=poses,
            )
        self.assertIn("no latent spread", logs.output[0])
        self.assertEqual(self.save_grid.call_count, 2)

    def test_missing_class_raises(self):
        with self.assertRaises(ValueError) as ctx:
            vis.pose_class_disentanglement_plot(
                self.dsize,
                self.lats,
                self.labels,
                "a,nope",
                self.poses,
                self.vae,
                "cpu",
            )
        self.assertIn("vis_class", str(ctx.exception))
        self.assertIn("nope", str(ctx.exception))
        self.assertIn("available ['a', 'b']", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            vis.latent_class_disentanglement_plot(
                self.dsize, self.lats, self.labels, "nope", self.vae, "cpu"
            )
        self.assertIn("vis_class", str(ctx.exception))
        self.save_grid.assert_not_called()

    def test_interpolation_grid_and_pose(self):
        labels = np.array(list("abcde") * 4)
        vis.interpolations_plot(
            self.dsize, self.lats, labels, self.vae, "cpu", poses=self.poses
        )
        z, pose = self.vae.decoder.calls[0]
        n = 10
        self.assertEqual(z.shape, (n * n, 3))
        self.assertEqual(self.save_grid.call_args[0][:2], (n, n))
        self.assertEqual(self.save_grid.call_args[0][4], "interpolations_trn")
        grid = z.numpy().reshape(n, n, 3)
        corners = [grid[0, 0], grid[n - 1, 0], grid[0, n - 1], grid[-1, -1]]
        rows = []
        for c in corners:
            hits = np.where(np.isclose(self.lats, c, atol=1e-5).all(axis=1))[0]
            self.assertEqual(len(hits), 1)
            rows.append(hits[0])
        # corners come from 4 distinct classes
        self.assertEqual(len(set(labels[rows])), 4)
        # same pose (the first corner's real pose) in every cell
        np.testing.assert_allclose(
            pose.numpy(),
            np.tile(self.poses[rows[0]], (n * n, 1)),
            rtol=1e-5,
        )

    def test_interpolation_without_pose_or_classes(self):
        labels = np.array(list("abcde") * 4)
        vis.interpolations_plot(
            self.dsize, self.lats, labels, self.vae, "cpu", poses=None
        )
        self.assertIsNone(self.vae.decoder.calls[0][1])
        self.vae.decoder.calls.clear()
        with self.assertLogs(level="WARNING"):
            vis.interpolations_plot(
                self.dsize, self.lats, np.array(["a"] * 20), self.vae, "cpu"
            )
        self.assertEqual(self.vae.decoder.calls, [])

    def test_interpolation_line_for_fewer_than_four_classes(self):
        labels = np.array(list("abc") * 6 + ["a", "b"])
        vis.interpolations_plot(
            self.dsize, self.lats, labels, self.vae, "cpu", poses=self.poses
        )
        z, pose = self.vae.decoder.calls[0]
        n = 10
        self.assertEqual(z.shape, (n, 3))
        self.assertEqual(self.save_grid.call_args[0][:2], (1, n))
        rows = []
        for c in (z[0].numpy(), z[-1].numpy()):
            hits = np.where(np.isclose(self.lats, c, atol=1e-5).all(axis=1))[0]
            self.assertEqual(len(hits), 1)
            rows.append(hits[0])
        # ends come from 2 distinct classes
        self.assertEqual(len(set(labels[rows])), 2)
        np.testing.assert_allclose(
            pose.numpy(),
            np.tile(self.poses[rows[0]], (n, 1)),
            rtol=1e-5,
        )

    def test_plot_labels_suffix(self):
        out = utils_vis.plot_labels(["a", "b", "test", "test"], "eval")
        self.assertEqual(list(out), ["a_eval", "b_eval", "eval", "eval"])
        self.assertEqual(
            list(utils_vis.plot_labels(["a", "test"], "test")),
            ["a_test", "test"],
        )
