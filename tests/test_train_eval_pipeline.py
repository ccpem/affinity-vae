import os
import random
import tempfile
import unittest

import torch

from avae import config
from run import run_pipeline
from tests import testdata_mrc, testdata_npy

# fixing random seeds so we dont get fail on mrc tests
torch.random.manual_seed(10)
random.seed(10)


def pipeline_params(datapath):
    return {
        "datapath": datapath,
        "datatype": "mrc",
        "split": 10,
        "batch": 25,
        "no_val_drop": True,
        "affinity": os.path.join(datapath, "affinity_fsc_10.csv"),
        "classes": os.path.join(datapath, "classes.csv"),
        "epochs": 1,
        "channels": 3,
        "depth": 4,
        "latent_dims": 8,
        "pose_dims": 3,
        "learning": 0.03,
        "beta_min": 0,
        "beta": 1,
        "beta_cycle": 1,
        "cyc_method_beta": "flat",
        "gamma_min": 0,
        "gamma": 1,
        "cyc_method_gamma": "flat",
        "recon_loss": "MSE",
        "gaussian_blur": True,
        "normalise": True,
        "shift_min": True,
        "rescale": 32,
        "classifier": "NN",
        "opt_method": "adam",
        "gpu_devices": "0",
        "freq_all": 1,
        "vis_all": False,
        "vis_format": "png",
    }


class TrainEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        """Test instantiation of the pipeline."""
        self.testdata_mrc = os.path.dirname(testdata_mrc.__file__)
        self.testdata_npy = os.path.dirname(testdata_npy.__file__)

        self.data_params = pipeline_params(self.testdata_mrc)

        self.data = config.load_config_params(local_args=self.data_params)

    def test_model_cnn_mrc(self):
        (
            n_dir_train,
            n_plots_train,
            n_latent_train,
            n_states_train,
        ) = helper_train_eval(self.data, eval=False)

        self.assertEqual(n_dir_train, 1)
        self.assertEqual(n_plots_train, 0)
        self.assertEqual(n_latent_train, 0)
        self.assertEqual(n_states_train, 2)

    def test_model_cnn_npy(self):
        self.data.datatype = "npy"
        self.data.datapath = self.testdata_npy
        self.data.affinity = os.path.join(self.testdata_npy, "affinity_an.csv")
        self.data.classes = os.path.join(self.testdata_npy, "classes.csv")
        (
            n_dir_train,
            n_plots_train,
            n_latent_train,
            n_states_train,
        ) = helper_train_eval(self.data, eval=False)

        self.assertEqual(n_dir_train, 1)
        self.assertEqual(n_plots_train, 0)
        self.assertEqual(n_latent_train, 0)
        self.assertEqual(n_states_train, 2)

    def test_model_nopose(self):
        self.data.model = "cnn"
        self.data.pose_dims = 0

        (
            n_dir_train,
            n_plots_train,
            n_latent_train,
            n_states_train,
        ) = helper_train_eval(self.data, eval=False)

        self.assertEqual(n_dir_train, 1)
        self.assertEqual(n_plots_train, 0)
        self.assertEqual(n_latent_train, 0)
        self.assertEqual(n_states_train, 2)

    def test_model_nogamma(self):
        self.data.model = "cnn"
        self.data.gamma = 0

        (
            n_dir_train,
            n_plots_train,
            n_latent_train,
            n_states_train,
        ) = helper_train_eval(self.data, eval=False)

        self.assertEqual(n_dir_train, 1)
        self.assertEqual(n_plots_train, 0)
        self.assertEqual(n_latent_train, 0)
        self.assertEqual(n_states_train, 2)


class VisEpochsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.testdata_mrc = os.path.dirname(testdata_mrc.__file__)
        self.data_params = pipeline_params(self.testdata_mrc)

    def _run(self, **overrides):
        """Train in a temp dir; return a snapshot of (dir, files) under plots/."""
        params = dict(self.data_params)
        params.update(overrides)
        data = config.load_config_params(local_args=params)
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix='avae-') as temp_dir:
            os.chdir(temp_dir)
            try:
                run_pipeline(data)
                return self._snapshot(temp_dir)
            finally:
                os.chdir(original_cwd)

    @staticmethod
    def _snapshot(temp_dir):
        plots = os.path.join(temp_dir, "plots")
        found = []
        for root, dirs, files in os.walk(plots):
            rel = os.path.relpath(root, plots)
            found.append((rel, sorted(files)))
        return found

    @staticmethod
    def _dirs(snapshot):
        return {rel for rel, _ in snapshot}

    def test_vis_epochs_freq(self):
        snap = self._run(epochs=2, vis_epochs=True, vis_rec=True, freq_rec=2)
        dirs = self._dirs(snap)
        self.assertIn(os.path.join("epochs", "epoch_0002"), dirs)
        self.assertNotIn(os.path.join("epochs", "epoch_0001"), dirs)

    def test_vis_int_without_eval_epoch(self):
        # z_test is only filled on freq_eval epochs; interpolation must not
        # try to concatenate it on other epochs.
        self._run(epochs=2, vis_int=True, freq_int=1, freq_eval=10)

    def test_vis_best_single_folder(self):
        snap = self._run(epochs=3, vis_best=True, vis_rec=True, freq_rec=10)
        best = [(r, f) for r, f in snap if r.startswith("best_epoch")]
        self.assertEqual(len(best), 1)
        self.assertTrue(any("recon" in name for name in best[0][1]))

    def test_vis_best_no_plots_enabled(self):
        off = {
            f"vis_{n}": False
            for n in (
                "acc",
                "rec",
                "emb",
                "dynamic",
                "sim",
                "int",
                "dis",
                "pos",
            )
        }
        with self.assertLogs(level="WARNING") as cm:
            snap = self._run(epochs=2, vis_best=True, **off)
        self.assertTrue(
            any(
                "vis_best is set but no per-epoch plots are enabled" in m
                for m in cm.output
            )
        )
        self.assertFalse(
            any(r.startswith("best_epoch") for r in self._dirs(snap))
        )


def helper_train_eval(
    data, eval=True, noplot=False, nolat=False, nostate=False
):
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix='avae-') as temp_dir:
        os.chdir(temp_dir)
        try:
            if eval:
                eval = [not eval, eval]
            else:
                eval = [eval]
            ret = []

            # run training
            for e in eval:
                data.evaluate = e
                if data.evaluate:
                    data.datapath = os.path.join(data.datapath, "test")

                run_pipeline(data)

                n_plots, n_latent, n_states = (0, 0, 0)
                n_dir = len(next(os.walk(temp_dir))[1])
                if os.path.exists(os.path.join(temp_dir, "plots")):
                    n_plots = (
                        len(os.listdir(os.path.join(temp_dir, "plots")))
                        if not noplot
                        else None
                    )
                if os.path.exists(os.path.join(temp_dir, "latents")):
                    n_latent = (
                        len(os.listdir(os.path.join(temp_dir, "latents")))
                        if not nolat
                        else None
                    )
                if os.path.exists(os.path.join(temp_dir, "states")):
                    n_states = (
                        len(os.listdir(os.path.join(temp_dir, "states")))
                        if not nostate
                        else None
                    )

                ret.extend([n_plots, n_latent, n_states])
            ret.insert(0, n_dir)
            return tuple(ret)
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
