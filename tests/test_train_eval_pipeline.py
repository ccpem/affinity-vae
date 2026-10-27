import os
import random
import re
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
    RUN_ID = "260101-0000"

    def setUp(self) -> None:
        self.testdata_mrc = os.path.dirname(testdata_mrc.__file__)
        self.data_params = pipeline_params(self.testdata_mrc)

    def _run(self, seed_files=(), **overrides):
        """Train in a temp dir; return a snapshot of (dir, files) under plots/.

        `seed_files` are paths (relative to the temp dir) created before
        training to mimic leftovers from an earlier run.
        """
        params = dict(self.data_params)
        params.setdefault("date_time_run", self.RUN_ID)
        params.update(overrides)
        data = config.load_config_params(local_args=params)
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix='avae-') as temp_dir:
            os.chdir(temp_dir)
            try:
                for seed in seed_files:
                    os.makedirs(os.path.dirname(seed), exist_ok=True)
                    open(seed, "w").close()
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

    @staticmethod
    def _val_lines(cm):
        return [m for m in cm.output if "Validation : Epoch" in m]

    def test_vis_epochs_with_best_and_leftovers(self):
        """Run A: per-epoch plots + best tracking + seeded leftovers."""
        rid = self.RUN_ID
        stale_e1 = os.path.join(
            "plots", "epochs", "epoch_0001", "stale_from_old_run.png"
        )
        stale_e2 = os.path.join(
            "plots", "epochs", "epoch_0002", "stale_from_old_run.png"
        )
        # leftover inside this run's own epoch-1 folder: epoch 1 saves an
        # interpolation there, so copying whole folders would pick it up
        stale_same_run = os.path.join(
            "plots", f"epochs_{rid}", "epoch_0001", "stale_same_run.png"
        )
        old_untagged = os.path.join("plots", "best_epoch0021", "old.png")
        other_run = os.path.join(
            "plots", "best_epoch0005_250101-0000", "old.png"
        )
        # freq_int=1 with freq_eval=10: interpolation on epochs without test
        # latents must not crash.
        with self.assertLogs(level="INFO") as cm:
            snap = dict(
                self._run(
                    seed_files=(
                        stale_e1,
                        stale_e2,
                        stale_same_run,
                        old_untagged,
                        other_run,
                    ),
                    epochs=3,
                    vis_epochs=True,
                    vis_best=True,
                    best_start=2,
                    vis_rec=True,
                    vis_acc=True,
                    vis_int=True,
                    freq_rec=10,
                    freq_acc=10,
                    freq_int=1,
                    freq_eval=10,
                    vis_emb=False,
                    vis_dynamic=False,
                    vis_sim=False,
                    vis_dis=False,
                    vis_pos=False,
                )
            )
        epochs_dir = f"epochs_{rid}"

        # epoch 1: only interpolation (freq_int=1) is due; no recon/best
        epoch1 = snap[os.path.join(epochs_dir, "epoch_0001")]
        self.assertTrue(any(f.startswith("interpolations") for f in epoch1))
        self.assertFalse(any("recon" in f for f in epoch1))
        self.assertEqual(
            [
                f
                for f in epoch1
                if not f.startswith("interpolations")
                and f != "stale_same_run.png"
            ],
            [],
            "epoch 1 should only contain interpolation output",
        )

        # exactly one best folder per kind, epoch >= 2, with recon files
        best_epochs = {}
        for kind in ("loss", "acc"):
            mine = [
                r
                for r in snap
                if re.fullmatch(rf"best_{kind}_epoch\d{{4}}_{rid}", r)
            ]
            self.assertEqual(len(mine), 1, f"best {kind} folders: {mine}")
            n = len(f"best_{kind}_epoch")
            best_epochs[kind] = int(mine[0][n : n + 4])
            self.assertGreaterEqual(best_epochs[kind], 2)
            best_recon = [f for f in snap[mine[0]] if "recon" in f]
            self.assertTrue(best_recon, f"best {kind} folder has no recon")

            # best epoch is copied to its epoch folder and to plots/
            epoch_files = snap[
                os.path.join(epochs_dir, f"epoch_{best_epochs[kind]:04d}")
            ]
            for name in best_recon:
                self.assertIn(name, epoch_files)
                self.assertIn(name, snap["."])

        # leftovers: not copied to plots/, and left untouched
        self.assertNotIn("stale_from_old_run.png", snap["."])
        self.assertNotIn("stale_same_run.png", snap["."])
        self.assertIn("stale_same_run.png", epoch1)
        for epoch in ("epoch_0001", "epoch_0002"):
            self.assertIn(
                "stale_from_old_run.png",
                snap[os.path.join("epochs", epoch)],
            )
        self.assertIn("old.png", snap["best_epoch0021"])
        self.assertIn("old.png", snap["best_epoch0005_250101-0000"])

        # validation log lines
        lines = self._val_lines(cm)
        self.assertEqual(len(lines), 3, lines)
        self.assertIn("Epoch: [1/3]", lines[0])
        self.assertNotIn("| Best:", lines[0])
        self.assertNotIn("| Best:", lines[1])
        self.assertIn("Epoch: [3/3]", lines[2])
        self.assertRegex(lines[2], r"\| Best: \d+ \d+% acc, 2 loss$")

    def test_vis_epochs_freq_without_best(self):
        """Run B: freq honoured, copy to plots/, best_start > epochs."""
        rid = self.RUN_ID
        with self.assertLogs(level="INFO") as cm:
            snap = dict(
                self._run(
                    epochs=2,
                    vis_epochs=True,
                    vis_best=True,
                    vis_rec=True,
                    freq_rec=2,
                    vis_acc=False,
                    vis_emb=False,
                    vis_dynamic=False,
                    vis_sim=False,
                    vis_int=False,
                    vis_dis=False,
                    vis_pos=False,
                )
            )
        epochs_dir = f"epochs_{rid}"
        self.assertNotIn(os.path.join(epochs_dir, "epoch_0001"), snap)
        self.assertIn(os.path.join(epochs_dir, "epoch_0002"), snap)
        recon = [
            f
            for f in snap[os.path.join(epochs_dir, "epoch_0002")]
            if "recon" in f
        ]
        self.assertTrue(recon, "no recon files in epoch_0002")
        for name in recon:
            self.assertIn(name, snap["."])
        self.assertFalse(any(r.startswith("best_epoch") for r in snap))
        self.assertFalse(any("| Best:" in m for m in self._val_lines(cm)))


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
