import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner

import run as run_module


class RunOutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.orig_cwd = os.getcwd()
        self.orig_handlers = list(logging.getLogger().handlers)

    def tearDown(self) -> None:
        os.chdir(self.orig_cwd)
        root = logging.getLogger()
        for handler in list(root.handlers):
            if handler not in self.orig_handlers and isinstance(
                handler, logging.FileHandler
            ):
                handler.close()
                root.removeHandler(handler)
        self.temp_dir.cleanup()

    def test_out_creates_dir_and_changes_cwd(self):
        out = os.path.join(self.temp_dir.name, "nested", "out")
        datapath = os.path.join(self.temp_dir.name, "data")
        os.makedirs(datapath)
        args = ["--out", out, "--datapath", datapath]
        # load_config_params inspects sys.argv to find CLI-supplied options
        with mock.patch.object(
            run_module, "run_pipeline"
        ) as pipeline, mock.patch.object(sys, "argv", ["run.py"] + args):
            result = CliRunner().invoke(run_module.run, args)
            cwd = os.getcwd()

        self.assertIsNone(result.exception)
        pipeline.assert_called_once()
        self.assertEqual(os.path.realpath(cwd), os.path.realpath(out))
        self.assertTrue(os.path.isdir(os.path.join(out, "logs")))

    def test_out_with_new_out_creates_results_dir_inside_out(self):
        out = os.path.join(self.temp_dir.name, "out")
        datapath = os.path.join(self.temp_dir.name, "data")
        os.makedirs(datapath)
        args = ["--out", out, "--new_out", "--datapath", datapath]
        with mock.patch.object(
            run_module, "run_pipeline"
        ) as pipeline, mock.patch.object(sys, "argv", ["run.py"] + args):
            result = CliRunner().invoke(run_module.run, args)
            cwd = os.getcwd()

        self.assertIsNone(result.exception)
        pipeline.assert_called_once()
        results = [d for d in os.listdir(out) if d.startswith("results_")]
        self.assertEqual(len(results), 1)
        run_dir = os.path.join(out, results[0])
        self.assertEqual(os.path.realpath(cwd), os.path.realpath(run_dir))
        self.assertTrue(os.path.isdir(os.path.join(run_dir, "logs")))
