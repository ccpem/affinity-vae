import os
import pathlib
import tempfile
import unittest

import tests
from avae.config import AffinityConfig, load_config_params, save_config_params
from tests import testdata_mrc, testdata_npy


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        """Setup data and output directories."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = os.path.join(
            os.path.dirname(tests.__file__), "avae-test-config.yml"
        )
        self.data_local = {
            "datapath": os.path.dirname(testdata_mrc.__file__),
            "affinity": os.path.join(
                os.path.dirname(testdata_mrc.__file__), "affinity_fsc_10.csv"
            ),
            "classes": os.path.join(
                os.path.dirname(testdata_npy.__file__), "classes.csv"
            ),
            "split": 5,
            "epochs": 150,
            "vis_sim": True,
        }

        self.datapath_local_missing = {
            "affinity": os.path.join(
                os.path.dirname(testdata_mrc.__file__), "affinity_fsc_10.csv"
            ),
            "classes": os.path.join(
                os.path.dirname(testdata_npy.__file__), "classes.csv"
            ),
            "split": 5,
            "epochs": 150,
        }

        self.data_local_fail_wrong_label = {
            "datapath": os.path.dirname(testdata_mrc.__file__),
            "affinity": os.path.join(
                os.path.dirname(testdata_mrc.__file__), "affinity_fsc_10.csv"
            ),
            "classes": os.path.join(
                os.path.dirname(testdata_npy.__file__), "classes.csv"
            ),
            "collect_meta": True,
            "batch": "25",
        }
        self.data_local_fail_wrong_type = {
            "batch": "25",
            "split": "test",
        }

        self.default_model = AffinityConfig(
            datapath=self.data_local["datapath"]
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_config_defaults(self):
        self.assertTrue(AffinityConfig.model_config["validate_default"])
        self.assertEqual(self.default_model.model, "cnn")

    def test_legacy_models_only_allow_existing_checkpoints(self):
        for model in ("u", "a", "b"):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    AffinityConfig(
                        datapath=self.data_local["datapath"], model=model
                    )

                self.assertEqual(
                    AffinityConfig(
                        datapath=self.data_local["datapath"],
                        model=model,
                        evaluate=True,
                    ).model,
                    model,
                )

                self.assertEqual(
                    AffinityConfig(
                        datapath=self.data_local["datapath"],
                        model=model,
                        restart=True,
                    ).model,
                    model,
                )

    def test_save_config(self):
        output_dir = os.path.join(self.temp_dir.name, "configs")

        output_path = save_config_params(self.default_model, output_dir)

        self.assertTrue(output_path.is_file())
        self.assertEqual(output_path.parent, pathlib.Path(output_dir))
        saved_model = load_config_params(config_file=output_path)
        self.assertEqual(saved_model, self.default_model)

    def test_validate_config(self):

        data = load_config_params(
            config_file=self.config, local_args=self.data_local
        )
        self.assertEqual(
            len(data.model_dump().items()),
            len(self.default_model.model_dump().items()),
        )
        self.assertEqual(str(data.affinity), self.data_local['affinity'])
        self.assertNotEqual(
            str(data.affinity), str(self.default_model.affinity)
        )
        self.assertEqual(data.split, self.data_local['split'])
        self.assertEqual(data.new_out, self.default_model.new_out)

        data_config_only = load_config_params(config_file=self.config)
        self.assertEqual(data_config_only.epochs, 1000)

        data_cli_frequency = load_config_params(
            config_file=self.config,
            local_args={"freq_all": 5, "freq_acc": 3},
        )
        self.assertEqual(data_cli_frequency.freq_acc, 3)
        self.assertEqual(data_cli_frequency.freq_rec, 5)

        data_local_data_only = load_config_params(local_args=self.data_local)
        self.assertEqual(data_local_data_only.epochs, 150)
        self.assertEqual(
            data_local_data_only.vis_all, self.default_model.vis_all
        )

    def test_validate_config_fail(self):

        with self.assertRaises(RuntimeError):
            load_config_params(local_args=self.data_local_fail_wrong_label)

        with self.assertRaises(RuntimeError):
            load_config_params(local_args=self.data_local_fail_wrong_type)

        with self.assertRaises(RuntimeError):
            load_config_params(local_args=self.datapath_local_missing)

        # wrong input for classifier
        self.data_local['classifier'] = 'LS'
        with self.assertRaises(RuntimeError):
            load_config_params(local_args=self.data_local)
