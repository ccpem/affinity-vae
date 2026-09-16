import os
import unittest

import numpy as np
import pandas as pd
import torch

from avae.decoders.decoders import Decoder
from avae.encoders.encoders import Encoder
from avae.loss import AVAELoss
from avae.models import AffinityVAE as avae
from tests import testdata_mrc

torch.manual_seed(0)


class LossTest(unittest.TestCase):
    def setUp(self) -> None:
        """Test instantiation of the loss."""

        self._orig_dir = os.getcwd()
        self.test_data = os.path.dirname(testdata_mrc.__file__)
        os.chdir(self.test_data)

        self.affinity = pd.read_csv("affinity_fsc_10.csv").to_numpy(
            dtype=np.float32
        )
        self.loss = AVAELoss(
            device=torch.device("cpu"),
            beta=[1],
            gamma=[1],
            lookup_aff=self.affinity,
            recon_loss="MSE",
        )
        self.encoder_3d = Encoder(
            capacity=8,
            depth=4,
            input_shape=(64, 64, 64),
            latent_dims=16,
            pose_dims=3,
            bnorm=True,
        )

        self.decoder_3d = Decoder(
            capacity=8,
            depth=4,
            input_shape=(64, 64, 64),
            latent_dims=16,
            pose_dims=3,
        )

        self.vae = avae(self.encoder_3d, self.decoder_3d)

    def tearDown(self):
        os.chdir(self._orig_dir)

    def test_loss_instatiation(self):
        """Test instantiation of the loss."""

        assert isinstance(self.loss, AVAELoss)

    def test_loss(self):

        x = torch.randn(14, 1, 64, 64, 64)

        x_hat, lat_mu, lat_logvar, lat, lat_pose = self.vae(x)
        total_loss, recon_loss, kldivergence, affin_loss = self.loss(
            x,
            x_hat,
            lat_mu,
            lat_logvar,
            0,
            batch_aff=torch.ones(14, dtype=torch.int),
        )

        self.assertGreaterEqual(total_loss.detach().numpy().item(0), 1.1171)
        self.assertGreater(recon_loss.detach().numpy().item(0), 1)
        self.assertGreater(recon_loss, kldivergence)
        self.assertGreater(total_loss, affin_loss)

    def test_loss_bvae(self):

        x = torch.randn(14, 1, 64, 64, 64)

        self.loss = AVAELoss(
            torch.device("cpu"),
            [1],
            [0],
            lookup_aff=self.affinity,
            recon_loss="MSE",
        )

        x_hat, lat_mu, lat_logvar, _, _ = self.vae(x)
        _, _, _, affin_loss = self.loss(
            x,
            x_hat,
            lat_mu,
            lat_logvar,
            0,
            batch_aff=torch.ones(14, dtype=torch.int),
        )
        self.assertEqual(affin_loss, 0)
