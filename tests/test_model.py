import unittest

import pytest
import torch

from avae.decoders.decoders import Decoder, DecoderA, DecoderB
from avae.decoders.differentiable import GaussianSplatDecoder
from avae.encoders.encoders import Encoder, EncoderA, EncoderB
from avae.models import AffinityVAE as avae
from avae.models import model_setup


class ModelInstanceTest(unittest.TestCase):
    def setUp(self) -> None:

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

        self.encoder_2d = Encoder(
            capacity=8,
            depth=4,
            input_shape=(64, 64),
            latent_dims=16,
            pose_dims=3,
        )
        self.decoder_2d = Decoder(
            capacity=8,
            depth=4,
            input_shape=(64, 64),
            latent_dims=16,
            pose_dims=3,
        )

        self.vae_2d = avae(self.encoder_2d, self.decoder_2d)

        self.x = torch.randn(14, 1, 64, 64, 64)

    def test_model_instance(self):
        """Test instantiation of the model."""

        assert isinstance(self.vae, avae)

    def test_model_3D(self):
        """Test that model is instantiated with 3D convolutions."""
        assert isinstance(self.vae.encoder.conv_enc[0], torch.nn.Conv3d)
        assert isinstance(
            self.vae.decoder.conv_dec[-1], torch.nn.ConvTranspose3d
        )

    def test_model_2D(self):
        """Test that model is instantiated with 2D convolutions."""

        assert isinstance(self.vae_2d.encoder.conv_enc[0], torch.nn.Conv2d)
        assert isinstance(
            self.vae_2d.decoder.conv_dec[-1], torch.nn.ConvTranspose2d
        )

    def test_model_forward(self):

        y = self.vae(self.x)

        self.assertEqual(self.x.shape, y[0].shape)
        self.assertEqual(torch.randn(14, 16).shape, y[1].shape)
        self.assertEqual(torch.randn(14, 16).shape, y[2].shape)
        self.assertEqual(torch.randn(14, 16).shape, y[3].shape)
        self.assertEqual(torch.randn(14, 3).shape, y[4].shape)

    def test_model_reproducibility(self):

        y1 = self.vae(self.x)
        y2 = self.vae(self.x)

        self.assertEqual(y1[0].all(), y2[0].all())
        self.assertEqual(y1[1].all(), y2[1].all())
        self.assertEqual(y1[2].all(), y2[2].all())
        self.assertEqual(y1[3].all(), y2[3].all())
        self.assertEqual(y1[4].all(), y2[4].all())


class ModelParamsTest(unittest.TestCase):
    def setUp(self):
        self.input_shape = (64, 64)
        self.input = torch.randn(14, 1, 64, 64)
        self.capacity_params = [9, 64]
        self.depth_params = [2, 4]
        self.filters_params = [[8, 16], [5, 9]]
        self.latent_params = [5, 32]
        self.pose_params = [(0, 2), (1, 3)]
        self.bnorm = []

    def test_model_exceptions(self):
        # no parameters
        with pytest.raises(TypeError) as errinfo:
            Encoder()
        assert (
            str(errinfo.value)
            == "Encoder.__init__() missing 1 required positional argument: 'input_shape'"
        )
        with pytest.raises(TypeError) as errinfo:
            Decoder()
        assert (
            str(errinfo.value)
            == "Decoder.__init__() missing 1 required positional argument: 'input_shape'"
        )

        # no capacity no filters scenario
        with pytest.raises(RuntimeError) as errinfo:
            Encoder(
                input_shape=self.input_shape,
            )
        assert (
            str(errinfo.value)
            == "Both 'capacity' and 'filters' parameters are None. Please pass one or the other to instantiate the network."
        )
        with pytest.raises(RuntimeError) as errinfo:
            Decoder(
                input_shape=self.input_shape,
            )
        assert (
            str(errinfo.value)
            == "Both 'capacity' and 'filters' parameters are None. Please pass one or the other to instantiate the network."
        )

        # capacity but no depth scenario - no longer needed since depth is now default to 4

        # 0 or negative value in filters
        with pytest.raises(RuntimeError) as errinfo:
            Encoder(
                filters=[0, -1],
                input_shape=self.input_shape,
            )
        assert (
            str(errinfo.value)
            == "Filter list cannot contain zeros or negative values."
        )
        with pytest.raises(RuntimeError) as errinfo:
            Decoder(
                filters=[0, -1],
                input_shape=self.input_shape,
            )
        assert (
            str(errinfo.value)
            == "Filter list cannot contain zeros or negative values."
        )

        # input too small for depth with filters
        with pytest.raises(AssertionError) as errinfo:
            Encoder(
                filters=[2, 4, 16, 32],
                input_shape=(8, 8),
            )
        assert (
            str(errinfo.value)
            == "Input size not compatible with --depth. Input must be divisible by 16."
        )
        with pytest.raises(AssertionError) as errinfo:
            Decoder(
                filters=[2, 4, 16, 32],
                input_shape=(8, 8),
            )
        assert (
            str(errinfo.value)
            == "Input size not compatible with --depth. Input must be divisible by 16."
        )

        # input too small for depth with capacity
        with pytest.raises(AssertionError) as errinfo:
            Encoder(
                capacity=8,
                depth=4,
                input_shape=(8, 8),
            )
        assert (
            str(errinfo.value)
            == "Input size not compatible with --depth. Input must be divisible by 16."
        )
        with pytest.raises(AssertionError) as errinfo:
            Decoder(
                capacity=8,
                depth=4,
                input_shape=(8, 8),
            )
        assert (
            str(errinfo.value)
            == "Input size not compatible with --depth. Input must be divisible by 16."
        )

        # negative depth
        with pytest.raises(RuntimeError) as errinfo:
            Encoder(
                capacity=8,
                depth=-1,
                input_shape=self.input_shape,
            )
        assert (
            str(errinfo.value)
            == "Parameter 'depth' cannot be a negative value."
        )
        with pytest.raises(RuntimeError) as errinfo:
            Decoder(
                capacity=8,
                depth=-1,
                input_shape=self.input_shape,
            )
        assert (
            str(errinfo.value)
            == "Parameter 'depth' cannot be a negative value."
        )

        # non-positive latent dims
        with pytest.raises(RuntimeError) as errinfo:
            Encoder(
                capacity=8,
                depth=4,
                latent_dims=0,
                input_shape=self.input_shape,
            )
        assert (
            str(errinfo.value)
            == "Parameter 'latent_dims' must be non-zero and positive."
        )
        with pytest.raises(RuntimeError) as errinfo:
            Decoder(
                capacity=8,
                depth=4,
                latent_dims=-1,
                input_shape=self.input_shape,
            )
        assert (
            str(errinfo.value)
            == "Parameter 'latent_dims' must be non-zero and positive."
        )

    def test_capacity(self):

        for dim in self.capacity_params:

            enc = Encoder(
                capacity=dim,
                depth=1,
                input_shape=self.input_shape,
                latent_dims=16,
                pose_dims=3,
            )
            dec = Decoder(
                capacity=dim,
                depth=1,
                input_shape=self.input_shape,
                latent_dims=16,
                pose_dims=3,
            )
            model = avae(enc, dec)
            output = model(self.input)[0]
            self.assertEqual(output.shape, self.input.shape)

    def test_depth(self):

        # no conv
        input = torch.randn(14, 1, 128)
        enc = Encoder(
            input_shape=(128,),
            depth=0,
        )
        dec = Decoder(
            input_shape=(128,),
            depth=0,
        )
        model = avae(enc, dec)
        output = model(input)[0]
        self.assertEqual(output.shape, input.shape)

        for dim in self.depth_params:

            enc = Encoder(
                capacity=8,
                depth=dim,
                input_shape=self.input_shape,
                latent_dims=16,
                pose_dims=3,
            )
            dec = Decoder(
                capacity=8,
                depth=dim,
                input_shape=self.input_shape,
                latent_dims=16,
                pose_dims=3,
            )
            model = avae(enc, dec)
            output = model(self.input)[0]
            self.assertEqual(output.shape, self.input.shape)

    def test_filters(self):

        for dim in self.filters_params:

            enc = Encoder(
                filters=dim,
                input_shape=self.input_shape,
                latent_dims=16,
                pose_dims=3,
            )
            dec = Decoder(
                filters=dim,
                input_shape=self.input_shape,
                latent_dims=16,
                pose_dims=3,
            )
            model = avae(enc, dec)
            output = model(self.input)[0]
            self.assertEqual(output.shape, self.input.shape)

        # uneven filters
        enc = Encoder(
            filters=dim,
            input_shape=self.input_shape,
            latent_dims=16,
            pose_dims=3,
        )
        dec = Decoder(
            filters=[16],
            input_shape=self.input_shape,
            latent_dims=16,
            pose_dims=3,
        )
        model = avae(enc, dec)
        output = model(self.input)[0]
        self.assertEqual(output.shape, self.input.shape)

    def test_latents(self):

        for dim in self.latent_params:

            enc = Encoder(
                filters=[8, 16],
                input_shape=self.input_shape,
                latent_dims=dim,
                pose_dims=3,
            )
            dec = Decoder(
                filters=[8, 16],
                input_shape=self.input_shape,
                latent_dims=dim,
                pose_dims=3,
            )
            model = avae(enc, dec)
            output = model(self.input)
            self.assertEqual(output[0].shape, self.input.shape)
            self.assertEqual(output[1].shape[1], dim)

    def test_pose(self):

        for dim, exp in self.pose_params:
            enc = Encoder(
                capacity=8,
                depth=4,
                input_shape=self.input_shape,
                latent_dims=16,
                pose_dims=dim,
                bnorm=True,
            )
            output = enc(self.input)
            self.assertEqual(len(output), exp)

    def test_batchnorm(self):

        enc = Encoder(
            filters=[8, 16], input_shape=self.input_shape, bnorm=True
        )
        dec = Decoder(
            filters=[8, 16], input_shape=self.input_shape, bnorm=False
        )
        model = avae(enc, dec)
        output = model(self.input)
        self.assertEqual(output[0].shape, self.input.shape)


class ModelSetupTest(unittest.TestCase):
    def setup_model(self, model_type, pose_dims=0, filters=None):
        return model_setup(
            model_type=model_type,
            input_shape=(8, 8),
            channels=2,
            depth=2,
            latent_dims=3,
            pose_dims=pose_dims,
            bnorm_encoder=True,
            bnorm_decoder=True,
            n_splats=4,
            gsd_conv_layers=0,
            device=torch.device("cpu"),
            filters=filters,
        )

    def test_cnn_custom_filters_and_batchnorm(self):
        model = self.setup_model("cnn", filters=[2, 4])
        inputs = torch.randn(2, 1, 8, 8)

        self.assertEqual(model.encoder.filters.tolist(), [2, 4])
        self.assertEqual(model.decoder.filters.tolist(), [2, 4])
        self.assertEqual(len(model.encoder.norm_enc), 2)
        self.assertEqual(len(model.decoder.norm_dec), 1)
        self.assertEqual(model(inputs)[0].shape, inputs.shape)

    def test_legacy_model_factories_remain_available(self):
        legacy_types = {
            "u": (Encoder, Decoder),
            "a": (EncoderA, DecoderA),
            "b": (EncoderB, DecoderB),
        }

        for model_type, expected_types in legacy_types.items():
            with self.subTest(model=model_type):
                model = self.setup_model(model_type, pose_dims=1)
                self.assertIsInstance(model.encoder, expected_types[0])
                self.assertIsInstance(model.decoder, expected_types[1])

    def test_gsd_uses_generic_encoder(self):
        model = self.setup_model("gsd", pose_dims=1, filters=[2, 4])

        self.assertIsInstance(model.encoder, Encoder)
        self.assertIsInstance(model.decoder, GaussianSplatDecoder)
