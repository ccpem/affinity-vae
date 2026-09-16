import abc
import logging

import numpy as np
import torch
import torch.nn as nn

from avae.base import AbstractAffinityVAE
from avae.decoders.decoders import Decoder, DecoderA, DecoderB
from avae.decoders.differentiable import GaussianSplatDecoder
from avae.encoders.encoders import Encoder, EncoderA, EncoderB

from .base import SpatialDims, dims_after_pooling, set_layer_dim


def model_setup(
    model_type: str,
    input_shape: tuple,
    channels: int,
    depth: int,
    latent_dims: int,
    pose_dims: int,
    bnorm_encoder: bool,
    bnorm_decoder: bool,
    n_splats: int,
    gsd_conv_layers: int,
    device: torch.device,
    filters: list | None = None,
):
    """Create the AffinityVAE model.

    Parameters
    ----------
    model_type : str
        The model to create: cnn or gsd. The u, a, and b aliases are retained
        for legacy checkpoints.
    input_shape : tuple
        The size of the input.
    channels : int
        The number of channels in the model.
    depth : int
        The depth of the model.
    latent_dims : int
        The number of latent dimensions.
    pose_dims : int
        The number of pose dimensions.
    bnorm_encoder : bool
        Whether to use batch normalisation in the encoder.
    bnorm_decoder : bool
        Whether to use batch normalisation in the decoder.
    n_splats : int
        The number of splats in the Gaussian Splat Decoder.
    gsd_conv_layers : int
        The number of convolutional layers in the Gaussian Splat Decoder.
    device : torch.device
        The device to use for training and inference.
    filters : list or None
        The filters to use in the model.

    """
    logging.info("\n")
    logging.info("############################################### MODEL")
    logging.info(f"Setting up the model...")

    if filters is not None:
        filters = np.asarray(filters, dtype=np.int64)

    if model_type == "a":
        encoder = EncoderA(
            input_shape,
            channels,
            depth,
            latent_dims,
            pose_dims,
            bnorm=bnorm_encoder,
        )
        decoder = DecoderA(
            input_shape,
            channels,
            depth,
            latent_dims,
            pose_dims,
            bnorm=bnorm_decoder,
        )
    elif model_type == "b":
        encoder = EncoderB(
            input_shape, channels, depth, latent_dims, pose_dims
        )
        decoder = DecoderB(
            input_shape, channels, depth, latent_dims, pose_dims
        )
    elif model_type in {"cnn", "u"}:
        encoder = Encoder(
            input_shape=input_shape,
            capacity=channels,
            filters=filters,
            depth=depth,
            latent_dims=latent_dims,
            pose_dims=pose_dims,
            bnorm=bnorm_encoder,
        )
        decoder = Decoder(
            input_shape=input_shape,
            capacity=channels,
            filters=filters,
            depth=depth,
            latent_dims=latent_dims,
            pose_dims=pose_dims,
            bnorm=bnorm_decoder,
        )
    elif model_type == "gsd":
        encoder = Encoder(
            input_shape=input_shape,
            capacity=channels,
            filters=filters,
            depth=depth,
            latent_dims=latent_dims,
            pose_dims=pose_dims,
            bnorm=bnorm_encoder,
        )
        decoder = GaussianSplatDecoder(
            input_shape,
            n_splats=n_splats,
            latent_dims=latent_dims,
            output_channels=gsd_conv_layers,
            device=device,
            pose_dims=pose_dims,
        )
    else:
        raise ValueError(
            "Invalid model type",
            model_type,
            "must be one of: cnn or gsd",
        )

    vae = AffinityVAE(encoder, decoder)

    logging.info(vae)

    return vae


# Core implementation of the AffinityVAE
class AffinityVAE(AbstractAffinityVAE):
    def __init__(self, encoder, decoder):
        super(AffinityVAE, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.pose != self.decoder.pose:
            logging.error("Encoder and decoder pose must be the same.")
            raise RuntimeError("Encoder and decoder pose must be the same.")

        self.pose = self.encoder.pose

    def forward(self, x):
        # encode
        if self.pose:
            latent_mu, latent_logvar, latent_pose = self.encoder(x)
        else:
            latent_mu, latent_logvar = self.encoder(x)
            latent_pose = None
        # reparametrise
        latent = self.reparametrise(latent_mu, latent_logvar)
        # decode
        x_recon = self.decoder(latent, latent_pose)  # pose set to None if pd=0

        return x_recon, latent_mu, latent_logvar, latent, latent_pose

    def reparametrise(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
