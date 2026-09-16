import typing

import torch

from avae.decoders.base import AbstractDecoder
from avae.decoders.spatial import (
    CartesianAxes,
    SpatialDims,
    axis_angle_to_quaternion,
    quaternion_to_rotation_matrix,
)


class GaussianSplatRenderer(torch.nn.Module):
    """Perform gaussian splatting."""

    def __init__(
        self,
        shape: typing.Tuple[int],
        *,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self._shape = shape
        self._ndim = len(shape)
        self.device = device

        if len(shape) not in (SpatialDims.TWO, SpatialDims.THREE):
            raise ValueError("Only 2D or 3D rotations are currently supported")

        grids = torch.meshgrid(
            *[torch.linspace(-1, 1, sz) for sz in shape],
            indexing="xy",
        )

        # add all zeros for z- if we have a 2d grid
        if len(shape) == SpatialDims.TWO:
            grids += (
                torch.zeros_like(
                    grids[0],
                ),
            )

        self.coords = (
            torch.stack([torch.ravel(grid) for grid in grids], axis=0)
            .transpose(0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

    def forward(
        self,
        splats: torch.Tensor,
        weights: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        splat_sigma_range: typing.Tuple[float, float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """Render the Gaussian splats in an image volume.

        Parameters
        ----------
        splats : tensor
            An (N, D, 3) tensor specifying the X,Y,Z coordinates of the D
            gaussians for the minibatch of N images.
        weights : tensor
            An (N, D, 1) tensor specifying the weights of the D gaussians for
            the minibatch of N images.In the range of 0 to 1.
        sigmas : tensor
            An (N, D, 1) tensor specifying the standard deviations of the D
            gaussians for the minibatch of N images. In the range of 0 to 1.
        splat_sigma_range : tuple
            The minimum and maximum values for sigma. Final sigma is calculated
            as sigmas * (max_sigma - min_sigma) + min_sigma.

        Returns
        -------
        x : tensor
            The rendered image volume.

        Notes
        -----
        This isn't very memory efficient since the GMM is evaluated for every
        voxel in the output. This means an (N, M*M*M, D) matrix, where M is the
        dimensions of the image volume (e.g. 32x32x32) and D is the number of
        gaussians (e.g. 1024). This leads to a matrix of 32 x 32768 x 1024
        for a minibatch of 32 volumes.
        """
        if torch.cuda.device_count() == 0 and self.device.type == 'cuda':
            self.device = torch.device('cpu')

        # scale the sigma values
        min_sigma, max_sigma = splat_sigma_range
        sigmas = sigmas * (max_sigma - min_sigma) + min_sigma

        # transpose keeping batch intact
        # coords_t = torch.swapaxes(self.coords, 1, 2)
        splats = splats.to(self.device)
        splats_t = torch.swapaxes(splats, 1, 2)
        splats_t = splats_t.to(self.device)
        self.coords = self.coords.to(
            self.device
        )  # calculate D^2 for all combinations of voxel and gaussian
        D_squared = torch.sum(
            self.coords[:, :, None, :] ** 2 + splats_t[:, None, :, :] ** 2,
            axis=-1,
        ) - 2 * torch.matmul(self.coords, splats)
        # scale the gaussians
        sigmas = 2.0 * sigmas[:, None, :] ** 2
        sigmas = sigmas.to(self.device)
        weights = weights.to(self.device)

        # now splat the gaussians
        x = torch.sum(
            weights[:, None, :] * torch.exp(-D_squared / sigmas),
            axis=-1,
        )

        return x.reshape((-1, *self._shape)).unsqueeze(1)


class SoftStep(torch.nn.Module):
    """Soft (differentiable) step function in the range of 0-1."""

    def __init__(self, *, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-self.k * x))


class GaussianSplatDecoder(AbstractDecoder):
    """Differentiable Gaussian splat decoder.

    Parameters
    ----------
    shape : tuple
        A tuple describing the output shape of the image data. Can be 2- or 3-
        dimensional. For example: (32, 32, 32)
    n_splats : int
        The number of Gaussians in the mixture model.
    latent_dims : int
        The dimensions of the latent representation.
    output_channels : int, optional
        The number of output channels in the final image volume. If not
        supplied, this will default to 1. If it is supplied, additional
        convolutions are applied to the GMM model.
    splat_sigma_range : tuple[float, float]
        The minimum and maximum sigma values for each splat. Useful to control
        the resolution of the final render.
    default_axis : CartesianAxes
        A default cartesian axis to use for rotation if the pose is provided by
        a rotation only. Default is Z, equivalent to a typical image rotation
        about the central axis.

    Notes
    -----
    Takes the latent code and pose estimate to generate a planar or volumetric
    image.  The code is used to position N symmetric gaussians in the image
    volume which are then rotated by an explicit rotation transform. These are
    rendered as an image by evaluating the list of gaussians as a GMM.

    The renderer is differentiable and can therefore be used during training.
    """

    def __init__(
        self,
        shape: typing.Tuple[int],
        *,
        n_splats: int = 128,
        latent_dims: int = 8,
        output_channels: int = 0,
        splat_sigma_range: typing.Tuple[float, float] = (0.02, 0.1),
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
        pose_dims: int,
    ):
        super().__init__()

        self._device = device
        # centroids should be in the range of (-1, 1)
        self.centroids = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats * 3),
            torch.nn.Tanh(),
        )
        # weights are effectively whether a splat is used or not
        # use a soft step function to make this `binary` (but differentiable)
        # NOTE(arl): not sure if this really makes any difference
        self.weights = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Tanh(),
            SoftStep(k=10.0),
        )
        # sigma ends up being scaled by `splat_sigma_range`
        self.sigmas = torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_splats),
            torch.nn.Sigmoid(),
        )
        # now set up the differentiable renderer
        self.configure_renderer(
            shape,
            splat_sigma_range=splat_sigma_range,
            default_axis=default_axis,
            device=device,
        )

        self._device = device
        self._ndim = len(shape)
        self._output_channels = output_channels
        """Decode the splats to retrieve the coordinates, weights and sigmas."""
        if pose_dims not in (1, 4):
            raise ValueError(
                "Pose needs to be either a single angle rotation about the "
                "`default_axis` or a full angle-axis representation in 3D. "
            )
        self.pose = not (pose_dims == 0)

        # add a final convolutional decoder to generate an image if the number
        # of output channels has been provided
        if output_channels != 0:
            conv = (
                torch.nn.Conv2d
                if self._ndim == SpatialDims.TWO
                else torch.nn.Conv3d
            )
            self._decoder = torch.nn.Sequential(
                conv(1, 32, 3, padding="same"),
                torch.nn.ReLU(),
                conv(32, 32, 3, padding="same"),
                torch.nn.ReLU(),
                conv(32, output_channels, 3, padding="same"),
            )

    def configure_renderer(
        self,
        shape: typing.Tuple[int],
        *,
        splat_sigma_range: typing.Tuple[float, float] = (0.02, 0.1),
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Reconfigure the renderer.

        Notes
        -----
        This might be useful to do once a model is trained. For example, one
        could change the resolution of the rendered image by changing the
        `shape` of the output.
        """
        self._shape = shape
        self._default_axis = default_axis.as_tensor()
        self._splatter = GaussianSplatRenderer(
            shape,
            device=device,
        )
        self._splat_sigma_range = splat_sigma_range

    def decode_splats(
        self, z: torch.Tensor, pose: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # predict the centroids for the splats
        splats = self.centroids(z).view(z.shape[0], 3, -1)
        weights = self.weights(z)
        sigmas = self.sigmas(z)

        # get the batch size
        batch_size = z.shape[0]

        # in the case where the encoded pose only has one dimension, we need to
        # use the pose as a rotation about the z-axis
        if pose.shape[-1] == 1:
            pose = torch.concat(
                [
                    pose,
                    torch.tile(self._default_axis, (batch_size, 1)),
                ],
                axis=-1,
            )

        # convert axis angles to quaternions
        assert pose.shape[-1] == 4, pose.shape
        quaternions = axis_angle_to_quaternion(pose, normalize=True)

        # convert the quaternions to rotation matrices
        rotation_matrices = quaternion_to_rotation_matrix(quaternions)

        # rotate the 3D points using the rotation matrices
        rotated_splats = torch.matmul(
            rotation_matrices,
            splats,
        )

        # use only the required spatial dimensions (batch, ndim, samples)
        # rotated_splats = rotated_splats[:, : self._ndim, :]

        return rotated_splats, weights, sigmas

    def forward(
        self,
        z: torch.Tensor,
        pose: torch.Tensor,
        *,
        use_final_convolution: bool = True,
    ) -> torch.Tensor:
        """Decode the latents to an image volume given an explicit transform.

        Parameters
        ----------
        z : tensor
            An (N, D) tensor specifying the D dimensional latent encodings for
            the minibatch of N images.
        pose : tensor
            An (N, 1 | 4) tensor specifying the pose in terms of a single
            rotation (assumed around the z-axis) or a full axis-angle rotation.
        use_final_convolution: bool
            Whether to apply the final convolutional layers to recover the image.
            This can be useful to inspect the underlying structure in a trained
            model.

        Returns
        -------
        x : tensor
            The decoded image from the latents and pose.
        """

        # decode the splats from the latents and pose
        splats, weights, sigmas = self.decode_splats(z, pose)

        x = self._splatter(
            splats, weights, sigmas, splat_sigma_range=self._splat_sigma_range
        )
        # if we're doing a final convolution, do it here
        if (
            self._output_channels is not None
            and self._output_channels != 0
            and use_final_convolution
        ):
            x = self._decoder(x)
        return x
