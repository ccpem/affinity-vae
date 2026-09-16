import enum
import typing

import torch


class SpatialDims(enum.IntEnum):
    TWO = 2
    THREE = 3


class CartesianAxes(enum.Enum):
    """Set of Cartesian axes as 3D vectors."""

    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.float32)


def axis_angle_to_quaternion(
    axis_angles: torch.Tensor, *, normalize: bool = True
) -> torch.Tensor:
    """Convert an axis angle to a rotation quaternion representation.

    Parameters
    ----------
    axis_angles : tensor
        An (N, 4) tensor specifying the axis angle representations for a batch.
        The order is (theta, x_hat, y_hat, z_hat).
    normalize : bool, default = True
        Whether to normalize the axes to a unit vector (recommended)

    Returns
    -------
    quaternions : tensor
        A (N, 4) tensor specifying the quaternion representations of the axis
        angles. The order is (q0, q1, q2, q3) where q0 is the real part, and
        (q1, q2, a3) are the imaginary parts.

    Notes
    -----

    """
    theta = axis_angles[:, 0].unsqueeze(-1)
    axis = axis_angles[:, 1:]

    if axis.shape[-1] not in (SpatialDims.THREE,):
        raise ValueError("Axis must be specified in three dimensions.")

    axis = torch.nn.functional.normalize(axis, dim=1) if normalize else axis

    real = torch.cos(theta / 2)
    imag = axis * torch.sin(theta / 2)

    quaternion = torch.concat([real, imag], axis=-1)

    return quaternion


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternion forms to rotation matrices.

    Parameters
    ----------
    quaternions : tensor
        A (N, 4) tensor specifying the quaternion representations of the axis
        angles. The order is (q0, q1, q2, q3).

    Returns
    -------
    rotation_matrices : tensor
        An (N, 3, 3) tensor specifying rotation matrices for each quaternion.
    """
    # extract the real and imaginary parts of the quaternions
    q0, q1, q2, q3 = torch.unbind(quaternions, dim=-1)

    # calculate the components of the rotation matrix
    R00 = q0**2 + q1**2 - q2**2 - q3**2
    R01 = 2 * (q1 * q2 - q0 * q3)
    R02 = 2 * (q1 * q3 + q0 * q2)
    R10 = 2 * (q1 * q2 + q0 * q3)
    R11 = q0**2 - q1**2 + q2**2 - q3**2
    R12 = 2 * (q2 * q3 - q0 * q1)
    R20 = 2 * (q1 * q3 - q0 * q2)
    R21 = 2 * (q2 * q3 + q0 * q1)
    R22 = q0**2 - q1**2 - q2**2 + q3**2

    # stack the components into the rotation matrix
    rotation_matrices = torch.stack(
        [
            torch.stack([R00, R01, R02], axis=-1),
            torch.stack([R10, R11, R12], axis=-1),
            torch.stack([R20, R21, R22], axis=-1),
        ],
        axis=-1,
    )

    return rotation_matrices


class RotatedCoordinates(torch.nn.Module):
    """Creates a homogeneous grid of rotated coordinates.

    Parameters
    ----------
    shape : tuple
        A tuple describing the output shape of the image data. Can be 2- or 3-
        dimensional.
    default_axis : CartesianAxes
        A default cartesian axis to use for rotation if the pose is provided by
        a rotation only. Default is Z, equivalent to a typical image rotation
        about the central axis.

    Notes
    -----
    Uses a quaternion representation:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    """

    def __init__(
        self,
        shape: typing.Tuple[int],
        *,
        default_axis: CartesianAxes = CartesianAxes.Z,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

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
            .unsqueeze(0)
            .to(device)
        )

        self.grids = torch.stack(grids, axis=0).unsqueeze(0).to(device)
        self._shape = shape
        self._ndim = len(shape)
        # self._default_axis = torch.tensor(default_axis.value, dtype=torch.float32)
        self._default_axis = default_axis.as_tensor()

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """Forward pass of spatial coordinate rotation.

        Parameters
        ----------
        pose : tensor
            An (N, D) tensor describing the angle and axis to rotate each grid
            by. D can either be 1 (i.e. just angle, assuming z-axis) or 4 (i.e.
            the angle and an axis to rotate around).

        Returns
        -------
        rotated_grids : tensor
            An (N, D, H, W) tensor, equivalent to a rotated meshgrid operation,
            where D is the axis dimension.
        """
        batch_size = pose.shape[0]

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
        # NOTE(arl): we should probably use rotation matrices OR quaternions
        # converting between them is not necessary
        rotation_matrices = quaternion_to_rotation_matrix(quaternions)

        # rotate the 3D points using the rotation matrices
        rotated_coords = torch.matmul(
            rotation_matrices,
            self.coords,
        )
        # use only the required spatial dimensions
        rotated_coords = rotated_coords[:, : self._ndim, :]

        # now create the equivalent of the rotated xy grid
        return rotated_coords.reshape((batch_size, self._ndim, *self._shape))
