"""
For each class representative, augument the data with
rotation and translation.
"""

import argparse
import os
import pathlib

import mrcfile
import numpy as np
import scipy.ndimage
import tqdm


def rescale(path_core, path_rescaled):
    path_rescaled = os.path.join(
        path_core.parent, path_core.stem + "_rescaled"
    )
    if not os.path.exists(path_rescaled):
        os.mkdir(path_rescaled)

    for x in list(path_core.iterdir()):
        print("Rescaling", x.stem)

        try:
            with mrcfile.open(pathlib.Path(x)) as mrc:
                nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
                new_x = scipy.ndimage.zoom(
                    mrc.data, (64 / nz, 64 / ny, 64 / nx)
                )
                new_x = (new_x - np.min(new_x)) / np.ptp(new_x)

                x_range, y_range, z_range = [
                    sum(
                        [
                            np.any(z_slice > 0.01)
                            for z_slice in np.rollaxis(new_x, dim)
                        ]
                    )
                    for dim in [2, 1, 0]
                ]

                new_x = np.where(new_x < 0.01, 0, new_x)

                with mrcfile.new(
                    os.path.join(path_rescaled, x.stem + ".mrc"),
                    overwrite=True,
                ) as mrc:
                    mrc.set_data(new_x)
                    mrc.header.label[1] = f"ratio_z_x={str(z_range/x_range)}"
                    mrc.header.label[2] = f"ratio_z_y={str(z_range/y_range)}"
                    mrc.header.nlabl = 3

        except ValueError:
            print(f"Problem with file: {x}")

        except IsADirectoryError:
            continue


def rotate_the_pokemino_1_axis(
    array, axes=(0, 1), theta=None, order=1, s=None
):
    """Rotates the Pokemino using scipy.ndimage.rotate"""

    assert (
        type(axes) == tuple
        and len(axes) == 2
        and all([type(i) == int for i in axes])
        and all([i in range(0, 3) for i in axes])
    ), "Incorrect axes parameter: pass a tuple of 2 axes."

    if not theta:
        np.random.seed(s)
        theta = np.random.choice([i for i in range(0, 360)])
    else:
        assert (isinstance(theta, int)) and theta in range(
            0, 360
        ), "Error: Pokemino3D.rotate_the_brick requires the value for theta in range <0, 360>."
    array = scipy.ndimage.rotate(
        array, angle=theta, axes=axes, order=order, reshape=False
    )

    return array, theta


def rotate_the_pokemino_3_axes(
    array, theta_x=None, theta_y=None, theta_z=None
):

    array, z_rot = rotate_the_pokemino_1_axis(
        array, axes=(1, 0), theta=theta_x
    )
    array, x_rot = rotate_the_pokemino_1_axis(
        array, axes=(2, 1), theta=theta_y
    )
    array, y_rot = rotate_the_pokemino_1_axis(
        array, axes=(0, 2), theta=theta_z
    )

    return array, x_rot, y_rot, z_rot


def shift_block(array, sh, axis=1):

    assert len(array.shape) == 3, "Pass a 3D array"
    assert type(sh) is int, "Pass a valid shift"
    assert sh < array.shape[axis], "You're asking for too much of a shift"

    if sh == 0:
        return array

    if sh < 0:
        if axis == 0:
            return np.concatenate(
                (array[-sh:, :, :], np.zeros_like(array[:-sh, :, :])), axis=0
            )
        if axis == 1:
            return np.concatenate(
                (array[:, -sh:, :], np.zeros_like(array[:, :-sh, :])), axis=1
            )
        if axis == 2:
            return np.concatenate(
                (array[:, :, -sh:], np.zeros_like(array[:, :, :-sh])), axis=2
            )

    if sh > 0:
        if axis == 0:
            return np.concatenate(
                (np.zeros_like(array[-sh:, :, :]), array[:-sh, :, :]), axis=0
            )
        if axis == 1:
            return np.concatenate(
                (np.zeros_like(array[:, -sh:, :]), array[:, :-sh, :]), axis=1
            )
        if axis == 2:
            return np.concatenate(
                (np.zeros_like(array[:, :, -sh:]), array[:, :, :-sh]), axis=2
            )


def add_label(mrc, param_name, param_value):

    label_count = mrc.header.nlabl
    mrc.header.label[label_count] = f"{param_name}={str(param_value)}"
    mrc.header.nlabl += 1


def read_rotate_translate_save_mrc(
    src_path, output_path, mrcs_list, n_pokeminos, nrot, ntrans
):

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for i in tqdm.tqdm(range(n_pokeminos)):

        protein = np.random.choice(mrcs)
        meta = []
        new_mrc = mrcfile.open(pathlib.Path(src_path, f"{protein}.mrc")).data

        if nrot == 1:
            new_mrc, theta_x = rotate_the_pokemino_1_axis(new_mrc)
            meta.append(str(theta_x))

        elif nrot == 3:
            new_mrc, theta_x, theta_y, theta_z = rotate_the_pokemino_3_axes(
                new_mrc
            )
            meta.append(str(theta_x))
            meta.append(str(theta_y))
            meta.append(str(theta_z))

        x_range, y_range, z_range = [
            [np.any(slice > 0.01) for slice in np.rollaxis(new_mrc, dim)]
            for dim in [0, 1, 2]
        ]

        if ntrans >= 1:
            # Shifting in x
            shift_x = int(
                np.random.choice(
                    range(
                        -x_range.index(True) + 1, x_range[::-1].index(True) - 1
                    )
                )
            )
            new_mrc = shift_block(new_mrc, shift_x, axis=0)
            meta.append(str(shift_x))

        if ntrans >= 2:
            # Shifting in y
            shift_y = int(
                np.random.choice(
                    range(
                        -y_range.index(True) + 1, y_range[::-1].index(True) - 1
                    )
                )
            )
            new_mrc = shift_block(new_mrc, shift_y, axis=1)
            meta.append(str(shift_y))

        if ntrans >= 3:
            # Shifting in z
            shift_z = int(
                np.random.choice(
                    range(
                        -z_range.index(True) + 1, z_range[::-1].index(True) - 1
                    )
                )
            )
            new_mrc = shift_block(new_mrc, shift_z, axis=2)
            meta.append(str(shift_z))

        meta = "-".join(meta)
        with mrcfile.new(
            Path(output_path / f"{protein}_{meta}.mrc"), overwrite=True
        ) as mrc:

            mrc.set_data(new_mrc)

            """if i == 0:
                    mrc.print_header()"""


parser = argparse.ArgumentParser()
parser.add_argument("--data")
args = parser.parse_args()
src_path = pathlib.Path(args.data)

# rescale
# output_path = Path(os.path.join(src_path.parent, src_path.stem+"_rescaled/"))
# rescale(src_path, output_path)

# random.seed(0)
# mrcs = random.sample([x.stem for x in list(src_path.iterdir())], k=20i)
mrcs = [x.stem for x in list(src_path.iterdir())]
n_pokeminos = 10000

# 1rot
# output_path = Path(os.path.join(src_path.parent, src_path.stem+"src_path.core+"_1rot/"))
# read_rotate_translate_save_mrc(src_path = src_path, output_path = output_path, mrcs_list = mrcs, n_pokeminos = n_pokeminos, nrot = 1, ntrans = 0)

# 3rot
output_path = pathlib.Path(
    os.path.join(src_path.parent, src_path.stem + "_3rot_10000/")
)
read_rotate_translate_save_mrc(
    src_path=src_path,
    output_path=output_path,
    mrcs_list=mrcs,
    n_pokeminos=n_pokeminos,
    nrot=3,
    ntrans=0,
)

# 1trans
# output_path = Path(os.path.join(src_path.parent, src_path.stem+"_1trans/"))
# read_rotate_translate_save_mrc(src_path = src_path, output_path = output_path, mrcs_list = mrcs, n_pokeminos = n_pokeminos, nrot = 0, ntrans = 1)

# 3trans
# output_path = Path(os.path.join(src_path.parent, src_path.stem+"_3trans/"))
# read_rotate_translate_save_mrc(src_path = src_path, output_path = output_path, mrcs_list = mrcs, n_pokeminos = n_pokeminos, nrot = 0, ntrans = 3)

# 1rot + 1trans
# output_path = Path(os.path.join(src_path.parent, src_path.stem+"_1rot_1trans/"))
# read_rotate_translate_save_mrc(src_path = src_path, output_path = output_path, mrcs_list = mrcs, n_pokeminos = n_pokeminos, nrot = 1, ntrans = 1)

# 3rot + 3trans
# output_path = Path(os.path.join(src_path.parent, src_path.stem+"_3rot_3trans/"))
# read_rotate_translate_save_mrc(src_path = src_path, output_path = output_path, mrcs_list = mrcs, n_pokeminos = n_pokeminos, nrot = 3, ntrans = 3)
