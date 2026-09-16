import copy
import logging
import os.path
import typing

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import numpy.typing as npt
import sklearn.linear_model
import sklearn.metrics
import sklearn.metrics.pairwise
import torch


def as_list(value: object) -> list:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def create_grid_for_plotting(
    rows: int, columns: int, dsize: tuple, padding: int = 0
) -> npt.NDArray:

    # define the dimensions for the napari grid

    if len(dsize) == 3:
        grid_for_napari = np.zeros(
            (
                rows * dsize[0],
                dsize[1] * columns + padding * columns,
                dsize[2],
            ),
            dtype=np.float32,
        )

    elif len(dsize) == 2:
        grid_for_napari = np.zeros(
            (
                rows * dsize[0],
                dsize[1] * columns + padding * columns,
            ),
            dtype=np.float32,
        )

    return grid_for_napari


def fill_grid_for_plottting(
    rows: int,
    columns: int,
    grid: npt.NDArray,
    dsize: tuple,
    array: npt.NDArray,
    padding: int = 0,
) -> npt.NDArray:

    if len(dsize) == 3:
        for j in range(columns):
            for i in range(rows):
                grid[
                    i * dsize[0] : (i + 1) * dsize[0],
                    j * (dsize[1] + padding) : (j + 1) * dsize[1]
                    + padding * j,
                    :,
                ] = array[i, j, :, :, :]

    elif len(dsize) == 2:
        for j in range(columns):
            for i in range(rows):
                grid[
                    i * dsize[0] : (i + 1) * dsize[0],
                    j * (dsize[1] + padding) : (j + 1) * dsize[1]
                    + padding * j,
                ] = array[i, j, :, :]
    return grid


def save_imshow_png(
    fname: str,
    array: npt.NDArray,
    cmap: str | None = None,
    min: float | None = None,
    max: float | None = None,
    writer: typing.Any = None,
    figname: str | None = None,
    epoch: int = 0,
    display: bool = False,
) -> None:
    if not display:
        if not os.path.exists("plots"):
            os.mkdir("plots")

        fig, _ = plt.subplots(figsize=(10, 10))
        plt.imshow(array, cmap=cmap, vmin=min, vmax=max)  # channels last

        plt.savefig("plots/" + fname)

        if writer:
            writer.add_figure(figname, fig, epoch)
    else:
        plt.imshow(array, cmap=cmap, vmin=min, vmax=max)  # channels last
        plt.show()

    plt.close()


def save_mrc_file(fname: str, array: npt.NDArray) -> None:
    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/" + fname, overwrite=True) as mrc:
        mrc.set_data(array)


def colour_per_class(classes: list) -> list:
    # Define the number of colors you want
    num_colors = len(classes)

    # Choose colormaps for combining
    cmap_1 = plt.get_cmap("tab20")
    cmap_2 = plt.get_cmap("Accent")
    cmap_3 = plt.get_cmap("Pastel1")
    cmap_4 = plt.get_cmap("Set1")

    # Combine the four colormaps
    combined_cmap = [cmap_1(i % 20) for i in range(20)]
    combined_cmap.extend([cmap_2(i % 8) for i in range(8)])
    combined_cmap.extend([cmap_3(i % 8) for i in range(8)])
    combined_cmap.extend([cmap_4(i % 8) for i in range(8)])

    # Create the colormap object
    custom_cmap = plt.cm.colors.ListedColormap(
        combined_cmap, name="custom_cmap"
    )

    # Generate a list of colors based on the modulo operation of i with respect to the number of colors in the combined colormap
    colours = [custom_cmap(i % len(combined_cmap)) for i in range(num_colors)]
    return colours


def pose_interpolation(
    enc: npt.NDArray,
    pos_dims: int,
    pose_mean: npt.NDArray,
    pose_std: npt.NDArray,
    dsize: tuple,
    number_of_samples: int,
    vae: torch.nn.Module,
    device: torch.device,
) -> npt.NDArray:

    """This function:
    1-  interpolates within each pose channels
        for the number_of_samples requested.
    2- returns all decoded images based on the input latent
        and the interpolated pose

    Parameters
    ----------
    enc: numpy array
        the latent encoding.
    pos_dims: int
        the pose channel dimension
    pose_mean: numpy array
        mean of each pose channel.
    pose_std: numpy array
        standard deviation of each pose channel.
    dsize: torch.size
        the dimension of the data. Example [32,32,32]
    number_of_samples: int
        number of samples to interpolate for.
    vae: torch.nn.Module
        Affinity vae model.
    device: torch.device
        Device to run the model on.
    """
    decoded_grid = []
    # Generate vectors representing single transversals along each lat_dim
    for p_dim in range(pos_dims):
        for grid_spot in range(number_of_samples):
            means = copy.deepcopy(pose_mean)
            means[p_dim] += pose_std[p_dim] * (-1.2 + 0.4 * grid_spot)

            pos = torch.from_numpy(np.array(means)).unsqueeze(0).to(device)
            lat = torch.from_numpy(np.array(enc)).unsqueeze(0).to(device)

            # Decode interpolated vectors
            with torch.no_grad():
                decoded_img = vae.decoder(lat, pos)

            decoded_grid.append(decoded_img.cpu().squeeze().numpy())

    decoded_grid = np.reshape(
        np.array(decoded_grid), (pos_dims, number_of_samples, *dsize)
    )

    return decoded_grid


def latent_space_similarity_mat(
    latent_space: npt.NDArray,
    class_labels: npt.NDArray,
    unique_classes: list,
    num_classes: int,
    plot_mode: str = "",
) -> npt.NDArray:
    """
    This function calculates the similarity (affinity) between classes in the latent space and builds a matrix.
    Parameters
    ----------
    latent_space: np.ndarray
        The latent space
    class_labels: np.array
        The labels of the latent space
    mode: str
        Mode of the calculation (train, test, val)
    epoch: int
        Epoch number for title
    display: bool
        When this variable is set to true, the function only dispalys the plot and doesnt save it.
    """
    # get same label order as affinity matrix
    cosine_sim_matrix = sklearn.metrics.pairwise.cosine_similarity(
        latent_space
    )

    cosine_sim_mat = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(i, num_classes):
            class_i_indices = np.where(class_labels == unique_classes[i])[0]
            class_j_indices = np.where(class_labels == unique_classes[j])[0]
            cosine_sims = cosine_sim_matrix[class_i_indices][
                :, class_j_indices
            ]
            if cosine_sims.size == 0:
                cosine_sim_mat[i, j] = 0.0
                cosine_sim_mat[j, i] = 0.0
                continue
            if plot_mode == "mean":
                cosine_sim_mat[i, j] = np.mean(cosine_sims)
                cosine_sim_mat[j, i] = cosine_sim_mat[
                    i, j
                ]  # symmetrical matrix
            if plot_mode == "std":
                cosine_sim_mat[i, j] = np.std(cosine_sims)
                cosine_sim_mat[j, i] = cosine_sim_mat[
                    i, j
                ]  # symmetrical matrix

    return cosine_sim_mat


def latest_file(path: str, extension: str) -> str:

    most_recent_file = ""
    most_recent_time = 0

    # iterate over the files in the directory using os.scandir
    for entry in os.scandir(path):
        if entry.name.lower().endswith(extension):
            # get the modification time of the file using entry.stat().st_mtime_ns
            mod_time = entry.stat().st_mtime_ns
            if (
                mod_time > most_recent_time
                and "eval" not in entry.name.lower()
            ):
                # update the most recent file and its modification time
                most_recent_file = entry.name
                most_recent_time = mod_time
    return most_recent_file
