import glob
import os.path
import pathlib
import shutil

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import numpy.typing as npt
import torch

SCREEN_HEIGHT_PX = 720  # top half of a 1440p screen
VECTOR_FORMATS = ("pdf", "svg", "eps")
PLOTS_ROOT = "plots"
_plots_dir = PLOTS_ROOT
_copy_to: tuple[str, ...] = ()
_run_id = ""  # run timestamp appended to epoch/best folders, set by set_run_id


def set_run_id(run_id: str) -> None:
    """Tag this run's epoch and best folders (e.g. with params.date_time_run)."""
    global _run_id
    _run_id = run_id


def vis_dir(epoch: int | None = None, best: str | None = None) -> str:
    """Folder for per-epoch plots: plots/, plots/epochs_<run id>/epoch_XXXX,
    or for best="loss"/"acc" plots/best_<kind>_epochXXXX_<run id>."""
    if epoch is None:
        return PLOTS_ROOT
    suffix = f"_{_run_id}" if _run_id else ""
    if best:
        return os.path.join(
            PLOTS_ROOT, f"best_{best}_epoch{epoch:04d}{suffix}"
        )
    return os.path.join(PLOTS_ROOT, f"epochs{suffix}", f"epoch_{epoch:04d}")


def set_vis_dir(
    epoch: int | None = None,
    best: str | None = None,
    copy_to: tuple[str, ...] = (),
) -> None:
    """Direct per-epoch plots to vis_dir(epoch, best) (None resets to plots/).

    Each per-epoch file saved there is also copied to the `copy_to` folders.
    """
    global _plots_dir, _copy_to
    _plots_dir = vis_dir(epoch, best)
    _copy_to = tuple(copy_to)


def copy_saved(path: str | pathlib.Path) -> None:
    """Copy a just-saved per-epoch file to the folders set by set_vis_dir."""
    if os.path.dirname(os.fspath(path)) != _plots_dir:
        return
    for target in _copy_to:
        os.makedirs(target, exist_ok=True)
        shutil.copy2(path, target)


def clear_best_dirs(kind: str) -> None:
    """Remove this run's earlier best folder of `kind` ("loss" or "acc") so
    only its latest is kept. Other runs' and legacy folders are left alone."""
    suffix = f"_{_run_id}" if _run_id else ""
    pattern = os.path.join(
        PLOTS_ROOT,
        f"best_{kind}_epoch[0-9][0-9][0-9][0-9]{glob.escape(suffix)}",
    )
    for path in glob.glob(pattern):
        shutil.rmtree(path, ignore_errors=True)


def plots_path(fname: str, per_epoch: bool = True) -> str:
    """Return the save path for a plot, creating its folder on demand."""
    base = _plots_dir if per_epoch else PLOTS_ROOT
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, fname)


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


def save_figure(
    fig: plt.Figure,
    path: str,
    vis_format: str,
    vis_print: bool = False,
    display: bool = False,
    cap_height: bool = True,
) -> None:
    """Save (or, with `display`, only show) a figure and close it.

    Vector formats take no dpi. `vis_print` saves at 300 dpi and full size.
    Otherwise, with `cap_height`, the dpi is lowered so the figure is at most
    SCREEN_HEIGHT_PX tall, as large figures cost time and space to rasterise.
    """
    if display:
        plt.show()
        plt.close(fig)
        return

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if vis_format not in VECTOR_FORMATS:
        dpi = plt.rcParams["figure.dpi"]
        if vis_print:
            dpi = 300
        elif cap_height:
            dpi = min(dpi, SCREEN_HEIGHT_PX / fig.get_figheight())
        kwargs["dpi"] = dpi
    if vis_format == "png":
        kwargs["pil_kwargs"] = {"compress_level": 3}
    fig.savefig(path, **kwargs)
    plt.close(fig)
    copy_saved(path)


def save_imshow_png(
    fname: str,
    array: npt.NDArray,
    cmap: str | None = None,
    min: float | None = None,
    max: float | None = None,
    display: bool = False,
    vis_print: bool = False,
) -> None:
    height, width = array.shape[:2]
    if width >= height:
        image_figure_size = (10, 10 * height / width)
    else:
        image_figure_size = (10 * width / height, 10)
    fig, ax = plt.subplots(figsize=image_figure_size)
    ax.imshow(array, cmap=cmap, vmin=min, vmax=max)  # channels last
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_figure(
        fig,
        plots_path(fname),
        fname.rsplit(".", 1)[-1],
        vis_print,
        display,
        cap_height=False,
    )


def save_mrc_file(fname: str, array: npt.NDArray) -> None:
    path = plots_path(fname)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(array)
    copy_saved(path)


def plot_labels(labels, suffix: str) -> npt.NDArray:
    """Labels for plots of test/eval data: `{class}_{suffix}`, or just
    `suffix` for unlabelled samples, i.e. files named `test_*` (label
    `test`). Plots only; accuracy and metadata keep the plain class names."""
    out = []
    for lab in np.asarray(labels, dtype=object).ravel():
        out.append(suffix if str(lab) == "test" else f"{lab}_{suffix}")
    return np.array(out, dtype=object)


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


def requested_classes(y: list, spec: str, name: str) -> list:
    """Classes listed in the comma separated `spec`; raise if any is not in `y`."""
    wanted = [c for c in spec.replace(" ", "").split(",") if c]
    available = sorted({str(c) for c in np.asarray(y).astype(str)})
    missing = [c for c in wanted if c not in available]
    if missing:
        raise ValueError(
            f"{name} contains classes that are not in the data: missing "
            f"{missing}, available {available}"
        )
    return wanted


def interpolate(
    start: npt.NDArray, end: npt.NDArray, steps: int
) -> npt.NDArray:
    """`steps` evenly spaced points on the straight line from `start` to `end`
    (both included). The vectors are the last axis; leading axes broadcast, so
    `start` and `end` of shape (..., D) give (..., steps, D)."""
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    start = np.asarray(start, dtype=np.float32)[..., None, :]
    end = np.asarray(end, dtype=np.float32)[..., None, :]
    return (1 - t) * start + t * end


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
    # The cosine similarity of two vectors is the dot product of their unit
    # length versions: cos(a, b) = u_a . u_b, with u = x / |x|. Averaging it
    # over all pairs of two classes therefore reduces to dot products of
    # per-class sums of unit vectors, so the N x N similarity matrix never
    # needs to be built:
    #   mean_ij = (S_i . S_j) / (n_i n_j),        S_c = sum of u over class c
    # For the std, the squared similarity (u_a . u_b)^2 equals the Frobenius
    # product <u_a u_a^T, u_b u_b^T>_F, so the mean squared similarity is
    #   E_ij[s^2] = <M_i, M_j>_F / (n_i n_j),    M_c = sum of u u^T in class c
    # and std_ij = sqrt(E_ij[s^2] - mean_ij^2).
    # Self-pairs are included on the diagonal, as in the full pairwise version.
    latent_space = np.asarray(latent_space, dtype=np.float64)
    class_labels = np.asarray(class_labels)
    # unit length latent vectors; zero vectors stay zero (similarity 0)
    norms = np.linalg.norm(latent_space, axis=1, keepdims=True)
    unit = latent_space / np.where(norms == 0, 1.0, norms)

    latent_dims = unit.shape[1]
    counts = np.zeros(num_classes)
    sums = np.zeros((num_classes, latent_dims))
    outer_sums = np.zeros((num_classes, latent_dims * latent_dims))
    for i in range(num_classes):
        class_unit = unit[class_labels == unique_classes[i]]
        counts[i] = len(class_unit)
        sums[i] = class_unit.sum(axis=0)
        if plot_mode == "std":
            outer_sums[i] = (class_unit.T @ class_unit).ravel()

    pair_counts = np.outer(counts, counts)
    has_pairs = pair_counts > 0
    safe_pair_counts = np.where(has_pairs, pair_counts, 1.0)
    mean_sim = (sums @ sums.T) / safe_pair_counts

    cosine_sim_mat = np.zeros((num_classes, num_classes))
    if plot_mode == "mean":
        cosine_sim_mat = mean_sim
    elif plot_mode == "std":
        second_moment = (outer_sums @ outer_sums.T) / safe_pair_counts
        cosine_sim_mat = np.sqrt(
            np.clip(second_moment - mean_sim**2, 0, None)
        )
    cosine_sim_mat[~has_pairs] = 0.0

    return cosine_sim_mat
