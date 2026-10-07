import copy
import logging
import os.path
import pathlib
import random
import typing
import warnings

import altair
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import PIL.Image
import scipy.stats
import sklearn.metrics
import sklearn.metrics.pairwise
import torch
import torchvision

from . import utils_learning
from .utils import (
    colour_per_class,
    create_grid_for_plotting,
    fill_grid_for_plottting,
    latent_space_similarity_mat,
    pose_interpolation,
    save_imshow_png,
    save_mrc_file,
)

MAX_CLASS_FIG_SIZE = 20
VECTOR_FORMATS = ("pdf", "svg", "eps")


def _class_plot_scale(
    num_classes: int, fig_size: int | None = None
) -> tuple[float, int, int]:
    """Return consistent figure and font sizes for class-based plots.

    The default figure grows by a quarter inch per class, capped at
    MAX_CLASS_FIG_SIZE inches since rendering cost grows with its area.
    Fonts are sized to one class row so tick labels do not overlap; save
    large matrices in a vector format (pdf/svg) to zoom in on them.
    """
    resolved_fig_size = (
        fig_size
        if fig_size is not None
        else min(max(6, num_classes / 4), MAX_CLASS_FIG_SIZE)
    )
    row_height_pt = 72 * resolved_fig_size / max(num_classes, 1)
    font_size = int(np.clip(0.7 * row_height_pt, 4, 14))
    return resolved_fig_size, font_size, max(14, font_size + 6)


def _loss_curve_figure(
    epochs: int,
    series: typing.Sequence[tuple[str | None, npt.ArrayLike, str | None, str]],
    path: str,
    vis_format: str = "png",
    vis_print: bool = False,
    title: str | None = None,
    ylabel: str = "Loss",
    log_scale: bool = True,
    linewidth: float | None = None,
    legend: bool = True,
) -> None:
    """Render and save one line-vs-epoch figure. `series` is a list of
    (label, values, colour, linestyle) tuples, one per line drawn."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.ticklabel_format(useOffset=False)

    for label, values, color, linestyle in series:
        ax.plot(
            range(1, epochs + 1),
            values,
            c=color,
            linestyle=linestyle,
            label=label,
            linewidth=linewidth,
        )

    if title:
        ax.set_title(title, fontsize=16, fontweight="normal")

    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("Epochs", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    if legend:
        ax.legend(fontsize=14)

    fig.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.savefig(f"{path}.{vis_format}", **({"dpi": 300} if vis_print else {}))
    plt.close(fig)


def _matrix_figure(
    fig_size: float,
    font_size: int,
    title_font_size: int,
    title: str,
    path: str,
    vis_format: str = "png",
    vis_print: bool = False,
    print_dpi: int = 300,
    *,
    data: npt.NDArray,
    class_labels: typing.Sequence,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
    values_format: str | None = None,
    value_font_size: int | None = None,
    highlight: npt.NDArray | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    display: bool = False,
) -> None:
    """Render and save (or display) one
    class x class matrix figure: image, optional per-cell values (colour
    picked for contrast against the cell, the way sklearn's
    ConfusionMatrixDisplay does), optional red-highlighted tick labels,
    colorbar, and title.
    """
    data = np.asarray(data)
    num_classes = len(class_labels)
    # Vector formats embed the matrix at its native class x class resolution
    # ("none" skips resampling), so they stay sharp at any size and dpi has
    # no effect on them.
    is_vector = vis_format in VECTOR_FORMATS

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(
        data,
        interpolation="none" if is_vector else "nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if values_format is not None:
        color_min, color_max = im.cmap(0.0), im.cmap(1.0)
        thresh = (data.max() + data.min()) / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                color = color_max if data[i, j] < thresh else color_min
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:{values_format}}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=value_font_size,
                )

    ax.set_xticks(np.arange(num_classes), labels=class_labels)
    ax.set_yticks(np.arange(num_classes), labels=class_labels)
    ax.set_ylim((num_classes - 0.5, -0.5))

    if highlight is not None:
        for is_highlighted, x_label, y_label in zip(
            highlight, ax.get_xticklabels(), ax.get_yticklabels()
        ):
            if is_highlighted:
                x_label.set_color("red")
                y_label.set_color("red")

    ax.tick_params(axis="x", rotation=90, labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=font_size)

    ax.set_title(title, fontsize=title_font_size, fontweight="normal")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=font_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=font_size)

    fig.tight_layout()

    if display:
        plt.show()
        return

    if not os.path.exists("plots"):
        os.mkdir("plots")
    fig.savefig(
        path,
        **({"dpi": print_dpi} if vis_print and not is_vector else {}),
        **(
            {"pil_kwargs": {"compress_level": 3}}
            if vis_format == "png"
            else {}
        ),
    )

    plt.close(fig)


def _save_grid_figure(
    rows: int,
    cols: int,
    dsize: tuple,
    images: npt.NDArray,
    name: str,
    vis_format: str = "png",
    padding: int = 0,
    display: bool = False,
    vis_print: bool = False,
) -> None:
    """Arrange `images` into a grid and save it: as an .mrc file for 3D
    data (`dsize` has 3 dims), or a raster image for 2D data."""
    grid_for_napari = create_grid_for_plotting(rows, cols, dsize, padding)
    grid_for_napari = fill_grid_for_plottting(
        rows, cols, grid_for_napari, dsize, images, padding
    )

    data_dim = len(dsize)
    if data_dim == 3:
        save_mrc_file(f"{name}.mrc", grid_for_napari)
    elif data_dim == 2:
        save_imshow_png(
            f"{name}.{vis_format}",
            grid_for_napari,
            display=display,
            vis_print=vis_print,
        )


def loss_plot(
    epochs: int,
    beta: float,
    gamma: float,
    train_loss: list[float],
    val_loss: list[float] | None = None,
    p: list | None = None,
    vis_format: str = "png",
    vis_print: bool = False,
) -> None:
    """Visualise loss over epochs.

    Parameters
    ----------
    epochs: int
        Number of epochs.
    beta: list[float]
        List of beta values.
    gamma: list[float]
        List of gamma values.
    train_loss: list
        Training loss over epochs.
    val_loss: list
        Validation loss over epochs.
    p: list
        List of 7 hyperparameters: batch size, depth, "
                "channel init, latent dimension, learning rate, beta, gamma.
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising loss ...\n")

    train_loss = np.transpose(np.asarray(train_loss))
    if val_loss is not None:
        val_loss = np.transpose(np.asarray(val_loss))

    cols = ["blue", "red", "green", "orange"]
    labs = [
        "Total loss",
        "Reconstruction loss",
        "KL divergence loss x BETA",
        "Affinity loss x GAMMA",
    ]
    vlabs = [
        "VAL Total loss",
        "VAL Reconstruction loss",
        "VAL KL divergence loss x BETA",
        "VAL Affinity loss x GAMMA",
    ]
    train_loss[-2] = train_loss[-2] * beta
    train_loss[-1] = train_loss[-1] * gamma
    if val_loss is not None:
        val_loss[-2] = val_loss[-2] * beta
        val_loss[-1] = val_loss[-1] * gamma

    if p is not None:
        if len(p) != 7:
            logging.warning(
                "\n\nWARNING: Function vis.loss_plot is expecting 'p' parameter "
                "to be a list of 7 hyperparameters: batch size, depth, "
                "channel init, latent dimension, learning rate, beta, gamma. "
                "Exiting.\n",
            )
            return
        title = (
            "bs: %d, d: %d, ch: %d, lat: %d, lr: %.3f, beta: %.1f, "
            "gamma: %.1f"
            % (
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
            )
        )
    else:
        title = "Loss"

    full_series = [(labs[i], train_loss[i], cols[i], "-") for i in range(4)]
    if val_loss is not None:
        full_series += [
            (vlabs[i], val_loss[i], cols[i], "--") for i in range(4)
        ]
    _loss_curve_figure(
        epochs, full_series, "plots/loss", vis_format, vis_print, title=title
    )

    # plotting only the total loss as it sometimes is a few order of magnitude higher than KLD and affinity losses
    total_series = [(labs[0], train_loss[0], cols[0], "-")]
    if val_loss is not None:
        total_series.append((vlabs[0], val_loss[0], cols[0], "--"))
    _loss_curve_figure(
        epochs, total_series, "plots/loss_total", vis_format, vis_print
    )


def plot_cyc_variable(
    array: list,
    variable_name: str,
    vis_format: str = "png",
    vis_print: bool = False,
):
    """Plot evolution of variable from the cyclical training

    Parameters
    ----------
    array : list
        List of values for the variable
    variable_name : str
        Name of the variable
    """
    logging.info(
        "################################################################",
    )
    logging.info(f"Visualising {variable_name} ...\n")
    _loss_curve_figure(
        len(array),
        [(None, array, None, "-")],
        f"plots/hyperaparam_{variable_name}",
        vis_format,
        vis_print,
        ylabel=rf"$\{variable_name}$",
        log_scale=False,
        linewidth=3,
        legend=False,
    )


def confusion_plot(
    y_train: npt.NDArray,
    ypred_train: npt.NDArray,
    y_val: npt.NDArray,
    ypred_val: npt.NDArray,
    classes: str | None = None,
    mode: str = "",
    epoch: int = 0,
    vis_format: str = "png",
    vis_print: bool = False,
):
    """Plot confusion matrix .

    Parameters
    ----------
    y_train: np.array
        Training labels.
    ypred_train: np.array
        Predicted training labels.
    y_val: np.array
        Validation labels (unseen data).
    ypred_val: np.array
        Predicted validation labels (unseen data).
    mode: str
        Added data mode to the name of the saved figures (e.g train, valid, eval).
    epoch: int
        Current epoch.
    """
    logging.info(
        "################################################################",
    )

    logging.info("Visualising accuracy: confusion and F1 scores ...\n")

    classes_list = np.unique(np.concatenate((y_train, ypred_train)))

    fig_size, font_size, title_font_size = _class_plot_scale(len(classes_list))
    value_font_size = min(font_size, 10)

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(
        y_train, ypred_train, labels=classes_list
    )

    # Convert confusion matrix to a DataFrame to be saved as csv file
    cm_df = pd.DataFrame(cm)

    train_support = cm.sum(axis=1)
    supported_train_classes = train_support > 0
    avg_accuracy = np.divide(
        cm.diagonal(),
        train_support,
        out=np.zeros_like(train_support, dtype=float),
        where=supported_train_classes,
    )

    # Normalize confusion matrix
    cmn = (
        np.divide(
            cm.astype(float),
            train_support[:, np.newaxis],
            out=np.zeros_like(cm, dtype=float),
            where=supported_train_classes[:, np.newaxis],
        )
        * 100
    )

    # Convert normalised confusion matrix to a DataFrame to be saved as csv file
    cmn_df = pd.DataFrame(cmn)

    with plt.rc_context({"font.weight": "bold", "font.size": font_size}):
        train_title = "Balanced accuracy at epoch {}: {:.1f}%".format(
            epoch + 1, np.mean(avg_accuracy[supported_train_classes]) * 100
        )

        _matrix_figure(
            fig_size,
            font_size,
            title_font_size,
            train_title,
            f"plots/confusion_train{mode}.{vis_format}",
            vis_format,
            vis_print,
            data=cm,
            class_labels=classes_list,
            cmap=plt.cm.Blues,
            values_format="d" if vis_print else None,
            value_font_size=value_font_size,
            xlabel="Predicted label",
            ylabel="True label",
        )

        _matrix_figure(
            fig_size,
            font_size,
            title_font_size,
            train_title,
            f"plots/confusion_train{mode}_norm.{vis_format}",
            vis_format,
            vis_print,
            data=cmn,
            class_labels=classes_list,
            cmap=plt.cm.Blues,
            vmin=0,
            vmax=100,
            values_format=".0f" if vis_print else None,
            value_font_size=value_font_size,
            xlabel="Predicted label (%)",
            ylabel="True label (%)",
        )

        # Save confusion matrix DataFrame to CSV
        cm_df.to_csv(f"plots/confusion_train{mode}.csv", index=False)

        # Save normalised confusion matrix DataFrame to CSV
        cmn_df.to_csv(f"plots/confusion_train{mode}_norm.csv", index=False)

    classes_list_eval = np.unique(np.concatenate((y_val, ypred_val)))

    if np.setdiff1d(classes_list_eval, classes_list).size > 0:
        ordered_class_eval = np.concatenate(
            (classes_list, np.setdiff1d(classes_list_eval, classes_list))
        )
    else:
        ordered_class_eval = classes_list

    cm_eval = sklearn.metrics.confusion_matrix(
        y_val, ypred_val, labels=ordered_class_eval
    )

    # Convert confusion matrix to a DataFrame to be saved as csv file
    cm_eval_df = pd.DataFrame(cm_eval)

    eval_support = cm_eval.sum(axis=1)
    supported_eval_classes = eval_support > 0
    if not np.all(supported_eval_classes):
        logging.warning(
            "Validation confusion matrix has no samples for classes %s. "
            "Their normalised rows will be set to zero and excluded from "
            "average per-class accuracy.",
            ordered_class_eval[~supported_eval_classes],
        )
    avg_accuracy_eval = np.divide(
        cm_eval.diagonal(),
        eval_support,
        out=np.zeros_like(eval_support, dtype=float),
        where=supported_eval_classes,
    )

    # Normalise the validation confusion matrix
    cmn_eval = (
        np.divide(
            cm_eval.astype(float),
            eval_support[:, np.newaxis],
            out=np.zeros_like(cm_eval, dtype=float),
            where=supported_eval_classes[:, np.newaxis],
        )
        * 100
    )

    # Convert confusion matrix to a DataFrame to be saved as csv file
    cmn_eval_df = pd.DataFrame(cmn_eval)

    # Save confusion matrix DataFrame to CSV
    cm_eval_df.to_csv(f"plots/confusion_valid{mode}.csv", index=False)

    # Save normalised confusion matrix DataFrame to CSV
    cmn_eval_df.to_csv(f"plots/confusion_valid{mode}_norm.csv", index=False)

    if mode == "_eval":
        figure_name = "plots/confusion_eval"
    elif mode == "":
        figure_name = "plots/confusion_valid"
    else:
        figure_name = f"plots/confusion_{mode}"

    with plt.rc_context(
        {
            "font.weight": "bold",
            "font.size": font_size,
        }
    ):
        eval_title = "Balanced accuracy at epoch {}: {:.1f}%".format(
            epoch + 1, np.mean(avg_accuracy_eval[supported_eval_classes]) * 100
        )

        _matrix_figure(
            fig_size,
            font_size,
            title_font_size,
            eval_title,
            figure_name + f".{vis_format}",
            vis_format,
            vis_print,
            data=cm_eval,
            class_labels=ordered_class_eval,
            cmap=plt.cm.Blues,
            values_format="d" if vis_print else None,
            value_font_size=value_font_size,
            xlabel="Predicted label",
            ylabel="True label",
        )

        _matrix_figure(
            fig_size,
            font_size,
            title_font_size,
            eval_title,
            f"{figure_name}_norm.{vis_format}",
            vis_format,
            vis_print,
            data=cmn_eval,
            class_labels=ordered_class_eval,
            cmap=plt.cm.Blues,
            vmin=0,
            vmax=100,
            values_format=".0f" if vis_print else None,
            value_font_size=value_font_size,
            xlabel="Predicted label (%)",
            ylabel="True label (%)",
        )


def f1_plot(
    y_train: npt.NDArray,
    ypred_train: npt.NDArray,
    y_val: npt.NDArray,
    ypred_val: npt.NDArray,
    mode: str = "",
    epoch: int = 0,
    vis_format: str = "png",
    vis_print: bool = False,
):
    """Plot F1 values using classes observed in the training labels.

    Evaluation classes unseen during training are excluded from the scores.

    Parameters
    ----------
    y_train: np.array
        Training labels.
    ypred_train: np.array
        Predicted training labels.
    y_val: np.array
        Validation labels (unseen data).
    ypred_val: np.array
        Predicted validation labels (unseen data).
    mode: str
        Added data mode to the name of the saved figures (e.g train, valid, eval).
    epoch: int
        Epoch number.
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising F1 scores ...\n")
    classes_list = np.unique(np.concatenate((y_train, ypred_train)))
    classes_list_eval = np.unique(np.concatenate((y_val, ypred_val)))
    fig_size, font_size, title_font_size = _class_plot_scale(len(classes_list))

    if np.setdiff1d(classes_list_eval, classes_list).size > 0:
        logging.info(
            f"Class {np.setdiff1d(classes_list_eval, classes_list)} will not be used to compute F1 values as it was unseen in training data."
        )

        index = np.argwhere(np.isin(y_val, classes_list)).ravel()

        y_val = np.array(y_val)[index].tolist()
        ypred_val = np.array(ypred_val)[index].tolist()

    train_f1_score = sklearn.metrics.f1_score(
        y_train,
        ypred_train,
        average=None,
        labels=classes_list,
        zero_division=0,
    ).tolist()
    valid_f1_score = sklearn.metrics.f1_score(
        y_val,
        ypred_val,
        average=None,
        labels=classes_list,
        zero_division=0,
    ).tolist()

    if mode == "_eval":
        label = "eval"
    else:
        label = "valid"

    f1_train_file = f"plots/f1_train{mode}.csv"
    f1_valid_file = f"plots/f1_{label}.csv"

    train_df = pd.DataFrame([train_f1_score], columns=classes_list)
    valid_df = pd.DataFrame([valid_f1_score], columns=classes_list)

    train_df["epoch"] = epoch + 1
    valid_df["epoch"] = epoch + 1

    train_df["f1_avg"] = np.mean(train_f1_score)
    valid_df["f1_avg"] = np.mean(valid_f1_score)

    if os.path.exists(f1_train_file) and epoch != 0:
        f1_train = pd.read_csv(f1_train_file)
        pd.concat([f1_train, train_df]).to_csv(f1_train_file, index=False)
    else:
        train_df.to_csv(f1_train_file, index=False)

    if os.path.exists(f1_valid_file) and epoch != 0:
        f1_valid = pd.read_csv(f1_valid_file)
        pd.concat([f1_valid, valid_df]).to_csv(f1_valid_file, index=False)
    else:
        valid_df.to_csv(f1_valid_file, index=False)

    with plt.rc_context(
        {
            "font.weight": "bold",
            "font.size": font_size,
        }
    ):
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        line_width = max(1.5, font_size / 6)
        plt.plot(
            classes_list,
            train_f1_score,
            label="train",
            marker="o",
            linewidth=line_width,
        )
        plt.plot(
            classes_list,
            valid_f1_score,
            label=label,
            marker="o",
            linewidth=line_width,
        )
        plt.xticks(rotation=45)
        plt.legend(loc="lower left")
        plt.title(
            "F1 Score at epoch {}".format(epoch + 1),
            fontsize=title_font_size,
            fontweight="normal",
        )
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(
            f"plots/f1{mode}.{vis_format}",
            **({"dpi": 300} if vis_print else {}),
        )

        plt.close()


def recon_plot(
    img: torch.Tensor,
    rec: torch.Tensor,
    label: list,
    data_dim: int,
    mode: str = "trn",
    display: bool = False,
    vis_format: str = "png",
    vis_print: bool = False,
) -> None:
    """Visualise reconstructions.

    Parameters
    ----------
    img: torch.Tensor
        Input images.
    rec: torch.Tensor
        Reconstructed images.
    label: list
        List of labels for reconstructed images.
    data_dim: int
        Dimensionality of the data.
    mode: str
        Type of image in the training set: trn or val.
    display: bool
        When this variable is set to true, the save_imshow_png function only dispalys the plot
        and does not save a png image. This is to allow the use of this function in jupyter notebook
        post calculation. This features is not yet implemented for save_mrc_file.
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising reconstructions " + mode + "...\n")

    fname_in = f"recon_{mode}_in.{vis_format}"
    fname_out = f"recon_{mode}_out.{vis_format}"

    if data_dim == 3:
        img_2d = img[:, :, img.shape[2] // 2, :, :]
        rec_2d = rec[:, :, rec.shape[2] // 2, :, :]
    elif data_dim == 2:
        img_2d = img
        rec_2d = rec

    img_2d = torchvision.utils.make_grid(img_2d.cpu(), 10, 2).numpy()
    rec_2d = torchvision.utils.make_grid(rec_2d.detach().cpu(), 10, 2).numpy()

    save_imshow_png(
        fname_in,
        np.transpose(img_2d, (1, 2, 0)),
        display=display,
        vis_print=vis_print,
    )
    save_imshow_png(
        fname_out,
        np.transpose(rec_2d, (1, 2, 0)),
        display=display,
        vis_print=vis_print,
    )

    if data_dim == 3:
        rec = rec.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        labels = np.asarray(label)[-len(img) :]
        dsize = rec.shape[-data_dim:]

        # The number of reconstruction and input images to be displayed in the .mrc output file
        number_of_random_samples = 10
        number_of_columns = 3
        padding = 0

        if len(labels) < number_of_random_samples * number_of_columns:
            # In there are not enough images for the stack, do one column only
            number_of_random_samples = min(
                number_of_random_samples, len(labels)
            )
            number_of_columns = 1

        # define the dimensions for the napari grid
        grid_for_napari = np.zeros(
            (
                number_of_random_samples * dsize[0],
                2 * dsize[1] * number_of_columns + padding * number_of_columns,
                dsize[2],
            ),
            dtype=np.float32,
        )

        reconstruction_index: list[dict[str, int | str]] = []
        for k in range(number_of_columns):
            # select 10 images at random
            selected_indices = np.random.choice(
                len(img), size=number_of_random_samples, replace=False
            )
            selected_img = img[selected_indices]
            selected_rec = rec[selected_indices]

            # stack the images together with their reconstruction
            rec_img = np.hstack((selected_img, selected_rec))

            # Create and save the mrc file with single transversals
            for j in range(2):
                for i in range(number_of_random_samples):
                    grid_for_napari[
                        i * dsize[0] : (i + 1) * dsize[0],
                        j * dsize[1]
                        + (dsize[1] * 2 + padding) * k : (j + 1) * dsize[1]
                        + (dsize[1] * 2 + padding) * k,
                        :,
                    ] = rec_img[i, j, :, :, :]

            reconstruction_index.extend(
                {
                    "grid_column": k,
                    "grid_row": i,
                    "batch_index": int(source_index),
                    "label": labels[source_index],
                }
                for i, source_index in enumerate(selected_indices)
            )

        save_mrc_file(f"recon_{mode}.mrc", grid_for_napari)
        pd.DataFrame(reconstruction_index).to_csv(
            f"plots/recon_{mode}.csv", index=False
        )


def latent_embed_plot_tsne(
    xs: npt.NDArray,
    ys: npt.NDArray,
    classes: list | None = None,
    mode: str = "",
    epoch: int = 0,
    perplexity: int = 40,
    marker_size: int = 24,
    l_w: int = 2,
    display: bool = False,
    vis_format: str = "png",
    vis_print: bool = False,
    embedding: npt.NDArray | None = None,
) -> None:
    """Plot static TSNE embedding.

    Parameters
    ----------
    xs: np.ndarray
        Array of latent vectors.
    ys: np.ndarray
        Array of labels.
    classes: list
        List of classes.
    mode: str
        Added data mode to the name of the saved figure (e.g train, valid, eval).
    epoch: int
        Current epoch
    perplexity: int
    display: bool
        When this variable is set to true, the save_imshow_png function only dispalys the plot
        and does not save a png image. This is to allow the use of this function in jupyter notebook
        post calculation. This features is not yet implemented for save_mrc_file.
    vis_format: str
        format of the image saved
    embedding: np.ndarray | None
        Precomputed embedding coordinates. If omitted, they are calculated
        from ``xs``.
    """
    logging.info(
        "################################################################",
    )
    if not mode:
        logging.info("Visualising static TSNE embedding...\n")
    else:
        logging.info("Visualising static TSNE embedding " + mode + "...\n")

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if xs.ndim != 2:
        raise ValueError("Embedding only accepts 2D arrays.")
    if xs.shape[-1] == 1:
        logging.warning(
            "Data contains 1 dimension, cannot create embedding,"
            " plotting histogram instead...\n"
        )
    if xs.shape[-1] == 2:
        logging.warning(
            "Data already contains 2 dimensions, cannot create"
            " embedding, plotting scatter of original data...\n"
        )

    lats = (
        utils_learning.tsne_embedding(xs, perplexity=perplexity)
        if embedding is None
        else np.asarray(embedding)
    )
    if len(lats) != len(xs):
        raise ValueError(
            "Embedding and latent vectors must have equal lengths."
        )

    if classes is None:
        classes = sorted(list(np.unique(ys)))
    else:
        if np.setdiff1d(ys, classes).size > 0:
            classes = list(
                np.concatenate((classes, np.setdiff1d(ys, classes)))
            )

    n_classes = len(classes)
    fig_size, font_size, title_font_size = _class_plot_scale(n_classes)

    if n_classes < 3:
        # If the number of classes are not moe than 3 the size of the figure would be too
        # small and matplotlib would through a singularity error
        fig, ax = plt.subplots(
            figsize=(int(n_classes / 2) + 7, int(n_classes / 2) + 5)
        )
    else:
        fig, ax = plt.subplots(
            figsize=(int(n_classes / 2) + 4, int(n_classes / 2) + 2)
        )
    marker_size = max(marker_size, int(marker_size * max(1, n_classes / 5)))
    # When the number of classes is less than 3 the image becomes two small
    colours = colour_per_class(classes)

    if xs.shape[-1] != 1:

        for mol_id, mol in enumerate(set(ys.tolist())):
            idx = np.where(np.array(ys.tolist()) == mol)[0]

            color = colours[classes.index(mol)]

            plt.scatter(
                lats[idx, 0],
                lats[idx, 1],
                s=marker_size,
                label=mol[:4],
                facecolor=color,
                edgecolor=color,
                alpha=0.5,
            )

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=font_size,
        )
        ax.set_xlabel("TSNE-1", fontsize=font_size)
        ax.set_ylabel("TSNE-2", fontsize=font_size)

    if xs.shape[-1] == 1:

        for mol_id, mol in enumerate(set(ys.tolist())):
            idx = np.where(np.array(ys.tolist()) == mol)[0]
            cols = colours[classes.index(mol)]
            plt.hist(
                lats[idx],
                100,
                color=cols,
                histtype="step",
                stacked=True,
                fill=False,
                label=mol[:4],
                linewidth=l_w,
            )
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=font_size,
        )
        ax.set_xlabel("dim 1", fontsize=font_size)
        ax.set_ylabel("freq", fontsize=font_size)

    ax.set_title(
        f"t-SNE Embedding at epoch: {epoch + 1}",
        fontsize=title_font_size,
        fontweight="normal",
    )
    ax.tick_params(axis="both", labelsize=font_size)
    fig.tight_layout()

    if not display:
        if not os.path.exists("plots"):
            os.mkdir("plots")
        plt.savefig(
            f"plots/embedding_TSNE{mode}.{vis_format}",
            **({"dpi": 300} if vis_print else {}),
        )
    else:
        plt.show()

    plt.close()


def dyn_latentembed_plot(
    df: pd.DataFrame,
    epoch: int,
    mode: str = "",
    embedding: npt.NDArray | None = None,
):
    """Plot dynamic TSNE embedding.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the latent vectors.
    epoch: int
        Current epoch.
    mode: str
        Added data mode to the name of the saved figure (e.g train, valid, eval).
    embedding: np.ndarray | None
        Precomputed embedding coordinates. If omitted, they are calculated
        from the latent columns in ``df``.
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising dynamic embedding...\n")

    latentspace = df[[col for col in df if col.startswith("lat")]].to_numpy()
    lat_emb = np.asarray(
        utils_learning.tsne_embedding(latentspace)
        if embedding is None
        else embedding
    )
    if len(lat_emb) != len(df):
        raise ValueError("Embedding and metadata must have equal lengths.")
    if lat_emb.shape[1] == 1:
        lat_emb = np.column_stack([lat_emb[:, 0], np.zeros(len(lat_emb))])
    titlex = "t-SNE-1"
    titley = "t-SNE-2"
    df["emb-x"], df["emb-y"] = np.array(lat_emb)[:, 0], np.array(lat_emb)[:, 1]

    # create column select options for radio buttons
    # add no std to radio select options - change value here for marker size!
    if "std-off" not in df.columns:
        df.insert(loc=0, column="std-off", value=np.zeros(len(lat_emb)) + 0.5)
    # add certainty averaged across all latent dimensions as an extra option
    std_dim_cols = [
        col
        for col in df.columns
        if col.startswith("std-") and col not in ("std-off", "std-avg")
    ]
    if std_dim_cols and "std-avg" not in df.columns:
        df["std-avg"] = df[std_dim_cols].mean(axis=1)
    opts = [col for col in df.columns if col.startswith("std")]

    # create radio buttons and bind to a folded column select
    bind_checkbox = altair.binding_radio(
        options=opts,
        labels=[
            "off"
            if i == "std-off"
            else "avg"
            if i == "std-avg"
            else str(int(i.split("-")[-1]) + 1)
            for i in opts
        ],
        name="Certainty of prediction per dimension:",
    )
    column_select = altair.selection_point(
        fields=["column"],
        bind=bind_checkbox,
        value=[{"column": "std-off"}],
        name="certainty",
    )

    # mode, class and in-chart selections (also work with shft+click for multi)
    selection = altair.selection_point(fields=["id"], on="mouseover")
    selection_mode = altair.selection_point(fields=["mode"])
    selection_class = altair.selection_point(fields=["id"])

    # in-chart color condition
    color = altair.condition(
        (selection & selection_mode & selection_class),
        altair.Color("id:N", legend=None),
        altair.value("lightgray"),
    )

    # tooltip disla on-mouseover
    tooltip = ["id", "meta", "mode", "image"]  # .append(opts)

    # main scatter plot
    scatter = (
        altair.Chart(df, title="shift+click for multi-select")
        .mark_point(size=100, opacity=0.5, filled=True)
        .transform_fold(fold=opts, as_=["column", "value"])
        .transform_filter(column_select)
        .encode(
            altair.X("emb-x", title=titlex),
            altair.Y("emb-y", title=titley),
            altair.Shape(
                "mode",
                scale=altair.Scale(
                    range=["circle", "square", "triangle", "diamond"]
                ),
                legend=None,
            ),
            altair.Tooltip(tooltip),
            color=color,
            size=altair.Size("value:Q", legend=None, aggregate="mean").scale(
                type="log"
            ),
        )
        .interactive()
        .properties(width=800, height=500)
        .add_params(selection)
        .add_params(selection_class)
        .add_params(selection_mode)
        .add_params(column_select)
    )

    # interactive class legend
    legend_class = (
        altair.Chart(df, title="Class")
        .mark_point(size=100, opacity=0.5, filled=True)
        .encode(
            y=altair.Y("id:N", axis=altair.Axis(title=None, orient="right")),
            color=altair.condition(
                selection_class,
                altair.Color("id:N"),
                altair.value("lightgrey"),
            ),
        )
        .add_params(selection_class)
    )

    # interactive mode legend
    legend_mode = (
        altair.Chart(df, title="Mode")
        .mark_point(size=100, opacity=0.5, filled=True)
        .encode(
            y=altair.Y("mode:N", axis=altair.Axis(title=None, orient="right")),
            shape=altair.Shape(
                "mode",
                scale=altair.Scale(
                    range=["circle", "square", "triangle", "diamond"]
                ),
                legend=None,
            ),
            color=altair.condition(
                selection_mode,
                altair.value("steelblue"),
                altair.value("lightgrey"),
            ),
        )
        .add_params(selection_mode)
    )

    # organise charts in window and configure fonts
    # class/mode selections are shared across the scatter and legend views;
    # Altair merges the duplicates and warns benignly, so silence that message.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Automatically deduplicated selection parameter.*",
        )
        chart = (
            (scatter | altair.vconcat(legend_class, legend_mode))
            .configure_axis(labelFontSize=20, titleFontSize=20)
            .configure_legend(labelFontSize=20, titleFontSize=20)
            .configure_title(fontSize=20)
        )

    # save charts and latent embedding
    if not os.path.exists("latents"):
        os.mkdir("latents")
    # save latentspace and ids
    chart.save(f"latents/latent_epoch_{epoch + 1}_{mode}.html")


def confidence_plot(x, y, s, suffix=None, vis_format="png", vis_print=False):
    logging.info(
        "################################################################",
    )
    logging.info(
        "Visualising class-average confidence metrics " + suffix + "...\n",
    )
    cmap = plt.get_cmap("jet")
    cols = [cmap(i) for i in np.linspace(0, 1, len(x[0]))]
    classes = np.unique(y)
    _, font_size, title_font_size = _class_plot_scale(len(classes))
    rows = len(classes) // 2
    if len(classes) % 2 != 0:
        rows += 1
    fig, ax = plt.subplots(len(classes), sharex=True, sharey=True)
    ax = np.atleast_1d(ax)
    for c, cl in enumerate(classes):
        mu_cl = np.take(x, np.where(np.array(y) == cl)[0], axis=0)
        var_cl = np.take(s, np.where(np.array(y) == cl)[0], axis=0)
        std_cl = np.exp(0.5 * var_cl)
        mu_cl = np.mean(mu_cl, axis=0)
        std_cl = np.mean(std_cl, axis=0)

        min_mu = np.min(mu_cl)
        max_mu = np.max(mu_cl)
        max_sig = np.max(std_cl)
        step = (2 * 4 * max_sig) / 100

        xs = np.arange(min_mu - (4 * max_sig), max_mu + (4 * max_sig), step)

        for i in range(len(mu_cl)):
            ax[c].plot(
                xs,
                scipy.stats.norm.pdf(xs, mu_cl[i], std_cl[i]),
                color=cols[i],
                label="lat" + str(i + 1),
            )
        ax[c].set_title(cl, fontsize=title_font_size, fontweight="normal")
    name = f"plots/confidence.{vis_format}"
    if suffix is not None:
        name = name[:-4] + "_" + suffix + name[-4:]
    handles, labels = ax[-1].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, bbox_to_anchor=(1.06, 0.9)
    )  # , loc="upper left")
    plt.tight_layout()
    fig.savefig(
        name,
        **({"dpi": 300} if vis_print else {}),
        bbox_extra_artists=(leg,),
        bbox_inches="tight",
    )
    plt.close()


def latent_space_similarity_plot(
    latent_space: npt.NDArray,
    class_labels: npt.NDArray,
    mode: str = "",
    epoch: int = 0,
    affinity_matrix: pathlib.Path | None = None,
    plot_mode: str = "mean",
    display: bool = False,
    font_size: int = 16,
    fig_size: int | None = None,
    dpi: int = 300,
    vis_format: str = "png",
    vis_print: bool = False,
) -> None:
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
    affinity_matrix: pathlib.Path | None
        File path to affinity matrix
    vis_format: str
        format of the image saved
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising the latent space similarity matrix ...\n")

    if affinity_matrix is None:
        unique_classes = np.unique(class_labels)
    else:
        classes_order = (
            pd.read_csv(affinity_matrix, header=0).columns.astype(str).tolist()
        )
        unique_classes_in_data = np.unique(class_labels)
        if np.setdiff1d(unique_classes_in_data, classes_order).size > 0:
            unique_classes = np.concatenate(
                (
                    classes_order,
                    np.setdiff1d(unique_classes_in_data, classes_order),
                )
            )
        else:
            unique_classes = classes_order

    num_classes = len(unique_classes)
    resolved_fig_size, font_size, title_font_size = _class_plot_scale(
        num_classes, fig_size
    )
    cosine_sim = latent_space_similarity_mat(
        latent_space,
        class_labels,
        unique_classes,
        num_classes,
        plot_mode=plot_mode,
    )

    # Visualize average cosine similarity matrix
    classes_in_data = set(np.asarray(class_labels).astype(str))
    highlight = np.array(
        [str(c) not in classes_in_data for c in unique_classes]
    )
    title = f"Average Cosine Similarity Matrix at epoch: {epoch + 1}"
    path = f"plots/similarity_mean{mode}.{vis_format}"

    if fig_size is None:
        with plt.rc_context(
            {
                "font.weight": "bold",
                "font.size": font_size,
            }
        ):
            _matrix_figure(
                resolved_fig_size,
                font_size,
                title_font_size,
                title,
                path,
                vis_format,
                vis_print,
                print_dpi=dpi,
                data=cosine_sim,
                class_labels=unique_classes,
                cmap="RdBu",
                vmin=-1,
                vmax=1,
                highlight=highlight,
                xlabel="Class Labels",
                ylabel="Class Labels",
                display=display,
            )
    else:
        _matrix_figure(
            resolved_fig_size,
            font_size,
            title_font_size,
            title,
            path,
            vis_format,
            vis_print,
            print_dpi=dpi,
            data=cosine_sim,
            class_labels=unique_classes,
            cmap="RdBu",
            vmin=-1,
            vmax=1,
            highlight=highlight,
            xlabel="Class Labels",
            ylabel="Class Labels",
            display=display,
        )


def latent_4enc_interpolate_plot(
    dsize: tuple,
    xs: torch.Tensor,
    ys: torch.Tensor,
    vae: torch.nn.Module,
    device: torch.device,
    plots_config: npt.NDArray,
    poses: list,
    display: bool = False,
    vis_format: str = "png",
    vis_print: bool = False,
) -> None:
    """Visualise the interpolation of latent space between 4 randomly selected encodings.
    The number of plots and the number of interpolation steps is modifyable.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    x: torch.Tensor
        A sample batch. we extract 4 random images from this
    xs: list
        the list of all latent vectors
    ys: list
        the list of all labels for each latent vector in xs
    vae: torch.nn.Module
        Affinity vae model.
    device: torch.device
        Device to run the model on.
    data_dim: int
        the size of spatial dimensions of the input data (2:2D, 3:3D)
    plots_config: np.array
        Array containing the number of plots to be generated and the number of interpolation steps.
    poses: list
        List of pose vectors.
    display: bool
        When this variable is set to true, the save_imshow_png function only dispalys the plot
        and does not save a png image. This is to allow the use of this function in jupyter notebook
        post calculation. This features is not yet implemented for save_mrc_file.
    vis_format: str
        format of the image saved
    """
    logging.info(
        "################################################################",
    )
    logging.info(
        "Visualising Latent Interpolation between 4 randomly selected encodings ...\n"
    )

    padding = 0
    data_dim = len(dsize)
    classes = np.unique(np.asarray(ys))
    latent_dim = xs.shape[1]

    # Number of plots (each have 4 random corners)
    plots_config = plots_config.replace(" ", "").split(",")

    # Number of interpolation steps
    num_steps = int(plots_config[1])

    if poses is not None:
        pose_mean = np.mean(poses)

    for num_fig in range(int(plots_config[0])):
        enc = []

        draw_four = random.sample(range(len(classes)), k=4)
        for idx in draw_four:
            lat = np.take(
                xs,
                random.sample(list(np.where(ys == classes[idx])[0]), k=1),
                axis=0,
            )
            enc.append(lat)

        enc = np.asarray(enc)
        alpha_values = torch.linspace(0, 1, num_steps)
        beta_values = torch.linspace(0, 1, num_steps)
        decoded_grid = []

        for h in alpha_values:
            for v in beta_values:

                # bilinear interpolation in the latent space
                interpolated_z = (
                    (1 - h) * (1 - v) * enc[0]
                    + h * (1 - v) * enc[1]
                    + (1 - h) * v * enc[2]
                    + h * v * enc[3]
                )

                # Decode the interpolated encoding to generate an image
                with torch.no_grad():
                    decoded_images = vae.decoder(
                        interpolated_z.view(-1, latent_dim).to(device=device),
                        (torch.zeros(1, poses[0].shape[0]) + pose_mean).to(
                            device=device
                        ),
                    )
                decoded_grid.append(decoded_images.cpu().squeeze().numpy())

        decoded_grid = np.reshape(
            np.array(decoded_grid), (num_steps, num_steps, *dsize)
        )

        _save_grid_figure(
            num_steps,
            num_steps,
            dsize,
            decoded_grid,
            f"latent_interpolate_{num_fig}",
            vis_format,
            padding,
            display=display,
            vis_print=vis_print,
        )


def latent_disentamglement_plot(
    dsize: tuple,
    lats: list,
    vae: torch.nn.Module,
    device: torch.device,
    poses: list | None = None,
    mode: str = "trn",
    vis_format: str = "png",
    vis_print: bool = False,
) -> None:
    """Visualise latent content disentanglement.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    lats: list
        List of latent vectors.
    vae: torch.nn.Module
        Affinity vae model.
    device: torch.device
        Device to run the model on.
    data_dim: int
        the size of spatial dimensions of the input data (2:2D, 3:3D)
    poses: list
        List of pose vectors.
    mode: str
        Mode of evaluation (trn: Training, vld: Validation, eval: Evaluation )
    """
    logging.info(
        "################################################################"
    )
    logging.info("Visualising latent content disentanglement ...\n")
    number_of_samples = 7
    padding = 0
    data_dim = len(dsize)
    latents = np.asarray(lats)
    if poses is not None:
        poses_space = np.asarray(poses)

    lat_means = np.mean(latents, axis=0)
    lat_stds = np.std(latents, axis=0)
    latent_dims = latents.shape[-1]

    if poses is not None:
        pos_means = np.mean(poses_space, axis=0)
        pos_dims = poses_space.shape[-1]

    recon_images = []

    # Generate vectors representing single transversals along each latent_dims
    for l_dim in range(latent_dims):
        for grid_spot in range(number_of_samples):
            means = copy.deepcopy(lat_means)
            # every 0.4 interval from -1.2 to 1.2 sigma
            means[l_dim] += lat_stds[l_dim] * (-1.2 + 0.4 * grid_spot)

            # Decode the current vector
            with torch.no_grad():
                current_lat_grid = torch.from_numpy(np.array([means])).to(
                    device
                )

                if poses is not None:
                    current_pos_grid = torch.from_numpy(
                        np.array([pos_means])
                    ).to(device)
                    current_recon = vae.decoder(
                        current_lat_grid, current_pos_grid
                    )
                else:
                    current_recon = vae.decoder(current_lat_grid, None)

            recon_images.append(current_recon.cpu().squeeze().numpy())

    # Combine the individual decoded images into a single array
    recon_images = np.array(recon_images)
    recon_images = np.reshape(
        recon_images, (latent_dims, number_of_samples, *dsize)
    )

    _save_grid_figure(
        latent_dims,
        number_of_samples,
        dsize,
        recon_images,
        f"disentanglement-latent_{mode}",
        vis_format,
        padding,
        vis_print=vis_print,
    )


def pose_class_disentanglement_plot(
    dsize: tuple,
    x: list,
    y: list,
    pose_vis_class: npt.NDArray,
    poses: list,
    vae: torch.nn.Module,
    device: torch.device,
    mode: str = "trn",
    number_of_samples: int = 7,
    specific_enc: npt.NDArray = None,
    display: bool = False,
    vis_format: str = "png",
    vis_print: bool = False,
):

    """Visualise Pose interpolation per class. This function creates a pose interpolatoion
    plot for all classes listed in pose_vis_class.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    x: list
        List of latent vectors.
    y: List
        List of the labels associated with each latent vector in x
    pose_vis_class: list
        Classes to be used for pose interpolation (a seperate pose interpolation figure would be created for each class)."
    poses: list
        List of pose vectors.
    vae: torch.nn.Module
        Affinity vae model.
    data_dim: int
        the size of spatial dimensions of the input data (2:2D, 3:3D)
    device: torch.device
        Device to run the model on.
    mode: str
        Mode of evaluation (trn: Training, vld: Validation, eval: Evaluation )
    number_of_samples: int = 7
        number of pose interpolation steps, prefarably odd
    display: bool
        When this variable is set to true, the save_imshow_png function only dispalys the plot
        and does not save a png image. This is to allow the use of this function in jupyter notebook
        post calculation. This features is not yet implemented for save_mrc_file.
    vis_format: str
        format of the image saved
    """
    logging.info(
        "Visualising pose disentanglement for each class {}...\n".format(
            "".join(pose_vis_class)
        )
    )
    if poses is None:
        logging.warning(
            "Pose interpolation cannot be done if pose dimension is not specified"
        )

    padding = 0
    data_dim = len(dsize)
    x = np.asarray(x)

    poses_space = np.asarray(poses)
    pos_dims = poses_space.shape[-1]
    pose_vis_class_list = pose_vis_class.replace(" ", "").split(",")

    for i in pose_vis_class_list:
        decoded_grid = []
        class_x = np.take(x, np.where(np.array(y) == i)[0], axis=0)
        class_x_indx = np.random.choice(class_x.shape[0])
        enc = class_x[class_x_indx, :]

        if specific_enc is not None:
            enc = specific_enc

        class_pos = np.take(poses_space, np.where(np.array(y) == i)[0], axis=0)
        class_pos_mean = np.mean(class_pos, axis=0)
        class_pos_stds = np.std(class_pos, axis=0)

        decoded_grid = pose_interpolation(
            enc,
            pos_dims,
            class_pos_mean,
            class_pos_stds,
            dsize,
            number_of_samples,
            vae,
            device,
        )

        _save_grid_figure(
            pos_dims,
            number_of_samples,
            dsize,
            decoded_grid,
            f"pose_interpolate_{mode}_{i}",
            vis_format,
            padding,
            display=display,
            vis_print=vis_print,
        )


def pose_disentanglement_plot(
    dsize: tuple,
    lats: list,
    poses: list,
    vae: torch.nn.Module,
    device: torch.device,
    label: str = "avg",
    mode: str = "trn",
    display: bool = False,
    vis_format: str = "png",
    vis_print: bool = False,
):
    """Visualise pose disentanglement.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    lats: list
        List of latent vectors.
    poses: list
        List of pose vectors.
    vae: torch.nn.Module
        Affinity vae model.
    data_dim: int
        the size of spatial dimensions of the input data (2:2D, 3:3D)
    device: torch.device
        Device to run the model on.
    display: bool
        When this variable is set to true, the save_imshow_png function only dispalys the plot
        and does not save a png image. This is to allow the use of this function in jupyter notebook
        post calculation. This features is not yet implemented for save_mrc_file.
    vis_format: str
        format of the image saved
    """
    logging.info(
        "################################################################",
    )
    if label == "avg":
        logging.info("Visualising pose disentanglement ...\n")
    else:
        logging.info(
            "Visualising pose disentanglement for class {}...\n".format(label)
        )

    number_of_samples = 7
    padding = 0
    data_dim = len(dsize)
    latents = np.asarray(lats)
    poses_space = np.asarray(poses)

    pos_means = np.mean(poses_space, axis=0)
    pos_stds = np.std(poses_space, axis=0)
    pos_dims = poses_space.shape[-1]

    lat_means = np.mean(latents, axis=0)

    recon = pose_interpolation(
        lat_means,
        pos_dims,
        pos_means,
        pos_stds,
        dsize,
        number_of_samples,
        vae,
        device,
    )

    _save_grid_figure(
        pos_dims,
        number_of_samples,
        dsize,
        recon,
        f"disentanglement-pose_{mode}_{label}",
        vis_format,
        padding,
        display=display,
        vis_print=vis_print,
    )


def interpolations_plot(
    dsize: tuple,
    lats: list,
    classes: list,
    vae: torch.nn.Module,
    device: torch.device,
    poses: list | None = None,
    mode: str = "trn",
    vis_format: str = "png",
    vis_print: bool = False,
) -> None:
    """Visualise interpolations.

    Parameters
    ----------
    dsize: torch.size
        the size of input.
    lats: list
        List of latent vectors.
    classes: list
        List of class labels.
    vae: torch.nn.Module
        Affinity vae model.
    device: torch.device
        Device to run the model on.
    poses: list
        List of pose vectors.
    mode: str
        Mode of evaluation (trn: Training, vld: Validation, eval: Evaluation )
    """
    logging.info(
        "################################################################",
    )
    logging.info("Visualising interpolations ...\n")
    data_dim = len(dsize)
    lats = np.asarray(lats)
    classes = np.asarray(classes)

    if poses is not None:
        poses_space = np.asarray(poses)

    class_ids = np.unique(classes)
    if len(class_ids) <= 3:
        logging.warning(
            "\n\nWARNING: Interpolation plot needs at least 4 distinct classes, "
            "cannot visualise interpolations. Exiting.\n",
        )
        return

    class_reps_lats = np.take(
        lats, [np.where(classes == i)[0][0] for i in class_ids], axis=0
    )
    class_reps_lats = np.asarray(class_reps_lats)
    if poses is not None:
        class_reps_poses = np.asarray(
            np.take(
                poses_space,
                [np.where(classes == i)[0][0] for i in class_ids],
                axis=0,
            )
        )

    draw_four = random.sample(list(enumerate(class_reps_lats)), k=4)
    inds, class_rep = list(zip(*draw_four))
    class_rep_lats = np.asarray(class_rep)
    latent_dim = class_rep_lats.shape[1]

    if poses is not None:
        class_rep_poses = np.asarray([class_reps_poses[i] for i in inds])
        poses_dim = class_rep_poses.shape[1]

    # Generate a gird of latent vectors interpolated between reps of four ids

    grid_size = 6
    alpha_values = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    beta_values = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    decoded_grid = []
    for i, h in enumerate(alpha_values):
        for j, v in enumerate(beta_values):

            # bilinear interpolation in the latent space
            interpolated_z = (
                (1 - h) * (1 - v) * class_rep_lats[0]
                + h * (1 - v) * class_rep_lats[1]
                + (1 - h) * v * class_rep_lats[2]
                + h * v * class_rep_lats[3]
            )
            interpolated_z_t = torch.from_numpy(
                np.asarray(interpolated_z, dtype=np.float32)
            ).view(-1, latent_dim)

            if poses is not None:
                interpolated_pose = (
                    (1 - h) * (1 - v) * class_rep_poses[0]
                    + h * (1 - v) * class_rep_poses[1]
                    + (1 - h) * v * class_rep_poses[2]
                    + h * v * class_rep_poses[3]
                )
                interpolated_pose_t = torch.from_numpy(
                    np.asarray(interpolated_pose, dtype=np.float32)
                ).view(-1, poses_dim)
            with torch.no_grad():
                if poses is not None:
                    decoded_images = vae.decoder(
                        interpolated_z_t.to(device=device),
                        interpolated_pose_t.to(device=device),
                    )
                else:
                    decoded_images = vae.decoder(
                        interpolated_z_t.to(device=device),
                        None,
                    )

            decoded_grid.append(decoded_images.cpu().squeeze().numpy())

    decoded_grid = np.reshape(
        np.array(decoded_grid), (grid_size, grid_size, *dsize)
    )

    _save_grid_figure(
        grid_size,
        grid_size,
        dsize,
        decoded_grid,
        f"interpolations_{mode}",
        vis_format,
        vis_print=vis_print,
    )


def plot_affinity_matrix(
    lookup: pd.DataFrame,
    all_classes: list,
    selected_classes: list,
    fig_size: int | None = None,
    vis_format: str = "png",
    vis_print: bool = False,
) -> None:
    """
    This function plots the Affinity matrix and highlights the
    classes selected for the given calculation.

    Parameters
    ----------
    all_classes: list
        All existing classes in the affinity matrix  affinity*.csv
    lookup: pandas.DataFrame
        The affinity matrix
    selected_classes : list
        All classes selected by the user for training in classes.csv
    vis_format: str
        format of the image saved
    """
    logging.info("\n")
    logging.info(
        "################################################################",
    )
    logging.info("Visualising affinity matrix ...\n")

    resolved_fig_size, font_size, title_font_size = _class_plot_scale(
        len(all_classes), fig_size
    )

    highlight = np.array([c not in selected_classes for c in all_classes])

    _matrix_figure(
        resolved_fig_size,
        font_size,
        title_font_size,
        "Affinity Matrix",
        f"plots/affinity_matrix.{vis_format}",
        vis_format,
        vis_print,
        data=lookup,
        class_labels=all_classes,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        highlight=highlight,
    )


def plot_classes_distribution(
    data: list,
    category: str,
    vis_format: str = "png",
    vis_print: bool = False,
) -> None:
    """Plot histogram with classes distribution

    Parameters
    ----------
    data : list
        List of classes
    category : str
        The category of the data (train, test, val)
    """

    logging.info(
        "################################################################",
    )
    logging.info("Visualising classes distribution " + category + "...\n")

    labels, counts = np.unique(data, return_counts=True)
    fig_size, font_size, title_font_size = _class_plot_scale(len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ticks = range(len(counts))
    ax.bar(ticks, counts, align="center", color="blue", alpha=0.5)
    ax.set_xticks(ticks, labels)
    ax.set_title(
        "Classes Distribution",
        fontsize=title_font_size,
        fontweight="normal",
    )
    ax.set_xlabel("Class", fontsize=font_size)
    ax.set_ylabel("Number of Entries", fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size, rotation=90)
    ax.tick_params(axis="y", labelsize=font_size)
    fig.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(
        f"plots/classes_distribution_{category}.{vis_format}",
        **({"dpi": 300} if vis_print else {}),
    )
    plt.close()


def _encoder(i: PIL.Image) -> str:
    """Encode PIL Image as base64 buffer.
    Parameters
    ----------
    i: PIL.Image
        Image to be encoded.

    Returns
    -------
    str
        Encoded image.

    """
    import base64
    import io

    with io.BytesIO() as buffer:
        i.thumbnail((110, 110))
        i.save(buffer, "PNG")
        data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

    return f"{data}"


def _decoder(i: str) -> PIL.Image:
    """Decode base64 buffer as PIL Image.
    Parameters
    ----------
    i: str
        Encoded image.

    Returns
    -------
    PIL.Image
        Decoded image.
    """
    import base64
    import io

    return PIL.Image.open(io.BytesIO(base64.b64decode(i)))


def format(im: PIL.Image, data_dim: int) -> list[str]:
    """Format PIL Image as Pandas compatible Altair image display.

    Parameters
    ----------
    im: PIL.Image
        Image to be formatted.
    data_dim: int
        Dimension of the data.

    Returns
    -------
    list[str]
        Formatted images compatible with Altair image display. Batch input
        returns one encoded item per sample, while single-sample input returns
        a one-element list.
    """
    if len(im.shape) == 5 and data_dim == 3:
        batch = True
        im = np.copy(
            im[:, :, im.shape[2] // 2, :, :]
            .squeeze(dim=1)
            .cpu()
            .detach()
            .numpy()
        )
    elif len(im.shape) == 4 and data_dim == 3:
        batch = False
        im = np.copy(
            im[:, im.shape[1] // 2, :, :].squeeze(dim=0).cpu().detach().numpy()
        )
    elif len(im.shape) == 4 and data_dim == 2:
        batch = True
        im = np.copy(im.squeeze(dim=1).cpu().detach().numpy())
        # .astype(np.uint8)
    elif len(im.shape) == 3 and data_dim == 2:
        batch = False
        im = np.copy(im.squeeze(dim=0).cpu().detach().numpy())
        # .astype(np.uint8)
    else:
        logging.warning(
            "\n\nWARNING: Wrong data format, please pass either a single "
            "unsqueezed tensor or a batch to image formatter. Exiting.\n",
        )
        return []
    im *= 255
    im = im.astype(np.uint8)
    if batch:
        # if batch we are adding a reconstruction to an input that already
        # exists in the DF, '&' is a separator.
        return ["&" + _encoder(PIL.Image.fromarray(i)) for i in im]
    else:
        return [_encoder(PIL.Image.fromarray(im))]


def merge(im: str) -> str | None:
    """Merge 2 base64 buffers as PIL Images and encode back to base64
    buffers.

    Parameters
    ----------
    im: str
        Input PIL Images to be merged.

    Returns
    -------
    str
        Merged image.
    """
    if im.startswith("data:image/"):
        return im

    i = im.split("&")
    if len(i) != 2:
        logging.warning(
            "\n\nWARNING: Image format corrupt. Number of images in meta_df: {}. "
            "Exiting. \n".format(len(i)),
        )
        return None

    im1 = _decoder(i[0])
    im2 = _decoder(i[1])

    new_image = PIL.Image.new("L", (im2.size[0] + im1.size[0], im2.size[1]))
    new_image.paste(im1, (0, 0))
    new_image.paste(im2, (im1.size[0], 0))
    data = _encoder(new_image)

    return f"data:image/png;base64,{data}"
