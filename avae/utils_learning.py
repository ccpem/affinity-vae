import logging
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.manifold
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import torch
import torch.distributed as dist


def accuracy(
    x_train: npt.NDArray,
    y_train: npt.NDArray,
    x_val: npt.NDArray,
    y_val: npt.NDArray,
    classifier: str = "NN",
) -> tuple[float, float, float, npt.NDArray, npt.NDArray]:
    """Compute classification accuracy and predictions for latent vectors."""
    logging.info(
        "############################################### Computing accuracy..."
    )
    labels = np.unique(np.concatenate((y_train, y_val)))
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(labels)

    training_classes = np.unique(y_train)
    validation_classes = np.unique(y_val)
    missing_from_validation = np.setdiff1d(
        training_classes, validation_classes
    )
    if missing_from_validation.size > 0:
        logging.warning(
            "Training classes %s are absent from validation data. They will "
            "be excluded from validation per-class metrics.",
            missing_from_validation,
        )

    unseen_in_training = np.setdiff1d(validation_classes, training_classes)
    if unseen_in_training.size > 0:
        logging.warning(
            "Validation classes %s were unseen during training. Overall "
            "validation accuracy includes them; seen-class accuracy excludes "
            "them.",
            unseen_in_training,
        )

    selected_indices = np.argwhere(np.isin(y_val, training_classes)).ravel()
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    parameters: dict[str, typing.Any]
    if classifier == "NN":
        parameters = {
            "hidden_layer_sizes": [
                (100, 50),
                (50, 20),
                (20, 10, 5),
                (100,),
                (50,),
            ],
        }
        method = sklearn.neural_network.MLPClassifier(
            max_iter=10000,
            activation="relu",
            solver="lbfgs",
            tol=1e-2,
            random_state=1,
            alpha=1,
        )
    elif classifier == "KNN":
        parameters = {"n_neighbors": range(1, 500, 100)}
        method = sklearn.neighbors.KNeighborsClassifier()
    else:
        raise ValueError("Invalid classifier type must be NN, KNN or LR")

    classifier_search = sklearn.model_selection.GridSearchCV(
        estimator=method,
        param_grid=parameters,
        scoring="f1_macro",
        cv=2,
        verbose=0,
    )
    fitted_classifier = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(), classifier_search
    )
    fitted_classifier.fit(x_train, y_train_encoded)
    logging.info(
        "Best parameters found for %s: %s",
        classifier,
        classifier_search.best_params_,
    )

    y_pred_train_encoded = fitted_classifier.predict(x_train)
    y_pred_val_encoded = fitted_classifier.predict(x_val)
    train_accuracy = sklearn.metrics.accuracy_score(
        y_train_encoded, y_pred_train_encoded
    )
    val_accuracy = sklearn.metrics.accuracy_score(
        y_val_encoded, y_pred_val_encoded
    )
    if selected_indices.size > 0:
        selected_val_accuracy = sklearn.metrics.accuracy_score(
            y_val_encoded[selected_indices],
            y_pred_val_encoded[selected_indices],
        )
    else:
        logging.warning(
            "No validation samples belong to classes seen during training; "
            "seen-class validation accuracy is unavailable."
        )
        selected_val_accuracy = float("nan")

    return (
        train_accuracy,
        val_accuracy,
        selected_val_accuracy,
        label_encoder.inverse_transform(y_pred_train_encoded),
        label_encoder.inverse_transform(y_pred_val_encoded),
    )


def tsne_embedding(xs: npt.NDArray, perplexity: int = 40) -> npt.NDArray:
    """Project latent vectors to at most two dimensions for plotting."""
    xs = np.asarray(xs)
    if xs.ndim != 2:
        raise ValueError("Embedding only accepts 2D arrays.")

    if xs.shape[-1] <= 2:
        return xs

    perplexity = min(perplexity, len(xs) - 1)
    logging.info(
        "############################################### Computing t-SNE..."
    )
    logging.info(
        "Samples: %d | Dimensions: %d | Perplexity: %d\n",
        len(xs),
        xs.shape[-1],
        perplexity,
    )
    return sklearn.manifold.TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=500,
        angle=0.7,
        n_jobs=-1,
        random_state=42,
    ).fit_transform(xs)


def format_meta_df(
    mode: str,
    filename_mode: list,
    meta_mode: list,
    x_mode: list,
    xhat_mode: list,
    z_mode: list,
    logvar_mode: list,
    pose: bool,
    pose_mode: list | None,
    y_mode: list | None = None,
) -> pd.DataFrame | None:
    """Build metadata DataFrame for a single mode."""
    if len(z_mode) == 0:
        return None

    n = len(z_mode)
    base_images = list(x_mode)
    if len(base_images) < n:
        base_images = base_images + [""] * (n - len(base_images))

    mode_meta = {
        "filename": list(filename_mode),
        "meta": list(meta_mode),
        "image": [str(base_images[i]) + str(xhat_mode[i]) for i in range(n)],
        "mode": [mode] * n,
    }

    if y_mode is not None:
        mode_meta["id"] = list(y_mode)

    lat_arr = np.asarray(z_mode)
    logvar_arr = np.asarray(logvar_mode)
    for d in range(lat_arr.shape[-1]):
        mode_meta[f"lat{d}"] = lat_arr[:, d]
        mode_meta[f"logvar-{d}"] = logvar_arr[:, d]
        mode_meta[f"std-{d}"] = np.exp(0.5 * logvar_arr[:, d])

    if pose and pose_mode is not None and len(pose_mode) > 0:
        pose_arr = np.asarray(pose_mode)
        for d in range(pose_arr.shape[-1]):
            mode_meta[f"pos{d}"] = pose_arr[:, d]

    return pd.DataFrame(mode_meta)


def build_meta_df(
    pose: bool,
    train: dict | None = None,
    val: dict | None = None,
    test: dict | None = None,
    eval_data: dict | None = None,
) -> pd.DataFrame:
    """Build metadata DataFrame from optional per-mode buffers."""
    local_meta_parts = []

    if train is not None:
        mode_meta_df = format_meta_df(
            mode="trn",
            filename_mode=train.get("filename", []),
            meta_mode=train.get("meta", []),
            x_mode=train.get("x", []),
            xhat_mode=train.get("xhat", []),
            z_mode=train.get("z", []),
            logvar_mode=train.get("logvar", []),
            pose=pose,
            pose_mode=train.get("pose"),
            y_mode=train.get("y", []),
        )
        if mode_meta_df is not None:
            local_meta_parts.append(mode_meta_df)

    if val is not None:
        mode_meta_df = format_meta_df(
            mode="val",
            filename_mode=val.get("filename", []),
            meta_mode=val.get("meta", []),
            x_mode=val.get("x", []),
            xhat_mode=val.get("xhat", []),
            z_mode=val.get("z", []),
            logvar_mode=val.get("logvar", []),
            pose=pose,
            pose_mode=val.get("pose"),
            y_mode=val.get("y", []),
        )
        if mode_meta_df is not None:
            local_meta_parts.append(mode_meta_df)

    if test is not None:
        mode_meta_df = format_meta_df(
            mode="tst",
            filename_mode=test.get("filename", []),
            meta_mode=test.get("meta", []),
            x_mode=test.get("x", []),
            xhat_mode=test.get("xhat", []),
            z_mode=test.get("z", []),
            logvar_mode=test.get("logvar", []),
            pose=pose,
            pose_mode=test.get("pose"),
            y_mode=test.get("y", []),
        )
        if mode_meta_df is not None:
            local_meta_parts.append(mode_meta_df)

    if eval_data is not None:
        mode_meta_df = format_meta_df(
            mode="evl",
            filename_mode=eval_data.get("filename", []),
            meta_mode=eval_data.get("meta", []),
            x_mode=eval_data.get("x", []),
            xhat_mode=eval_data.get("xhat", []),
            z_mode=eval_data.get("z", []),
            logvar_mode=eval_data.get("logvar", []),
            pose=pose,
            pose_mode=eval_data.get("pose"),
            y_mode=eval_data.get("y"),
        )
        if mode_meta_df is not None:
            local_meta_parts.append(mode_meta_df)

    if local_meta_parts:
        return pd.concat(local_meta_parts, ignore_index=False)

    return pd.DataFrame()


def combine_meta_df(
    meta_df: pd.DataFrame, rank_zero: bool, world_size: int
) -> pd.DataFrame:
    """Combine per-rank metadata DataFrames into one DataFrame on rank zero."""
    if not (dist.is_available() and dist.is_initialized() and world_size > 1):
        return meta_df

    gathered_meta = [None] * world_size if rank_zero else None
    dist.gather_object(meta_df, gathered_meta, dst=0)

    if not rank_zero:
        return pd.DataFrame()

    if gathered_meta is None:
        return pd.DataFrame()

    non_empty_meta = [
        df
        for df in gathered_meta
        if isinstance(df, pd.DataFrame) and not df.empty
    ]
    if non_empty_meta:
        return pd.concat(non_empty_meta, ignore_index=False)

    return pd.DataFrame()


def combine_accuracy_data(
    z_train: list,
    y_train: list,
    z_val: list,
    y_val: list,
    rank_zero: bool,
    world_size: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray] | None:
    """Combine per-rank latent vectors and labels on rank zero."""
    local_data = (z_train, y_train, z_val, y_val)
    if not (dist.is_available() and dist.is_initialized() and world_size > 1):
        return tuple(np.asarray(values) for values in local_data)

    gathered_data: list[tuple[list, list, list, list] | None] | None = (
        [None] * world_size if rank_zero else None
    )
    dist.gather_object(local_data, gathered_data, dst=0)

    if not rank_zero or gathered_data is None:
        return None

    complete_data: list[tuple[list, list, list, list]] = []
    for rank_data in gathered_data:
        if rank_data is None:
            raise RuntimeError(
                "Accuracy data was not gathered from every rank."
            )
        complete_data.append(rank_data)
    return tuple(
        np.concatenate(
            [np.asarray(rank_data[index]) for rank_data in complete_data]
        )
        for index in range(len(local_data))
    )


def log_progress(message: str) -> None:
    """Log a single console line that overwrites the previous one."""
    logger = logging.getLogger()
    stream_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]

    if not stream_handlers:
        logger.info(message)
        return

    original_terminators = [handler.terminator for handler in stream_handlers]
    try:
        for handler in stream_handlers:
            handler.terminator = "\r"
        logger.info(message)
    finally:
        for handler, terminator in zip(stream_handlers, original_terminators):
            handler.terminator = terminator


def configure_optimiser(
    opt_method: str, model: torch.nn.Module, learning_rate: float
):
    """
    Configure the optimiser for the training.

    Parameters
    ----------
    opt_method : str
        Optimisation method.
    model : torch.nn.Module
        Model to be trained.
    learning_rate : float
        Learning rate for the optimiser.

    Returns
    -------
    optimizer : torch.optim
        Optimiser for the training.
    """
    if opt_method == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate  # , weight_decay=1e-5
        )
    elif opt_method == "sgd":
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=learning_rate  # , weight_decay=1e-5
        )
    elif opt_method == "asgd":
        optimizer = torch.optim.aSGD(
            params=model.parameters(), lr=learning_rate  # , weight_decay=1e-5
        )
    else:
        raise ValueError(
            "Invalid optimisation method",
            opt_method,
            "must be adam or sgd if you have other methods in mind, this can be easily added to the train.py",
        )

    return optimizer
