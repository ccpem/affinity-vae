import napari
import numpy as np
import pandas as pd
import torch

from avae.napari import GenerativeAffinityVAEWidget


def setup_napari():
    """Setup the napari viewer."""
    BOX_SIZE = 32
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 2
    viewer.axes.visible = True
    viewer.add_image(
        np.zeros((BOX_SIZE, BOX_SIZE), dtype=np.float32),
        colormap="inferno",
        rendering="iso",
        name="Reconstruction",
        depiction="volume",
    )
    return viewer


def load_model(model_fn, meta_fn, device="cpu"):
    """Load the model and meta data."""
    print("Loading model")
    checkpoint = torch.load(model_fn, map_location=torch.device(device))
    model = checkpoint["model_class_object"]

    model.load_state_dict(checkpoint["model_state_dict"])

    meta_df = pd.read_pickle(meta_fn)

    model.to(device)
    model.eval()
    return model, meta_df


def run_napari(model_fn, meta_fn, ldim=None, pdim=None, manifold="umap"):
    """Run the napari viewer for model."""

    viewer = setup_napari()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, meta_df = load_model(model_fn, meta_fn, device=device)

    if pdim is not None:
        pose_dims = pdim
    else:
        try:
            pose_dims = model.encoder.pose_fc.out_features
        except AttributeError:
            raise AttributeError(
                "Model does not have pose attributes, please specify manually."
            )

    if ldim is not None:
        latent_dims = ldim
    else:
        try:
            latent_dims = model.encoder.mu.out_features
        except AttributeError:
            raise AttributeError(
                "Model does not have latent attributes, please specify manually."
            )

    widget = GenerativeAffinityVAEWidget(
        viewer,
        model,
        device=device,
        meta_df=meta_df,
        pose_dims=pose_dims,
        latent_dims=latent_dims,
        manifold=manifold,
    )
    viewer.window.add_dock_widget(widget, name="AffinityVAE")
    napari.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_file", help="Path to model state file.", required=True
    )
    parser.add_argument(
        "--meta_file", help="Path to the meta file.", required=True
    )
    parser.add_argument(
        "--manifold",
        help="Manifold to use for latent space. This can be either `umap` or `load`.",
        required=False,
        default="umap",
    )
    parser.add_argument(
        "--pose_dims",
        help="Number of pose dimensions. This will overwrite the internal model value.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--latent_dims",
        help="Number of latent dimensions. This will overwrite the internal model value.",
        default=None,
        required=False,
    )

    args = parser.parse_args()

    model_fn = args.model_file
    meta_fn = args.meta_file
    pdim = args.pose_dims
    ldim = args.latent_dims
    manifold = args.manifold

    if manifold != "umap" and manifold != "load":
        raise ValueError("Manifold must be either \"umap\" or \"load\".")

    print("Running napari viewer with model: {}".format(model_fn))
    run_napari(model_fn, meta_fn, ldim, pdim, manifold)
