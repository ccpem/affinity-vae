# code @quantumjot extracted from https://github.com/quantumjot/vne/blob/broadcast/vne/utils/napari.py

import enum
import typing

import matplotlib.backends.backend_qt5agg
import matplotlib.figure
import matplotlib.pyplot as plt
import napari
import numpy as np
import numpy.typing as npt
import pandas as pd
import qtpy
import torch
import umap


class CartesianAxes(enum.Enum):
    """Set of Cartesian axes as 3D vectors."""

    X = (1, 0, 0)
    Y = (0, 1, 0)
    Z = (0, 0, 1)

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.float32)


MAX_UMAP = 100.0


def process(
    model: torch.nn.Module,
    z: tuple,
    pose: float,
    *,
    device: torch.device = torch.device("cpu"),
) -> npt.NDArray:
    z = torch.tensor(
        np.array([z], dtype=np.float32),
        device=device,
    )
    pose = torch.tensor(
        np.array([pose], dtype=np.float32),
        device=device,
    )

    with torch.inference_mode():
        x = model.decoder(z, pose)

    return x.squeeze().cpu().numpy()


def scale_from_slider(x, min_val: float, max_val: float) -> float:
    scaled_value = x / 1000.0
    scaled_range = max_val - min_val
    return min_val + (scaled_value) * scaled_range


def scale_to_slider(x, min_val: float, max_val: float) -> float:
    scaled_range = max_val - min_val
    scaled_value = (x - min_val) / scaled_range
    return np.clip(((scaled_value)) * 1000, 0, 1000).astype(int)


class MplCanvas(matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        with plt.style.context("dark_background"):
            fig = matplotlib.figure.Figure(
                figsize=(width, height), dpi=dpi, frameon=False, facecolor="k"
            )
            self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class GenerativeAffinityVAEWidget(qtpy.QtWidgets.QWidget):
    def __init__(
        self,
        napari_viewer: napari.Viewer,
        model: torch.nn.Module,
        meta_df: pd.DataFrame,  # DataFrame containing both latents and embeddings
        pose_dims: int = 1,  # Number of pose dimensions
        *,
        latent_dims: int = 8,
        reconstruction_layer_name: str = "Reconstruction",
        device: torch.device = torch.device("cpu"),
        manifold: str = "umap",  # use manifold or load embeddings from metafile
    ) -> None:
        super().__init__()

        self.viewer = napari_viewer
        self._model = model
        self._layer = self.viewer.layers[reconstruction_layer_name]
        self._device = device
        self._meta_df = meta_df  # Store the DataFrame
        self.pose_dims = int(pose_dims)
        self._latent_dims = int(latent_dims)  # Number of latent dimensions
        self.cartesian = True  # Keeping this as false and as a placeholder as we havent implemented cartesian pose space yet

        self._main_layout = qtpy.QtWidgets.QVBoxLayout()
        self._tabs = qtpy.QtWidgets.QTabWidget()
        self._widgets: dict = {}

        self.add_pose_widget()
        self._main_layout.addWidget(self._tabs, stretch=0)
        self.add_latent_widget()
        self.add_manifold_widget()

        # Expand the main widget
        self.setLayout(self._main_layout)
        self._main_layout.addStretch(stretch=1)
        self.setMinimumWidth(400)
        self.manifold = manifold
        self._load_data()

        self.set_embedding(
            embedding=self._embedding, labels=self._centroids_dict
        )  # Add labels to the manifold
        self.inverse_map_manifold_to_z()  # Update the reconstruction

    def _load_data(self):
        latent_space = self._meta_df[
            [col for col in self._meta_df.columns if col.startswith("lat")]
        ].to_numpy()  # Assuming the column name for latent variables is 'latent'
        labels = self._meta_df["id"]

        pose_space = self._meta_df[
            [col for col in self._meta_df.columns if col.startswith("pos")]
        ].to_numpy()

        self._latent_range_min = latent_space.min(axis=0)
        self._latent_range_max = latent_space.max(axis=0)
        self._pose_range_min = pose_space.min(axis=0)
        self._pose_range_max = pose_space.max(axis=0)

        if self.manifold == "umap":
            self._mapper = umap.UMAP(random_state=42, n_jobs=1)
            self._embedding = self._mapper.fit_transform(latent_space)
        elif self.manifold == "load":
            self._embedding = self._meta_df[
                [col for col in self._meta_df if col.startswith("emb")]
            ].to_numpy()

        # Combine embeddings and labels into a structured array
        data = np.array(
            list(zip(self._embedding[:, 0], self._embedding[:, 1], labels)),
            dtype=[('X', float), ('Y', float), ('Label', object)],
        )

        # Calculate centroids and store as a dictionary
        self._centroids_dict = {}
        for label in np.unique(labels):
            mask = data['Label'] == label
            self._centroids_dict[label] = [
                np.mean(data[mask]['X']),
                np.mean(data[mask]['Y']),
            ]

    def add_pose_widget(self) -> None:
        """Add widgets to manipulate the model pose space."""
        pose_axes = qtpy.QtWidgets.QComboBox()
        if self.cartesian:
            pose_axes.addItems(["X", "Y", "Z"])
        else:
            for dim in range(self.pose_dims):
                pose_axes.addItems(["pose" + str(dim)])
        pose_value = qtpy.QtWidgets.QSlider(qtpy.QtCore.Qt.Horizontal)
        pose_value.setRange(0, 1000)
        pose_value.setValue(500)
        pose_value.setMinimumWidth(300)

        pose_widgets = {
            "axes": pose_axes,
            "theta": pose_value,
        }

        self._widgets.update(pose_widgets)

        layout = qtpy.QtWidgets.QFormLayout()
        for label, widget in pose_widgets.items():
            if label == "theta":
                widget.valueChanged.connect(self.update_reconstruction)
            label_widget = qtpy.QtWidgets.QLabel(label)
            layout.addRow(label_widget, widget)

        pose_widget = qtpy.QtWidgets.QGroupBox("Pose")
        pose_widget.setLayout(layout)

        self._main_layout.addWidget(pose_widget)

    def add_latent_widget(self) -> None:
        """Add widgets to manipulate the model latent space."""

        def _z_widget(idx):
            z_value = qtpy.QtWidgets.QSlider(qtpy.QtCore.Qt.Horizontal)
            z_value.setRange(0, 1000)
            z_value.setValue(500)
            z_value.setMinimumWidth(300)
            return z_value

        latent_widgets = {
            f"z{idx}": _z_widget(idx) for idx in range(self._latent_dims)
        }

        self._widgets.update(latent_widgets)

        layout = qtpy.QtWidgets.QFormLayout()
        for label, widget in latent_widgets.items():
            widget.valueChanged.connect(self.update_reconstruction)
            label_widget = qtpy.QtWidgets.QLabel(label)
            layout.addRow(label_widget, widget)

        tab = qtpy.QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "Latents")

    def add_manifold_widget(self) -> None:
        manifold_widget = MplCanvas()
        manifold_widget.axes.set_title("Latent manifold")
        layout = qtpy.QtWidgets.QVBoxLayout()
        layout.addWidget(manifold_widget)
        tab = qtpy.QtWidgets.QWidget()
        tab.setLayout(layout)

        manifold_widget.figure.canvas.mpl_connect(
            "button_press_event", self.inverse_map_manifold_to_z
        )
        self._tabs.addTab(tab, "Manifold")

        self._widgets.update(
            {
                "manifold": manifold_widget,
            }
        )

    def get_pose(self) -> npt.NDArray:

        if self.cartesian:
            theta = scale_from_slider(
                self._widgets["theta"].value(), np.pi, 2 * np.pi
            )
            axis = CartesianAxes[str(self._widgets["axes"].currentText())]
            return np.array([theta, *axis.value], dtype=np.float32)

        else:
            theta = scale_from_slider(
                self._widgets["theta"].value(),
                self._pose_range_min,
                self._pose_range_max,
            )
            return np.array(theta, dtype=np.float32)

    def get_state(self) -> tuple[float, npt.NDArray]:
        pose = self.get_pose()
        z_values = np.array(
            [
                self._widgets[f"z{idx}"].value()
                for idx in range(self._latent_dims)
            ]
        )
        z = scale_from_slider(
            z_values, self._latent_range_min, self._latent_range_max
        )
        return pose, z

    def inverse_map_manifold_to_z(self, event: typing.Any = None) -> None:

        if event is None:
            pt = self._embedding[0].reshape(
                1, -1
            )  # Use the first point in the manifold as a test point
        else:
            pt = np.array([event.xdata, event.ydata]).reshape(1, -1)

        # Retrieve latent variables based on the clicked index in the DataFrame
        if self.manifold == "umap" and self._mapper is not None:
            inv_transformed_points = self._mapper.inverse_transform(pt)[0]
        else:
            clicked_index = self.get_clicked_index(pt)
            inv_transformed_points = self._meta_df.iloc[clicked_index][
                [col for col in self._meta_df.columns if col.startswith("lat")]
            ].values  # Assuming the column name for latent variables is 'latent'

            # for debugging
            print('Data point clicked:')
            print(self._meta_df.iloc[clicked_index]["id"])
        print("Latent variables clicked:")
        print(inv_transformed_points)

        pose = self.get_pose()
        self._layer.data = process(
            self._model, inv_transformed_points, pose, device=self._device
        )

        transformed = [
            scale_to_slider(
                pt,
                self._latent_range_min[z_dim],
                self._latent_range_max[z_dim],
            )
            for z_dim, pt in enumerate(np.squeeze(inv_transformed_points))
        ]

        for idx in range(self._latent_dims):
            slider = self._widgets[f"z{idx}"]
            with qtpy.QtCore.QSignalBlocker(slider):
                slider.setValue(transformed[idx])

    def get_clicked_index(self, test_pt) -> str | None:
        # Helper method to retrieve the index of the clicked point in the DataFrame
        clicked_index = None
        try:
            distances = np.sqrt(
                np.sum(
                    (self._meta_df[['emb-x', 'emb-y']] - test_pt) ** 2, axis=1
                )
            )
            # Find the index of the point with the minimum distance
            clicked_index = np.argmin(distances)
        except Exception as e:
            print(f"Error finding index: {e}")
        return clicked_index

    def update_reconstruction(self) -> None:
        print('Updating reconstruction')
        pose, z = self.get_state()
        print(pose)
        print(z)
        self._layer.data = process(self._model, z, pose, device=self._device)

    def set_embedding(
        self,
        embedding: npt.NDArray,
        *,
        labels: dict | npt.NDArray | None = None,
    ) -> None:
        from matplotlib import colormaps
        from scipy.stats import gaussian_kde

        kernel = gaussian_kde(embedding.T)

        xmin, xmax = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
        ymin, ymax = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1

        X, Y = np.meshgrid(
            np.linspace(xmin, xmax, 512),
            np.linspace(ymin, ymax, 512),
            indexing="xy",
        )
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)

        self._widgets["manifold"].axes.imshow(
            np.flipud(Z),
            extent=[xmin, xmax, ymin, ymax],
            cmap=colormaps.get_cmap("inferno"),
        )

        if labels is None:
            return

        with plt.style.context("dark_background"):
            for label, centroid in labels.items():
                self._widgets["manifold"].axes.text(
                    centroid[0], centroid[1], str(label)
                )

        self._widgets["manifold"].axes.set_aspect("equal")
        self._widgets["manifold"].axes.get_xaxis().set_label_text("UMAP-1")
        self._widgets["manifold"].axes.get_yaxis().set_label_text("UMAP-2")
