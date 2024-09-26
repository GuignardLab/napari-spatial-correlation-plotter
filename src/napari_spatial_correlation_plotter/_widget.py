"""
Copyright (c) 2022, DFG Cluster of Excellence "Physics of Life" TU Dresden: Robert Haase, Laura Zigutyte, Marcello Zoccoler, Ryan Savill, Johannes Müller and Max Planck Institute of Molecular Physiology Dortmund: Thorsten Wagner
"""
import os
from pathlib import Path as PathL
from time import time

import matplotlib.pyplot as plt
import napari
import numpy as np
from magicgui.widgets import Container, EmptyWidget, create_widget
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector
from napari.layers import Image, Labels, Layer
from napari.utils import DirectLabelColormap
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication, QIcon
from skimage.measure import regionprops
from tapenade.analysis.spatial_correlation import SpatialCorrelationPlotter
from tapenade.preprocessing import masked_gaussian_smooth_dense_two_arrays_gpu
from vispy.color import Color

from napari_spatial_correlation_plotter._nice_colormap import get_nice_colormap

ICON_ROOT = PathL(__file__).parent / "icons"

"""
This widget takes HEAVY inspiration from the napari-clusters-plotter plugin [1],
most notably the SelectFromCollection, MplCanvas and FigureToolbar classes.
The PlotterWidget class has been modified for the specific use case of this
plugin, but the core functionalites have been adapted directly.

napari-clusters-plotter source code is available at
https://github.com/BiAPoL/napari-clusters-plotter/tree/main


[1] Zigutyte, L., Savill, R., Müller, J., Zoccoler, M., Wagner, T., & Haase, R. (2023).
napari-clusters-plotter. Zenodo. https://doi.org/10.5281/zenodo.5884657
"""

# TODO:
# - add log scale to heatmap colors


colors = get_nice_colormap()
cmap = [Color(hex_name).RGBA.astype("float") / 255 for hex_name in colors]


def in_bbox(min_x, max_x, min_y, max_y, xys):
    mins = np.array([min_x, min_y]).reshape(1, 2)
    maxs = np.array([max_x, max_y]).reshape(1, 2)

    foo = np.logical_and(xys >= mins, xys <= maxs)

    return np.logical_and(foo[:, 0], foo[:, 1])


# Class below was based upon matplotlib lasso selection example:
# https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html
class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.
    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.
    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, parent, ax, xys, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.parent = parent
        # self.collection = collection
        # self.alpha_other = alpha_other

        # self.xys = collection.get_offsets()
        self.xys = xys
        # self.Npts = len(self.xys)

        self.lasso = LassoSelector(ax, onselect=self.onselect, button=1)
        self.ind = []
        self.ind_mask = []

    def onselect(self, verts):
        verts = np.array(verts)
        min_x, min_y = np.min(verts, axis=0)
        max_x, max_y = np.max(verts, axis=0)

        ind_mask = in_bbox(min_x, max_x, min_y, max_y, self.xys)

        path = Path(verts)
        # ind_mask = np.where(
        #     ind_mask, path.contains_points(self.xys[ind_mask]), False
        # )
        ind_mask[ind_mask] = path.contains_points(self.xys[ind_mask])
        self.ind_mask = ind_mask

        self.canvas.draw_idle()
        # self.selected_coordinates = self.xys[self.ind].data

        if self.parent.manual_clustering_method is not None:
            self.parent.manual_clustering_method(self.ind_mask)

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


class MplCanvas(FigureCanvas):
    def __init__(
        self,
        xys,
        parent=None,
        width=7,
        height=4,
        manual_clustering_method=None,
        create_selectors=False,
    ):

        self.xys = xys

        if parent is None:
            self.fig = Figure(figsize=(width, height))
            self.axes = self.fig.add_subplot(111)
        else:
            self.fig = parent
            if len(self.fig.axes) == 0:
                self.fig.add_subplot(111)
            self.axes = self.fig.axes[0]
            # figure size
            self.fig.set_size_inches(width, height)
        self.manual_clustering_method = manual_clustering_method
        self.fig.tight_layout()

        super().__init__(self.fig)

        self.reset_params(create_selectors=create_selectors, xys=xys)

    def reset_params(self, create_selectors, xys):
        self.axes = self.fig.axes[0]

        if len(self.axes.collections) == 0:
            self.pts = self.axes.scatter([], [])

        self.pts = self.axes.collections[0]

        self.fig.patch.set_facecolor("#262930")

        # changing color of plot background to napari main window color
        if create_selectors:
            self.axes.set_facecolor("white")
        else:
            self.axes.set_facecolor("#262930")

        # changing colors of all axes
        self.axes.spines["bottom"].set_color("white")
        self.axes.spines["top"].set_color("white")
        self.axes.spines["right"].set_color("white")
        self.axes.spines["left"].set_color("white")
        self.axes.xaxis.label.set_color("white")
        self.axes.yaxis.label.set_color("white")

        # changing colors of axes labels
        self.axes.tick_params(axis="x", colors="white")
        self.axes.tick_params(axis="y", colors="white")

        # COLORBAR
        # extract already existing colobar from figure
        if len(self.fig.axes) > 1:
            cb = self.axes.images[0].colorbar
            cb_label = cb.ax.get_ylabel()
            # set colorbar label plus label color
            cb.set_label(cb_label, color="white")

            # set colorbar tick color
            cb.ax.yaxis.set_tick_params(color="white")

            # set colorbar edgecolor
            cb.outline.set_edgecolor("white")

            # set colorbar ticklabels
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

        if create_selectors:
            self.selector = SelectFromCollection(self, self.axes, xys)
            # Rectangle
            self.rectangle_selector = RectangleSelector(
                self.axes,
                self.draw_rectangle,
                useblit=True,
                props=dict(edgecolor="#1f77b4", fill=False),
                button=3,  # right button
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=False,
            )

    def draw_rectangle(self, eclick, erelease):
        """eclick and erelease are the press and release events"""
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)

        self.rect_ind_mask = in_bbox(min_x, max_x, min_y, max_y, self.xys)

        if self.manual_clustering_method is not None:
            self.manual_clustering_method(self.rect_ind_mask)

    def reset(self):
        self.axes.clear()
        self.is_pressed = None


class FigureToolbar(NavigationToolbar):
    def __init__(self, canvas):
        super().__init__(canvas, None)
        self.canvas = canvas

    def _update_buttons_checked(self):
        super()._update_buttons_checked()
        # changes pan/zoom icons depending on state (checked or not)
        if "pan" in self._actions:
            if self._actions["pan"].isChecked():
                self._actions["pan"].setIcon(
                    QIcon(os.path.join(ICON_ROOT, "Pan_checked.png"))
                )
            else:
                self._actions["pan"].setIcon(
                    QIcon(os.path.join(ICON_ROOT, "Pan.png"))
                )
        if "zoom" in self._actions:
            if self._actions["zoom"].isChecked():
                self._actions["zoom"].setIcon(
                    QIcon(os.path.join(ICON_ROOT, "Zoom_checked.png"))
                )
            else:
                self._actions["zoom"].setIcon(
                    QIcon(os.path.join(ICON_ROOT, "Zoom.png"))
                )

    def save_figure(self):
        self.canvas.fig.patch.set_facecolor("white")

        # changing color of plot background to napari main window color
        self.canvas.axes.set_facecolor("white")

        # changing colors of all axes
        self.canvas.axes.spines["bottom"].set_color("black")
        self.canvas.axes.spines["top"].set_color("black")
        self.canvas.axes.spines["right"].set_color("black")
        self.canvas.axes.spines["left"].set_color("black")
        self.canvas.axes.xaxis.label.set_color("black")
        self.canvas.axes.yaxis.label.set_color("black")

        # changing colors of axes labels
        self.canvas.axes.tick_params(axis="x", colors="black")
        self.canvas.axes.tick_params(axis="y", colors="black")

        # COLORBAR
        # extract already existing colobar from figure
        if len(self.canvas.fig.axes) > 0:
            cb = self.canvas.axes.images[0].colorbar
            cb_label = cb.ax.get_ylabel()
            # set colorbar label plus label color
            cb.set_label(cb_label, color="black")

            # set colorbar tick color
            cb.ax.yaxis.set_tick_params(color="black")

            # set colorbar edgecolor
            cb.outline.set_edgecolor("black")

            # set colorbar ticklabels
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="black")

        super().save_figure()

        self.canvas.fig.patch.set_facecolor("#262930")

        # changing color of plot background to napari main window color
        self.canvas.axes.set_facecolor("#262930")

        # changing colors of all axes
        self.canvas.axes.spines["bottom"].set_color("white")
        self.canvas.axes.spines["top"].set_color("white")
        self.canvas.axes.spines["right"].set_color("white")
        self.canvas.axes.spines["left"].set_color("white")
        self.canvas.axes.xaxis.label.set_color("white")
        self.canvas.axes.yaxis.label.set_color("white")

        # changing colors of axes labels
        self.canvas.axes.tick_params(axis="x", colors="white")
        self.canvas.axes.tick_params(axis="y", colors="white")

        # COLORBAR
        # extract already existing colobar from figure
        if len(self.canvas.fig.axes) > 0:
            cb = self.canvas.axes.images[0].colorbar
            cb_label = cb.ax.get_ylabel()
            # set colorbar label plus label color
            cb.set_label(cb_label, color="white")

            # set colorbar tick color
            cb.ax.yaxis.set_tick_params(color="white")

            # set colorbar edgecolor
            cb.outline.set_edgecolor("white")

            # set colorbar ticklabels
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

        self.canvas.draw()


class PlotterWidget(Container):
    def __init__(self, napari_viewer):
        super().__init__()

        self.cluster_ids = None
        self._viewer = napari_viewer

        self.cluster_labels_layer = None
        self.quantityX_smoothed_layer = None
        self.quantityY_smoothed_layer = None

        self.quantityX_labels_choices_displayed = False
        self.quantityY_labels_choices_displayed = False

        self.histogram_displayed = False

        self.figure = None

        self.labels_method_choices = ["cellular density", "volume fraction"]
        self._hidden_features = {}

        # Canvas Widget that displays the 'figure', it takes the 'figure' instance
        if True:
            self.graphics_widget = MplCanvas(
                manual_clustering_method=self.manual_clustering_method,
                xys=None,
            )
            self.toolbar = FigureToolbar(self.graphics_widget)

            self.toolbar.native = self.toolbar
            self.toolbar._explicitly_hidden = False
            self.toolbar.name = ""

            self.graphics_widget.native = self.graphics_widget
            self.graphics_widget._explicitly_hidden = False
            self.graphics_widget.name = ""

            self.graph_container = Container(
                widgets=[
                    self.toolbar,
                    self.graphics_widget,
                ],
                labels=False,
            )

            self.quantityX_layer_combo = create_widget(
                annotation=Layer,
                label="Quantity X",
                options={"choices": self._image_labels_layers_filter},
            )

            self.quantityX_layer_combo.changed.connect(
                self._update_quantities_labels_choices
            )

            self.quantityX_labels_choices_combo = create_widget(
                widget_type="ComboBox",
                options={"choices": self.labels_method_choices},
            )
            self.quantityX_labels_choices_container = Container(
                widgets=[self.quantityX_labels_choices_combo],
                labels=False,
                layout="horizontal",
            )

            self.quantityY_layer_combo = create_widget(
                annotation=Layer,
                label="Quantity Y",
                options={"choices": self._image_labels_layers_filter},
            )

            self.quantityY_layer_combo.changed.connect(
                self._update_quantities_labels_choices
            )

            self.quantityY_labels_choices_combo = create_widget(
                widget_type="ComboBox",
                options={"choices": self.labels_method_choices},
            )
            self.quantityY_labels_choices_container = Container(
                widgets=[self.quantityY_labels_choices_combo],
                labels=False,
                layout="horizontal",
            )

            self.mask_layer_combo = create_widget(
                annotation=Image,
                label="Mask layer",
                options={
                    "nullable": True,
                    "choices": self._bool_layers_filter,
                },
            )

            self.labels_layer_combo = create_widget(
                annotation=Labels,
                label="Labels layer",
                options={"nullable": True},
            )

            self.blur_sigma_slider = create_widget(
                widget_type="IntSlider",
                label="Blur sigma",
                options={"min": 0, "max": 50, "value": 1},
            )

            # self.blur_sigma_slider.changed.connect(self.sigma_changed)
            self.run_button = create_widget(
                widget_type="PushButton",
                label="Compute correlation heatmap",
            )

            self.run_button.clicked.connect(self.run)

            self.show_individual_cells_checkbox = create_widget(
                annotation=bool,
                label="Show individual cells",
            )

            self.show_individual_cells_checkbox.changed.connect(
                self.parameters_changed
            )

            self.show_linear_fit_checkbox = create_widget(
                annotation=bool,
                label="Show linear fit",
            )

            self.show_linear_fit_checkbox.changed.connect(
                self.parameters_changed
            )

            #! normalize is currently broken with manual selection
            # self.normalize_quantities_checkbox = create_widget(
            #     annotation=bool, label="Normalize quantities",
            # )

            # self.normalize_quantities_checkbox.changed.connect(self.parameters_changed)

            self.display_quadrants = create_widget(
                annotation=bool,
                label="Display quadrants",
            )

            self.display_quadrants.changed.connect(self.parameters_changed)

            self.options_container1 = Container(
                widgets=[
                    self.show_individual_cells_checkbox,
                    self.show_linear_fit_checkbox,
                ],
                labels=False,
                layout="horizontal",
            )
            self.options_container2 = Container(
                widgets=[
                    # self.normalize_quantities_checkbox, #! normalize is currently broken with manual selection
                    self.display_quadrants,
                ],
                labels=False,
                layout="horizontal",
            )

            self.heatmap_binsX = create_widget(
                widget_type="IntSlider",
                label="X",
                value=40,
                options={"min": 2, "max": 100, "tracking": True},
            )

            self.heatmap_binsX.changed.connect(self.parameters_changed)

            self.heatmap_binsY = create_widget(
                widget_type="IntSlider",
                label="Y",
                value=40,
                options={"min": 2, "max": 100, "tracking": True},
            )

            self.heatmap_binsY.changed.connect(self.parameters_changed)

            self.heatmap_bins_container = Container(
                widgets=[
                    self.heatmap_binsX,
                    self.heatmap_binsY,
                ],
                labels=True,
                label="Heatmap bins",
                layout="horizontal",
            )

            self.percentilesX = create_widget(
                widget_type="FloatRangeSlider",
                label="X",
                options={
                    "min": 0,
                    "max": 100,
                    "value": [0, 100],
                    "tracking": True,
                },
            )

            self.percentilesX.changed.connect(self.parameters_changed)

            self.percentilesY = create_widget(
                widget_type="FloatRangeSlider",
                label="Y",
                options={
                    "min": 0,
                    "max": 100,
                    "value": [0, 100],
                    "tracking": True,
                },
            )

            self.percentilesY.changed.connect(self.parameters_changed)

            self.percentiles_container = Container(
                widgets=[
                    self.percentilesX,
                    self.percentilesY,
                ],
                labels=True,
                label="Percentiles",
                layout="horizontal",
            )

            parameters_text = EmptyWidget(label="<u>Parameters:</u>")

            display_parameters_text = EmptyWidget(
                label="<u>Display Parameters:</u>"
            )

            self.extend(
                [
                    parameters_text,
                    self.quantityX_layer_combo,
                    self.quantityY_layer_combo,
                    self.mask_layer_combo,
                    self.labels_layer_combo,
                    self.blur_sigma_slider,
                    self.run_button,
                    self.graph_container,
                    display_parameters_text,
                    self.options_container1,
                    self.options_container2,
                    self.heatmap_bins_container,
                    self.percentiles_container,
                ]
            )

        # takes care of case where this isn't set yet directly after init
        self.plot_cluster_name = None

        self.id = 0

    def manual_clustering_method(self, inside):

        inside = np.array(inside)  # leads to errors sometimes otherwise
        if len(inside) == 0:
            return  # if nothing was plotted yet, leave

        clustering_ID = "MANUAL_CLUSTER_ID"

        modifiers = QGuiApplication.keyboardModifiers()
        if (
            modifiers == Qt.ShiftModifier
            and clustering_ID in self._hidden_features.keys()
        ):
            former_clusters = self._hidden_features[clustering_ID]
            former_clusters[inside] = np.max(former_clusters) + 1
            self._hidden_features.update({clustering_ID: former_clusters})
        else:
            self._hidden_features[clustering_ID] = inside.astype(int)

        # redraw the whole plot
        self.draw_cluster_labels(
            self._hidden_features,
            plot_cluster_name=clustering_ID,
        )

    def run(self):
        # Check if all necessary layers are specified
        if self.quantityX_layer_combo.value is None:
            napari.utils.notifications.show_warning(
                "Please specify quantityX_layer"
            )
            return
        else:
            self.quantityX = self.quantityX_layer_combo.value.data
            self.quantityX_label = self.quantityX_layer_combo.value.name
            if isinstance(self.quantityX_layer_combo.value, Labels):
                self.quantityX_colormap = "inferno"
            else:
                self.quantityX_colormap = (
                    self.quantityX_layer_combo.value.colormap
                )
            self.quantityX_is_labels = isinstance(
                self.quantityX_layer_combo.value, Labels
            )
            self.quantityX_labels_choice = (
                self.quantityX_labels_choices_combo.value
            )

        if self.quantityY_layer_combo.value is None:
            napari.utils.notifications.show_warning(
                "Please specify quantityY_layer"
            )
            return
        else:
            self.quantityY = self.quantityY_layer_combo.value.data
            self.quantityY_label = self.quantityY_layer_combo.value.name
            if isinstance(self.quantityY_layer_combo.value, Labels):
                self.quantityY_colormap = "inferno"
            else:
                self.quantityY_colormap = (
                    self.quantityY_layer_combo.value.colormap
                )
            self.quantityY_is_labels = isinstance(
                self.quantityY_layer_combo.value, Labels
            )
            self.quantityY_labels_choice = (
                self.quantityY_labels_choices_combo.value
            )

        if self.mask_layer_combo.value is not None:
            self.mask = self.mask_layer_combo.value.data
        else:
            self.mask = None

        if self.labels_layer_combo.value is not None:
            self.labels_image = self.labels_layer_combo.value.data
        else:
            self.labels_image = None
            if self.mask is not None:
                self.argwheres = np.argwhere(self.mask)
            else:
                shape = self.quantityX.shape
                # use np.mgrid
                self.argwheres = (
                    np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
                    .reshape(3, -1)
                    .T
                )

        # Blur the layers
        smoothedX, smoothedY = self._smooth_quantities(
            self.quantityX,
            self.quantityX_is_labels,
            self.quantityX_labels_choice,
            self.quantityY,
            self.quantityY_is_labels,
            self.quantityY_labels_choice,
            self.mask,
        )

        self._update_smoothed_layers(
            smoothedX,
            self.quantityX_colormap,
            smoothedY,
            self.quantityY_colormap,
        )
        self.plot_from_smoothed(
            smoothedX,
            self.quantityX_is_labels,
            self.quantityX_label,
            self.quantityX_labels_choice,
            smoothedY,
            self.quantityY_is_labels,
            self.quantityY_label,
            self.quantityY_labels_choice,
            self.mask,
            self.labels_image,
        )

        # Set a parameter "self.histogram_displayed" to True
        self.histogram_displayed = True

        if self.cluster_labels_layer is not None:
            self.cluster_labels_layer.data = np.zeros_like(
                self.cluster_labels_layer.data
            )

    def _update_smoothed_layers(
        self, blurredX, X_colormap, blurredY, Y_colormap
    ):
        if (
            self.quantityX_smoothed_layer is None
            or self.quantityX_smoothed_layer not in self._viewer.layers
        ):
            self.quantityX_smoothed_layer = self._viewer.add_image(
                blurredX, colormap=X_colormap
            )
        else:
            self.quantityX_smoothed_layer.data = blurredX
            self.quantityX_smoothed_layer.contrast_limits = (
                blurredX.min(), blurredX.max()
            )
            self.quantityX_smoothed_layer.refresh()

        if (
            self.quantityY_smoothed_layer is None
            or self.quantityY_smoothed_layer not in self._viewer.layers
        ):
            self.quantityY_smoothed_layer = self._viewer.add_image(
                blurredY, colormap=Y_colormap
            )
        else:
            self.quantityY_smoothed_layer.data = blurredY
            self.quantityY_smoothed_layer.contrast_limits = (
                blurredY.min(), blurredY.max()
            )
            self.quantityY_smoothed_layer.refresh()

    def _update_quantities_labels_choices(self, event):

        if isinstance(self.quantityX_layer_combo.value, Labels):
            if not self.quantityX_labels_choices_displayed:

                self.insert(
                    self.index(self.quantityX_layer_combo) + 1,
                    self.quantityX_labels_choices_container,
                )

                self.quantityX_labels_choices_displayed = True

        else:
            if self.quantityX_labels_choices_displayed:

                self.remove(self.quantityX_labels_choices_container)
                self.quantityX_labels_choices_displayed = False

        if isinstance(self.quantityY_layer_combo.value, Labels):
            if not self.quantityY_labels_choices_displayed:

                self.insert(
                    self.index(self.quantityY_layer_combo) + 1,
                    self.quantityY_labels_choices_container,
                )

                self.quantityY_labels_choices_displayed = True

        else:
            if self.quantityY_labels_choices_displayed:

                self.remove(self.quantityY_labels_choices_container)
                self.quantityY_labels_choices_displayed = False

    def _transform_labels_to_density(self, labels, method):
        self.test_value = True
        if method == self.labels_method_choices[0]:
            props = regionprops(labels)
            centroids = np.array([prop.centroid for prop in props]).astype(int)

            labels = np.zeros(labels.shape, dtype=bool)

            labels[centroids[:, 0], centroids[:, 1], centroids[:, 2]] = True

            return labels

        elif method == self.labels_method_choices[1]:

            return labels.astype(bool)

    def _smooth_quantities(
        self,
        quantityX,
        quantityX_is_labels,
        quantityX_labels_choice,
        quantityY,
        quantityY_is_labels,
        quantityY_labels_choice,
        mask,
    ):

        masks_volume = []

        if quantityX_is_labels:
            quantityX = self._transform_labels_to_density(
                quantityX, quantityX_labels_choice
            )
            masks_volume.append(None)
        if quantityY_is_labels:
            quantityY = self._transform_labels_to_density(
                quantityY, quantityY_labels_choice
            )
            masks_volume.append(None)

        if mask is None or all([elem is None for elem in masks_volume]):
            masks_volume = None

        if self.blur_sigma_slider.value > 0:
            smoothedX, smoothedY = masked_gaussian_smooth_dense_two_arrays_gpu(
                datas=[quantityX, quantityY],
                sigmas=self.blur_sigma_slider.value,
                mask=mask,
                masks_for_volume=masks_volume,
            )
        else:
            smoothedX, smoothedY = quantityX, quantityY

        return smoothedX, smoothedY

    def plot_from_smoothed(
        self,
        smoothedX,
        quantityX_is_labels,
        quantityX_label,
        quantityX_labels_choice,
        smoothedY,
        quantityY_is_labels,
        quantityY_label,
        quantityY_labels_choice,
        mask,
        labels,
    ):
        # Construct HeatmapPlotter
        self.heatmap_plotter = SpatialCorrelationPlotter(
            quantity_X=smoothedX,
            quantity_Y=smoothedY,
            mask=mask,
            labels=labels,
        )

        if quantityX_is_labels:
            labelX = quantityX_labels_choice
        else:
            labelX = quantityX_label

        if quantityY_is_labels:
            labelY = quantityY_labels_choice
        else:
            labelY = quantityY_label

        # Get figure from HeatmapPlotter
        figure, _ = self.heatmap_plotter.get_heatmap_figure(
            bins=(self.heatmap_binsX.value, self.heatmap_binsY.value),
            show_individual_cells=self.show_individual_cells_checkbox.value,
            show_linear_fit=self.show_linear_fit_checkbox.value,
            # normalize_quantities=self.normalize_quantities_checkbox.value,
            normalize_quantities=False,  #! normalize is currently broken with manual selection
            percentiles_X=self.percentilesX.value,
            percentiles_Y=self.percentilesY.value,
            figsize=self.graphics_widget.figure.get_size_inches(),
            label_X=labelX,
            label_Y=labelY,
        )

        # Display figure in graphics_widget
        self.plot_heatmap(figure)

    def plot_heatmap(self, figure):

        if self.figure is not None:
            plt.close(self.figure)
        self.figure = figure

        # labels_layer_exists = self.labels_layer_combo.value is not None

        xys = self.heatmap_plotter.xys

        self.graphics_widget = MplCanvas(
            parent=figure,
            manual_clustering_method=self.manual_clustering_method,
            create_selectors=True,  # labels_layer_exists,
            xys=xys,
        )
        self.toolbar = FigureToolbar(self.graphics_widget)

        self.toolbar.native = self.toolbar
        self.toolbar._explicitly_hidden = False
        self.toolbar.name = ""

        self.graphics_widget.native = self.graphics_widget
        self.graphics_widget._explicitly_hidden = False
        self.graphics_widget.name = ""

        new_graph_container = Container(
            widgets=[
                self.toolbar,
                self.graphics_widget,
            ],
            labels=False,
        )

        widget_index = self.index(self.graph_container)
        self.remove(self.graph_container)
        self.insert(widget_index, new_graph_container)
        self.graph_container = new_graph_container
        self.graphics_widget.draw()

    def parameters_changed(self):
        if self.histogram_displayed:

            labelX = (
                self.quantityX_labels_choice
                if self.quantityX_is_labels
                else self.quantityX_label
            )
            labelY = (
                self.quantityY_labels_choice
                if self.quantityY_is_labels
                else self.quantityY_label
            )

            t0 = time()
            # Get figure from HeatmapPlotter
            figure, _ = self.heatmap_plotter.get_heatmap_figure(
                bins=(self.heatmap_binsX.value, self.heatmap_binsY.value),
                show_individual_cells=self.show_individual_cells_checkbox.value,
                show_linear_fit=self.show_linear_fit_checkbox.value,
                # normalize_quantities=self.normalize_quantities_checkbox.value, #! normalize is currently broken with manual selection
                normalize_quantities=False,
                percentiles_X=self.percentilesX.value,
                percentiles_Y=self.percentilesY.value,
                figsize=self.graphics_widget.figure.get_size_inches(),
                label_X=labelX,
                label_Y=labelY,
                display_quadrants=self.display_quadrants.value,
            )
            print("Time to get figure:", time() - t0)

            # Display figure in graphics_widget -> Create a method "self.plot"
            self.plot_heatmap(figure)

    def _image_labels_layers_filter(self, wdg):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, Image | Labels)
        ]

    def _bool_layers_filter(self, wdg):
        return [
            layer
            for layer in self._viewer.layers
            if (isinstance(layer, Image) and layer.data.dtype == bool)
        ]

    def draw_cluster_labels(
        self,
        features,
        plot_cluster_name=None,
    ):
        """
        Takes the manually selected points and plot the cluster on the labels image
        """

        # self.analysed_layer = self.labels_select.value
        # labels_layer = self.labels_layer_combo.value
        # mask_layer = self.mask_layer_combo.value
        # self.graphics_widget.reset()

        # fill all prediction nan values with -1
        self.cluster_ids = features[plot_cluster_name]  # .fillna(-1)

        self.graphics_widget.selector.disconnect()
        self.graphics_widget.selector = SelectFromCollection(
            self.graphics_widget,
            self.graphics_widget.axes,
            xys=self.graphics_widget.xys,
        )

        # generate dictionary mapping each prediction to its respective color
        # list cycling with  % introduced for all labels except hdbscan noise points (id = -1)
        cmap_dict = {
            int(prediction + 1): (
                cmap[int(prediction) % len(cmap)]
                if prediction > 0
                else [0, 0, 0, 0]
            )
            for prediction in range(np.max(self.cluster_ids) + 1)
        }
        # take care of background label
        cmap_dict[None] = [0, 0, 0, 0]

        napari_cmap = DirectLabelColormap(color_dict=cmap_dict)

        keep_selection = list(self._viewer.layers.selection)

        if self.labels_image is not None:
            cluster_image = self.generate_cluster_image_from_labels(
                self.labels_image, self.cluster_ids
            )

        elif self.mask is not None:
            cluster_image = self.generate_cluster_image_from_points(
                self.argwheres,
                self.cluster_ids,
                shape=self.quantityX.shape,
            )
        else:
            cluster_image = self.generate_cluster_image_from_points(
                self.argwheres,
                self.cluster_ids,
                shape=self.quantityX_layer_combo.value.data.shape,
            )

        # if the cluster image layer doesn't yet exist make it
        # otherwise just update it
        if (
            self.cluster_labels_layer is None
            or self.cluster_labels_layer not in self._viewer.layers
        ):
            # visualising cluster image
            self.cluster_labels_layer = self._viewer.add_labels(
                cluster_image,  # self.analysed_layer.data
                colormap=napari_cmap,  # cluster_id_dict
                name="clustered labels",
                opacity=1,
            )
        else:
            # updating data
            self.cluster_labels_layer.data = cluster_image
            self.cluster_labels_layer.colormap = napari_cmap

        self._viewer.layers.selection.clear()
        for s in keep_selection:
            self._viewer.layers.selection.add(s)

    def generate_cluster_image_from_labels(self, label_image, predictionlist):
        props = regionprops(label_image)

        cluster_image = np.zeros(label_image.shape, dtype="uint8")

        argwheres = np.argwhere(predictionlist > 0).flatten()

        for index in argwheres:
            prop = props[index]
            roi_data = label_image[prop.slice]
            cluster_image[prop.slice][roi_data == prop.label] = (
                predictionlist[index] + 1
            )

        return cluster_image

    def generate_cluster_image_from_points(
        self, argwheres, predictionlist, shape
    ):

        cluster_image = np.zeros(shape, dtype="uint8")
        points_to_display = argwheres[predictionlist > 0]

        cluster_image[tuple(points_to_display.T)] = (
            predictionlist[predictionlist > 0] + 1
        )

        return cluster_image
