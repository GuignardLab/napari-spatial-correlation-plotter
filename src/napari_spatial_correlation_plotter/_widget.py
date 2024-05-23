
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path as PathL
from magicgui.widgets import Container, create_widget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector
from napari.layers import Labels, Image, Layer
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication, QIcon
from skimage.measure import regionprops
from organoid.analysis.spatial_correlation._spatial_correlation_plotter import SpatialCorrelationPlotter
from napari_spatial_correlation_plotter._nice_colormap import get_nice_colormap
from vispy.color import Color
from napari.utils import DirectLabelColormap
from pyngs.dense_smoothing_3d_gpu import gaussian_smooth_dense_two_arrays_gpu



ICON_ROOT = PathL(__file__).parent / "icons"


# get colormap as rgba array

colors = get_nice_colormap()
cmap = [Color(hex_name).RGBA.astype("float") / 255 for hex_name in colors]

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

    def __init__(self, parent, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.parent = parent
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.lasso = LassoSelector(ax, onselect=self.onselect, button=1)
        self.ind = []
        self.ind_mask = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind_mask = path.contains_points(self.xys)
        self.ind = np.nonzero(self.ind_mask)[0]

        self.canvas.draw_idle()
        self.selected_coordinates = self.xys[self.ind].data

        if self.parent.manual_clustering_method is not None:
            self.parent.manual_clustering_method(self.ind_mask)

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4, 
                 manual_clustering_method=None, create_selectors=False):

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

        self.reset_params(create_selectors=create_selectors)


    def reset_params(self, create_selectors):
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
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")

        if create_selectors:
            self.selector = SelectFromCollection(self, self.axes, self.pts)
            # Rectangle
            self.rectangle_selector = RectangleSelector(
                self.axes,
                self.draw_rectangle,
                useblit=True,
                props=dict(edgecolor='#1f77b4', fill=False),
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
        self.xys = self.pts.get_offsets()
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        self.rect_ind_mask = [
            min_x <= x <= max_x and min_y <= y <= max_y
            for x, y in zip(self.xys[:, 0], self.xys[:, 1])
        ]
        if self.manual_clustering_method is not None:
            self.manual_clustering_method(self.rect_ind_mask)

    def reset(self):
        self.axes.clear()
        self.is_pressed = None

class MyNavigationToolbar(NavigationToolbar):
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
                self._actions["pan"].setIcon(QIcon(os.path.join(ICON_ROOT, "Pan.png")))
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
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="black")

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
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")

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

        self.labels_method_choices = ['cellular density', 'volume fraction']
        

        # Canvas Widget that displays the 'figure', it takes the 'figure' instance
        if True:
            self.graphics_widget = MplCanvas(
                manual_clustering_method=self.manual_clustering_method
            )
            self.toolbar = MyNavigationToolbar(self.graphics_widget)

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
                labels=False
            )

            self.quantityX_layer_combo = create_widget(
                annotation=Layer, label="quantityX_layer",
                options={'choices': self._image_labels_layers_filter}
            )

            self.quantityX_layer_combo.changed.connect(
                self._update_quantities_labels_choices
            )

            self.quantityX_labels_choices_combo = create_widget(
                widget_type="ComboBox",
                options={'choices': self.labels_method_choices}
            )
            self.quantityX_labels_choices_container = Container(
                widgets=[self.quantityX_labels_choices_combo],
                labels=False,
                layout='horizontal'
            )

            

            self.quantityY_layer_combo = create_widget(
                annotation=Layer, label="quantityY_layer",
                options={'choices': self._image_labels_layers_filter}
            )

            self.quantityY_layer_combo.changed.connect(
                self._update_quantities_labels_choices
            )

            self.quantityY_labels_choices_combo = create_widget(
                widget_type="ComboBox",
                options={'choices': self.labels_method_choices}
            )
            self.quantityY_labels_choices_container = Container(
                widgets=[self.quantityY_labels_choices_combo],
                labels=False,
                layout='horizontal'
            )

            self.mask_layer_combo = create_widget(
                annotation=Image, label="Mask layer",
                options={'nullable': True, 'choices': self._bool_layers_filter}
            )

            self.labels_layer_combo = create_widget(
                annotation=Labels, label="Labels layer",
                options={'nullable': True}
            )

            self.blur_sigma_slider = create_widget(
                widget_type="IntSlider", label="Blur sigma", 
                options={'min':1, 'max':50, 'value':10}
            )

            # self.blur_sigma_slider.changed.connect(self.sigma_changed)

            self.run_button = create_widget(
                widget_type="PushButton", label="Run",
            )

            self.run_button.clicked.connect(self.run)

            self.show_individual_cells_checkbox = create_widget(
                annotation=bool, label="Show individual cells",
            )

            self.show_individual_cells_checkbox.changed.connect(self.parameters_changed)

            self.show_linear_fit_checkbox = create_widget(
                annotation=bool, label="Show linear fit",
            )

            self.show_linear_fit_checkbox.changed.connect(self.parameters_changed)

            self.normalize_quantities_checkbox = create_widget(
                annotation=bool, label="Normalize quantities",
            )

            self.normalize_quantities_checkbox.changed.connect(self.parameters_changed)

            self.options_container = Container(
                widgets=[
                    self.show_individual_cells_checkbox,
                    self.show_linear_fit_checkbox,
                    self.normalize_quantities_checkbox,
                ],
                labels=False,
                layout='horizontal'
            )

            self.heatmap_bins_X = create_widget(
                widget_type="IntSlider", label="X",
                value=20, options={'min':2, 'max':100, 'tracking': False}
            )

            self.heatmap_bins_X.changed.connect(self.parameters_changed)

            self.heatmap_bins_Y = create_widget(
                widget_type="IntSlider", label="Y",
                value=20, options={'min':2, 'max':100, 'tracking': False}
            )

            self.heatmap_bins_Y.changed.connect(self.parameters_changed)

            self.heatmap_bins_container = Container(
                widgets=[
                    self.heatmap_bins_X,
                    self.heatmap_bins_Y,
                ],
                labels=True,
                label='heatmap bins',
                layout='horizontal'
            )

            self.percentiles_X = create_widget(
                widget_type="FloatRangeSlider", label="X",
                options={'min':0, 'max':100, 'value':[0,100],}
            )

            self.percentiles_X.changed.connect(self.parameters_changed)

            self.percentiles_Y = create_widget(
                widget_type="FloatRangeSlider", label="Y",
                options={'min':0, 'max':100, 'value':[0,100],}
            )

            self.percentiles_Y.changed.connect(self.parameters_changed)

            self.percentiles_container = Container(
                widgets=[
                    self.percentiles_X,
                    self.percentiles_Y,
                ],
                labels=True,
                label='Percentiles',
                layout='horizontal'
            )

            self.test_button = create_widget(
                widget_type="PushButton", label="Test",
            )
            self.test_value=False
            self.test_button.clicked.connect(self.test_function)

            self.extend(
                [
                    self.quantityX_layer_combo,
                    self.quantityY_layer_combo,
                    self.mask_layer_combo,
                    self.labels_layer_combo,
                    self.blur_sigma_slider,
                    self.run_button,
                    self.graph_container,
                    self.options_container,
                    self.heatmap_bins_container,
                    self.percentiles_container,
                    self.test_button,
                ]
            )

        # takes care of case where this isn't set yet directly after init
        self.plot_cluster_name = None

        self.id=0

    def manual_clustering_method(self, inside):
        # return

        inside = np.array(inside)  # leads to errors sometimes otherwise

        labels_layer = self.labels_layer_combo.value

        if len(inside) == 0:
            return  # if nothing was plotted yet, leave
        clustering_ID = "MANUAL_CLUSTER_ID"

        features = self.get_layer_tabular_data(labels_layer)

        modifiers = QGuiApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier and clustering_ID in features.keys():
            former_clusters = features[clustering_ID].to_numpy()
            former_clusters[inside] = np.max(former_clusters) + 1
            features.update(pd.DataFrame(former_clusters, columns=[clustering_ID]))
        else:
            features[clustering_ID] = inside.astype(int)
        self.add_column_to_layer_tabular_data(
            labels_layer, clustering_ID, features[clustering_ID]
        )

        # redraw the whole plot
        self.draw_cluster_labels(
            features,
            plot_cluster_name=clustering_ID,
        )

    def test_function(self):
        print(self.test_value)

    def run(self):
        # Check if all necessary layers are specified
        if self.quantityX_layer_combo.value is None:
            warnings.warn("Please specify quantityX_layer")
            return
        if self.quantityY_layer_combo.value is None:
            warnings.warn("Please specify quantityY_layer")
            return

        # Blur the layers
        smoothed_X, smoothed_Y = self._smooth_quantities()

        self._update_smoothed_layers(smoothed_X, smoothed_Y)
        self.plot_from_smoothed(smoothed_X, smoothed_Y)

        # Set a parameter "self.histogram_displayed" to True
        self.histogram_displayed = True

    def _update_smoothed_layers(self, blurred_X, blurred_Y):
        if (
            self.quantityX_smoothed_layer is None or \
            self.quantityX_smoothed_layer not in self._viewer.layers
        ):
            self.quantityX_smoothed_layer = viewer.add_image(
                blurred_X,
                colormap=self.quantityX_layer_combo.value.colormap
            )
        else:
            self.quantityX_smoothed_layer.data = blurred_X

        if (
            self.quantityY_smoothed_layer is None or \
            self.quantityY_smoothed_layer not in self._viewer.layers
        ):
            self.quantityY_smoothed_layer = viewer.add_image(
                blurred_Y,
                colormap=self.quantityX_layer_combo.value.colormap
            )
        else:
            self.quantityY_smoothed_layer.data = blurred_Y

    def _update_quantities_labels_choices(self, event):

        if isinstance(self.quantityX_layer_combo.value, Labels):
            if not self.quantityX_labels_choices_displayed:

                self.insert(
                    self.index(self.quantityX_layer_combo) + 1, 
                    self.quantityX_labels_choices_container
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
                    self.quantityY_labels_choices_container
                )
                
                self.quantityY_labels_choices_displayed = True
        
        else:
            if self.quantityY_labels_choices_displayed:

                self.remove(self.quantityY_labels_choices_container)
                self.quantityY_labels_choices_displayed = False
            
    
    # def _smooth_parallel_func(
    #         self, 
    #         input, 
    #         is_temporal,
    #         dim_space,
    #         sigmas,
    #         n_job
    #     ):
    #     data, mask, mask_for_volume = input
    #     return gaussian_smooth_dense(
    #         data_dense=data, 
    #         is_temporal=is_temporal, 
    #         dim_space=dim_space, 
    #         sigmas=sigmas, 
    #         mask=mask, 
    #         mask_for_volume=mask_for_volume, 
    #         n_job=n_job
    #     )
    
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

            

    def _smooth_quantities(self):

        quantityX_layer = self.quantityX_layer_combo.value
        quantityY_layer = self.quantityY_layer_combo.value

        quantityX = quantityX_layer.data
        quantityY = quantityY_layer.data

        masks_volume = []

        if isinstance(quantityX_layer, Labels):
            quantityX = self._transform_labels_to_density(
                quantityX, 
                self.quantityX_labels_choices_combo.value
            )
            masks_volume.append(None)
        if isinstance(quantityY_layer, Labels):
            quantityY = self._transform_labels_to_density(
                quantityY, 
                self.quantityY_labels_choices_combo.value
            )
            masks_volume.append(None)

        mask = self.mask_layer_combo.value.data if self.mask_layer_combo.value is not None else None
            
        if mask is None or all([elem is None for elem in masks_volume]):
            masks_volume = None

        smoothed_X, smoothed_Y = gaussian_smooth_dense_two_arrays_gpu(
            datas=[quantityX, quantityY],
            sigmas=self.blur_sigma_slider.value,
            mask=mask,
            masks_for_volume=masks_volume,
        )
        

        # func_parallel = partial(
        #     gaussian_smooth_dense,
        #     is_temporal=False,
        #     dim_space=3,
        #     sigmas=self.blur_sigma_slider.value,
        #     mask=blur_mask,
        #     mask_for_volume=blur_mask_volume,
        #     n_job=1
        # )

        # if self.quantityX_labels_choices_displayed:
        #     if self.quantityX_labels_choices_combo.value == 'centroid_density':
        #         quantity_X = ...
        #     elif self.quantityX_labels_choices_combo.value == 'volume density':
        #         quantity_X = ...
        # else:
        #     quantity_X = self.quantityX_layer_combo.value.data


        # func_parallel = partial(
            # self._smooth_parallel_func,
            # is_temporal=False,
            # dim_space=3,
            # sigmas=self.blur_sigma_slider.value,
            # mask=blur_mask,
            # mask_for_volume=blur_mask_volume,
            # n_job=1
        # )

        # smoothed_X, smoothed_Y = list(process_map(
            # func_parallel,
            # [self.quantityX_layer_combo.value.data, self.quantityY_layer_combo.value.data],
            # max_workers=2,
            # chunksize=1,
            # desc="Smoothing quantities"
        # ))

        # smoothed_X = gaussian_smooth_dense(
        #     self.quantityX_layer_combo.value.data,
        #     is_temporal=False,
        #     dim_space=3,
        #     sigmas=self.blur_sigma_slider.value,
        #     mask=blur_mask,
        #     mask_for_volume=blur_mask_volume,
        #     n_job=1
        # )

        # smoothed_Y = gaussian_smooth_dense(
        #     self.quantityY_layer_combo.value.data,
        #     is_temporal=False,
        #     dim_space=3,
        #     sigmas=self.blur_sigma_slider.value,
        #     mask=blur_mask,
        #     mask_for_volume=blur_mask_volume,
        #     n_job=1
        # )

        return smoothed_X, smoothed_Y

    def plot_from_smoothed(self, smoothed_X, smoothed_Y):
        # Construct HeatmapPlotter
        self.heatmap_plotter = SpatialCorrelationPlotter(
            quantity_X=smoothed_X,
            quantity_Y=smoothed_Y,
            mask=self.mask_layer_combo.value.data if self.mask_layer_combo.value is not None else None,
            labels=self.labels_layer_combo.value.data if self.labels_layer_combo.value is not None else None,
        )

        if isinstance(self.quantityX_layer_combo.value, Labels):
            label_X = self.quantityX_labels_choices_combo.value
        else:
            label_X = self.quantityX_layer_combo.value.name
        
        if isinstance(self.quantityY_layer_combo.value, Labels):
            label_Y = self.quantityY_labels_choices_combo.value
        else:
            label_Y = self.quantityY_layer_combo.value.name

        # Get figure from HeatmapPlotter
        figure, _ = self.heatmap_plotter.get_heatmap_figure(
            bins=(self.heatmap_bins_X.value, self.heatmap_bins_Y.value),
            show_individual_cells=self.show_individual_cells_checkbox.value,
            show_linear_fit=self.show_linear_fit_checkbox.value,
            normalize_quantities=self.normalize_quantities_checkbox.value,
            percentiles_X=self.percentiles_X.value,
            percentiles_Y=self.percentiles_Y.value,
            figsize=self.graphics_widget.figure.get_size_inches(),
            label_X=label_X,
            label_Y=label_Y,
        )

        # Display figure in graphics_widget -> Create a method "self.plot"
        self.plot_heatmap(figure)

    def plot_heatmap(self, figure):

        if self.figure is not None:
            plt.close(self.figure)
        self.figure = figure

        labels_layer_exists = self.labels_layer_combo.value is not None

        self.graphics_widget = MplCanvas(
            figure, manual_clustering_method=self.manual_clustering_method,
            create_selectors=labels_layer_exists
        )
        self.toolbar = MyNavigationToolbar(self.graphics_widget)

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
            labels=False
        )

        widget_index = self.index(self.graph_container)
        self.remove(self.graph_container)
        self.insert(widget_index, new_graph_container)
        self.graph_container = new_graph_container
        self.graphics_widget.draw()

    def sigma_changed(self):
        if self.histogram_displayed:

            smoothed_X, smoothed_Y = self._smooth_quantities()
            self._update_smoothed_layers(smoothed_X, smoothed_Y)
            
            self.plot_from_smoothed(smoothed_X, smoothed_Y)


    def parameters_changed(self):
        if self.histogram_displayed:
            
            # Get figure from HeatmapPlotter
            figure, _ = self.heatmap_plotter.get_heatmap_figure(
                bins=(self.heatmap_bins_X.value, self.heatmap_bins_Y.value),
                show_individual_cells=self.show_individual_cells_checkbox.value,
                show_linear_fit=self.show_linear_fit_checkbox.value,
                normalize_quantities=self.normalize_quantities_checkbox.value,
                percentiles_X=self.percentiles_X.value,
                percentiles_Y=self.percentiles_Y.value,
                figsize=self.graphics_widget.figure.get_size_inches(),
                label_X=self.quantityX_layer_combo.value.name,
                label_Y=self.quantityY_layer_combo.value.name,
            )

            # Display figure in graphics_widget -> Create a method "self.plot"
            self.plot_heatmap(figure)

    def _image_labels_layers_filter(self, wdg):
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, Image) or isinstance(layer, Labels)
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
        labels_layer = self.labels_layer_combo.value
        # self.graphics_widget.reset()
    
        # fill all prediction nan values with -1 -> turns them
        # into noise points
        self.cluster_ids = features[plot_cluster_name].fillna(-1)

        # get long colormap from function
        if len(labels_layer.data.shape) > 3:
            warnings.warn("Image dimensions too high for processing!")

        self.graphics_widget.selector.disconnect()
        self.graphics_widget.selector = SelectFromCollection(
            self.graphics_widget,
            self.graphics_widget.axes,
            self.graphics_widget.pts,
        )

        # generate dictionary mapping each prediction to its respective color
        # list cycling with  % introduced for all labels except hdbscan noise points (id = -1)
        cmap_dict = {
            int(prediction + 1): (
                cmap[int(prediction) % len(cmap)]
                if prediction > 0
                else [0, 0, 0, 0]
            )
            for prediction in self.cluster_ids
        }
        # take care of background label
        cmap_dict[None] = [0, 0, 0, 0]

        napari_cmap = DirectLabelColormap(color_dict=cmap_dict)

        keep_selection = list(self._viewer.layers.selection)


        if labels_layer is not None and labels_layer.ndim <= 3:
            cluster_image = self.generate_cluster_image(
                labels_layer.data, self.cluster_ids.tolist()
            )
        else:
            warnings.warn("Image dimensions too high for processing!")
            return

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
                scale=labels_layer.scale,
                opacity=1
            )
        else:
            # updating data
            self.cluster_labels_layer.data = cluster_image
            self.cluster_labels_layer.colormap = napari_cmap

        self._viewer.layers.selection.clear()
        for s in keep_selection:
            self._viewer.layers.selection.add(s)

    def get_layer_tabular_data(self, layer):
        if hasattr(layer, "properties") and layer.properties is not None:
            return pd.DataFrame(layer.properties)
        if hasattr(layer, "features") and layer.features is not None:
            return layer.features
        return None


    def add_column_to_layer_tabular_data(self, layer, column_name, data):
        if hasattr(layer, "properties"):
            layer.properties[column_name] = data
        if hasattr(layer, "features"):
            layer.features.loc[:, column_name] = data


    def generate_cluster_image(self, label_image, predictionlist):
        props = regionprops(label_image)

        cluster_image = np.zeros(label_image.shape, dtype='uint8')

        for i, prop in enumerate(props):
            if predictionlist[i] > 0:
                prop_slice = prop.slice

                roi_data = label_image[prop_slice]

                cluster_image[prop_slice][roi_data==prop.label] = predictionlist[i]+1

        return cluster_image



if __name__ == "__main__":
    import napari
    from napari.layers import Labels, Image
    import scipy
    import tifffile

    path_to_data = '/home/jvanaret/data/data_paper_valentin/morphology/processed'
    data = tifffile.imread(f'{path_to_data}/ag1_norm.tif')
    mask = tifffile.imread(f'{path_to_data}/ag1_mask.tif')
    labels = tifffile.imread(f'{path_to_data}/ag1_norm_labels.tif') 


    # heatmap, xedges, yedges = np.histogram2d(
    #         data[mask].ravel(),
    #         data[mask].ravel()+2,
    #         bins=(20,20),
    #     )
    
    # props = regionprops(labels, intensity_image=data)
    # cellular_X = [np.mean(prop.image_intensity[prop.image]) for prop in props]
    # cellular_X2 = [np.median(prop.image_intensity[prop.image]) for prop in props]
    
    # plt.imshow(heatmap.T, origin='lower',
    #            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]],
    #            aspect='auto', interpolation='none')
    
    # plt.scatter(cellular_X, cellular_X, c='red', s=0.01)
    # plt.scatter(cellular_X2, np.array(cellular_X2)-0.5, c='blue', s=0.01)
    # plt.show()

    viewer = napari.Viewer()

    mask_layer = viewer.add_image(mask)
    image_layer = viewer.add_image(data)
    image_layer2 = viewer.add_image(scipy.ndimage.gaussian_filter(data,10))
    labels_layer = viewer.add_labels(labels)

    widget = PlotterWidget(viewer)
    viewer.window.add_dock_widget(widget)

    widget.mask_layer_combo.value = mask_layer
    widget.labels_layer_combo.value = labels_layer
    widget.quantityX_layer_combo.value = image_layer
    widget.quantityY_layer_combo.value = image_layer2

    napari.run()