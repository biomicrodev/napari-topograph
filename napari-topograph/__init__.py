import os
from pathlib import Path
from typing import List, Optional

import dask.array as da
import numpy as np
from bmd_common.images import read_image, PyramidGroup, Pyramid
from qtpy.QtCore import Qt

from .plugin import TopographCells, _TopographPixels, TopographPixels

os.environ["NAPARI_ASYNC"] = "1"

import napari
from napari import Viewer
from napari.layers import Shapes, Image
from napari.utils.events import Event
from napari.utils.status_messages import format_float
from napari._qt.qthreading import thread_worker
from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget


def is_rectangle(points: np.ndarray, *, decimals: int = 3) -> bool:
    points = points.round(decimals=decimals)
    yy = np.unique(points[:, 0])
    xx = np.unique(points[:, 1])

    return (len(yy) == 2) and (len(xx) == 2)


def build_plugin_(viewer: Viewer) -> None:
    """
    This part of the code sort of ended up as the controller for the plugin.
    """
    # validate state
    image_layers = [l for l in viewer.layers if isinstance(l, Image)]
    if len(image_layers) < 1:
        return
    layer: Image = image_layers[0]  # the image to visualize is the first
    image: np.ndarray = layer.data
    if image.ndim != 3:
        return
    if image.shape[2] <= 1:
        return

    n_channels = image.shape[2]

    # set up dock widgets
    topograph_pixels = _TopographPixels(n_channels)
    topograph_cells = TopographCells()

    viewer.window.add_dock_widget(
        topograph_pixels.view, name="TopographPixels", area="right"
    )
    # viewer.window.add_dock_widget(topograph_cells, name='TopographCells', area='right')

    init_rect = np.asarray([[0, 0], [0, 500], [500, 500], [500, 0]])

    shapes_layer: Shapes = viewer.add_shapes(
        [init_rect],
        shape_type="rectangle",
        name="Annotations",
        ndim=2,
        edge_color=[(1, 1, 1, 1)],  # black
        face_color=[(0, 0, 0, 0)],  # clear
    )

    _shape: Optional[np.ndarray] = None

    def update_plots():
        # at least one rectangle
        shapes: List[np.ndarray] = shapes_layer.data
        shapes: List[np.ndarray] = [s for s in shapes if is_rectangle(s)]
        if len(shapes) == 0:
            return

        # check if it's changed at all
        shape: np.ndarray = shapes[0]
        nonlocal _shape
        if (shape == _shape).all():
            return
        _shape = np.copy(shape)

        # get shape extents
        ymin, ymax = shape[:, 0].min(), shape[:, 0].max()
        ymin, ymax = int(round(ymin)), int(round(ymax))
        # ymin, ymax = int(round(ymin)), int(round(ymax)) + 1

        xmin, xmax = shape[:, 1].min(), shape[:, 1].max()
        xmin, xmax = int(round(xmin)), int(round(xmax))
        # xmin, xmax = int(round(xmin)), int(round(xmax)) + 1

        # clip extents
        ymin = min(max(ymin, 0), image.shape[0])
        ymax = max(min(ymax, image.shape[0]), 0)

        xmin = min(max(xmin, 0), image.shape[1])
        xmax = max(min(xmax, image.shape[1]), 0)

        # get desired region of image
        region = image[ymin:ymax, xmin:xmax, :]
        topograph_pixels.plotLayout.updatePlots(region)

    def on_change(event: Event) -> None:
        event_type = getattr(event, "type", None)
        if event_type != "set_data":
            return

        update_plots()

    shapes_layer.events.connect(on_change)
    update_plots()


def build_plugin(viewer: Viewer) -> None:
    # prepare viewer window
    viewer.window.qt_viewer.layerButtons.deleteButton.setDisabled(True)

    # add annotation layer with a single window
    init_rect = np.asarray([[0, 0], [500, 0], [500, 500], [0, 500]])
    annotation_layer: Shapes = viewer.add_shapes(
        [init_rect],
        shape_type="rectangle",
        name="Annotations",
        ndim=2,
        edge_color=[(1, 1, 1, 1)],
        face_color=[(0, 0, 0, 0)],
    )

    topograph_pixels = TopographPixels(viewer.layers)
    dock_widget: QtViewerDockWidget = viewer.window.add_dock_widget(
        topograph_pixels.view, name="TopographPixels", area="right"
    )
    viewer.window._qt_window.resizeDocks([dock_widget], [300], Qt.Horizontal)

    # update annotation rectangle
    _rect: Optional[np.ndarray] = None

    def update_plots() -> None:
        """
        Note that all coordinate values are in world coordinates; they correspond to
        *microns*. It is the responsibility of whoever is interacting with the image to
        convert these values from microns to pixels.
        """

        # at least one rectangle
        shapes: List[np.ndarray] = annotation_layer.data
        rects: List[np.ndarray] = [s for s in shapes if is_rectangle(s)]
        if len(rects) == 0:
            return

        # did rect change?
        rect = rects[0]
        nonlocal _rect
        if (_rect == rect).all():
            return
        _rect = np.copy(rect)

        # polygon to rectangle
        ymin, ymax = rect[:, 0].min(), rect[:, 0].max()
        ymin, ymax = int(round(ymin)), int(round(ymax)) + 1

        xmin, xmax = rect[:, 1].min(), rect[:, 1].max()
        xmin, xmax = int(round(xmin)), int(round(xmax)) + 1

        topograph_pixels.update_plots(xmin, xmax, ymin, ymax)

    @thread_worker
    def run_update():
        topograph_pixels.runButton.setDisabled(True)
        topograph_pixels.runButton.setText("Running...")
        update_plots()
        topograph_pixels.runButton.setEnabled(True)
        topograph_pixels.runButton.setText("Update")

    def run():
        worker = run_update()
        worker.start()

    topograph_pixels.runButton.clicked.connect(run)

    # viewer.layers.events.moved.connect(update_plots)
    #
    # def on_change(event: Event) -> None:
    #     event_type = getattr(event, "type", None)
    #     if event_type != "set_data":
    #         return
    #
    #     update_plots()
    #
    # annotation_layer.events.connect(on_change)

    # updating edge width
    _edge_width: Optional[float] = None

    def update_edge_width() -> None:
        edge_width = min(5 / viewer.camera.zoom, 200)

        nonlocal _edge_width
        if _edge_width == edge_width:
            return
        _edge_width = edge_width

        # current_edge_width.setter from `shapes.py`
        annotation_layer._current_edge_width = edge_width
        for i in range(len(annotation_layer._data_view.shapes)):
            annotation_layer._data_view.update_edge_width(i, edge_width)
        annotation_layer.status = format_float(annotation_layer.current_edge_width)
        annotation_layer.events.edge_width()

    def on_camera_event(event: Event):
        if event.type != "zoom":
            return

        update_edge_width()

    viewer.camera.events.connect(on_camera_event)
    update_edge_width()


# def build_plugin2(viewer: Viewer) -> None:
#     topograph_cells = TopographCells()
#     viewer.window.add_dock_widget(
#         topograph_cells.view, name="TopographCells", area="right"
#     )


def run():
    path = Path("/path/to/file")
    pyramid_group: PyramidGroup = read_image(path)

    with napari.gui_qt():
        viewer = napari.Viewer()

        background: Optional[Pyramid] = pyramid_group.background
        if background is not None:
            background: List[da.Array] = [
                layer.rechunk({0: 1, 1: 1024, 2: 1024}) for layer in background
            ]
            viewer.add_image(background, name="background")

        for i, pyramid in enumerate(pyramid_group.pyramids[:2]):
            pyramid: Pyramid

            mpp = pyramid.mpp
            mpp = (1, mpp.y, mpp.x)

            offset = pyramid.offset
            offset = (1, offset.y, offset.x)

            layers: List[da.Array] = [
                layer.rechunk({0: 1, 1: 1024, 2: 1024}) for layer in pyramid.layers
            ]

            viewer.add_image(
                layers, name=f"Pyramid {i + 1}", scale=mpp, translate=offset
            )

        build_plugin(viewer)
