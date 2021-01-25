from typing import List, Optional

import napari
import numpy as np
import skimage.data
from napari import Viewer
from napari.layers import Shapes, Image
from napari.utils.events import Event

from .plugin import TopographPixels, TopographCells


def is_rectangle(points: np.ndarray, *, decimals: int = 3) -> bool:
    points = points.round(decimals=decimals)
    yy = np.unique(points[:, 0])
    xx = np.unique(points[:, 1])

    return (len(yy) == 2) and (len(xx) == 2)


def build_plugin(viewer: Viewer) -> None:
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
    topograph_pixels = TopographPixels(n_channels)
    topograph_cells = TopographCells()

    viewer.window.add_dock_widget(
        topograph_pixels, name="TopographPixels", area="right"
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
        ymin, ymax = int(round(ymin)), int(round(ymax)) + 1

        xmin, xmax = shape[:, 1].min(), shape[:, 1].max()
        xmin, xmax = int(round(xmin)), int(round(xmax)) + 1

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


def run():
    image: np.ndarray = skimage.data.astronaut()
    with napari.gui_qt():
        viewer = Viewer()
        viewer.add_image(image)
        build_plugin(viewer)
