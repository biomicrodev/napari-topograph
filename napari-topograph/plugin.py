from dataclasses import dataclass
from typing import Tuple, Dict, List

import colorcet as cc
import dask.array as da
import numpy as np
import pyqtgraph as pg
from dask.delayed import Delayed
from napari.components import LayerList
from napari.layers import Image
from qtpy.QtWidgets import (
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QHBoxLayout,
)

from .colors import hex2uint8
from .histograms import dask_hist1d, dask_hist2d
from .items import BarPlotItem, cc_cmaps


@dataclass(frozen=True)
class Pyramid:
    images: List[da.Array]
    offset: Tuple[float, ...]
    mpp: Tuple[float, ...]


@dataclass(frozen=True)
class Layer:
    image: da.Array
    offset: Tuple[float, ...]
    mpp: Tuple[float, ...]


class CorrelationPlotLayout(QGridLayout):
    def __init__(
        self,
        n: int,
        *args,
        labels: List[str] = None,
        cmap: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # validate labels
        if (labels is not None) and (n != len(labels)):
            raise RuntimeError(f"Mismatch between inputs: {labels}, {n}")
        if labels is None:
            labels = [f"Channel {i + 1}" for i in range(n)]

        if cmap is not None:
            if cmap not in cc_cmaps:
                raise NameError(
                    "Undefined colormap, should be one of the following: "
                    + ", ".join([f'"{i}"' for i in cc_cmaps])
                    + "."
                )
        else:
            cmap = "fire"

        lut = cc.palette[cmap]
        lut = [hex2uint8(v) for v in lut]

        self.n = n

        self.items: Dict[Tuple[int, int], pg.GraphicsObject] = {}
        self.viewBoxes: Dict[Tuple[int, int], pg.ViewBox] = {}

        for row in range(n):
            for col in range(n):
                # on diagonal
                if row == col:
                    item = BarPlotItem()

                    viewBox = pg.ViewBox(lockAspect=False)
                    plot = pg.PlotWidget(viewBox=viewBox)
                    plot.getPlotItem().setLogMode(x=False, y=True)
                    plot.addItem(item)

                # off diagonal
                else:
                    item = pg.ImageItem()
                    item.setLookupTable(lut)

                    viewBox = pg.ViewBox(lockAspect=True)
                    plot = pg.PlotWidget(viewBox=viewBox)
                    plot.addItem(item)

                viewBox.setMouseEnabled(x=True, y=True)

                self.items[(row, col)] = item
                self.viewBoxes[(row, col)] = viewBox

                self.addWidget(plot, row, col)

                if col == 0:
                    plot.getPlotItem().setLabel("left", text=labels[row])
                if row == (n - 1):
                    plot.getPlotItem().setLabel("bottom", text=labels[col])

        # link axes
        # for row in range(self.n):
        #     for col in range(self.n):
        #         if row == col:
        #             continue
        #         if row == 1 and col == 0:
        #             continue
        #         self.viewBoxes[(row, col)].setXLink(self.viewBoxes[(1, 0)])
        #         self.viewBoxes[(row, col)].setYLink(self.viewBoxes[(1, 0)])

    def updatePlots(self, histograms: Dict[Tuple[int, int], np.ndarray]) -> None:
        for row in range(self.n):
            for col in range(self.n):
                if row == col:
                    hist: np.ndarray = histograms[(row, col)]
                    item: BarPlotItem = self.items[(row, col)]
                    item.setData(list(range(hist.shape[0])), hist, width=1.0)

                elif row > col:
                    hist: np.ndarray = histograms[(row, col)]

                    item: pg.ImageItem = self.items[(row, col)]
                    item.setImage(image=hist)

                    itemT: pg.ImageItem = self.items[(col, row)]
                    itemT.setImage(image=hist.T)

                viewBox = self.viewBoxes[(row, col)]
                viewBox.updateAutoRange()


class _TopographPixels:
    def __init__(self, n_channels: int):
        self.plotLayout = CorrelationPlotLayout(n_channels)

        self.view = QWidget()
        self.view.setLayout(self.plotLayout)


class TopographPixels:
    def __init__(self, layers: LayerList):
        self.layers = layers
        image_layers = [layer for layer in self.layers if isinstance(layer, Image)]
        self.n_channels = max([im.data[0].shape[0] for im in image_layers])

        # init view
        self.runButton = QPushButton()
        self.runButton.setText("Update")
        self.runButton.setEnabled(True)

        self.plotLayout = CorrelationPlotLayout(self.n_channels)

        layout = QVBoxLayout()
        layout.addWidget(self.runButton, stretch=0)
        layout.addLayout(self.plotLayout, stretch=1)
        self.view = QWidget()
        self.view.setLayout(layout)

        # self._upper_left: Optional[np.ndarray] = None
        # self._lower_right: Optional[np.ndarray] = None

    def update_plots(self, xmin_w: int, xmax_w: int, ymin_w: int, ymax_w: int) -> None:
        """
        We have to be careful not to mix up world and pixel coords. World coordinates
        are explicitly indicated as such.

        TODO: there is a substantial bug here where any overlapping regions between
        dask arrays are doubly counted.
        """

        min_w = np.asarray([ymin_w, xmin_w])
        max_w = np.asarray([ymax_w, xmax_w])

        histograms: Dict[Tuple[int, int], np.ndarray] = {}

        regions: List[da.Array] = []

        image_layers = [layer for layer in self.layers if isinstance(layer, Image)]
        for layer in image_layers:
            mpp: np.ndarray = layer.scale[1:]
            translate_w: np.ndarray = layer.translate[1:]

            # translate rect bounds from global to image
            upper_left_w = np.subtract(min_w, translate_w)
            lower_right_w = np.subtract(max_w, translate_w)

            # scale rect bounds from world to pixel coordinates
            upper_left = np.divide(upper_left_w, mpp)
            lower_right = np.divide(lower_right_w, mpp)

            # make them integers
            upper_left = np.round(upper_left).astype(int)
            lower_right = np.round(lower_right).astype(int)
            lower_right += 1

            # clip to image bounds
            image: da.Array = layer.data[0]
            upper_left = np.minimum(np.maximum(upper_left, 0), image.shape[1:])
            lower_right = np.minimum(np.maximum(lower_right, 0), image.shape[1:])

            # stop early
            if np.any(upper_left == lower_right):
                continue
            # if (
            #     np.all(self._upper_left == upper_left)
            #     and np.all(self._lower_right == lower_right)
            # ) or np.any(upper_left == lower_right):
            #     continue
            # self._upper_left = upper_left
            # self._lower_right = lower_right

            # crop
            region = image[
                :,
                upper_left[0] : lower_right[0],
                upper_left[1] : lower_right[1],
            ]
            regions.append(region)

        hist1d: Delayed = da.sum(
            da.stack(
                [
                    da.stack(
                        [
                            dask_hist1d(region[i, ...], range=(0, 256), bins=256)[0]
                            for i in range(self.n_channels)
                        ]
                    )
                    for region in regions
                ]
            ),
            axis=0,
        )

        indices = [
            (row, col)
            for row in range(self.n_channels)
            for col in range(self.n_channels)
            if row > col
        ]
        hist2d: Delayed = da.sum(
            da.stack(
                [
                    da.stack(
                        [
                            dask_hist2d(
                                x=region[col, ...],
                                y=region[row, ...],
                                range=[(0, 256), (0, 256)],
                                bins=256,
                            )
                            for row, col in indices
                        ]
                    )
                    for region in regions
                ]
            ),
            axis=0,
        )

        hist1d, hist2d = da.compute(hist1d, hist2d)
        hist1d: np.ndarray
        hist2d: np.ndarray
        hist1d = np.log10(hist1d)
        hist1d[~np.isfinite(hist1d)] = 0
        hist2d = np.log10(hist2d)
        hist2d[~np.isfinite(hist2d)] = 0

        for row in range(self.n_channels):
            histograms[(row, row)] = hist1d[row, ...]

        for i, (row, col) in enumerate(indices):
            histograms[(row, col)] = hist2d[i, ...]

        self.plotLayout.updatePlots(histograms)


class TopographCells(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.segmentButton = QPushButton("Segment cells")
        self.segmentButton.setEnabled(True)

        self.statusText = QLabel("Not running")

        segmentLayout = QHBoxLayout()
        segmentLayout.addWidget(self.segmentButton)
        segmentLayout.addWidget(self.statusText)

        layout = QVBoxLayout()
        layout.addLayout(segmentLayout)
        layout.addStretch(1)

        self.view = QWidget()
        self.view.setLayout(layout)
