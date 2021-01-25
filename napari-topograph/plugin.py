from typing import Tuple, Dict, List

import colorcet as cc
import numpy as np
import pyqtgraph as pg
from fast_histogram import histogram2d, histogram1d
from qtpy.QtWidgets import QWidget, QGridLayout, QLayout, QLayoutItem

from .colors import hex2uint8
from .items import BarPlotItem, cc_cmaps


def clearLayout(layout: QLayout) -> None:
    if layout.count() == 0:
        return

    item: QLayoutItem = layout.takeAt(0)
    while item is not None:
        if item.widget() is not None:
            item.widget().deleteLater()
        elif item.layout() is not None:
            item.layout().deleteLater()

        item: QLayoutItem = layout.takeAt(0)


class CorrelationPlotLayout(QGridLayout):
    def __init__(
        self,
        n: int,
        *args,
        bins: int = 256,
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
        self.bins = bins

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

                if row == (n - 1):
                    plot.getPlotItem().setLabel("bottom", text=labels[col])

        # for row in range(self.n):
        #     for col in range(self.n):
        #         if row == col:
        #             continue
        #         if row == 1 and col == 0:
        #             continue
        #         self.viewBoxes[(row, col)].setXLink(self.viewBoxes[(1, 0)])
        #         self.viewBoxes[(row, col)].setYLink(self.viewBoxes[(1, 0)])

    def updatePlots(self, image: np.ndarray) -> None:
        for row in range(self.n):
            for col in range(self.n):
                item = self.items[(row, col)]

                if row == col:
                    hist: np.ndarray = histogram1d(
                        image[..., row], bins=self.bins, range=[0, self.bins]
                    )
                    hist = np.log10(hist)
                    # hist: np.ma.MaskedArray = np.ma.log10(hist)
                    # hist: np.ndarray = hist.filled(0)

                    item: BarPlotItem
                    item.setData(list(range(self.bins)), hist)

                else:
                    hist: np.ndarray = histogram2d(
                        image[..., col].flatten(),
                        image[..., row].flatten(),
                        bins=self.bins,
                        range=[(0, self.bins), (0, self.bins)],
                    )
                    hist = np.ma.log10(hist)
                    hist[~np.isfinite(hist)] = 0
                    # hist: np.ma.MaskedArray = np.ma.log10(hist)
                    # hist: np.ndarray = hist.filled(0)

                    item: pg.ImageItem
                    item.setImage(image=hist)

                viewBox = self.viewBoxes[(row, col)]
                viewBox.updateAutoRange()


class TopographPixels(QWidget):
    def __init__(self, n_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plotLayout = CorrelationPlotLayout(n_channels)
        self.setLayout(self.plotLayout)


class TopographCells(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
