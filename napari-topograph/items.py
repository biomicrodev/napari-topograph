"""
A collection of modified GraphicsObjects from pyqtgraph. These were written to allow
updating plots directly instead of having to re-instantiate the class.

TODO: follow the matplotlib API more closely
"""

from typing import Optional, Tuple

import colorcet as cc
import numpy as np
from matplotlib import cbook
from pyqtgraph import GraphicsObject, getConfigOption, mkBrush, mkPen
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPicture, QColor, QPainter, QPainterPath
from .colors import hex2uint8, cc_cmaps


def validate(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[Optional[np.ndarray], ...]:
    if (x is None) and (y is None) and (z is None):
        return (None, None, None)

    if (x is None) and (y is None) and (z is not None):
        x = np.arange(0, z.shape[0] + 1, 1)
        y = np.arange(0, z.shape[1] + 1, 1)
        x, y = np.meshgrid(x, y, indexing="ij")
        return x, y, z

    if (x is not None) and (y is not None) and (z is not None):
        if (x.shape[0] != z.shape[0] + 1) or (x.shape[1] != z.shape[1] + 1):
            raise ValueError("The dimensions of x should be one greater than that of z")
        if (y.shape[0] != z.shape[0] + 1) or (y.shape[1] != z.shape[1] + 1):
            raise ValueError("The dimensions of y should be one greater than that of z")

        return x, y, z

    raise ValueError("Data must be sent as (z) or (x, y, z)")


class PColorRectItem(GraphicsObject):
    def __init__(
        self,
        *,
        x: np.ndarray = None,
        y: np.ndarray = None,
        z: np.ndarray = None,
        **kwargs,
    ):
        super().__init__()

        self.qpicture: Optional[QPicture] = None
        self.axisOrder = getConfigOption("imageAxisOrder")

        self.edgecolors = kwargs.get("edgecolors", None)
        self.antialiasing = kwargs.get("antialiasing", False)

        # set up colormap
        if "cmap" in kwargs.keys():
            if kwargs["cmap"] in cc_cmaps:
                cmap = kwargs["cmap"]
            else:
                raise NameError(
                    "Undefined colormap, should be one of the following: "
                    + ", ".join([f'"{i}"' for i in cc_cmaps])  # TODO: make this legible
                    + "."
                )
        else:
            cmap = "fire"

        lut = cc.palette[cmap]
        lut = [QColor(*hex2uint8(v)) for v in lut]
        self.lut = lut

        # set up data
        self.setData(x, y, z)

    def setData(self, x: np.ndarray = None, y: np.ndarray = None, z: np.ndarray = None):
        self.x, self.y, self.z = validate(x, y, z)

        shapeChanged = False
        if self.qpicture is None:
            shapeChanged = True
        elif (x is None) and (y is None) and (z is not None):
            if (z.shape[0] != self.x[:, 1][-1]) or (z.shape[1] != self.y[0][-1]):
                shapeChanged = True
        elif (x is not None) and (y is not None) and (z is not None):
            if np.any(self.x != x) or np.any(self.y != y):
                shapeChanged = True

        # initialize painting
        self.qpicture = QPicture()
        p = QPainter(self.qpicture)
        p.setPen(
            mkPen(
                self.edgecolors if self.edgecolors is not None else QColor(0, 0, 0, 0)
            )
        )
        if self.antialiasing:
            p.setRenderHint(QPainter.HighQualityAntialiasing)

        # transform
        self.z = np.ma.masked_array(self.z, mask=~np.isfinite(self.z))
        norm = self.z - self.z.min()
        norm /= norm.max()
        norm *= len(self.lut) - 1
        norm = norm.astype(int)

        # plot
        for yi in range(norm.shape[0]):
            for xi in range(norm.shape[1]):
                if norm.mask[yi, xi]:
                    continue  # TODO: support bad colors

                c = self.lut[norm[yi, xi]]
                p.setBrush(mkBrush(c))

                rect = QRectF(
                    QPointF(self.x[yi + 1, xi + 1], self.y[yi + 1, xi + 1]),
                    QPointF(self.x[yi, xi], self.y[yi, xi]),
                )
                p.drawRect(rect)

        # finish painting
        p.end()
        self.update()

        self.prepareGeometryChange()
        if shapeChanged:
            self.informViewBoundsChanged()

    def paint(self, p, *args):
        if self.z is None:
            return

        p.drawPicture(0, 0, self.qpicture)

    def setBorder(self, b):
        self.border = mkPen(b)
        self.update()

    def width(self):
        if self.x is None:
            return None
        return np.max(self.x)

    def height(self):
        if self.y is None:
            return None
        return np.max(self.y)

    def boundingRect(self):
        if self.qpicture is None:
            return QRectF(0.0, 0.0, 0.0, 0.0)
        return QRectF(self.qpicture.boundingRect())


class BarPlotItem(GraphicsObject):
    """
    Has a similar method signature as `matplotlib.axes.hist`.
    """

    def __init__(self):
        super().__init__()

        self.qpicture: Optional[QPicture] = None

    def setData(
        self,
        x,
        height,
        width=0.8,
        bottom=None,
        *,
        align="center",
        fillcolor=None,
        edgecolor=None,
    ) -> None:
        # assume vertical
        cbook._check_in_list(["center", "edge"], align=align)

        y = bottom
        if y is None:
            y = 0

        x, height, width, y = np.broadcast_arrays(np.atleast_1d(x), height, width, y)

        if align == "center":
            try:
                left = x - width / 2
            except TypeError as e:
                raise TypeError(
                    f"the dtypes of parameters x ({x.dtype}) "
                    f"and width ({width.dtype}) "
                    f"are incompatible"
                ) from e
        elif align == "edge":
            left = x
        else:
            raise RuntimeError(f"unknown align mode {align}")
        bottom = y

        # prepare to draw
        if edgecolor is None:
            edgecolor = (128, 128, 128)  # getConfigOption("foreground")
        pen = mkPen(edgecolor)

        if fillcolor is None:
            fillcolor = (128, 128, 128)
        brush = mkBrush(fillcolor)

        self.qpicture = QPicture()
        self.path = QPainterPath()

        p = QPainter(self.qpicture)
        p.setPen(pen)
        p.setBrush(brush)

        # draw
        rects = zip(left, bottom, width, height)
        for l, b, w, h in rects:
            rect = QRectF(l, b, w, h)
            p.drawRect(rect)
            self.path.addRect(rect)

        p.end()
        self.prepareGeometryChange()

    def paint(self, p: QPainter, *args) -> None:
        if self.qpicture is None:
            return
        self.qpicture.play(p)

    def boundingRect(self) -> QRectF:
        if self.qpicture is None:
            return QRectF(0.0, 0.0, 0.0, 0.0)
        return QRectF(self.qpicture.boundingRect())
