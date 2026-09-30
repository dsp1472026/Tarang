"""
Static plot snapshots for the report (spectrum, constellation) as PNG bytes, so the report view,
the HTML export and the PDF carry the same figures. Plots are drawn with pyqtgraph off-screen at
2x resolution (sharp in print) and are never shown; GUI thread only.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pyqtgraph as pg

from core.signal_buffer import SignalBuffer
from gui import theme
from gui.qt_compat import QtCore, QtGui, QtWidgets
from gui.widgets.spectrum_plot import SpectrumPlotWidget

FIG_W, FIG_H = 560, 340          # logical size in the report
SCALE = 2                        # render resolution factor

Figure = Tuple[str, bytes, int]  # (caption, PNG bytes, display width px)


def _png(widget, w: int, h: int) -> bytes:
    widget.resize(w, h)
    pix = QtGui.QPixmap(w * SCALE, h * SCALE)
    pix.setDevicePixelRatio(SCALE)
    pix.fill(QtGui.QColor(theme.PANEL))
    QtWidgets.QWidget.render(widget, pix)          # the plot view overrides render() with a scene-painter signature
    ba = QtCore.QByteArray()
    buf = QtCore.QBuffer(ba)
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    pix.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def _plot(title: str) -> pg.PlotWidget:
    w = pg.PlotWidget(title=title, background=theme.PANEL)
    w.showGrid(x=True, y=True, alpha=0.25)
    w.setMenuEnabled(False)
    w.hideButtons()
    return w


def spectrum_figure(buf: SignalBuffer) -> bytes:
    x = buf.samples[: 1 << 21]                  # same cap as the Signals tab
    data = SpectrumPlotWidget.compute(SignalBuffer(samples=np.asarray(x), sample_rate=buf.sample_rate,
                                                   center_frequency=buf.center_frequency), nperseg=2048)
    if data is None:
        return b""
    f, p = data
    w = _plot("Power Spectral Density")
    if buf.sample_rate:
        w.setLabel("bottom", "Frequency", units="Hz")
    else:
        w.setLabel("bottom", "Frequency (normalised)")
    w.setLabel("left", "Power", units="dB")
    w.plot(f, p, pen=pg.mkPen(theme.PRIMARY, width=1.4))
    return _png(w, FIG_W, FIG_H)


def constellation_figure(points: np.ndarray, max_points: int = 5000) -> bytes:
    pts = np.asarray(points)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    w = _plot("Constellation Diagram")
    w.setLabel("bottom", "In-Phase (I)")
    w.setLabel("left", "Quadrature (Q)")
    w.setAspectLocked(True)
    w.addItem(pg.ScatterPlotItem(x=pts.real, y=pts.imag, size=4, pen=pg.mkPen(None),
                                 brush=pg.mkBrush(37, 99, 235, 140)))
    return _png(w, FIG_H + 40, FIG_H)


def render_figures(results: Dict[str, Any]) -> List[Figure]:
    """Figures for one analysis run; a plot that cannot be drawn is left out (never raises)."""
    out: List[Figure] = []
    buf = results.get("buffer")
    if buf is not None and buf.is_valid():
        try:
            png = spectrum_figure(buf)
            if png:
                out.append(("Spectrum of the capture (Welch PSD)", png, FIG_W))
        except Exception:                               # noqa: BLE001 - a missing figure never breaks the report
            pass
    demod = results.get("demod_result")
    points = getattr(demod, "constellation_points", None)
    if points is not None and len(points) > 0:         # FSK has no constellation
        try:
            out.append(("Constellation of the demodulated symbols", constellation_figure(points), FIG_H + 40))
        except Exception:                               # noqa: BLE001
            pass
    return out
