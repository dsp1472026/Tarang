"""
Tab 13 - Recovered Image: a file carried in a SYNC | LENGTH | PAYLOAD | CRC32 frame
(postdemod/image_payload.py), from any PSK / QAM / FSK receiver. Shows the image scaled to fit,
the frame details and evidence, and saves it (encoded formats byte-exact, raw pixels as PNG).

Display: a transmitted image is often small (a 256-px thumbnail of a screenshot), so the view
enlarges it with Lanczos resampling (smooth, readable text) by default; 'Pixels' shows the
received samples as blocks and 'Native size' shows them 1:1. Enlarging cannot add detail that
was not transmitted; send a larger image (e.g. max dimension 1024) for a sharper result.
"""

import os

import numpy as np

from gui.qt_compat import (QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSplitter, Qt,
                           QtGui, QVBoxLayout, QWidget)
from gui.widgets.tables import make_text, wrap_label


def _qimage(px: np.ndarray):
    """(H, W, C) uint8 -> QImage (copied, so the numpy buffer may be freed)."""
    px = np.ascontiguousarray(px)
    h, w, c = px.shape
    F = QtGui.QImage.Format
    fmt = {1: F.Format_Grayscale8, 3: F.Format_RGB888, 4: F.Format_RGBA8888}[c]
    return QtGui.QImage(px.data, w, h, w * c, fmt).copy()


class ImageTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.source_path = ""
        main = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.header = wrap_label(QLabel("Recovered image: awaiting analysis"))
        self.header.setStyleSheet("font-weight: bold; color: #0F766E; font-size: 13px;")
        self.header.setWordWrap(True)
        bar.addWidget(self.header, 1)
        bar.addWidget(QLabel("View:"))
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Smooth (Lanczos)", "Pixels (nearest)", "Native size"])
        self.view_mode.currentIndexChanged.connect(lambda *_: self._rescale())
        bar.addWidget(self.view_mode)
        self.save_btn = QPushButton("Save image...")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        bar.addWidget(self.save_btn)
        self.save_big_btn = QPushButton("Save enlarged x4...")
        self.save_big_btn.setEnabled(False)
        self.save_big_btn.setToolTip("Lanczos x4 enlargement for viewing / reports (interpolated, no new detail)")
        self.save_big_btn.clicked.connect(self.save_enlarged)
        bar.addWidget(self.save_big_btn)
        main.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        vb = QGroupBox("Image")
        vl = QVBoxLayout(vb)
        self.view = QLabel("No image in this capture.")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(200, 150)
        # the scaled pixmap must not become the label's minimum size (the window could then no longer
        # shrink, the same class of problem as the 126 618-px 'Unable to set geometry' error)
        from gui.qt_compat import QtWidgets
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.view.setStyleSheet("background-color: #FFFFFF; color: #64748B; border: 1px solid #E2E8F0; border-radius: 6px;")
        vl.addWidget(self.view)
        split.addWidget(vb)
        eb = QGroupBox("Frame && evidence")
        el = QVBoxLayout(eb)
        self.info = make_text("#64748B")
        el.addWidget(self.info)
        split.addWidget(eb)
        split.setSizes([700, 400])
        main.addWidget(split)
        self._pixmap = None

    def clear(self, reason: str = "no image / file frame found in the demodulated bits"):
        self.image = None
        self._pixmap = None
        self.header.setText(f"Recovered image: {reason}")
        self.view.setPixmap(QtGui.QPixmap())
        self.view.setText("No image in this capture.")
        self.info.clear()
        self.save_btn.setEnabled(False)
        self.save_big_btn.setEnabled(False)

    def update_image(self, image, source_path: str = ""):
        if image is None:
            self.clear()
            return
        self.image, self.source_path = image, source_path
        self.header.setText(("Recovered image: " if image.is_image else "Recovered file: ") + image.describe())
        if image.is_image:
            self._pixmap = QtGui.QPixmap.fromImage(_qimage(image.pixels))
            self._rescale()
        else:
            self._pixmap = None
            self.view.setPixmap(QtGui.QPixmap())
            self.view.setText("Payload is not an image (saved as binary).")
        lines = [f"Format        : {image.format}",
                 f"Size          : {image.width} x {image.height}" + (f" x {image.channels}" if image.channels else "")
                 if image.is_image else f"Size          : -",
                 f"Payload bytes : {len(image.payload)}",
                 f"CRC-32        : {'VALID' if image.crc_ok else 'NOT VERIFIED'} (received 0x{image.crc_received:08X}, "
                 f"computed 0x{image.crc_computed:08X})",
                 f"FEC           : {image.fec}",
                 f"Hypothesis    : {image.hypothesis}",
                 f"Sync          : bit {image.sync_bit}, {image.sync_errors}/32 bit errors",
                 "", "Evidence:"]
        lines += [f"[{e.source}] {e.description}" for e in image.evidence]
        self.info.setText("\n".join(lines))
        self.save_btn.setEnabled(True)
        self.save_big_btn.setEnabled(image.is_image)

    def _rescale(self):
        if self._pixmap is None or self.image is None or self.image.pixels is None:
            return
        size = self.view.size()
        pm = self._pixmap
        mode = self.view_mode.currentIndex()
        if mode == 2:                                  # native size
            self.view.setPixmap(pm)
            return
        if mode == 1:                                  # blocks: the received samples as they are
            self.view.setPixmap(pm.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.FastTransformation))
            return
        # Lanczos (sharper than Qt's bilinear for small images with text)
        px = self.image.pixels
        h, w = px.shape[:2]
        f = min(size.width() / w, size.height() / h)
        if f <= 1.0:
            self.view.setPixmap(pm.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
            return
        from PIL import Image
        mode_s = {1: "L", 3: "RGB", 4: "RGBA"}[px.shape[2]]
        big = Image.fromarray(px[:, :, 0] if px.shape[2] == 1 else px, mode_s).resize(
            (max(1, int(w * f)), max(1, int(h * f))), Image.LANCZOS)
        arr = np.asarray(big)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        self.view.setPixmap(QtGui.QPixmap.fromImage(_qimage(arr)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def save_enlarged(self):
        if self.image is None or self.image.pixels is None:
            return
        from PIL import Image
        stem = os.path.splitext(os.path.basename(self.source_path or "capture"))[0]
        path, _ = QFileDialog.getSaveFileName(self, "Save enlarged image", f"{stem}_recovered_x4.png", "PNG (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            px = self.image.pixels
            im = Image.fromarray(px[:, :, 0] if px.shape[2] == 1 else px)
            im.resize((im.width * 4, im.height * 4), Image.LANCZOS).save(path)
            self.header.setText(f"Saved (x4 Lanczos): {path}")

    def save(self):
        if self.image is None:
            return
        stem = os.path.splitext(os.path.basename(self.source_path or "capture"))[0]
        ext = self.image.extension
        path, _ = QFileDialog.getSaveFileName(self, "Save recovered image", f"{stem}_recovered{ext}",
                                              f"{self.image.format} (*{ext});;All files (*)")
        if path:
            if not os.path.splitext(path)[1]:
                path += ext
            self.image.save(path)
            self.header.setText(f"Saved: {path}")
