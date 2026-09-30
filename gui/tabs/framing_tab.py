"""
Tab 11 - Framing (L11): frame layout found by CRC validation (sync, length field, CRC type /
byte order / coverage), every frame with its CRC check, and the decoded payload (text + hex).
"""

from gui.qt_compat import QGroupBox, QLabel, QSplitter, Qt, QVBoxLayout, QWidget
from gui.widgets.tables import fill_table, make_table, make_text, wrap_label


class FramingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main = QVBoxLayout(self)
        self.header = wrap_label(QLabel("Framing: awaiting analysis"))
        self.header.setStyleSheet("font-weight: bold; color: #0F766E; font-size: 13px;")
        self.header.setWordWrap(True)
        main.addWidget(self.header)
        self.layout_label = wrap_label(QLabel(""))
        self.layout_label.setWordWrap(True)
        main.addWidget(self.layout_label)
        split = QSplitter(Qt.Orientation.Vertical)
        fb = QGroupBox("Frames")
        fl = QVBoxLayout(fb)
        self.table = make_table(["#", "Start bit", "Sync", "Sync errors", "Length (B)", "CRC", "CRC received",
                                 "CRC computed", "Valid", "Payload (text)"])
        self.table.itemSelectionChanged.connect(self._on_select)
        fl.addWidget(self.table)
        split.addWidget(fb)
        pb = QGroupBox("Payload")
        pl = QVBoxLayout(pb)
        self.text = make_text("#1E293B")
        pl.addWidget(self.text)
        split.addWidget(pb)
        eb = QGroupBox("Evidence")
        el = QVBoxLayout(eb)
        self.ev_text = make_text("#64748B")
        el.addWidget(self.ev_text)
        split.addWidget(eb)
        split.setSizes([220, 260, 120])
        main.addWidget(split)
        self._frames = []

    def clear(self, reason: str = "no bitstream"):
        self.header.setText(f"Framing: {reason}")
        self.layout_label.setText("")
        self.table.setRowCount(0)
        self.text.clear()
        self.ev_text.clear()
        self._frames = []

    def update_postdemod(self, post, fallback_text=None):
        if post is None or post.best is None:
            self.clear("not run")
            if fallback_text:
                self.text.setText(f"Receiver payload (no frame structure): {fallback_text}")
            return
        fa = post.best.frames
        self._frames = fa.frames
        if fa.sync is None or not fa.frames:
            self.header.setText("Framing: no library sync word in the decoded stream - frame boundaries unknown")
            self.layout_label.setText("")
            self.table.setRowCount(0)
            self.text.setText(f"Receiver payload (text heuristic, unverified): {fallback_text}" if fallback_text
                              else "No frames.")
            self.ev_text.setText("\n".join(e.description for e in fa.evidence))
            return
        self.header.setText(f"Framing: {fa.valid_frames}/{len(fa.frames)} frame(s) CRC-valid "
                            f"(sync '{fa.sync.name}'{', inverted' if fa.sync.inverted else ''})")
        self.layout_label.setText("Layout: " + (fa.layout.describe(fa.sync.length) if fa.layout else
                                                "no length/CRC layout validated") +
                                  f"   [{fa.layouts_tested} layouts tested]")
        rows = []
        for f in fa.frames:
            rows.append([f.index, f.start_bit, f.sync_name, f.sync_errors, f.length if f.length is not None else "-",
                         f.crc_name or "-", "-" if f.crc_received is None else f"0x{f.crc_received:X}",
                         "-" if f.crc_computed is None else f"0x{f.crc_computed:X}",
                         {True: "VALID", False: "FAIL", None: "n/a"}[f.crc_ok], f.text[:80]])
        fill_table(self.table, rows, [bool(f.crc_ok) for f in fa.frames])
        self._show_frame(0)
        self.ev_text.setText("\n".join(f"[{e.source}] {e.description}" for e in fa.evidence))

    def _on_select(self):
        r = self.table.currentRow()
        if r >= 0:
            self._show_frame(r)

    def _show_frame(self, i):
        if not self._frames or i >= len(self._frames):
            return
        f = self._frames[i]
        hexs = f.payload.hex(" ").upper()
        lines = [f"Frame {f.index}: {len(f.payload)} bytes, CRC "
                 f"{'VALID' if f.crc_ok else 'n/a' if f.crc_ok is None else 'FAIL'}", "", "Text:", f.text, "", "Hex:"]
        lines += [hexs[i:i + 48] for i in range(0, len(hexs), 48)]
        self.text.setText("\n".join(lines))
