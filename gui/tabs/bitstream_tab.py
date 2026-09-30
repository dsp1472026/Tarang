"""
Tab 8 - Bitstream (L8): statistics, periodicity, sync-word search (Chiani & Martini metrics),
GF(2) rank profile and the blind convolutional-code estimate.
"""

from typing import Optional

import numpy as np
import pyqtgraph as pg

from bitstream import forensics, sync_search
from bitstream.framing import FramingForensics
from bitstream.patterns import PatternDetector
from gui.qt_compat import QGroupBox, QHBoxLayout, QLabel, QPushButton, QSplitter, Qt, QVBoxLayout, QWidget
from gui.widgets.background import BackgroundRunner
from gui.widgets.tables import fill_table, make_table, make_text, wrap_label

SYNC_HEADERS = ["Sync word", "Len", "Best offset", "Hamming", "Hard corr", "Soft corr", "LRT", "GLRT",
                "Threshold (max errors)", "Occurrences", "Period", "P(emulation)", "Polarity", "Detection"]


class BitstreamTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_bits: Optional[np.ndarray] = None
        self.current_llrs: Optional[np.ndarray] = None
        self._runner = BackgroundRunner(self)          # forensics off the GUI thread
        self.init_ui()

    def init_ui(self):
        main = QVBoxLayout(self)
        top = QHBoxLayout()
        self.status_label = wrap_label(QLabel("Status: awaiting bitstream..."))
        self.status_label.setStyleSheet("font-weight: bold; color: #0F766E;")
        self.run_btn = QPushButton("Re-run analysis on current bits")
        self.run_btn.clicked.connect(self.run_forensics)
        top.addWidget(self.status_label, 1)
        top.addWidget(self.run_btn)
        main.addLayout(top)

        split = QSplitter(Qt.Orientation.Vertical)
        sync_box = QGroupBox("Sync-word search (sliding hard / soft correlation, LRT, GLRT; threshold from "
                             "false-alarm probability 1e-3)")
        sl = QVBoxLayout(sync_box)
        self.sync_table = make_table(SYNC_HEADERS)
        sl.addWidget(self.sync_table)
        split.addWidget(sync_box)

        lower = QSplitter(Qt.Orientation.Horizontal)
        stats_box = QGroupBox("Statistics, periodicity, structure")
        stl = QVBoxLayout(stats_box)
        self.stats_text = make_text("#1E293B")
        stl.addWidget(self.stats_text)
        lower.addWidget(stats_box)

        plots = QWidget()
        pl = QVBoxLayout(plots)
        pl.setContentsMargins(0, 0, 0, 0)
        self.ac_plot = pg.PlotWidget(title="Bit autocorrelation (bipolar)")
        self.ac_plot.setLabel("bottom", "Lag", units="bits")
        self.ac_curve = self.ac_plot.plot(pen=pg.mkPen("#2563EB", width=1.2))
        self.rank_plot = pg.PlotWidget(title="GF(2) rank deficiency (L - rank)/L vs row length L")
        self.rank_plot.setLabel("bottom", "Row length L", units="bits")
        self.rank_bars = pg.BarGraphItem(x=[], height=[], width=0.8, brush="#14B8A6")
        self.rank_plot.addItem(self.rank_bars)
        pl.addWidget(self.ac_plot)
        pl.addWidget(self.rank_plot)
        lower.addWidget(plots)
        split.addWidget(lower)
        split.setSizes([260, 420])
        main.addWidget(split)

    # ------------------------------------------------------------------
    def clear(self, reason: str = "No bitstream."):
        self.current_bits = self.current_llrs = None
        self.status_label.setText(f"Status: {reason}")
        self.sync_table.setRowCount(0)
        self.stats_text.clear()
        self.ac_curve.setData([], [])
        self.rank_bars.setOpts(x=[], height=[])

    def set_bitstream(self, bits: Optional[np.ndarray], llrs: Optional[np.ndarray] = None):
        """Manual path (conditioning tab / no Phase-4 result): analyse these bits directly."""
        if bits is None or len(bits) == 0:
            self.clear()
            return
        self.current_bits = np.asarray(bits, dtype=np.uint8)
        self.current_llrs = None if llrs is None else np.asarray(llrs, dtype=np.float64)
        self.run_forensics()

    def run_forensics(self):
        """Sync-word library search, rank profile, periodicity, autocorrelation and randomness of the
        current bits. Over a million bits this took ~8 s and froze the window ('Not responding'); a
        thread still stalled it for 1-2 s (the numpy calls hold the GIL), so it runs in a worker
        process; the tab shows a progress note until the result arrives."""
        bits = self.current_bits
        if bits is None or len(bits) == 0:
            return
        llrs = self.current_llrs
        self.status_label.setText(f"Status: analysing {len(bits):,} bits in the background...")

        def show(r):
            rnd, per, ac, syncs, prof, blind, frame_len = r
            self._show(bits, "current bits", rnd, per, ac, syncs, prof, blind, frame_len)

        self._runner.submit_process(forensics.run, (bits, llrs), show,
                                    lambda msg: self.status_label.setText(f"Status: analysis failed ({msg})"))

    def update_postdemod(self, post):
        """Phase-4 result: show the L8 report of the primary hypothesis."""
        if post is None:
            self.clear("no post-demodulation result")
            return
        bs = post.bitstream
        # the analysed stream is the receiver's; keep it for the manual re-run button
        self._show(None, bs.hypothesis, bs.randomness, bs.periodicity, bs.autocorr, bs.sync, bs.rank_profile,
                   bs.blind_conv, bs.frame_length, n_bits=bs.n_bits, hex_preview=bs.preview_hex)

    def _show(self, bits, hyp, rnd, per, ac, syncs, prof, blind, frame_len, n_bits=None, hex_preview=None):
        n = n_bits if n_bits is not None else len(bits)
        det = sync_search.best_detection(syncs)
        self.status_label.setText(f"Status: {n} bits ({hyp})" + (f" - sync '{det.name}' detected at bit "
                                                                 f"{det.offset}" if det else " - no library sync word"))
        rows, hl = [], []
        for s in syncs:
            rows.append([s.name, s.length, s.offset, f"{s.hamming}/{s.length}", round(s.hard_corr, 3),
                         round(s.soft_corr, 3), round(s.lrt, 1), round(s.glrt, 1), s.threshold, len(s.occurrences),
                         s.period or "-", f"{s.p_emulation:.2e}", "inverted" if s.inverted else "normal",
                         s.confidence if s.detected else "not detected"])
            hl.append(s.detected)
        fill_table(self.sync_table, rows, hl)

        L = ["=== RANDOMNESS ===",
             f"Bits              : {n}",
             f"Ones fraction     : {rnd.get('bit_balance', 0) * 100:.2f} %",
             f"Binary entropy    : {rnd.get('binary_entropy', 0):.4f} (1.0 = max)",
             f"Byte entropy      : {rnd.get('byte_entropy', 0):.4f} / 8 bits",
             f"Randomness score  : {rnd.get('randomness_score', 0):.4f}", "",
             "=== PERIODICITY (autocorrelation peaks) ==="]
        L += [f"lag {lag:5d} bits   r = {c:+.4f}" for lag, c in per[:8]] or ["no prominent peaks"]
        L += ["", f"Frame length estimate: {frame_len if frame_len else 'unknown'}"
              + (" (sync recurrence)" if det and det.period else " (autocorrelation)" if frame_len else ""), "",
              "=== LINEAR STRUCTURE (GF(2) rank) ==="]
        defic = {k: v for k, v in prof.items() if v > 0}
        L.append("rank-deficient row lengths: " + (", ".join(str(k) for k in list(defic)[:16]) if defic else "none "
                                                   "(no linear code structure visible without de-interleaving)"))
        L.append(blind.description if blind else "blind convolutional estimate: not applicable")
        if bits is not None:
            pats = PatternDetector.find_repeated_patterns(bits, min_len=16, max_len=32, min_count=2)
            L += ["", "=== REPEATED PATTERNS ==="]
            L += [f"0x{p['hex']}  len {p['length']}  x{p['count']}" for p in pats[:5]] or ["none"]
            bursts = FramingForensics.detect_bursts(bits, window_size=32)
            L.append(f"activity bursts: {len(bursts)}")
            hex_preview = np.packbits(bits[:256 // 8 * 8]).tobytes().hex(" ").upper()
        if hex_preview:
            L += ["", "=== FIRST 32 BYTES (MSB first, byte-aligned at bit 0) ===", hex_preview]
        self.stats_text.setText("\n".join(L))
        if ac is not None and len(ac):
            self.ac_curve.setData(np.arange(len(ac)), np.asarray(ac))
        ks = sorted(prof)
        self.rank_bars.setOpts(x=ks, height=[prof[k] for k in ks], width=0.8)
