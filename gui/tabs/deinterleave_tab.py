"""
Tab 9 - De-interleaving (L9): best hypothesis with its evidence, candidate table, the
Burel-Gautier rank-deficiency map (offset x deficiency at the estimated period) and the
rank profile. A hypothesis is shown as validated only when FEC decoding + CRC confirm it.
"""

import numpy as np
import pyqtgraph as pg

from gui.qt_compat import QGroupBox, QLabel, QSplitter, Qt, QVBoxLayout, QWidget
from gui.widgets.tables import fill_table, make_table, make_text, wrap_label

HEADERS = ["Bit stream", "Type", "Parameters", "Offset (bits)", "Structure strength", "Evidence", "Validation"]


class DeinterleaveTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main = QVBoxLayout(self)
        self.header = wrap_label(QLabel("De-interleaving: awaiting analysis"))
        self.header.setStyleSheet("font-weight: bold; color: #0F766E; font-size: 13px;")
        self.header.setWordWrap(True)
        main.addWidget(self.header)
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QSplitter(Qt.Orientation.Vertical)
        best_box = QGroupBox("Best hypothesis and evidence")
        bl = QVBoxLayout(best_box)
        self.best_text = make_text("#0F766E")
        bl.addWidget(self.best_text)
        left.addWidget(best_box)
        cand_box = QGroupBox("Candidates (screened by FEC parity checks / sync; decoded and CRC-checked)")
        cl = QVBoxLayout(cand_box)
        self.table = make_table(HEADERS)
        cl.addWidget(self.table)
        left.addWidget(cand_box)
        split.addWidget(left)
        plots = QWidget()
        pl = QVBoxLayout(plots)
        pl.setContentsMargins(0, 0, 0, 0)
        self.heat_plot = pg.PlotWidget(title="Burel-Gautier: rank deficiency vs alignment offset (estimated period)")
        self.heat_plot.setLabel("bottom", "Offset", units="bits")
        self.heat_plot.setLabel("left", "(L - rank)/L")
        self.heat_curve = self.heat_plot.plot(pen=None, symbol="s", symbolSize=5, symbolBrush="#2563EB")
        self.prof_plot = pg.PlotWidget(title="Rank profile of the received stream (rows of L bits)")
        self.prof_plot.setLabel("bottom", "Row length L", units="bits")
        self.prof_bars = pg.BarGraphItem(x=[], height=[], width=0.8, brush="#14B8A6")
        self.prof_plot.addItem(self.prof_bars)
        pl.addWidget(self.heat_plot)
        pl.addWidget(self.prof_plot)
        split.addWidget(plots)
        split.setSizes([700, 450])
        main.addWidget(split)

    def clear(self, reason: str = "no bitstream"):
        self.header.setText(f"De-interleaving: {reason}")
        self.best_text.clear()
        self.table.setRowCount(0)
        self.heat_curve.setData([], [])
        self.prof_bars.setOpts(x=[], height=[])

    def update_postdemod(self, post):
        if post is None:
            self.clear("not run")
            return
        best = post.best
        d = post.deint_best
        validated = bool(best and best.crc_frames and d.kind != "none")
        if d.kind == "none":
            head = "no interleaver" + (" (FEC + CRC validate without de-interleaving)" if post.verified else "")
        else:
            head = f"{d.kind} interleaver {d.params}" + (" - VALIDATED" if validated else " - unverified")
        self.header.setText(f"De-interleaving decision: {head}")
        L = ["Best hypothesis:"]
        if d.kind == "block":
            L += ["  Type        : rectangular block interleaver",
                  f"  Dimensions  : {d.params}",
                  "  Orientation : transmitter writes rows, reads columns (= columns/rows of the transposed shape)",
                  f"  Bit offset  : {d.offset}"]
        elif d.kind == "conv":
            L += ["  Type        : convolutional (Forney) interleaver", f"  Parameters  : {d.params}",
                  f"  Bit offset  : {d.offset}"]
        elif d.kind == "diag":
            L += ["  Type        : diagonal (helical) block interleaver", f"  Dimensions  : {d.params}",
                  "  Orientation : transmitter writes rows, reads along the wrapped diagonals",
                  f"  Bit offset  : {d.offset}"]
        elif d.kind == "pr":
            L += ["  Type        : pseudo-random interleaver", f"  Parameters  : {d.params}",
                  f"  Bit offset  : {d.offset}"]
        else:
            L += ["  Type        : none detected / not needed"]
        L += ["", "Evidence:"]
        if d.kind != "none":
            L.append(f"  Screening   : {d.detail} (strength {d.strength:.2f})")
        pe = post.period_estimate
        if pe is not None and pe.period:
            L.append(f"  Rank method : period P = {pe.period}, best offset {pe.offset}, deficiency "
                     f"{pe.deficiency:.2f}, K-S {'n/a' if pe.ks is None else f'{pe.ks:.2f}'} (Burel-Gautier / "
                     f"Wee-Choi-Jeong)")
        else:
            L.append("  Rank method : no period found (needs N > P^2 bits; larger interleavers are hypothesised "
                     "from the burst length)")
        if best is not None:
            L.append(f"  FEC         : {best.fec_name} ({best.fec_family})")
            L.append(f"  CRC         : {best.crc_frames} valid frame(s)")
        L.append(f"  Search      : {post.hypotheses_tested} hypotheses, {post.seconds:.1f} s")
        L.append("")
        L.append(f"Confidence: {'high (validated by FEC + CRC)' if validated else 'n/a' if d.kind == 'none' else 'low (not validated)'}")
        L += ["", "Search log:"] + [f"  {m}" for m in post.log[-12:]]
        self.best_text.setText("\n".join(L))
        rows = [c.as_row() for c in post.deint_candidates]
        fill_table(self.table, rows, [bool(c.validated) for c in post.deint_candidates])
        if pe is not None and pe.period and pe.period in pe.heat:
            v = pe.heat[pe.period]
            self.heat_plot.setTitle(f"Burel-Gautier: rank deficiency vs offset at P = {pe.period}")
            self.heat_curve.setData(np.arange(len(v)), v)
        else:
            self.heat_plot.setTitle("Burel-Gautier: no period estimate (stream too short for P^2 bits)")
            self.heat_curve.setData([], [])
        prof = post.bitstream.rank_profile
        ks = sorted(prof)
        self.prof_bars.setOpts(x=ks, height=[prof[k] for k in ks], width=0.8)
