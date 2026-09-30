"""
Tab 10 - FEC (L10): FEC-family probabilities (uncoded / convolutional / block), the decoded
hypothesis with its evidence (syndrome, Viterbi re-encoding mismatch, RS corrections, CRC),
all tested candidates, the blind convolutional estimate and the code library.
"""

from gui.qt_compat import QGroupBox, QLabel, QSplitter, Qt, QVBoxLayout, QWidget
from gui.widgets.tables import fill_table, make_table, make_text, wrap_label
from fec import block_codes, concatenated, convolutional, ldpc, reed_solomon

NEXT_ACTION = {"Uncoded": "direct frame / sync / CRC analysis",
               "Convolutional code": "rate / polynomial hypotheses, soft Viterbi, re-encoding check",
               "Block code (Hamming / RS)": "syndrome decoding / RS profiles x alignments, block-interleaver search",
               "LDPC": "codeword alignment by syndrome, normalised min-sum; blind parity-check recovery",
               "Concatenated (RS + convolutional)": "Viterbi, outer byte de-interleaver x RS profiles, iterative "
                                                    "RS -> Viterbi feedback"}


class FECTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main = QVBoxLayout(self)
        self.header = wrap_label(QLabel("FEC: awaiting analysis"))
        self.header.setStyleSheet("font-weight: bold; color: #0F766E; font-size: 13px;")
        self.header.setWordWrap(True)
        main.addWidget(self.header)
        split = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)
        fam_box = QGroupBox("FEC family classification")
        fl = QVBoxLayout(fam_box)
        self.fam_table = make_table(["FEC family", "Probability", "Next action"])
        fl.addWidget(self.fam_table)
        top.addWidget(fam_box)
        best_box = QGroupBox("Best FEC hypothesis and evidence")
        bl = QVBoxLayout(best_box)
        self.best_text = make_text("#0F766E")
        bl.addWidget(self.best_text)
        top.addWidget(best_box)
        top.setSizes([420, 700])
        split.addWidget(top)
        cand_box = QGroupBox("Decoding candidates")
        cl = QVBoxLayout(cand_box)
        self.table = make_table(["Bit stream", "Family", "Code", "Setting", "Metric", "Result", "CRC-valid frames"])
        cl.addWidget(self.table)
        split.addWidget(cand_box)
        lib_box = QGroupBox("Code library searched")
        ll = QVBoxLayout(lib_box)
        self.lib_text = make_text("#64748B")
        ll.addWidget(self.lib_text)
        split.addWidget(lib_box)
        split.setSizes([300, 230, 150])
        main.addWidget(split)
        self.lib_text.setText("\n".join(
            [f"{c.name:44s} {c.describe()}" for c in convolutional.LIBRARY]
            + [f"{p.name:44s} {p.describe()}" for p in reed_solomon.PROFILES]
            + [f"{b.name:44s} {b.describe()}" for b in block_codes.LIBRARY]
            + [f"{c.name:44s} {c.describe()}" for c in ldpc.LIBRARY]
            + ["LDPC (unknown code)                          blind: parity checks recovered from the null space of "
               "aligned codewords (needs more codewords than n, error-free blocks)"]
            + [f"Concatenated {k:31s} {v.describe()}" for k, v in concatenated.PROFILES.items()]
            + ["Turbo / BCH / polar                          not identified"]))

    def clear(self, reason: str = "no bitstream"):
        self.header.setText(f"FEC: {reason}")
        self.fam_table.setRowCount(0)
        self.best_text.clear()
        self.table.setRowCount(0)

    def update_postdemod(self, post):
        if post is None:
            self.clear("not run")
            return
        fam = post.family
        rows = sorted(fam.probabilities.items(), key=lambda kv: -kv[1])
        fill_table(self.fam_table, [[k, f"{v:.2f}", NEXT_ACTION.get(k, "")] for k, v in rows],
                   [k == fam.family for k, _ in rows])
        best = post.best
        ok = post.verified
        name = best.fec_name if best else "none"
        self.header.setText(f"FEC decision: {'uncoded' if name == 'none' else name} - "
                            + ("VERIFIED by CRC" if ok else "unverified"))
        L = ["Best FEC hypothesis"]
        if best is None:
            L.append("  none")
        else:
            L += [f"  Family      : {best.fec_family}", f"  Code        : {name}",
                  f"  Description : {best.fec_detail}",
                  f"  Bit stream  : {best.hypothesis}"]
            if best.deint.kind != "none":
                L.append(f"  After       : de-interleaver {best.deint.kind} {best.deint.params} (offset {best.deint.offset})")
            L += ["", "Evidence"]
            L += [f"  [{e.source}] {e.description} (score {e.score:.2f})" for e in best.evidence]
            L += [f"  [{e.source}] {e.description} (score {e.score:.2f})" for e in fam.evidence]
            L.append(f"  CRC-valid frames: {best.crc_frames}")
        bc = post.bitstream.blind_conv
        L += ["", "Blind convolutional-code estimate (rank profile of the received stream):",
              "  " + (bc.description if bc else "not applicable (no rate-k/n rank pattern; interleaved, block-coded "
                                                "or uncoded)")]
        L += ["", f"Final confidence: {'0.95+ (decoded and CRC-validated)' if ok else 'low (no CRC validation)'}"]
        self.best_text.setText("\n".join(L))
        fill_table(self.table, [f.as_row() for f in post.fec_candidates],
                   [f.crc_frames > 0 for f in post.fec_candidates])
