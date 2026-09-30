"""
Demodulation GUI Tab.
Displays Symbol Decision Bitstreams, Error Vector Magnitude (EVM %),
and 1 SPS Synchronized Constellation Diagram.
"""

from typing import Dict, Any
from gui.qt_compat import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QTextEdit
from gui.widgets.tables import short_text, wrap_label
from gui.widgets.constellation_plot import ConstellationPlotWidget
from demodulation.demodulator import DemodulationResult


class DemodulationTab(QWidget):
    """
    GUI Tab displaying Layer 6 Symbol Demodulation & EVM metrics.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # Left Column: Constellation Plot
        plot_group = QGroupBox("1 SPS Synchronized Constellation Diagram")
        plot_layout = QVBoxLayout(plot_group)
        self.constellation_plot = ConstellationPlotWidget()
        plot_layout.addWidget(self.constellation_plot)
        layout.addWidget(plot_group, stretch=1)

        # Right Column: Bitstream & Metrics Panel
        right_group = QGroupBox("Demodulated Output && Metrics")
        right_layout = QVBoxLayout(right_group)

        self.evm_label = wrap_label(QLabel("EVM: N/A"))
        self.evm_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #D97706;")
        right_layout.addWidget(self.evm_label)

        self.payload_label = wrap_label(QLabel("Decoded Payload: N/A"))
        self.payload_label.setWordWrap(True)
        self.payload_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0F766E;")
        right_layout.addWidget(self.payload_label)

        self.bit_count_label = wrap_label(QLabel("Bit Count: 0 bits"))
        self.bit_count_label.setStyleSheet("font-size: 13px; color: #1E293B;")
        right_layout.addWidget(self.bit_count_label)

        right_layout.addWidget(QLabel("Demodulated Bitstream (Hex / Binary Preview):"))
        self.bitstream_text = QTextEdit()
        self.bitstream_text.setReadOnly(True)
        self.bitstream_text.setStyleSheet("font-family: monospace; font-size: 12px; background-color: #FFFFFF; color: #0F766E;")
        right_layout.addWidget(self.bitstream_text)

        layout.addWidget(right_group, stretch=1)

    def clear(self, reason: str = ""):
        self.constellation_plot.update_plot(None)
        self.evm_label.setText("EVM: N/A")
        self.bit_count_label.setText("Total Demodulated Bits: 0 bits")
        self.bitstream_text.setText(reason or "No bits demodulated.")

    def update_payload(self, text, crc_valid: bool, modulation: str = ""):
        if text is None:
            self.payload_label.setText(f"Decoded Payload: N/A ({modulation}: demodulator not available yet)")
            return
        self.payload_label.setText(f"Decoded Payload ({modulation}, CRC {'VALID' if crc_valid else 'n/a or FAIL'}): "
                                   f"{short_text(repr(text), 300)}")

    def update_demodulation(self, demod_res: DemodulationResult):
        if not demod_res:
            return

        # 1. Update Constellation Plot
        if len(demod_res.constellation_points) > 0:
            self.constellation_plot.update_plot(demod_res.constellation_points)

        # 2. Update EVM & Bit Count
        if len(demod_res.constellation_points) > 0:
            self.evm_label.setText(f"Error Vector Magnitude (EVM): {demod_res.evm_pct:.2f}%")
        else:                                   # FSK: no constellation; value is the tone-decision error rate
            self.evm_label.setText(f"FSK decision error estimate: {demod_res.evm_pct:.2f}%")
        self.bit_count_label.setText(f"Total Demodulated Bits: {len(demod_res.bits)} bits")

        # 3. Format Bitstream Preview
        bits = demod_res.bits
        if len(bits) > 0:
            bit_str = "".join(str(b) for b in bits[:512])
            if len(bits) > 512:
                bit_str += f"\n... [{len(bits) - 512} more bits]"
            self.bitstream_text.setText(bit_str)
        else:
            self.bitstream_text.setText("No bits demodulated.")
