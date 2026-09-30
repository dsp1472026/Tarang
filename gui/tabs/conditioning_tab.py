from typing import Optional
import numpy as np
from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QLabel, QTextEdit
)
from gui.widgets.tables import wrap_label
from conditioning.bit_order import BitOrderConverter
from conditioning.differential import DifferentialDecoder
from conditioning.descrambler import LFSRDescrambler
from conditioning.alignment import BitAlignment


class ConditioningTab(QWidget):
    """
    GUI Tab for Layer 7 Bitstream Conditioning:
    Supports bit ordering, differential decoding, LFSR descrambling, and byte alignment.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_bits: Optional[np.ndarray] = None
        self.conditioned_bits: Optional[np.ndarray] = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left Column: Controls & Configuration Panel
        ctrl_group = QGroupBox("Bitstream Conditioning Controls")
        ctrl_layout = QFormLayout(ctrl_group)

        self.bit_order_combo = QComboBox()
        self.bit_order_combo.addItems(BitOrderConverter.candidate_generator())

        self.diff_combo = QComboBox()
        self.diff_combo.addItems(DifferentialDecoder.candidate_generator())

        self.descrambler_combo = QComboBox()
        self.descrambler_combo.addItems(LFSRDescrambler.candidate_generator())

        self.alignment_combo = QComboBox()
        self.alignment_combo.addItems([str(c) for c in BitAlignment.candidate_generator()])

        self.apply_btn = QPushButton("Apply Conditioning")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self.on_apply_clicked)

        ctrl_layout.addRow("Bit Ordering:", self.bit_order_combo)
        ctrl_layout.addRow("Differential Mode:", self.diff_combo)
        ctrl_layout.addRow("LFSR Descrambler:", self.descrambler_combo)
        ctrl_layout.addRow("Bit Offset Shift:", self.alignment_combo)
        ctrl_layout.addRow(self.apply_btn)

        main_layout.addWidget(ctrl_group, stretch=1)

        # Right Column: Displays & Previews
        disp_group = QGroupBox("Conditioned Bitstream && Output Preview")
        disp_layout = QVBoxLayout(disp_group)

        self.status_label = wrap_label(QLabel("Status: Awaiting demodulated bitstream..."))
        self.status_label.setStyleSheet("font-weight: bold; color: #D97706;")
        disp_layout.addWidget(self.status_label)

        disp_layout.addWidget(QLabel("Conditioned Binary Bitstream:"))
        self.bits_text = QTextEdit()
        self.bits_text.setReadOnly(True)
        self.bits_text.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #FFFFFF; color: #0F766E;")
        disp_layout.addWidget(self.bits_text)

        disp_layout.addWidget(QLabel("Packed Hex Dump & ASCII Preview:"))
        self.hex_text = QTextEdit()
        self.hex_text.setReadOnly(True)
        self.hex_text.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #FFFFFF; color: #2563EB;")
        disp_layout.addWidget(self.hex_text)

        main_layout.addWidget(disp_group, stretch=2)

    def set_bitstream(self, bits: np.ndarray):
        """
        Receives raw demodulated bits from pipeline.
        """
        if bits is None or len(bits) == 0:
            self.raw_bits = None
            self.conditioned_bits = None
            self.status_label.setText("Status: No bits available.")
            self.bits_text.clear()
            self.hex_text.clear()
            return

        self.raw_bits = np.asarray(bits, dtype=np.uint8)
        self.status_label.setText(f"Loaded Bitstream: {len(self.raw_bits)} bits")
        self.on_apply_clicked()

    def on_apply_clicked(self):
        if self.raw_bits is None or len(self.raw_bits) == 0:
            return

        bits = self.raw_bits.copy()

        # 1. Alignment Shift
        shift = int(self.alignment_combo.currentText())
        bits = BitAlignment.shift_bits(bits, shift=shift)

        # 2. Differential Decoding
        diff_mode = self.diff_combo.currentText()
        bits = DifferentialDecoder.process(bits, mode=diff_mode)

        # 3. LFSR Descrambling
        desc_preset = self.descrambler_combo.currentText()
        bits = LFSRDescrambler.process(bits, preset=desc_preset)

        # 4. Bit Ordering & Packing
        order_mode = self.bit_order_combo.currentText()
        bits = BitOrderConverter.convert_bit_order(bits, mode=order_mode)

        self.conditioned_bits = bits

        # Update Binary Text Display
        bit_str = "".join(str(b) for b in bits[:1024])
        if len(bits) > 1024:
            bit_str += f"\n... [{len(bits) - 1024} more bits]"
        self.bits_text.setText(bit_str)

        # Update Hex Dump & ASCII Text
        packed_bytes = BitOrderConverter.pack_bits(bits, mode="MSB")
        hex_dump = packed_bytes.hex().upper()
        formatted_hex = " ".join([hex_dump[i:i+2] for i in range(0, len(hex_dump), 2)])

        try:
            ascii_str = "".join([chr(b) if 32 <= b <= 126 else "." for b in packed_bytes])
        except Exception:
            ascii_str = ""

        preview = f"HEX ({len(packed_bytes)} bytes):\n{formatted_hex[:512]}\n\nASCII TEXT:\n{ascii_str[:256]}"
        self.hex_text.setText(preview)
        self.status_label.setText(f"Conditioned Bitstream: {len(bits)} bits ({len(packed_bytes)} bytes)")
