"""
Synchronization GUI Tab.
Displays Carrier Frequency/Phase Lock (Costas Loop), AGC Normalization,
and Gardner Symbol Timing Recovery metrics.
"""

from typing import Dict, Any
from gui.qt_compat import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView
from gui.widgets.tables import wrap_label
from synchronization.agc import AGCResult
from synchronization.costas_loop import CostasResult
from synchronization.timing_recovery import TimingRecoveryResult


class SyncTab(QWidget):
    """
    GUI Tab displaying Layer 5 Carrier & Symbol Synchronization results.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.header_label = wrap_label(QLabel("Synchronization Status: Awaiting Analysis..."))
        self.header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0F766E;")
        layout.addWidget(self.header_label)

        # Sync Parameters Table
        table_group = QGroupBox("Carrier && Timing Synchronization Metrics")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Stage", "Parameter / Metric", "Value", "Method / Description"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.table)
        layout.addWidget(table_group)

    def update_sync(self, agc_res: AGCResult, costas_res: CostasResult, timing_res: TimingRecoveryResult):
        self.header_label.setText("Synchronization Complete: Carrier & Timing Locked")
        self.table.setRowCount(3)

        # 1. AGC
        if agc_res:
            self.table.setItem(0, 0, QTableWidgetItem("AGC"))
            self.table.setItem(0, 1, QTableWidgetItem("Normalized Gain"))
            self.table.setItem(0, 2, QTableWidgetItem(f"{agc_res.gain_applied:.4f} linear"))
            self.table.setItem(0, 3, QTableWidgetItem("Power Normalization to Unity"))

        # 2. Costas Loop
        if costas_res:
            self.table.setItem(1, 0, QTableWidgetItem("Costas Loop"))
            self.table.setItem(1, 1, QTableWidgetItem("Residual CFO"))
            self.table.setItem(1, 2, QTableWidgetItem(f"{costas_res.cfo_residual_hz:.2f} Hz"))
            self.table.setItem(1, 3, QTableWidgetItem("M-th Power Phase Lock"))

        # 3. Timing Recovery
        if timing_res:
            self.table.setItem(2, 0, QTableWidgetItem("Gardner TED"))
            self.table.setItem(2, 1, QTableWidgetItem("Output Symbols (1 SPS)"))
            self.table.setItem(2, 2, QTableWidgetItem(f"{len(timing_res.buffer.samples)} symbols"))
            self.table.setItem(2, 3, QTableWidgetItem("Linear Fractional Interpolation"))

    def clear_sync(self, message: str):
        self.header_label.setText(message)
        self.table.setRowCount(0)
