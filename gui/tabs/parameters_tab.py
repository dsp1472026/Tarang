from typing import Dict
from gui.qt_compat import (
    QWidget, QVBoxLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QTextEdit, QLabel, Qt
)
from core.evidence import Parameter


class ParametersTab(QWidget):
    """
    Parameters Tab: Displays extracted parameters with explicit provenance
    (KNOWN/ESTIMATED/ASSUMED/UNKNOWN), confidence scores, and supporting evidence.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameters_map: Dict[str, Parameter] = {}
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Parameter Provenance Table
        table_group = QGroupBox("Signal Parameters Provenance Table")
        table_layout = QVBoxLayout(table_group)

        self.param_table = QTableWidget(0, 6)
        self.param_table.setHorizontalHeaderLabels([
            "Parameter Name", "Value", "Unit", "Provenance (Source)", "Estimation Method", "Confidence"
        ])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.param_table.itemSelectionChanged.connect(self.on_param_selected)

        table_layout.addWidget(self.param_table)

        # Evidence Details Inspector
        evidence_group = QGroupBox("Supporting Evidence Inspector")
        evidence_layout = QVBoxLayout(evidence_group)

        self.evidence_text = QTextEdit()
        self.evidence_text.setReadOnly(True)
        evidence_layout.addWidget(self.evidence_text)

        splitter.addWidget(table_group)
        splitter.addWidget(evidence_group)

        main_layout.addWidget(splitter)

    def update_parameters(self, parameters: Dict[str, Parameter]):
        self.parameters_map = parameters
        self.param_table.setRowCount(0)
        self.evidence_text.clear()

        for name, param in parameters.items():
            row = self.param_table.rowCount()
            self.param_table.insertRow(row)

            val_str = f"{param.value:,.4f}" if isinstance(param.value, (int, float)) else str(param.value)
            conf_str = f"{param.confidence:.2f}" if param.confidence is not None else "N/A"

            self.param_table.setItem(row, 0, QTableWidgetItem(param.name))
            item = QTableWidgetItem(val_str if len(val_str) <= 200 else val_str[:200] + " ...")
            item.setToolTip(val_str[:2000])
            self.param_table.setItem(row, 1, item)
            self.param_table.setItem(row, 2, QTableWidgetItem(param.unit))
            self.param_table.setItem(row, 3, QTableWidgetItem(param.source))
            self.param_table.setItem(row, 4, QTableWidgetItem(param.method or "N/A"))
            self.param_table.setItem(row, 5, QTableWidgetItem(conf_str))

    def on_param_selected(self):
        selected_rows = self.param_table.selectedItems()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        param_name = self.param_table.item(row, 0).text()

        param = self.parameters_map.get(param_name)
        if not param:
            return

        lines = [
            f"=== EVIDENCE DETAILS FOR PARAMETER: {param.name.upper()} ===",
            f"Value       : {param.value} {param.unit}",
            f"Provenance  : {param.source}",
            f"Method      : {param.method}",
            f"Confidence  : {param.confidence}",
            "\n--- SUPPORTING EVIDENCE ITEMS ---"
        ]

        for i, ev in enumerate(param.evidence, 1):
            lines.append(f"[{i}] Description: {ev.description}")
            lines.append(f"    Source     : {ev.source}")
            lines.append(f"    Score      : {ev.score:.2f}")
            lines.append(f"    Metadata   : {ev.metadata}\n")

        self.evidence_text.setText("\n".join(lines))
