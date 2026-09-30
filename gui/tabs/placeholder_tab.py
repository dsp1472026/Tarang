from gui.qt_compat import QWidget, QVBoxLayout, QLabel, Qt


class GenericPlaceholderTab(QWidget):
    """
    Placeholder tab widget for pipeline phases coming up in Phase 2, 3, and 4.
    """

    def __init__(self, tab_name: str, phase_num: int, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(f"<b>{tab_name}</b><br><br>This module will be activated in Phase {phase_num}.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
