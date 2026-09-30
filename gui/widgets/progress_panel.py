from gui.qt_compat import QWidget, QHBoxLayout, QProgressBar, QLabel, QPushButton, pyqtSignal
from gui.widgets.tables import wrap_label


class ProgressPanel(QWidget):
    """
    Bottom progress bar and status panel with cancellation support.
    """

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.status_label = wrap_label(QLabel("Ready"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)

        self.layout.addWidget(self.status_label, 2)
        self.layout.addWidget(self.progress_bar, 3)
        self.layout.addWidget(self.cancel_btn, 0)

    def start_task(self, task_name: str):
        self.status_label.setText(task_name)
        self.progress_bar.setValue(0)
        self.cancel_btn.setEnabled(True)

    def update_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def finish_task(self, message: str = "Ready"):
        self.progress_bar.setValue(100)
        self.status_label.setText(message)
        self.cancel_btn.setEnabled(False)
