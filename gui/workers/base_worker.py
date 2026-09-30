from gui.qt_compat import QThread, pyqtSignal
from typing import Any, Optional


class BaseWorker(QThread):
    """
    Base QThread worker for executing asynchronous DSP operations without blocking PyQt6 GUI.
    """

    # Signals
    progress = pyqtSignal(int, str)  # percent (0-100), message
    finished = pyqtSignal(object)    # result object
    error = pyqtSignal(str)          # error message
    cancelled = pyqtSignal()         # worker cancelled signal

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_cancelled = False

    def cancel(self):
        """
        Signals the worker to cancel processing as soon as safe.
        """
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def run(self):
        """
        Subclasses should override work() method instead of run()
        to ensure proper error handling and cancellation checks.
        """
        self._is_cancelled = False
        try:
            result = self.work()
            if self._is_cancelled:
                self.cancelled.emit()
            else:
                self.finished.emit(result)
        except Exception as e:
            if self._is_cancelled:                     # work() stopped at a cancellation check
                self.cancelled.emit()
            else:
                self.error.emit(str(e))

    def work(self) -> Any:
        """
        Override this method in derived workers.
        """
        raise NotImplementedError("Subclasses must implement work()")
