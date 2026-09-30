"""
Run a tab's heavy computation off the GUI thread.

Filling the tabs after an analysis ran everything on the GUI thread: the Bitstream tab's forensics
over a million bits (sync-word library search, GF(2) rank profile, autocorrelation, ~8 s) and the
Detection tab's spectrum + spectrogram of a 2 M-sample capture (~1 s). Windows then marked the window
'Not responding' for a few seconds. A tab now submits the computation here and applies the result
when it arrives; only the latest submission of each runner is applied (an older one finishing late
is dropped), so a new analysis never shows stale plots.
"""

from typing import Callable, Optional

from gui.qt_compat import QThread, QtCore, pyqtSignal


_POOL = None


def _pool():
    """One shared worker process, started on first use (its start-up happens in the background too)."""
    global _POOL
    if _POOL is None:
        import concurrent.futures
        import multiprocessing
        _POOL = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    return _POOL


class _Task(QThread):
    done_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[], object], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done_ok.emit(self._fn())
        except Exception as e:                          # noqa: BLE001 - reported to the tab, never raised
            self.failed.emit(f"{type(e).__name__}: {e}")


class BackgroundRunner(QtCore.QObject):
    """One runner per tab (or per plot): submit(fn, on_done) computes fn() in a worker thread and calls
    on_done(result) on the GUI thread, unless a newer submission was made in the meantime."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generation = 0
        self._tasks = set()

    def submit(self, fn: Callable[[], object], on_done: Callable[[object], None],
               on_error: Optional[Callable[[str], None]] = None) -> None:
        self._generation += 1
        gen = self._generation
        task = _Task(fn, self)
        task.done_ok.connect(lambda res, g=gen: self._deliver(g, on_done, res))
        if on_error is not None:
            task.failed.connect(lambda msg, g=gen: self._deliver(g, on_error, msg))
        task.finished.connect(lambda t=task: self._forget(t))
        self._tasks.add(task)                           # keep a reference until the thread has finished
        task.start()

    def submit_process(self, fn: Callable, args: tuple, on_done: Callable[[object], None],
                       on_error: Optional[Callable[[str], None]] = None) -> None:
        """Like submit(), but fn(*args) runs in a separate process (fn must be a module-level function,
        args picklable). For work that holds Python's GIL in long numpy calls: a thread would still stall
        the GUI for the length of each call, a process cannot."""
        self._generation += 1
        gen = self._generation
        fut = _pool().submit(fn, *args)
        timer = QtCore.QTimer(self)

        def poll():
            if not fut.done():
                return
            timer.stop()
            timer.deleteLater()
            try:
                value = fut.result()
            except Exception as e:                      # noqa: BLE001 - reported to the tab
                if on_error is not None:
                    self._deliver(gen, on_error, f"{type(e).__name__}: {e}")
                return
            self._deliver(gen, on_done, value)

        timer.timeout.connect(poll)
        timer.start(100)

    def _deliver(self, gen, cb, value):
        if gen == self._generation:
            cb(value)

    def _forget(self, task):
        self._tasks.discard(task)
        task.deleteLater()

    def busy(self) -> bool:
        return any(t.isRunning() for t in self._tasks)
