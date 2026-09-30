"""
Unit Test for MainWindow and GUI Tabs.
"""

import sys
import pytest
from gui.qt_compat import QApplication
from gui.main_window import MainWindow


def test_main_window_initialization():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = MainWindow()

    assert window.tabs.count() == 14
    assert window.tabs.tabText(0) == "1. Input && Setup"          # '&&' = a literal '&' (not a shortcut)
    assert window.tabs.tabText(12) == "13. Image"
    assert window.tabs.tabText(13) == "14. Signals"
    assert window.tabs.tabText(1) == "2. Signal Detection"
    assert window.tabs.tabText(2) == "3. Parameters"
    assert window.tabs.tabText(3) == "4. Modulation"
    assert window.tabs.tabText(4) == "5. Sync"
    assert window.tabs.tabText(6) == "7. Conditioning"
    assert window.tabs.tabText(7) == "8. Bitstream"
    assert window.modulation_tab is not None
    assert window.sync_tab is not None
    assert window.demodulation_tab is not None
    assert window.conditioning_tab is not None
    assert window.bitstream_tab is not None
