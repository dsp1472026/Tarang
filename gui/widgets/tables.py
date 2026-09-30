"""Small helpers shared by the Phase 4 tabs (read-only result tables, monospace text panes)."""

from typing import List, Optional, Sequence

from gui.qt_compat import QHeaderView, QTableWidget, QTableWidgetItem, QTextEdit, Qt, QColor

MONO = "font-family: Consolas, monospace; font-size: 12px;"


def make_table(headers: Sequence[str], stretch_last: bool = True) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(list(headers))
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if stretch_last:
        hh.setStretchLastSection(True)
    return t


def fill_table(t: QTableWidget, rows: List[Sequence], highlight: Optional[List[bool]] = None):
    t.setRowCount(0)
    for i, row in enumerate(rows):
        t.insertRow(i)
        for j, v in enumerate(row):
            if isinstance(v, float):
                s = f"{v:.4g}"
            else:
                s = str(v)
            item = QTableWidgetItem(s)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if highlight and highlight[i]:
                item.setBackground(QColor(204, 251, 241))
                item.setForeground(QColor(19, 78, 74))
            t.setItem(i, j, item)


def wrap_label(label):
    """A label for dynamic text: wraps, and never forces the window wider than the screen.
    Word wrap alone cannot break a long unbroken string (a decoded binary payload), whose
    minimum width then became the window's (126 618 px: 'Unable to set geometry')."""
    from gui.qt_compat import QtWidgets
    label.setWordWrap(True)
    sp = label.sizePolicy()
    sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Ignored)
    label.setSizePolicy(sp)
    label.setMinimumWidth(50)
    return label


def short_text(text, limit: int = 400) -> str:
    """Display form of a possibly huge string: the first `limit` characters and a count of the rest."""
    s = str(text)
    return s if len(s) <= limit else s[:limit] + f" ... [{len(s) - limit:,} more characters]"


def make_text(color: str = "#1E293B") -> QTextEdit:
    w = QTextEdit()
    w.setReadOnly(True)
    w.setStyleSheet(MONO + f" background-color: #FFFFFF; color: {color};")
    return w
