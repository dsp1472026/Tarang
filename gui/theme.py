"""
Application colour theme: a light, professional palette (white cards on a soft slate background,
blue primary, teal accents), applied as one Qt style sheet plus matching pyqtgraph defaults. Call
apply_theme(app) once, before building widgets (plots pick up the pyqtgraph defaults when they are created).
"""

from gui.qt_compat import QApplication, QColor, QtGui

# palette tokens
PRIMARY = "#2563EB"     # blue: primary actions, selection, focus
TEAL = "#0F766E"        # headings, section titles
TEAL_LIGHT = "#14B8A6"  # secondary accent
BG = "#F8FAFC"          # window background
PANEL = "#FFFFFF"       # cards: group boxes, tab pages
PANEL_2 = "#FFFFFF"     # inputs, tables
SURFACE = "#F1F5F9"     # table headers, hover fills, tracks
BORDER = "#E2E8F0"
BORDER_STRONG = "#CBD5E1"
TEXT = "#1E293B"
MUTED = "#64748B"
SELECT = "#DBEAFE"      # selection background (light blue)
GOOD = "#16A34A"
WARN = "#D97706"
BAD = "#DC2626"
ACCENT = TEAL           # kept for older call sites: accent text
ACCENT_2 = PRIMARY

# per-signal colours (plots, table badges): muted, distinguishable on white
SERIES = ["#2563EB", "#0F766E", "#D97706", "#7C3AED", "#DC2626", "#0891B2", "#65A30D", "#DB2777"]

STYLE = f"""
QWidget {{ background-color: {BG}; color: {TEXT}; font-family: "Segoe UI", "Inter", sans-serif; font-size: 10pt; }}
QMainWindow, QDialog {{ background-color: {BG}; }}
QToolTip {{ background-color: {PANEL}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; padding: 4px; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {PANEL}; top: -1px; }}
QTabBar::tab {{ background: {BG}; color: {MUTED}; padding: 6px 9px; margin-right: 1px;
               border: 1px solid {BORDER}; border-bottom: none; border-top-left-radius: 7px; border-top-right-radius: 7px; }}
QTabBar::tab:hover {{ color: {TEXT}; background: {SURFACE}; }}
QTabBar::tab:selected {{ color: {PRIMARY}; background: {PANEL}; font-weight: 600; border-color: {BORDER};
                        border-top: 2px solid {PRIMARY}; }}

QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {TEAL}; font-weight: 600; background: {BG}; }}

QLabel {{ background: transparent; }}

QPushButton {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; border-radius: 6px;
              padding: 6px 14px; font-weight: 600; }}
QPushButton:hover {{ background: {SURFACE}; color: {PRIMARY}; border-color: {PRIMARY}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ background: {SURFACE}; color: #94A3B8; border-color: {BORDER}; }}
QPushButton#primary {{ background: {PRIMARY}; color: #FFFFFF; border: 1px solid {PRIMARY}; font-size: 11pt; padding: 9px; }}
QPushButton#primary:hover {{ background: #1D4ED8; border-color: #1D4ED8; }}
QPushButton#primary:pressed {{ background: #1E40AF; }}
QPushButton#primary:disabled {{ background: #93C5FD; border-color: #93C5FD; color: #FFFFFF; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {PANEL_2}; border: 1px solid {BORDER_STRONG}; border-radius: 6px;
                                                  padding: 5px 8px; selection-background-color: {SELECT}; selection-color: {TEXT}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {PRIMARY}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {BORDER_STRONG}; selection-background-color: {SELECT};
                               selection-color: {TEXT}; }}

QTextEdit, QPlainTextEdit, QTextBrowser {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;
                                          selection-background-color: {SELECT}; selection-color: {TEXT}; }}

QTableWidget, QTableView {{ background: {PANEL_2}; alternate-background-color: {BG}; gridline-color: {BORDER};
                            border: 1px solid {BORDER}; border-radius: 6px; selection-background-color: {SELECT};
                            selection-color: {TEXT}; }}
QHeaderView::section {{ background: {SURFACE}; color: #334155; padding: 6px; border: none;
                        border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER}; font-weight: 600; }}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}

QProgressBar {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 7px; text-align: center; color: {TEXT};
                height: 18px; }}
QProgressBar::chunk {{ border-radius: 6px; background: {TEAL_LIGHT}; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #94A3B8; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER_STRONG}; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: #94A3B8; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:hover {{ background: {PRIMARY}; }}
QRadioButton::indicator:checked, QCheckBox::indicator:checked {{ background: {PRIMARY}; border: 1px solid {PRIMARY}; border-radius: 3px; }}
QMessageBox {{ background: {PANEL}; }}
"""


def apply_theme(app: QApplication) -> None:
    """Style sheet + palette + pyqtgraph defaults. Safe to call more than once."""
    if app is None or app.property("sih_theme_applied"):
        return
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    for role, col in ((roles.Window, BG), (roles.WindowText, TEXT), (roles.Base, PANEL_2), (roles.AlternateBase, BG),
                      (roles.Text, TEXT), (roles.Button, PANEL_2), (roles.ButtonText, TEXT), (roles.Highlight, SELECT),
                      (roles.HighlightedText, TEXT), (roles.ToolTipBase, PANEL_2), (roles.ToolTipText, TEXT)):
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
    try:
        import pyqtgraph as pg
        pg.setConfigOptions(background=PANEL, foreground=MUTED, antialias=True)
    except Exception:                                  # noqa: BLE001 - plots keep their defaults
        pass
    app.setProperty("sih_theme_applied", True)
