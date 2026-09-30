"""
Qt Compatibility Abstraction Layer supporting PyQt6, PySide6, and PyQt5.
Fixes Windows DLL path lookup conflicts and provides robust fallback across all Qt bindings.
"""
import sys
import os
import site

# 1. Automatically detect workspace .venv site-packages if present
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
venv_site = os.path.join(workspace_root, ".venv", "Lib", "site-packages")

site_dirs = []
if os.path.exists(venv_site):
    site_dirs.append(venv_site)
    if venv_site not in sys.path:
        sys.path.insert(0, venv_site)

try:
    site_dirs.extend(site.getsitepackages())
except Exception:
    pass

# 2. Add Qt DLL paths to Windows DLL search directory
if sys.platform == "win32":
    try:
        paths = os.environ.get("PATH", "").split(os.pathsep)
        cleaned_paths = []
        for p in paths:
            plower = p.lower()
            if any(env_name in plower for env_name in ["radioconda", "anaconda", "miniconda"]):
                if any(sub in plower for sub in ["library", "qt", "bin"]):
                    continue
            cleaned_paths.append(p)

        qt_pkg_subdirs = [
            "PyQt5",
            os.path.join("PyQt5", "Qt5", "bin"),
            "PySide6",
            "PySide6_Essentials",
            "shiboken6",
            "PyQt6",
            os.path.join("PyQt6", "Qt6", "bin")
        ]

        for s_dir in site_dirs:
            for sub in qt_pkg_subdirs:
                dll_dir = os.path.join(s_dir, sub)
                if os.path.exists(dll_dir):
                    cleaned_paths.insert(0, dll_dir)
                    try:
                        os.add_dll_directory(dll_dir)
                    except Exception:
                        pass

        os.environ["PATH"] = os.pathsep.join(cleaned_paths)
    except Exception:
        pass

# 3. Try PyQt5, PyQt6, PySide6
QT_BINDING = None
err_msgs = []

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import QThread, pyqtSignal, Qt
    QT_BINDING = "PyQt5"
except Exception as e:
    err_msgs.append(f"PyQt5: {e}")

if QT_BINDING is None:
    try:
        from PyQt6 import QtWidgets, QtCore, QtGui
        from PyQt6.QtCore import QThread, pyqtSignal, Qt
        QT_BINDING = "PyQt6"
    except Exception as e:
        err_msgs.append(f"PyQt6: {e}")

if QT_BINDING is None:
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from PySide6.QtCore import QThread, Signal as pyqtSignal, Qt
        QT_BINDING = "PySide6"
    except Exception as e:
        err_msgs.append(f"PySide6: {e}")

if QT_BINDING is None:
    raise ImportError(
        "Could not load any Qt binding (PyQt5, PyQt6, or PySide6).\n"
        + "\n".join(err_msgs)
        + "\nPlease install PyQt5, PyQt6, or PySide6."
    )

# Force pyqtgraph to use the successfully loaded Qt library
os.environ["QT_LIB"] = QT_BINDING

# Export standard classes
QApplication = QtWidgets.QApplication
QMainWindow = QtWidgets.QMainWindow
QWidget = QtWidgets.QWidget
QVBoxLayout = QtWidgets.QVBoxLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QFormLayout = QtWidgets.QFormLayout
QGroupBox = QtWidgets.QGroupBox
QLineEdit = QtWidgets.QLineEdit
QComboBox = QtWidgets.QComboBox
QPushButton = QtWidgets.QPushButton
QFileDialog = QtWidgets.QFileDialog
QLabel = QtWidgets.QLabel
QTextEdit = QtWidgets.QTextEdit
QTabWidget = QtWidgets.QTabWidget
QMessageBox = QtWidgets.QMessageBox
QSplitter = QtWidgets.QSplitter
QTableWidget = QtWidgets.QTableWidget
QTableWidgetItem = QtWidgets.QTableWidgetItem
QHeaderView = QtWidgets.QHeaderView
QProgressBar = QtWidgets.QProgressBar
QDialog = QtWidgets.QDialog
QRadioButton = QtWidgets.QRadioButton
QButtonGroup = QtWidgets.QButtonGroup
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QSpinBox = QtWidgets.QSpinBox
QTextBrowser = QtWidgets.QTextBrowser
QColor = QtGui.QColor

# Printing / PDF export (report tab); None when the binding ships without QtPrintSupport
try:
    if QT_BINDING == "PyQt5":
        from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
    elif QT_BINDING == "PyQt6":
        from PyQt6.QtPrintSupport import QPrinter, QPrintDialog
    else:
        from PySide6.QtPrintSupport import QPrinter, QPrintDialog
except Exception:
    QPrinter = QPrintDialog = None
# QtGui itself is also importable from here (QImage / QPixmap for the image tab)
