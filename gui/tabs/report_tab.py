"""
Tab 12 - Report: evidence-based report of the whole run (HTML view) with export to
HTML, Markdown, JSON and PDF, and printing. The view and the HTML/PDF exports include spectrum and
constellation snapshots of the run.
"""

import os

from gui.qt_compat import (QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPrintDialog, QPrinter, QPushButton,
                           QTextBrowser, QVBoxLayout, QWidget, QtGui)
from gui.widgets.report_figures import render_figures
from gui.widgets.tables import wrap_label
from reporting.report import build_report, save, to_html


class ReportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = None
        self.figures = []
        main = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.status = wrap_label(QLabel("Report: run an analysis first"))
        self.status.setStyleSheet("font-weight: bold; color: #0F766E;")
        bar.addWidget(self.status, 1)
        for label, ext in (("Export HTML", ".html"), ("Export Markdown", ".md"), ("Export JSON", ".json")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, e=ext: self.export(e))
            bar.addWidget(b)
        self.pdf_btn = QPushButton("Export PDF")
        self.pdf_btn.clicked.connect(self.export_pdf)
        bar.addWidget(self.pdf_btn)
        self.print_btn = QPushButton("Print")
        self.print_btn.clicked.connect(self.print_report)
        bar.addWidget(self.print_btn)
        if QPrinter is None:                           # Qt build without print support
            for b in (self.pdf_btn, self.print_btn):
                b.setEnabled(False)
                b.setToolTip("Qt print support is not available in this installation")
        main.addLayout(bar)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(False)
        main.addWidget(self.view)

    def clear(self):
        self.report = None
        self.figures = []
        self.status.setText("Report: run an analysis first")
        self.view.clear()

    def update_results(self, results):
        try:
            self.report = build_report(results)
        except Exception as e:                        # the report must never break the GUI
            self.report = None
            self.figures = []
            self.status.setText(f"Report: could not be built ({e})")
            return
        r = self.report["result"]
        self.status.setText(f"Report: {r['status']} - {r['processing_chain']}")
        self.figures = render_figures(results)
        self.view.setHtml(to_html(self.report, self.figures))

    def export(self, ext: str):
        if not self._check_ready():
            return
        stem = os.path.splitext(self.report["input"]["file_name"] or "analysis")[0]
        filt = {".html": "HTML (*.html)", ".md": "Markdown (*.md)", ".json": "JSON (*.json)"}[ext]
        path, _ = QFileDialog.getSaveFileName(self, "Export report", f"{stem}_report{ext}", filt)
        if path:
            if not path.lower().endswith(ext):
                path += ext
            save(self.report, path, self.figures)
            self.status.setText(f"Report saved: {path}")

    def _check_ready(self) -> bool:
        if not self.report:
            QMessageBox.information(self, "Report", "Run an analysis first.")
            return False
        return True

    def _print_to(self, printer) -> None:
        """Lay the report out on the printer's pages (a fresh document, independent of the view's width)."""
        doc = QtGui.QTextDocument()
        doc.setHtml(to_html(self.report, self.figures))
        (doc.print_ if hasattr(doc, "print_") else doc.print)(printer)

    def export_pdf(self):
        if not self._check_ready():
            return
        stem = os.path.splitext(self.report["input"]["file_name"] or "analysis")[0]
        path, _ = QFileDialog.getSaveFileName(self, "Export report as PDF", f"{stem}_report.pdf", "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        try:
            self._print_to(printer)
        except Exception as e:                        # e.g. file locked by a PDF viewer
            QMessageBox.warning(self, "Export PDF", f"Could not write the PDF:\n{e}")
            return
        self.status.setText(f"Report saved: {path}")

    def print_report(self):
        if not self._check_ready():
            return
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        dlg = QPrintDialog(printer, self)
        dlg.setWindowTitle("Print report")
        if dlg.exec() == QPrintDialog.DialogCode.Accepted:
            self._print_to(printer)
            self.status.setText("Report sent to the printer")
