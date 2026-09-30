import os
from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QComboBox, QPushButton, QFileDialog, QLabel, QTextEdit, QMessageBox, pyqtSignal
)
from gui.widgets.tables import wrap_label
from ingestion.format_detector import FormatDetector
from ingestion.wav_reader import WAVReader
from gui.widgets.iq_duration_dialog import IQSetupDialog
from ingestion.iq_formats import DISPLAY_NAMES, normalize_datatype


class InputTab(QWidget):
    """
    Input Tab: File selection, automatic format detection display,
    duration-based sample rate calculation, and ingest controls.
    """

    ingest_requested = pyqtSignal(dict)  # Emits config dictionary to trigger pipeline worker

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. File Selection Group
        file_group = QGroupBox("Signal File Selection")
        file_layout = QHBoxLayout(file_group)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select .IQ, .wav, .bin, or .sigmf-meta file...")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)

        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)

        # 2. Format & Reader Configuration Group
        config_group = QGroupBox("Signal && Reader Parameters")
        config_layout = QFormLayout(config_group)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["AUTO", "IQ", "WAV", "SIGMF"])

        self.dtype_combo = QComboBox()
        self.dtype_combo.addItem("auto", None)
        for key, label in DISPLAY_NAMES.items():         # CU8, CS8/CI8, CS16/CI16/SC16, CF32/FC32, CF64/FC64
            self.dtype_combo.addItem(label, key)

        self.endian_combo = QComboBox()
        self.endian_combo.addItems(["little", "big"])

        self.iq_order_combo = QComboBox()
        self.iq_order_combo.addItems(["IQ", "QI"])

        self.byte_offset_edit = QLineEdit("0")
        self.sample_rate_edit = QLineEdit("1000000")  # Default 1 MHz

        self.fs_status_label = wrap_label(QLabel("Sample Rate Status: Pending file selection..."))
        self.fs_status_label.setStyleSheet("font-weight: bold; color: #2563EB;")

        self.center_freq_edit = QLineEdit("0")
        self.threshold_edit = QLineEdit("6.0")        # Threshold dB for SOI detection

        self.read_mode_label = wrap_label(QLabel("Read Mode: MONO_ANALYTIC (mono WAV \u2192 analytic signal via Hilbert transform)"))
        self.read_mode_label.setStyleSheet("font-weight: bold; color: #B45309; background-color: #FFFBEB; padding: 5px 8px; border: 1px solid #FDE68A; border-radius: 6px;")
        self.read_mode_label.setVisible(False)

        config_layout.addRow("File Format:", self.format_combo)
        config_layout.addRow(self.read_mode_label)
        config_layout.addRow("Data Type (IQ):", self.dtype_combo)
        config_layout.addRow("Endianness:", self.endian_combo)
        config_layout.addRow("I/Q Order:", self.iq_order_combo)
        config_layout.addRow("Byte Offset:", self.byte_offset_edit)
        config_layout.addRow("Sample Rate (Hz):", self.sample_rate_edit)
        config_layout.addRow("Resolution Source:", self.fs_status_label)
        config_layout.addRow("Center Frequency (Hz):", self.center_freq_edit)
        config_layout.addRow("SOI Detection Threshold (dB):", self.threshold_edit)

        # 3. Process Button
        self.process_btn = QPushButton("Ingest && Analyze Signal")
        self.process_btn.setObjectName("primary")          # accent style from gui/theme.py
        self.process_btn.clicked.connect(self.on_process_clicked)

        # 4. Summary Text Area
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)

        main_layout.addWidget(file_group)
        main_layout.addWidget(config_group)
        main_layout.addWidget(self.process_btn)
        main_layout.addWidget(QLabel("Metadata & File Info Summary:"))
        main_layout.addWidget(self.summary_text)

    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Signal File",
            "",
            "All Signal Files (*.iq *.wav *.bin *.raw *.sigmf-meta *.sigmf-data);;IQ Files (*.iq *.bin *.raw);;WAV Files (*.wav);;SigMF Files (*.sigmf-meta);;All Files (*)"
        )
        if filename:
            self.file_path_edit.setText(filename)
            self.handle_file_selected(filename)

    def handle_file_selected(self, file_path: str):
        if not os.path.exists(file_path):
            return

        fmt = FormatDetector.detect_format(file_path)

        self.sample_rate_source = None
        self.blind_sps = None
        if fmt == "WAV":
            try:
                meta = WAVReader.parse_header(file_path)
                channels = meta.get("channels", 1)
                wav_buf = WAVReader.read(file_path)
                fs = wav_buf.sample_rate
                if fs and fs > 0:
                    self.sample_rate_edit.setText(str(int(fs)))
                    self.fs_status_label.setText(f"Auto-detected from WAV header: {fs:,.0f} Hz")
                    self.fs_status_label.setStyleSheet("font-weight: bold; color: #16A34A;")
                    self.format_combo.setCurrentText("WAV")

                if channels == 1:
                    self.read_mode_label.setVisible(True)
                    self.dtype_combo.setVisible(False)
                    self.endian_combo.setVisible(False)
                    self.iq_order_combo.setVisible(False)
                    self.byte_offset_edit.setVisible(False)
                else:
                    self.read_mode_label.setVisible(False)
                    self.dtype_combo.setVisible(True)
                    self.endian_combo.setVisible(True)
                    self.iq_order_combo.setVisible(True)
                    self.byte_offset_edit.setVisible(True)
            except Exception as e:
                self.fs_status_label.setText(f"WAV Header Read Error: {str(e)}")

        elif fmt == "IQ":
            dialog = IQSetupDialog(file_path, parent=self)
            if dialog.exec() == IQSetupDialog.Accepted:
                iq_cfg = dialog.get_config()
                self.format_combo.setCurrentText("IQ")
                self.apply_iq_setup(iq_cfg)
            else:
                self.summary_text.setText("File ingestion setup cancelled by user.")

    def apply_iq_setup(self, iq_cfg):
        """Copy the IQ setup dialog's result (type, endianness, header, sample rate and its source) to the form."""
        fs = iq_cfg["sample_rate"]
        self.sample_rate_source = iq_cfg.get("sample_rate_source", "duration")
        self.blind_sps = iq_cfg.get("blind_sps")
        # full precision: truncating to whole Hz would add a sample-clock error for short recordings
        self.sample_rate_edit.setText("" if fs is None else f"{fs:.10g}")
        text = {"duration": f"Computed sample rate: {fs or 0:,.3f} Hz (from user-provided duration)",
                "blind": f"ASSUMED sample rate: {fs or 0:,.3f} Hz (blind: {self.blind_sps or 0:.4f} samples/symbol x "
                         f"standard symbol rate)",
                "unknown": "Sample rate UNKNOWN: results in normalised units"}[self.sample_rate_source]
        self.fs_status_label.setText(text)
        self.fs_status_label.setStyleSheet("font-weight: bold; color: " +
                                           ("#2563EB;" if self.sample_rate_source == "duration" else "#D97706;"))
        key, endian = normalize_datatype(iq_cfg["dtype"])
        idx = self.dtype_combo.findData(key)
        if idx >= 0:
            self.dtype_combo.setCurrentIndex(idx)
        self.endian_combo.setCurrentText(endian)
        self.byte_offset_edit.setText(str(int(iq_cfg.get("byte_offset", 0) or 0)))

    def on_process_clicked(self):
        file_path = self.file_path_edit.text().strip()
        if not file_path or not os.path.exists(file_path):
            self.summary_text.setText("Error: Selected file path does not exist.")
            return

        fmt_type = FormatDetector.detect_format(file_path)
        no_rate = not self.sample_rate_edit.text().strip() or float(self.sample_rate_edit.text() or 0) <= 0
        if fmt_type == "IQ" and no_rate and getattr(self, "sample_rate_source", None) != "unknown":
            # Prompt setup dialog if sample rate not set
            dialog = IQSetupDialog(file_path, parent=self)
            if dialog.exec() == IQSetupDialog.Accepted:
                iq_cfg = dialog.get_config()
                self.apply_iq_setup(iq_cfg)
            else:
                return

        try:
            sample_rate = float(self.sample_rate_edit.text()) if self.sample_rate_edit.text().strip() else None
            center_freq = float(self.center_freq_edit.text()) if self.center_freq_edit.text().strip() else 0.0
            byte_offset = int(self.byte_offset_edit.text()) if self.byte_offset_edit.text().strip() else 0
            thresh_db = float(self.threshold_edit.text()) if self.threshold_edit.text().strip() else 6.0
        except ValueError as e:
            self.summary_text.setText(f"Invalid numeric input in settings: {str(e)}")
            return

        fmt = self.format_combo.currentText()
        force_fmt = None if fmt == "AUTO" else fmt

        config = {
            "file_path": file_path,
            "force_format": force_fmt,
            "dtype": self.dtype_combo.currentData() or "auto",
            "endianness": self.endian_combo.currentText(),
            "iq_order": self.iq_order_combo.currentText(),
            "byte_offset": byte_offset,
            "sample_rate": sample_rate,
            # duration | blind | unknown (IQ setup dialog), header (WAV), manual (typed in the field)
            "sample_rate_source": ("header" if fmt_type == "WAV" else
                                   getattr(self, "sample_rate_source", None) or "manual"),
            "blind_sps": getattr(self, "blind_sps", None),
            "center_frequency": center_freq,
            "threshold_db": thresh_db,
        }

        self.ingest_requested.emit(config)

    def display_buffer_summary(self, buffer):
        if not buffer or not buffer.is_valid():
            self.summary_text.setText("Failed to load valid signal buffer.")
            return

        if buffer.sample_rate and buffer.sample_rate > 0:
            self.sample_rate_edit.setText(str(int(buffer.sample_rate)))

        summary = [
            f"=== SIGNAL BUFFER SUMMARY ===",
            f"Source Format : {buffer.source_format}",
            f"Sample Count  : {len(buffer):,}",
            f"Sample Rate   : {buffer.sample_rate / 1e6:.4f} MHz ({buffer.sample_rate:,.0f} Hz)" if buffer.sample_rate else "Sample Rate   : UNKNOWN",
            f"Center Freq   : {buffer.center_frequency / 1e6:.4f} MHz",
            f"Start Time    : {buffer.start_time:.4f} s",
            f"Metadata      : {buffer.metadata}"
        ]
        self.summary_text.setText("\n".join(summary))
