"""
IQ File Setup & Duration Dialog.
Provides automatic sample format detection display with confidence scoring,
format override dropdown, recording duration & unit input (s, ms, µs, ns),
and real-time sample rate calculation & validation.
"""

import os
from typing import Dict, Any, Optional
from gui.qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QComboBox, QPushButton, QMessageBox, Qt, QRadioButton, QButtonGroup, QApplication
)
from ingestion.format_detector import FormatDetector, IQDetectionResult
from ingestion.iq_formats import DISPLAY_NAMES, FORMATS, datatype_tag, normalize_datatype


class IQSetupDialog(QDialog):
    """
    Dialog displayed when an IQ file is loaded to confirm sample format & recording duration.
    """

    UNIT_CONVERSION = {
        "Seconds (s)": 1.0,
        "Milliseconds (ms)": 1e-3,
        "Microseconds (µs)": 1e-6,
        "Nanoseconds (ns)": 1e-9
    }

    # CU8 2, CS8 2, CS16 4, CF32 8, CF64 16 bytes per complex (I+Q) sample
    BYTES_PER_COMPLEX_SAMPLE = {key: spec[1] for key, spec in FORMATS.items()}

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        self.setWindowTitle("IQ File Setup — Sample Format & Sample Rate")
        self.setMinimumWidth(540)

        # Run auto-detection
        self.detection_result: IQDetectionResult = FormatDetector.auto_detect_iq_format(file_path)

        self.selected_format = normalize_datatype(self.detection_result.format_name)[0]   # cu8 | cs8 | cs16 | cf32 | cf64
        self.detected_format = self.selected_format
        self.header_offset = int(self.detection_result.header_offset or 0)
        self.computed_fs: Optional[float] = None
        self.mode = "duration"                       # "duration" | "blind" | "unknown"
        self.blind = None                            # BlindRateEstimate after "Estimate"

        self.init_ui()
        self.update_calculations()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. File Info Group
        file_group = QGroupBox("IQ File Information")
        file_layout = QFormLayout(file_group)

        self.file_label = QLabel(os.path.basename(self.file_path))
        self.file_label.setStyleSheet("font-weight: bold;")
        self.size_label = QLabel(f"{self.file_size:,} bytes")

        file_layout.addRow("File Path:", self.file_label)
        file_layout.addRow("File Size:", self.size_label)

        # 2. Format Auto-Detection Result Group
        detect_group = QGroupBox("Automatic Sample Format Detection")
        detect_layout = QVBoxLayout(detect_group)

        self.result_box = QLabel()
        self.update_detection_box_style()

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Sample Format:"))

        self.format_combo = QComboBox()
        for key, label in DISPLAY_NAMES.items():
            self.format_combo.addItem(f"{label} - {FORMATS[key][1]} B/sample", key)
        idx = self.format_combo.findData(self.selected_format)
        if idx >= 0:
            self.format_combo.setCurrentIndex(idx)
        self.format_combo.currentIndexChanged.connect(self.on_format_changed)

        format_layout.addWidget(self.format_combo)

        detect_layout.addWidget(self.result_box)
        detect_layout.addLayout(format_layout)

        # 3. How to obtain the sample rate
        mode_group = QGroupBox("Sample rate source")
        mode_layout = QVBoxLayout(mode_group)
        self.rb_duration = QRadioButton("Enter the recording duration (exact: samples / duration)")
        self.rb_blind = QRadioButton("Estimate from the signal (blind: measured samples/symbol x a standard "
                                     "symbol rate; marked ASSUMED)")
        self.rb_unknown = QRadioButton("Unknown: keep every result in normalised units (cycles/sample, "
                                       "samples/symbol)")
        self.rb_duration.setChecked(True)
        self.mode_buttons = QButtonGroup(self)
        for i, rb in enumerate((self.rb_duration, self.rb_blind, self.rb_unknown)):
            self.mode_buttons.addButton(rb, i)
            mode_layout.addWidget(rb)
            rb.toggled.connect(self.on_mode_changed)

        # 3b. Blind estimate
        self.blind_group = QGroupBox("Blind estimate (cyclostationary symbol-rate line)")
        blind_layout = QVBoxLayout(self.blind_group)
        row = QHBoxLayout()
        self.estimate_btn = QPushButton("Estimate from signal")
        self.estimate_btn.clicked.connect(self.run_blind_estimate)
        row.addWidget(self.estimate_btn)
        row.addStretch()
        blind_layout.addLayout(row)
        self.blind_label = QLabel("Not estimated yet.")
        self.blind_label.setWordWrap(True)
        blind_layout.addWidget(self.blind_label)
        self.blind_combo = QComboBox()
        self.blind_combo.currentIndexChanged.connect(self.update_calculations)
        blind_layout.addWidget(QLabel("Consistent (sample rate, symbol rate) pairs - pick one:"))
        blind_layout.addWidget(self.blind_combo)
        self.blind_group.setEnabled(False)

        # 3c. Recording Duration Group
        duration_group = QGroupBox("Recording Duration (Time Period)")
        self.duration_group = duration_group
        duration_layout = QHBoxLayout(duration_group)

        self.duration_edit = QLineEdit("1.0")
        self.duration_edit.setPlaceholderText("Enter recording time...")
        self.duration_edit.textChanged.connect(self.update_calculations)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Seconds (s)", "Milliseconds (ms)", "Microseconds (µs)", "Nanoseconds (ns)"])
        self.unit_combo.currentIndexChanged.connect(self.update_calculations)

        duration_layout.addWidget(QLabel("Duration:"))
        duration_layout.addWidget(self.duration_edit)
        duration_layout.addWidget(self.unit_combo)

        # 4. Computed Sample Rate & Validation Status
        calc_group = QGroupBox("Sample Rate && Validation")
        calc_layout = QVBoxLayout(calc_group)

        self.calc_fs_label = QLabel("Computed sample rate: N/A")
        self.calc_fs_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2563EB;")

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #DC2626; font-weight: bold;")
        self.warning_label.setWordWrap(True)

        calc_layout.addWidget(self.calc_fs_label)
        calc_layout.addWidget(self.warning_label)

        # 5. Buttons
        btn_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Signal")
        self.analyze_btn.setObjectName("primary")
        self.analyze_btn.clicked.connect(self.on_accept)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addWidget(self.cancel_btn)

        main_layout.addWidget(file_group)
        main_layout.addWidget(detect_group)
        main_layout.addWidget(mode_group)
        main_layout.addWidget(duration_group)
        main_layout.addWidget(self.blind_group)
        main_layout.addWidget(calc_group)
        main_layout.addLayout(btn_layout)

    def update_detection_box_style(self):
        res = self.detection_result
        if res.is_low_confidence:
            bg_color = "#FFFBEB"
            text_color = "#B45309"
            status_text = f"⚠️ Low Confidence Detection: {self._detected_label()} ({res.confidence:.1f}% confidence)\nMethod: {res.method}. Please verify format."
        else:
            bg_color = "#F0FDF4"
            text_color = "#15803D"
            status_text = f"✓ Auto-detected Format: {self._detected_label()} ({res.confidence:.1f}% confidence)\nMethod: {res.method}"

        self.result_box.setText(status_text)
        self.result_box.setStyleSheet(
            f"background-color: {bg_color}; color: {text_color}; padding: 8px; border-radius: 4px; font-weight: bold;"
        )

    def _detected_label(self) -> str:
        res = self.detection_result
        key = normalize_datatype(res.format_name)[0]
        extra = []
        if FORMATS[key][1] > 2:
            extra.append(f"{res.endianness}-endian")
        if res.header_offset:
            extra.append(f"{res.header_offset}-byte header skipped")
        return DISPLAY_NAMES[key] + (f" [{', '.join(extra)}]" if extra else "")

    def on_format_changed(self):
        self.selected_format = self.format_combo.currentData()
        self.blind = None                              # the samples change with the format
        self.blind_combo.clear()
        self.blind_label.setText("Not estimated yet (sample format changed).")
        self.update_calculations()

    def on_mode_changed(self):
        self.mode = ("duration" if self.rb_duration.isChecked() else "blind" if self.rb_blind.isChecked()
                     else "unknown")
        self.duration_group.setEnabled(self.mode == "duration")
        self.blind_group.setEnabled(self.mode == "blind")
        self.update_calculations()

    def _read_samples(self):
        from ingestion.iq_formats import decode_iq_bytes
        from estimation.blind_sample_rate import MAX_SAMPLES
        bps = self.BYTES_PER_COMPLEX_SAMPLE.get(self.selected_format, 4)
        endian = self.detection_result.endianness if self.selected_format == self.detected_format else "little"
        with open(self.file_path, "rb") as fh:
            fh.seek(self.header_offset)
            raw = fh.read(MAX_SAMPLES * bps)
        return decode_iq_bytes(raw, self.selected_format, endian)

    def run_blind_estimate(self):
        from estimation.blind_sample_rate import estimate
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.blind = estimate(self._read_samples())
        except Exception as e:                            # the dialog must survive a failed estimate
            self.blind = None
            self.blind_label.setText(f"Estimation failed: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        b = self.blind
        self.blind_combo.clear()
        if b.sps is None:
            self.blind_label.setText("No symbol-rate line found: " + "; ".join(b.notes))
        else:
            self.blind_label.setText(
                f"Measured {b.sps:.4f} samples/symbol ({b.family}; {b.method}; confidence {b.confidence:.2f}).\n"
                "The samples fix only this ratio: every pair below explains the signal equally. Pairs whose "
                "sample rate is a standard SDR/audio rate are listed first.")
            for h in b.hypotheses:
                self.blind_combo.addItem(h.label(), h.sample_rate)
        self.update_calculations()

    def update_calculations(self):
        if self.mode == "unknown":
            self.computed_fs = None
            self.calc_fs_label.setText("Sample rate: UNKNOWN (results in normalised units)")
            self.warning_label.setText("")
            return
        if self.mode == "blind":
            fs = self.blind_combo.currentData() if self.blind_combo.count() else None
            self.computed_fs = float(fs) if fs else None
            self.calc_fs_label.setText(f"Assumed sample rate: {self.computed_fs:,.1f} Hz (blind hypothesis)"
                                       if self.computed_fs else "Assumed sample rate: press 'Estimate from signal'")
            self.warning_label.setText("ASSUMED value: the absolute sample rate cannot be measured from the "
                                       "samples; frequencies and rates in Hz / Bd scale with this choice."
                                       if self.computed_fs else "")
            return
        fmt = self.selected_format
        bytes_per_sample = self.BYTES_PER_COMPLEX_SAMPLE.get(fmt, 4)

        duration_str = self.duration_edit.text().strip()
        unit_str = self.unit_combo.currentText()
        unit_factor = self.UNIT_CONVERSION.get(unit_str, 1.0)

        warnings = []

        # Check divisibility
        data_bytes = max(self.file_size - self.header_offset, 0)
        if data_bytes % bytes_per_sample != 0:
            warnings.append(f"⚠️ Warning: Data size ({data_bytes:,} B) is not a multiple of {bytes_per_sample} bytes ({fmt.upper()}). Possible header or corruption.")

        try:
            val = float(duration_str) if duration_str else 0.0
            if val <= 0:
                self.calc_fs_label.setText("Computed sample rate: Please enter duration > 0")
                self.computed_fs = None
                return

            duration_sec = val * unit_factor
            total_complex = data_bytes // bytes_per_sample       # complex (I+Q) samples in the recording
            fs = total_complex / duration_sec
            self.computed_fs = fs

            # Format fs display
            if fs >= 1e9:
                fs_str = f"{fs / 1e9:.4f} GHz"
            elif fs >= 1e6:
                fs_str = f"{fs / 1e6:.4f} MHz"
            elif fs >= 1e3:
                fs_str = f"{fs / 1e3:.4f} kHz"
            else:
                fs_str = f"{fs:.2f} Hz"

            self.calc_fs_label.setText(f"Computed sample rate: {fs_str} ({fs:,.0f} Hz) (from user-provided duration)")

            # Range validation: 1 Hz to 100 GHz
            if fs < 1.0 or fs > 100.0e9:
                warnings.append("⚠️ Computed sample rate seems unrealistic (< 1 Hz or > 100 GHz). Please verify duration and format.")

        except ValueError:
            self.calc_fs_label.setText("Computed sample rate: Invalid duration input")
            self.computed_fs = None

        self.warning_label.setText("\n".join(warnings))

    def on_accept(self):
        if self.mode == "unknown":
            self.accept()
            return
        if self.mode == "blind" and not self.computed_fs:
            QMessageBox.warning(self, "No estimate", "Press 'Estimate from signal' and pick a sample rate, "
                                                     "or choose another sample-rate source.")
            return
        if self.computed_fs is None or self.computed_fs <= 0:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid positive duration to calculate the sample rate.")
            return

        if self.computed_fs < 1.0 or self.computed_fs > 100.0e9:
            reply = QMessageBox.question(
                self,
                "Unrealistic Sample Rate",
                f"The computed sample rate ({self.computed_fs:,.0f} Hz) is outside typical range [1 Hz, 100 GHz]. Proceed anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self.accept()

    def get_config(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            # canonical tag (cu8 / cs8 / cs16_le / cf32_be ...); the detected endianness is kept only
            # while the user leaves the detected type selected
            "dtype": datatype_tag(self.selected_format, self.detection_result.endianness
                                  if self.selected_format == self.detected_format else "little"),
            "byte_offset": self.header_offset,
            "sample_rate": self.computed_fs,
            "sample_rate_source": self.mode,               # duration | blind | unknown
            "blind_sps": self.blind.sps if (self.mode == "blind" and self.blind) else None,
            "blind_method": self.blind.method if (self.mode == "blind" and self.blind) else None,
            "duration_input": float(self.duration_edit.text() or 0),
            "duration_unit": self.unit_combo.currentText(),
            "detection_confidence": self.detection_result.confidence,
            "detection_method": self.detection_result.method
        }
