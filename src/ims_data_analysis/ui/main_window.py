from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import SVGExporter
from PyQt5 import QtCore
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QFontComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QTabWidget,
)

from ims_data_analysis.analysis.metrics import compute_ko, detect_all_peaks, detect_nearest_peak
from ims_data_analysis.analysis.mode_transform import build_mode_view
from ims_data_analysis.analysis.vsims_optimizer import extract_optimized_trace
from ims_data_analysis.io.h5_loader import H5LoadError, load_h5_experiment
from ims_data_analysis.io.user_settings import SpectrumStyleSettings, UserSettings, load_user_settings, save_user_settings
from ims_data_analysis.models import LoadedExperiment, ModeView, OperationMode


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IMS Data Analysis 2026")
        self.resize(1550, 980)

        self.spectrum_curve_color = "#000000"
        self.baseline_color = "#4cacf7"
        self.peak_color = "#e67700"
        self.cursor_color = "#ff6b6b"
        self.spectrum_background_color = "#ffffff"
        self.spectrum_text_color = "#000000"
        self.spectrum_font_family = "Arial"
        self.spectrum_font_size = 11
        self.spectrum_title_text = "Selected Spectrum"
        self.spectrum_show_title = True

        self.loaded: LoadedExperiment | None = None
        self.mode_view: ModeView | None = None
        self.current_row: int = 0
        self.cursor_x: float = 0.0
        self.optimized_cursor_x: float = 0.0
        self._active_spectrum_target = "selected"
        self.user_settings: UserSettings = load_user_settings()
        self._selected_style = self.user_settings.selected_spectrum_style
        self._optimized_style = self.user_settings.optimized_spectrum_style
        if self.user_settings.vs_step_table_target in {"selected", "optimized"}:
            self._active_spectrum_target = self.user_settings.vs_step_table_target

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left_control_panel = QWidget()
        control_layout = QVBoxLayout(left_control_panel)

        file_group = QGroupBox("Data")
        file_form = QFormLayout(file_group)
        self.load_btn = QPushButton("Load H5 File")
        self.load_btn.clicked.connect(self._on_load_h5)
        self.mode_value = QLabel("-")
        self.created_value = QLabel("-")
        self.metadata_voltage_value = QLabel("-")
        file_form.addRow(self.load_btn)
        file_form.addRow("Mode:", self.mode_value)
        file_form.addRow("Created:", self.created_value)
        file_form.addRow("Metadata Voltage (kV):", self.metadata_voltage_value)
        control_layout.addWidget(file_group)

        param_group = QGroupBox("Analysis Parameters")
        param_form = QFormLayout(param_group)

        self.pressure_spin = self._spin(300.0, 2000.0, 760.0, 1.0)
        self.temperature_spin = self._spin(-20.0, 200.0, 25.0, 0.1)
        self.length_spin = self._spin(0.1, 200.0, 10.0, 0.1)
        self.gate_mult_spin = self._spin(0.001, 100.0, 1.0, 0.01)
        self.time_add_spin = self._spin(-200.0, 200.0, 0.0, 0.1)

        self.noise_start_spin = self._spin(0.0, 100000.0, 0.0, 0.1)
        self.noise_end_spin = self._spin(0.0, 100000.0, 5.0, 0.1)
        self.min_prom_spin = self._spin(0.0, 1e9, 10.0, 1.0)
        self.min_snr_spin = self._spin(0.0, 1e6, 3.0, 0.1)

        self.peak_mode_combo = QComboBox()
        self.peak_mode_combo.addItems(["All peaks", "Nearest to cursor"])
        self.subtract_baseline_checkbox = QCheckBox("Subtract baseline")

        param_form.addRow("Pressure P (Torr):", self.pressure_spin)
        param_form.addRow("Temperature T (C):", self.temperature_spin)
        param_form.addRow("Drift Length L (cm):", self.length_spin)
        param_form.addRow("Gate Multiplier:", self.gate_mult_spin)
        param_form.addRow("Stepped VSIMS Time Add (ms):", self.time_add_spin)
        param_form.addRow("Noise Start (ms):", self.noise_start_spin)
        param_form.addRow("Noise End (ms):", self.noise_end_spin)
        param_form.addRow("Min Prominence:", self.min_prom_spin)
        param_form.addRow("Min SNR:", self.min_snr_spin)
        param_form.addRow("Peak Detection:", self.peak_mode_combo)
        param_form.addRow(self.subtract_baseline_checkbox)
        control_layout.addWidget(param_group)

        metadata_group = QGroupBox("Metadata Overrides")
        metadata_form = QFormLayout(metadata_group)
        self.mode_override_checkbox = QCheckBox("Override mode from metadata")
        self.mode_override_combo = QComboBox()
        self.mode_override_combo.addItems(
            [
                OperationMode.DTIMS.value,
                OperationMode.FTIMS.value,
                OperationMode.SWEPT_FTIMS.value,
                OperationMode.STEPPED_VSIMS.value,
                OperationMode.SWEPT_VSIMS.value,
            ]
        )
        self.voltage_override_checkbox = QCheckBox("Override voltage metadata")
        self.voltage_override_spin = self._spin(0.0, 100.0, 0.0, 0.01)
        self.voltage_override_missing_only_checkbox = QCheckBox("Apply voltage override only when metadata missing")
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self._save_settings)

        metadata_form.addRow(self.mode_override_checkbox)
        metadata_form.addRow("Forced mode:", self.mode_override_combo)
        metadata_form.addRow(self.voltage_override_checkbox)
        metadata_form.addRow("Override voltage (kV):", self.voltage_override_spin)
        metadata_form.addRow(self.voltage_override_missing_only_checkbox)
        metadata_form.addRow(self.save_settings_btn)
        control_layout.addWidget(metadata_group)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        self.export_spectrum_btn = QPushButton("Export Current Spectrum SVG")
        self.export_spectrum_csv_btn = QPushButton("Export Current Spectrum CSV")
        self.export_table_csv_btn = QPushButton("Export Peak Table CSV")
        self.export_heatmap_btn = QPushButton("Export Current Heatmap SVG")
        for _btn in (
            self.export_spectrum_btn,
            self.export_spectrum_csv_btn,
            self.export_table_csv_btn,
            self.export_heatmap_btn,
        ):
            _btn.setMinimumHeight(56)
        self.export_spectrum_btn.clicked.connect(self._export_spectrum_svg)
        self.export_spectrum_csv_btn.clicked.connect(self._export_spectrum_csv)
        self.export_table_csv_btn.clicked.connect(self._export_table_csv)
        self.export_heatmap_btn.clicked.connect(self._export_heatmap_svg)
        export_layout.addWidget(self.export_spectrum_btn)
        export_layout.addWidget(self.export_spectrum_csv_btn)
        export_layout.addWidget(self.export_table_csv_btn)
        export_layout.addWidget(self.export_heatmap_btn)

        style_group = QGroupBox("Spectrum Style")
        style_form = QFormLayout(style_group)
        self.curve_color_btn = QPushButton()
        self.curve_color_btn.clicked.connect(lambda: self._pick_color("spectrum_curve_color", self.curve_color_btn))
        self.baseline_color_btn = QPushButton()
        self.baseline_color_btn.clicked.connect(lambda: self._pick_color("baseline_color", self.baseline_color_btn))
        self.peak_color_btn = QPushButton()
        self.peak_color_btn.clicked.connect(lambda: self._pick_color("peak_color", self.peak_color_btn))
        self.cursor_color_btn = QPushButton()
        self.cursor_color_btn.clicked.connect(lambda: self._pick_color("cursor_color", self.cursor_color_btn))
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.clicked.connect(
            lambda: self._pick_color("spectrum_background_color", self.bg_color_btn)
        )
        self.text_color_btn = QPushButton()
        self.text_color_btn.clicked.connect(lambda: self._pick_color("spectrum_text_color", self.text_color_btn))
        self.font_family_combo = QFontComboBox()
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(7, 30)
        self.font_size_spin.setValue(self.spectrum_font_size)
        self.title_edit = QLineEdit(self.spectrum_title_text)
        self.show_title_checkbox = QCheckBox("Display spectrum title")
        self.show_title_checkbox.setChecked(self.spectrum_show_title)
        self.export_peak_highlights_checkbox = QCheckBox("Include detected peak highlights in spectrum SVG")
        self.export_peak_highlights_checkbox.setChecked(True)
        self.font_family_combo.currentFontChanged.connect(self._on_spectrum_font_changed)
        self.font_size_spin.valueChanged.connect(self._on_spectrum_font_changed)
        self.title_edit.textChanged.connect(self._on_spectrum_title_changed)
        self.show_title_checkbox.toggled.connect(self._on_spectrum_title_changed)

        style_form.addRow("Trace Color:", self.curve_color_btn)
        style_form.addRow("Baseline Color:", self.baseline_color_btn)
        style_form.addRow("Peak Marker Color:", self.peak_color_btn)
        style_form.addRow("Cursor Color:", self.cursor_color_btn)
        style_form.addRow("Background Color:", self.bg_color_btn)
        style_form.addRow("Text/Axis Color:", self.text_color_btn)
        style_form.addRow("Font Family:", self.font_family_combo)
        style_form.addRow("Font Size:", self.font_size_spin)
        style_form.addRow("Title Text:", self.title_edit)
        style_form.addRow(self.show_title_checkbox)
        style_form.addRow(self.export_peak_highlights_checkbox)

        control_layout.addStretch(1)
        left_control_panel.setMinimumWidth(340)

        self.heatmap_panel = QWidget()
        heatmap_layout = QHBoxLayout(self.heatmap_panel)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_layout.setSpacing(6)

        self.heat_lut = pg.HistogramLUTWidget(orientation="vertical", gradientPosition="right")
        self.heat_lut.setMinimumWidth(90)
        self.heat_lut.item.gradient.loadPreset("spectrum")

        self.heat_plot = pg.PlotWidget(title="Navigable Heatmap")
        self.heat_plot.setLabel("bottom", "X")
        self.heat_plot.setLabel("left", "Y")
        self.heat_image = pg.ImageItem()
        self.heat_plot.addItem(self.heat_image)
        self.heat_lut.item.setImageItem(self.heat_image)

        heatmap_layout.addWidget(self.heat_plot, stretch=1)
        heatmap_layout.addWidget(self.heat_lut)

        self.heat_overlay_curve = pg.PlotDataItem(pen=pg.mkPen(color="#ffd43b", width=2, style=QtCore.Qt.DashLine))
        self.heat_plot.addItem(self.heat_overlay_curve)
        self.heat_cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffffff", width=3))
        self.heat_cursor_line.setZValue(10)
        self.heat_plot.addItem(self.heat_cursor_line)
        self.heat_plot.scene().sigMouseClicked.connect(self._on_heat_click)

        self.spectrum_plot = pg.PlotWidget(title="Selected Spectrum")
        self.spectrum_plot.setMouseEnabled(x=False, y=False)
        self.spectrum_plot.setMenuEnabled(False)
        self.spectrum_plot.setLabel("bottom", "X")
        self.spectrum_plot.setLabel("left", "Signal")
        self.spectrum_curve = self.spectrum_plot.plot([], [], pen=pg.mkPen(color=self._selected_style.curve_color, width=2))
        self.noise_markers = pg.ScatterPlotItem(
            size=4,
            symbol="o",
            pen=pg.mkPen(color=self._selected_style.baseline_color, width=1),
            brush=pg.mkBrush(self._color_with_alpha(self._selected_style.baseline_color, 170)),
        )
        self.spectrum_plot.addItem(self.noise_markers)
        self.peak_markers = pg.ScatterPlotItem(
            size=11,
            symbol="o",
            pen=pg.mkPen(color=self._selected_style.peak_color, width=2),
            brush=pg.mkBrush(255, 236, 153, 200),
        )
        self.spectrum_plot.addItem(self.peak_markers)
        self.cursor_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(self._selected_style.cursor_color, width=1.5))
        self.spectrum_plot.addItem(self.cursor_line)
        self.spectrum_plot.scene().sigMouseClicked.connect(self._on_spectrum_click)

        self.optimized_plot = pg.PlotWidget(title="Stepped VSIMS Optimized Spectrum")
        self.optimized_plot.setMouseEnabled(x=False, y=False)
        self.optimized_plot.setMenuEnabled(False)
        self.optimized_plot.setLabel("bottom", "Optimized Drift Time (ms)")
        self.optimized_plot.setLabel("left", "Signal")
        self.optimized_curve = self.optimized_plot.plot([], [], pen=pg.mkPen(color=self._optimized_style.curve_color, width=2))
        self.optimized_peak_markers = pg.ScatterPlotItem(
            size=10,
            symbol="o",
            pen=pg.mkPen(color=self._optimized_style.peak_color, width=2),
            brush=pg.mkBrush(255, 236, 153, 200),
        )
        self.optimized_plot.addItem(self.optimized_peak_markers)
        self.optimized_cursor_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(self._optimized_style.cursor_color, width=1.5),
        )
        self.optimized_plot.addItem(self.optimized_cursor_line)
        self.optimized_plot.scene().sigMouseClicked.connect(self._on_optimized_spectrum_click)

        self.peak_table = QTableWidget(0, 7)
        self.peak_table.setHorizontalHeaderLabels(
            ["Peak", "Position", "Intensity", "SNR", "Ko", "FWHM", "Resolving Power"]
        )
        self.peak_table.horizontalHeader().setStretchLastSection(True)

        self.selected_axis_controls = self._build_axis_controls("Spectrum Axis Controls", "selected")

        self.selected_plot_page = QWidget()
        selected_layout = QVBoxLayout(self.selected_plot_page)
        selected_layout.setContentsMargins(0, 0, 0, 0)
        selected_layout.addWidget(self.spectrum_plot)
        selected_layout.addWidget(self.selected_axis_controls)
        selected_layout.addWidget(self.peak_table)

        self.optimized_peak_table = QTableWidget(0, 7)
        self.optimized_peak_table.setHorizontalHeaderLabels(
            ["Optimized t (ms)", "Voltage (kV)", "Intensity", "SNR", "Ko", "FWHM", "Resolving Power"]
        )
        self.optimized_peak_table.horizontalHeader().setStretchLastSection(True)

        self.optimized_axis_controls = self._build_axis_controls("Optimized Axis Controls", "optimized")

        self.optimized_plot_page = QWidget()
        optimized_layout = QVBoxLayout(self.optimized_plot_page)
        optimized_layout.setContentsMargins(0, 0, 0, 0)
        optimized_layout.addWidget(self.optimized_plot)
        optimized_layout.addWidget(self.optimized_axis_controls)
        optimized_layout.addWidget(self.optimized_peak_table)

        self.plot_tabs = QTabWidget()
        self.plot_tabs.addTab(self.heatmap_panel, "Heatmap")
        self.plot_tabs.addTab(self.selected_plot_page, "Spectrum")
        self.plot_tabs.addTab(self.optimized_plot_page, "Optimized Spectrum")
        self.plot_tabs.currentChanged.connect(self._on_plot_tab_changed)

        right_control_panel = QWidget()
        right_control_layout = QVBoxLayout(right_control_panel)
        right_control_layout.setContentsMargins(0, 0, 0, 0)
        right_control_layout.addWidget(export_group)
        right_control_layout.addWidget(style_group)
        right_control_layout.addStretch(1)
        right_control_panel.setMinimumWidth(360)

        layout.addWidget(left_control_panel)
        layout.addWidget(self.plot_tabs, stretch=1)
        layout.addWidget(right_control_panel)

        for widget in [
            self.pressure_spin,
            self.temperature_spin,
            self.length_spin,
            self.gate_mult_spin,
            self.time_add_spin,
            self.noise_start_spin,
            self.noise_end_spin,
            self.min_prom_spin,
            self.min_snr_spin,
        ]:
            widget.valueChanged.connect(self._refresh_analysis)
        self.peak_mode_combo.currentIndexChanged.connect(self._refresh_analysis)
        self.subtract_baseline_checkbox.toggled.connect(self._on_baseline_subtraction_toggled)
        self.mode_override_checkbox.toggled.connect(self._on_override_controls_changed)
        self.mode_override_combo.currentIndexChanged.connect(self._on_override_controls_changed)
        self.voltage_override_checkbox.toggled.connect(self._on_override_controls_changed)
        self.voltage_override_spin.valueChanged.connect(self._on_override_controls_changed)
        self.voltage_override_missing_only_checkbox.toggled.connect(self._on_override_controls_changed)

        self._apply_loaded_settings_to_controls()
        self._load_style_controls_from_target()
        self._apply_all_spectrum_styles()
        self._update_plot_tab_availability()
        self._activate_preferred_plot_tab()

    def _spin(self, minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(5)
        return spin

    def _color_with_alpha(self, color_hex: str, alpha: int) -> QColor:
        color = QColor(color_hex)
        color.setAlpha(int(np.clip(alpha, 0, 255)))
        return color

    def _build_axis_controls(self, title: str, target: str) -> QWidget:
        group = QGroupBox(title)
        form = QFormLayout(group)

        x_min_spin = self._spin(-1e12, 1e12, 0.0, 0.1)
        x_max_spin = self._spin(-1e12, 1e12, 1.0, 0.1)
        y_min_spin = self._spin(-1e12, 1e12, 0.0, 0.1)
        y_max_spin = self._spin(-1e12, 1e12, 1.0, 0.1)
        reset_x_btn = QPushButton("Reset X to Data")
        reset_y_btn = QPushButton("Reset Y to Data")

        x_row = QWidget()
        x_layout = QHBoxLayout(x_row)
        x_layout.setContentsMargins(0, 0, 0, 0)
        x_layout.addWidget(QLabel("Min"))
        x_layout.addWidget(x_min_spin)
        x_layout.addWidget(QLabel("Max"))
        x_layout.addWidget(x_max_spin)
        x_layout.addWidget(reset_x_btn)

        y_row = QWidget()
        y_layout = QHBoxLayout(y_row)
        y_layout.setContentsMargins(0, 0, 0, 0)
        y_layout.addWidget(QLabel("Min"))
        y_layout.addWidget(y_min_spin)
        y_layout.addWidget(QLabel("Max"))
        y_layout.addWidget(y_max_spin)
        y_layout.addWidget(reset_y_btn)

        form.addRow("X Axis:", x_row)
        form.addRow("Y Axis:", y_row)

        controls = {
            "widget": group,
            "x_min": x_min_spin,
            "x_max": x_max_spin,
            "y_min": y_min_spin,
            "y_max": y_max_spin,
        }
        setattr(self, f"_{target}_axis_controls", controls)

        x_min_spin.valueChanged.connect(lambda *_: self._apply_axis_ranges(target))
        x_max_spin.valueChanged.connect(lambda *_: self._apply_axis_ranges(target))
        y_min_spin.valueChanged.connect(lambda *_: self._apply_axis_ranges(target))
        y_max_spin.valueChanged.connect(lambda *_: self._apply_axis_ranges(target))
        reset_x_btn.clicked.connect(lambda: self._reset_axis_to_data(target, "x"))
        reset_y_btn.clicked.connect(lambda: self._reset_axis_to_data(target, "y"))
        return group

    def _pick_color(self, attr_name: str, button: QPushButton) -> None:
        if self._active_style_target() is None:
            return
        current = QColor(getattr(self, attr_name))
        chosen = QColorDialog.getColor(current, self, "Select Color")
        if not chosen.isValid():
            return
        setattr(self, attr_name, chosen.name())
        button.setText(chosen.name())
        self._store_active_style_from_controls()
        self._apply_active_spectrum_style()

    def _on_spectrum_font_changed(self, *_args) -> None:
        self.spectrum_font_family = self.font_family_combo.currentFont().family()
        self.spectrum_font_size = int(self.font_size_spin.value())
        self._store_active_style_from_controls()
        self._apply_active_spectrum_style()

    def _on_spectrum_title_changed(self, *_args) -> None:
        self.spectrum_title_text = self.title_edit.text().strip()
        self.spectrum_show_title = self.show_title_checkbox.isChecked()
        self._store_active_style_from_controls()
        self._apply_active_spectrum_style()

    def _sync_color_button_labels(self) -> None:
        self.curve_color_btn.setText(self.spectrum_curve_color)
        self.baseline_color_btn.setText(self.baseline_color)
        self.peak_color_btn.setText(self.peak_color)
        self.cursor_color_btn.setText(self.cursor_color)
        self.bg_color_btn.setText(self.spectrum_background_color)
        self.text_color_btn.setText(self.spectrum_text_color)

    def _current_plot_tab_kind(self) -> str | None:
        if not hasattr(self, "plot_tabs"):
            return None
        index = self.plot_tabs.currentIndex()
        if index == 0:
            return "heatmap"
        if index == 1:
            return "selected"
        if index == 2:
            return "optimized"
        return None

    def _active_style_target(self) -> str | None:
        kind = self._current_plot_tab_kind()
        if kind in {"selected", "optimized"}:
            return kind
        return None

    def _active_peak_table(self) -> QTableWidget:
        return self.optimized_peak_table if self._active_style_target() == "optimized" else self.peak_table

    def _active_spectrum_plot(self) -> pg.PlotWidget:
        return self.optimized_plot if self._active_style_target() == "optimized" else self.spectrum_plot

    def _axis_controls_for_target(self, target: str) -> dict[str, object]:
        return getattr(self, f"_{target}_axis_controls")

    def _update_axis_controls_from_data(self, target: str, x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0 or y.size == 0:
            return
        controls = self._axis_controls_for_target(target)
        for key, values in (("x_min", x), ("x_max", x), ("y_min", y), ("y_max", y)):
            spin = controls[key]
            spin.blockSignals(True)
            try:
                spin.setValue(float(np.min(values)) if key.endswith("min") else float(np.max(values)))
            finally:
                spin.blockSignals(False)
        self._apply_axis_ranges(target)

    def _sync_axis_controls_to_current_data(self, target: str) -> None:
        if self.mode_view is None:
            return
        if target == "optimized":
            x_data, y_data = self.optimized_curve.getData()
        else:
            x_data = self.mode_view.x_axis
            y_data = self._display_row(self.current_row)
        x_arr = np.asarray([] if x_data is None else x_data, dtype=np.float64)
        y_arr = np.asarray([] if y_data is None else y_data, dtype=np.float64)
        self._update_axis_controls_from_data(target, x_arr, y_arr)

    def _apply_axis_ranges(self, target: str) -> None:
        controls = self._axis_controls_for_target(target)
        plot_widget = self.spectrum_plot if target == "selected" else self.optimized_plot
        x_min = min(float(controls["x_min"].value()), float(controls["x_max"].value()))
        x_max = max(float(controls["x_min"].value()), float(controls["x_max"].value()))
        y_min = min(float(controls["y_min"].value()), float(controls["y_max"].value()))
        y_max = max(float(controls["y_min"].value()), float(controls["y_max"].value()))
        plot_widget.setXRange(x_min, x_max, padding=0)
        plot_widget.setYRange(y_min, y_max, padding=0)

    def _reset_axis_to_data(self, target: str, axis: str) -> None:
        x_data, y_data = self._current_curve_data(target)
        if x_data.size == 0 or y_data.size == 0:
            return
        controls = self._axis_controls_for_target(target)
        if axis == "x":
            controls["x_min"].blockSignals(True)
            controls["x_max"].blockSignals(True)
            try:
                controls["x_min"].setValue(float(np.min(x_data)))
                controls["x_max"].setValue(float(np.max(x_data)))
            finally:
                controls["x_min"].blockSignals(False)
                controls["x_max"].blockSignals(False)
        else:
            controls["y_min"].blockSignals(True)
            controls["y_max"].blockSignals(True)
            try:
                controls["y_min"].setValue(float(np.min(y_data)))
                controls["y_max"].setValue(float(np.max(y_data)))
            finally:
                controls["y_min"].blockSignals(False)
                controls["y_max"].blockSignals(False)
        self._apply_axis_ranges(target)

    def _current_curve_data(self, target: str) -> tuple[np.ndarray, np.ndarray]:
        if target == "optimized":
            x_data, y_data = self.optimized_curve.getData()
        else:
            x_data, y_data = self.spectrum_curve.getData()
        x_arr = np.asarray([] if x_data is None else x_data, dtype=np.float64)
        y_arr = np.asarray([] if y_data is None else y_data, dtype=np.float64)
        return x_arr, y_arr

    def _style_for_target(self, target: str) -> SpectrumStyleSettings:
        return self._optimized_style if target == "optimized" else self._selected_style

    def _store_active_style_from_controls(self) -> None:
        target = self._active_style_target()
        if target is None:
            return
        style = self._style_for_target(target)
        style.curve_color = self.spectrum_curve_color
        style.baseline_color = self.baseline_color
        style.peak_color = self.peak_color
        style.cursor_color = self.cursor_color
        style.background_color = self.spectrum_background_color
        style.text_color = self.spectrum_text_color
        style.font_family = self.spectrum_font_family
        style.font_size = self.spectrum_font_size
        style.title_text = self.spectrum_title_text
        style.show_title = self.spectrum_show_title

    def _load_style_controls_from_target(self) -> None:
        target = self._active_style_target() or self._active_spectrum_target
        style = self._style_for_target(target)
        self.spectrum_curve_color = style.curve_color
        self.baseline_color = style.baseline_color
        self.peak_color = style.peak_color
        self.cursor_color = style.cursor_color
        self.spectrum_background_color = style.background_color
        self.spectrum_text_color = style.text_color
        self.spectrum_font_family = style.font_family
        self.spectrum_font_size = int(style.font_size)
        self.spectrum_title_text = style.title_text
        self.spectrum_show_title = bool(style.show_title)

        self.curve_color_btn.setText(self.spectrum_curve_color)
        self.baseline_color_btn.setText(self.baseline_color)
        self.peak_color_btn.setText(self.peak_color)
        self.cursor_color_btn.setText(self.cursor_color)
        self.bg_color_btn.setText(self.spectrum_background_color)
        self.text_color_btn.setText(self.spectrum_text_color)

        self.font_family_combo.blockSignals(True)
        self.font_size_spin.blockSignals(True)
        self.title_edit.blockSignals(True)
        self.show_title_checkbox.blockSignals(True)
        try:
            self.font_family_combo.setCurrentFont(QFont(self.spectrum_font_family))
            self.font_size_spin.setValue(self.spectrum_font_size)
            self.title_edit.setText(self.spectrum_title_text)
            self.show_title_checkbox.setChecked(self.spectrum_show_title)
        finally:
            self.font_family_combo.blockSignals(False)
            self.font_size_spin.blockSignals(False)
            self.title_edit.blockSignals(False)
            self.show_title_checkbox.blockSignals(False)

    def _update_spectrum_controls_enabled(self) -> None:
        active_target = self._active_style_target()
        enabled = active_target is not None
        for widget in [
            self.curve_color_btn,
            self.baseline_color_btn,
            self.peak_color_btn,
            self.cursor_color_btn,
            self.bg_color_btn,
            self.text_color_btn,
            self.font_family_combo,
            self.font_size_spin,
            self.title_edit,
            self.show_title_checkbox,
            self.export_spectrum_btn,
            self.export_spectrum_csv_btn,
            self.export_table_csv_btn,
            self.selected_axis_controls,
            self.optimized_axis_controls,
        ]:
            widget.setEnabled(enabled)

    def _update_plot_tab_availability(self) -> None:
        is_step_vsims = self.mode_view is not None and self.mode_view.mode == OperationMode.STEPPED_VSIMS
        self.plot_tabs.setTabEnabled(2, is_step_vsims)
        if not is_step_vsims and self.plot_tabs.currentIndex() == 2:
            self.plot_tabs.setCurrentIndex(1)

    def _on_plot_tab_changed(self, index: int) -> None:
        if index == 0:
            self._active_spectrum_target = self._active_spectrum_target if self._active_spectrum_target in {"selected", "optimized"} else "selected"
            self._update_spectrum_controls_enabled()
            return

        self._active_spectrum_target = "optimized" if index == 2 else "selected"
        self._load_style_controls_from_target()
        self._update_spectrum_controls_enabled()
        self._apply_active_spectrum_style()

    def _apply_style_to_plot(self, plot_widget: pg.PlotWidget, curve_item: pg.PlotDataItem, peak_item: pg.ScatterPlotItem, cursor_item: pg.InfiniteLine | None, style: SpectrumStyleSettings, x_label: str, selected: bool) -> None:
        curve_item.setPen(pg.mkPen(color=style.curve_color, width=2))
        peak_item.setPen(pg.mkPen(color=style.peak_color, width=2))
        if cursor_item is not None:
            cursor_item.setPen(pg.mkPen(style.cursor_color, width=1.5))
        plot_widget.setBackground(style.background_color)

        font = QFont(style.font_family, style.font_size)
        css_size = f"{style.font_size}pt"
        text_style = {"color": style.text_color, "font-size": css_size}
        plot_widget.setLabel("bottom", x_label, **text_style)
        plot_widget.setLabel("left", "Signal", **text_style)
        if style.show_title:
            title_text = style.title_text or ("Selected Spectrum" if selected else "Optimized Spectrum")
            plot_widget.setTitle(
                (
                    f"<span style='color:{style.text_color};"
                    f" font-family:{style.font_family}; font-size:{css_size};'>"
                    f"{title_text}"
                    "</span>"
                )
            )
        else:
            plot_widget.setTitle("")
        plot_widget.getAxis("bottom").setPen(pg.mkPen(style.text_color))
        plot_widget.getAxis("left").setPen(pg.mkPen(style.text_color))
        plot_widget.getAxis("bottom").setTextPen(pg.mkPen(style.text_color))
        plot_widget.getAxis("left").setTextPen(pg.mkPen(style.text_color))
        plot_widget.getAxis("bottom").setStyle(tickFont=font)
        plot_widget.getAxis("left").setStyle(tickFont=font)

    def _apply_all_spectrum_styles(self) -> None:
        self._apply_style_to_plot(
            self.spectrum_plot,
            self.spectrum_curve,
            self.peak_markers,
            self.cursor_line,
            self._selected_style,
            self.mode_view.x_label if self.mode_view else "X",
            True,
        )
        self.noise_markers.setPen(pg.mkPen(color=self._selected_style.baseline_color, width=1))
        self.noise_markers.setBrush(pg.mkBrush(self._color_with_alpha(self._selected_style.baseline_color, 170)))
        self._apply_style_to_plot(
            self.optimized_plot,
            self.optimized_curve,
            self.optimized_peak_markers,
            self.optimized_cursor_line,
            self._optimized_style,
            "Optimized Drift Time (ms)",
            False,
        )

    def _apply_active_spectrum_style(self) -> None:
        target = self._active_style_target()
        if target is None:
            return
        style = self._style_for_target(target)
        self._apply_style_to_plot(
            self.spectrum_plot if target == "selected" else self.optimized_plot,
            self.spectrum_curve if target == "selected" else self.optimized_curve,
            self.peak_markers if target == "selected" else self.optimized_peak_markers,
            self.cursor_line if target == "selected" else self.optimized_cursor_line,
            style,
            self.mode_view.x_label if (target == "selected" and self.mode_view is not None) else "Optimized Drift Time (ms)",
            target == "selected",
        )
        if target == "selected":
            self.noise_markers.setPen(pg.mkPen(color=style.baseline_color, width=1))
            self.noise_markers.setBrush(pg.mkBrush(self._color_with_alpha(style.baseline_color, 170)))

    def _update_heatmap_cursor(self) -> None:
        if self.mode_view is None or self.mode_view.mode != OperationMode.STEPPED_VSIMS or self.mode_view.voltage_axis_kv is None:
            self.heat_cursor_line.setVisible(False)
            return

        self.heat_cursor_line.setVisible(True)
        row = int(np.clip(self.current_row, 0, self.mode_view.voltage_axis_kv.size - 1))
        self.heat_cursor_line.setPos(float(self.mode_view.voltage_axis_kv[row]))

    def _activate_preferred_plot_tab(self) -> None:
        if self.mode_view is not None and self.mode_view.mode == OperationMode.STEPPED_VSIMS and self._active_spectrum_target == "optimized":
            self.plot_tabs.setCurrentIndex(2)
        else:
            self.plot_tabs.setCurrentIndex(1)

    def _apply_loaded_settings_to_controls(self) -> None:
        self.mode_override_checkbox.setChecked(self.user_settings.mode_override_enabled)
        index = self.mode_override_combo.findText(self.user_settings.mode_override_value)
        if index >= 0:
            self.mode_override_combo.setCurrentIndex(index)
        self.voltage_override_checkbox.setChecked(self.user_settings.voltage_override_enabled)
        self.voltage_override_spin.setValue(self.user_settings.voltage_override_kv)
        self.voltage_override_missing_only_checkbox.setChecked(
            self.user_settings.voltage_override_only_when_missing
        )

    def _apply_loaded_experiment_parameters(self) -> None:
        if self.loaded is None:
            return

        cfg = self.loaded.config
        updates = [
            (self.pressure_spin, cfg.pressure_torr),
            (self.temperature_spin, cfg.temperature_c),
            (self.length_spin, cfg.length_cm),
            (self.gate_mult_spin, cfg.gate_multiplier),
        ]
        for widget, value in updates:
            if value is None:
                continue
            widget.blockSignals(True)
            try:
                widget.setValue(float(value))
            finally:
                widget.blockSignals(False)

        self._refresh_analysis()

    def _capture_settings_from_controls(self) -> UserSettings:
        self._store_active_style_from_controls()
        return UserSettings(
            mode_override_enabled=self.mode_override_checkbox.isChecked(),
            mode_override_value=self.mode_override_combo.currentText(),
            voltage_override_enabled=self.voltage_override_checkbox.isChecked(),
            voltage_override_kv=float(self.voltage_override_spin.value()),
            voltage_override_only_when_missing=self.voltage_override_missing_only_checkbox.isChecked(),
            vs_step_table_target=self._active_spectrum_target,
            selected_spectrum_style=self._selected_style,
            optimized_spectrum_style=self._optimized_style,
        )

    def _effective_mode(self) -> OperationMode | None:
        if not self.user_settings.mode_override_enabled:
            return None
        try:
            return OperationMode(self.user_settings.mode_override_value)
        except ValueError:
            return None

    def _effective_voltage_kv(self, metadata_voltage_kv: float | None) -> float | None:
        if not self.user_settings.voltage_override_enabled:
            return metadata_voltage_kv
        if self.user_settings.voltage_override_only_when_missing and metadata_voltage_kv is not None:
            return metadata_voltage_kv
        return float(self.user_settings.voltage_override_kv)

    def _save_settings(self) -> None:
        self.user_settings = self._capture_settings_from_controls()
        save_user_settings(self.user_settings)
        if self.loaded is not None:
            self._rebuild_mode_view_from_settings()
        self._apply_all_spectrum_styles()
        self._refresh_analysis()

    def _on_override_controls_changed(self, _value=None) -> None:
        self.user_settings = self._capture_settings_from_controls()
        if self.loaded is not None:
            self._rebuild_mode_view_from_settings()
        self._refresh_analysis()

    def _mode_label(self, mode: OperationMode) -> str:
        labels = {
            OperationMode.DTIMS: "DTIMS",
            OperationMode.FTIMS: "Stepped FTIMS",
            OperationMode.SWEPT_FTIMS: "Sweep FTIMS",
            OperationMode.STEPPED_VSIMS: "Stepped VSIMS",
            OperationMode.SWEPT_VSIMS: "Sweep VSIMS",
        }
        return labels.get(mode, mode.value)

    def _rebuild_mode_view_from_settings(self) -> None:
        if self.loaded is None:
            return
        override_mode = self._effective_mode()
        self.mode_view = build_mode_view(self.loaded, mode_override=override_mode)
        if self.mode_view is None:
            return
        mode_text = self._mode_label(self.mode_view.mode)
        if override_mode is not None:
            mode_text = f"{mode_text} (override)"
        self.mode_value.setText(mode_text)
        self._load_style_controls_from_target()
        self._apply_loaded_experiment_parameters()
        self._set_default_noise_window_from_mode_view()
        self._apply_all_spectrum_styles()
        self._update_plot_tab_availability()
        self._render_heatmap()
        self._sync_axis_controls_to_current_data("optimized")
        self._select_row(self.current_row)
        self._sync_axis_controls_to_current_data("selected")
        self._activate_preferred_plot_tab()

    def _on_load_h5(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open IMS H5", str(Path.home()), "H5 Files (*.h5 *.hdf5)")
        if not file_path:
            return

        try:
            self.loaded = load_h5_experiment(file_path)
        except H5LoadError as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Unexpected error: {exc}")
            return

        self.mode_view = build_mode_view(self.loaded)
        self.mode_value.setText(self.loaded.mode_label)
        self.created_value.setText(self.loaded.created_at or "-")
        self.current_row = 0
        self._load_style_controls_from_target()
        self._apply_loaded_experiment_parameters()
        self._set_default_noise_window_from_mode_view()
        self._apply_all_spectrum_styles()
        self._update_plot_tab_availability()
        self._render_heatmap()
        self._sync_axis_controls_to_current_data("optimized")
        self._select_row(0)
        self._sync_axis_controls_to_current_data("selected")
        self._activate_preferred_plot_tab()

    def _set_default_noise_window_from_mode_view(self) -> None:
        if self.mode_view is None or self.mode_view.x_axis.size == 0:
            return
        x_max = float(np.max(self.mode_view.x_axis))
        x_min = float(np.min(self.mode_view.x_axis))
        start = max(x_min, x_max - 10.0)
        self.noise_start_spin.blockSignals(True)
        self.noise_end_spin.blockSignals(True)
        try:
            self.noise_start_spin.setValue(start)
            self.noise_end_spin.setValue(x_max)
        finally:
            self.noise_start_spin.blockSignals(False)
            self.noise_end_spin.blockSignals(False)

    def _render_heatmap(self) -> None:
        if self.mode_view is None:
            return

        x = self.mode_view.x_axis
        y = self.mode_view.y_axis
        z = self._display_heatmap_matrix()
        if z.size == 0:
            self.heat_image.setImage(np.empty((0, 0)))
            return

        if self.mode_view.mode == OperationMode.STEPPED_VSIMS and self.mode_view.voltage_axis_kv is not None:
            x = self.mode_view.voltage_axis_kv
            y = self.mode_view.x_axis
            z = z.T
            self.heat_plot.setLabel("bottom", "Stepped Voltage (kV)")
            self.heat_plot.setLabel("left", "Drift Time (ms)")
            self.heat_image.setImage(z, autoLevels=True, axisOrder="row-major")
        else:
            self.heat_plot.setLabel("bottom", self.mode_view.x_label)
            self.heat_plot.setLabel("left", self.mode_view.y_label)
            self.heat_image.setImage(z, autoLevels=True)

        x_min = float(np.min(x))
        x_max = float(np.max(x))
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        rect = QtCore.QRectF(x_min, y_min, max(1e-9, x_max - x_min), max(1e-9, y_max - y_min))
        self.heat_image.setRect(rect)
        self._update_heatmap_cursor()
        self._update_vsims_overlay_and_trace()

    def _nearest_axis_index(self, axis: np.ndarray, value: float) -> int:
        if axis.size == 0:
            return 0
        return int(np.argmin(np.abs(axis - value)))

    def _on_heat_click(self, event) -> None:
        if self.mode_view is None or self.mode_view.heatmap.size == 0:
            return
        pos = self.heat_plot.getViewBox().mapSceneToView(event.scenePos())
        if self.mode_view.mode == OperationMode.STEPPED_VSIMS and self.mode_view.voltage_axis_kv is not None:
            row = self._nearest_axis_index(self.mode_view.voltage_axis_kv, float(pos.x()))
        else:
            row = self._nearest_axis_index(self.mode_view.y_axis, float(pos.y()))
        self._select_row(row)

    def _on_spectrum_click(self, event) -> None:
        if self.mode_view is None or self.mode_view.heatmap.size == 0:
            return
        pos = self.spectrum_plot.getViewBox().mapSceneToView(event.scenePos())
        self.cursor_x = float(pos.x())
        self.cursor_line.setPos(self.cursor_x)
        self._refresh_analysis()

    def _on_optimized_spectrum_click(self, event) -> None:
        if self.mode_view is None or self.mode_view.heatmap.size == 0:
            return
        pos = self.optimized_plot.getViewBox().mapSceneToView(event.scenePos())
        self.optimized_cursor_x = float(pos.x())
        self.optimized_cursor_line.setPos(self.optimized_cursor_x)
        self._refresh_analysis()

    def _on_baseline_subtraction_toggled(self, _checked: bool) -> None:
        if self.mode_view is None:
            return
        self._render_heatmap()
        self._select_row(self.current_row)

    def _is_fft_mode(self) -> bool:
        if self.mode_view is None:
            return False
        return self.mode_view.mode in {OperationMode.FTIMS, OperationMode.SWEPT_FTIMS}

    def _baseline_bounds(self) -> tuple[float, float]:
        low = float(self.noise_start_spin.value())
        high = float(self.noise_end_spin.value())
        return (low, high) if low <= high else (high, low)

    def _subtract_baseline_row(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        low, high = self._baseline_bounds()
        if x.size == 0 or y.size == 0:
            return np.asarray(y, dtype=np.float64).copy()
        baseline_mask = (x >= low) & (x <= high)
        y_arr = np.asarray(y, dtype=np.float64)
        if not np.any(baseline_mask):
            return y_arr.copy()
        baseline_mean = float(np.mean(y_arr[baseline_mask]))
        return y_arr - baseline_mean

    def _display_heatmap_matrix(self) -> np.ndarray:
        if self.mode_view is None:
            return np.empty((0, 0), dtype=np.float64)

        matrix = np.asarray(self.mode_view.heatmap, dtype=np.float64)
        if matrix.size == 0:
            return matrix

        # Remove FFT-only polarity inversion effects for stepped/swept FTIMS.
        if self._is_fft_mode():
            matrix = np.abs(matrix)

        if not self.subtract_baseline_checkbox.isChecked():
            return matrix

        x = np.asarray(self.mode_view.x_axis, dtype=np.float64)
        if matrix.ndim != 2:
            return matrix

        rows = [self._subtract_baseline_row(x, matrix[row_idx, :]) for row_idx in range(matrix.shape[0])]
        return np.asarray(rows, dtype=np.float64)

    def _display_row(self, row: int) -> np.ndarray:
        matrix = self._display_heatmap_matrix()
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return np.asarray([], dtype=np.float64)
        row_idx = int(np.clip(row, 0, matrix.shape[0] - 1))
        return matrix[row_idx, :]

    def _display_row_count(self) -> int:
        if self.mode_view is None:
            return 0
        if self.mode_view.heatmap.ndim == 2:
            return int(self.mode_view.heatmap.shape[0])
        return 0

    def _select_row(self, row: int) -> None:
        if self.mode_view is None or self.mode_view.heatmap.size == 0:
            return

        display_rows = self._display_row_count()
        if display_rows <= 0:
            return
        row = int(np.clip(row, 0, display_rows - 1))
        self.current_row = row
        x = self.mode_view.x_axis
        y = self._display_row(row)

        self.spectrum_curve.setData(x, y)
        self.spectrum_plot.setLabel("bottom", self.mode_view.x_label)
        self._apply_all_spectrum_styles()

        if x.size:
            if self.cursor_x < float(np.min(x)) or self.cursor_x > float(np.max(x)):
                self.cursor_x = float(x[len(x) // 2])
            self.cursor_line.setPos(self.cursor_x)
        self._update_heatmap_cursor()

        voltage_kv = self._metadata_voltage_kv_for_row(row)
        self.metadata_voltage_value.setText("-" if voltage_kv is None else f"{voltage_kv:.4f}")
        self._refresh_analysis()

    def _metadata_voltage_kv_for_row(self, row: int) -> float | None:
        if self.loaded is None or self.mode_view is None:
            return None

        cfg = self.loaded.config
        mode = cfg.operation_mode

        if mode == OperationMode.STEPPED_VSIMS:
            if self.mode_view.voltage_axis_kv is not None and row < self.mode_view.voltage_axis_kv.size:
                return float(self.mode_view.voltage_axis_kv[row])

        if mode == OperationMode.SWEPT_VSIMS:
            return float(cfg.swept_vsims_config.v_add_kv)

        if mode == OperationMode.STEPPED_VSIMS:
            return float(cfg.vsims_config.initial_voltage_kv)

        return None

    def _refresh_analysis(self) -> None:
        if self.mode_view is None or self.mode_view.heatmap.size == 0:
            self.noise_markers.setData([], [])
            self.peak_markers.setData([], [])
            self.optimized_peak_markers.setData([], [])
            self.peak_table.setRowCount(0)
            self.optimized_peak_table.setRowCount(0)
            return

        x = self.mode_view.x_axis
        y = self._display_row(self.current_row)
        metadata_voltage_kv = self._metadata_voltage_kv_for_row(self.current_row)
        analysis_voltage_kv = self._effective_voltage_kv(metadata_voltage_kv)
        self.metadata_voltage_value.setText("-" if analysis_voltage_kv is None else f"{analysis_voltage_kv:.4f}")

        pressure = float(self.pressure_spin.value())
        temperature = float(self.temperature_spin.value())
        length = float(self.length_spin.value())
        gate_mult = float(self.gate_mult_spin.value())
        noise_start = float(self.noise_start_spin.value())
        noise_end = float(self.noise_end_spin.value())

        self._update_noise_markers(x, y, noise_start, noise_end)

        if self.peak_mode_combo.currentText() == "All peaks":
            peaks = detect_all_peaks(
                x=x,
                y=y,
                noise_start=noise_start,
                noise_end=noise_end,
                min_prominence=float(self.min_prom_spin.value()),
                min_snr=float(self.min_snr_spin.value()),
                pressure_torr=pressure,
                temperature_c=temperature,
                length_cm=length,
                voltage_kv=0.0 if analysis_voltage_kv is None else analysis_voltage_kv,
                gate_multiplier=gate_mult,
            )
        else:
            peaks = detect_nearest_peak(
                x=x,
                y=y,
                cursor_x=self.cursor_x,
                noise_start=noise_start,
                noise_end=noise_end,
                pressure_torr=pressure,
                temperature_c=temperature,
                length_cm=length,
                voltage_kv=0.0 if analysis_voltage_kv is None else analysis_voltage_kv,
                gate_multiplier=gate_mult,
            )

        self._update_peak_markers(peaks)
        self._populate_peak_table(peaks)
        self._update_vsims_overlay_and_trace()

    def _update_peak_markers(self, peaks) -> None:
        if not peaks:
            self.peak_markers.setData([], [])
            return
        x = [peak.x_position for peak in peaks]
        y = [peak.intensity for peak in peaks]
        self.peak_markers.setData(x, y)

    def _update_noise_markers(self, x: np.ndarray, y: np.ndarray, noise_start: float, noise_end: float) -> None:
        low = min(noise_start, noise_end)
        high = max(noise_start, noise_end)
        mask = (x >= low) & (x <= high)
        if not np.any(mask):
            self.noise_markers.setData([], [])
            return
        self.noise_markers.setData(x[mask], y[mask])

    def _populate_peak_table(self, peaks) -> None:
        self.peak_table.setHorizontalHeaderLabels(
            ["Peak", "Position", "Intensity", "SNR", "Ko", "FWHM", "Resolving Power"]
        )
        self.peak_table.setRowCount(len(peaks))
        for row, peak in enumerate(peaks):
            cells = [
                str(row + 1),
                f"{peak.x_position:.5f}",
                f"{peak.intensity:.5f}",
                f"{peak.snr_linear:.2f} ({peak.snr_db:.2f} dB)",
                "--" if peak.ko is None else f"{peak.ko:.5f}",
                f"{peak.fwhm:.5f}",
                f"{peak.resolving_power:.5f}",
            ]
            for col, value in enumerate(cells):
                self.peak_table.setItem(row, col, QTableWidgetItem(value))

    def _update_vsims_overlay_and_trace(self) -> None:
        if self.mode_view is None or self.loaded is None:
            return

        if self.loaded.config.operation_mode != OperationMode.STEPPED_VSIMS:
            self.heat_overlay_curve.setData([], [])
            self.optimized_curve.setData([], [])
            self.optimized_peak_markers.setData([], [])
            self.optimized_peak_table.setRowCount(0)
            return

        if self.mode_view.voltage_axis_kv is None or self.mode_view.voltage_axis_kv.size == 0:
            self.heat_overlay_curve.setData([], [])
            self.optimized_curve.setData([], [])
            self.optimized_peak_markers.setData([], [])
            self.optimized_peak_table.setRowCount(0)
            return

        voltage_axis, topt_values, optimized_trace = extract_optimized_trace(
            heatmap=self._display_heatmap_matrix(),
            x_time_ms=self.mode_view.x_axis,
            y_voltage_kv=self.mode_view.voltage_axis_kv,
            pulse_width_ms=self.loaded.config.pulse_width_ms,
            temperature_c=float(self.temperature_spin.value()),
            gate_multiplier=float(self.gate_mult_spin.value()),
            time_add_ms=float(self.time_add_spin.value()),
        )
        self.heat_overlay_curve.setData(voltage_axis, topt_values)
        self.optimized_curve.setData(topt_values, optimized_trace)
        if topt_values.size:
            if self.optimized_cursor_x < float(np.min(topt_values)) or self.optimized_cursor_x > float(np.max(topt_values)):
                self.optimized_cursor_x = float(topt_values[len(topt_values) // 2])
            self.optimized_cursor_line.setPos(self.optimized_cursor_x)
        self._update_vsims_optimized_peaks(topt_values, optimized_trace, voltage_axis)
        self._update_heatmap_cursor()
        self.optimized_plot.show()

    def _update_vsims_optimized_peaks(self, optimized_x: np.ndarray, optimized_y: np.ndarray, voltage_axis: np.ndarray) -> None:
        if optimized_x.size == 0 or optimized_y.size == 0 or voltage_axis.size == 0:
            self.optimized_peak_markers.setData([], [])
            self.peak_table.setRowCount(0)
            return

        if self.peak_mode_combo.currentText() == "All peaks":
            peaks = detect_all_peaks(
                x=optimized_x,
                y=optimized_y,
                noise_start=float(self.noise_start_spin.value()),
                noise_end=float(self.noise_end_spin.value()),
                min_prominence=float(self.min_prom_spin.value()),
                min_snr=float(self.min_snr_spin.value()),
                pressure_torr=float(self.pressure_spin.value()),
                temperature_c=float(self.temperature_spin.value()),
                length_cm=float(self.length_spin.value()),
                voltage_kv=0.0,
                gate_multiplier=float(self.gate_mult_spin.value()),
            )
        else:
            peaks = detect_nearest_peak(
                x=optimized_x,
                y=optimized_y,
                cursor_x=self.optimized_cursor_x,
                noise_start=float(self.noise_start_spin.value()),
                noise_end=float(self.noise_end_spin.value()),
                pressure_torr=float(self.pressure_spin.value()),
                temperature_c=float(self.temperature_spin.value()),
                length_cm=float(self.length_spin.value()),
                voltage_kv=0.0,
                gate_multiplier=float(self.gate_mult_spin.value()),
            )

        if not peaks:
            self.optimized_peak_markers.setData([], [])
            self.optimized_peak_table.setRowCount(0)
            return

        marker_x = [peak.x_position for peak in peaks]
        marker_y = [peak.intensity for peak in peaks]
        self.optimized_peak_markers.setData(marker_x, marker_y)

        rows = []
        for peak in peaks:
            nearest_idx = int(np.argmin(np.abs(optimized_x - peak.x_position)))
            voltage_kv = float(voltage_axis[nearest_idx])
            ko = compute_ko(
                drift_time_ms=float(peak.x_position),
                pressure_torr=float(self.pressure_spin.value()),
                temperature_c=float(self.temperature_spin.value()),
                length_cm=float(self.length_spin.value()),
                voltage_kv=voltage_kv,
                gate_multiplier=float(self.gate_mult_spin.value()),
            )
            rows.append(
                [
                    f"{peak.x_position:.5f}",
                    f"{voltage_kv:.4f}",
                    f"{peak.intensity:.5f}",
                    f"{peak.snr_linear:.2f} ({peak.snr_db:.2f} dB)",
                    "--" if ko is None else f"{ko:.5f}",
                    f"{peak.fwhm:.5f}",
                    f"{peak.resolving_power:.5f}",
                ]
            )

        self.optimized_peak_table.setHorizontalHeaderLabels(
            ["Optimized t (ms)", "Voltage (kV)", "Intensity", "SNR", "Ko", "FWHM", "Resolving Power"]
        )
        self.optimized_peak_table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for col_index, value in enumerate(row_values):
                self.optimized_peak_table.setItem(row_index, col_index, QTableWidgetItem(value))

    def _export_plot_svg(self, plot_widget: pg.PlotWidget, caption: str, hide_cursor: bool = False) -> None:
        out_path, _ = QFileDialog.getSaveFileName(self, caption, str(Path.home() / "figure.svg"), "SVG Files (*.svg)")
        if not out_path:
            return

        path = Path(out_path)
        if path.suffix.lower() != ".svg":
            path = path.with_suffix(".svg")

        cursor_item = self.cursor_line if plot_widget is self.spectrum_plot else self.optimized_cursor_line
        cursor_visible = cursor_item.isVisible()
        if hide_cursor:
            cursor_item.setVisible(False)
        try:
            exporter = SVGExporter(plot_widget.plotItem)
            exporter.export(str(path))
        finally:
            if hide_cursor:
                cursor_item.setVisible(cursor_visible)

    def _export_spectrum_svg(self) -> None:
        target = self._active_style_target()
        if target is None:
            QMessageBox.warning(self, "Export Error", "Open a spectrum tab before exporting SVG.")
            return
        baseline_visible = self.noise_markers.isVisible()
        peaks_visible = self.peak_markers.isVisible()
        self.noise_markers.setVisible(False)
        self.peak_markers.setVisible(peaks_visible and self.export_peak_highlights_checkbox.isChecked())
        try:
            plot_widget = self.spectrum_plot if target == "selected" else self.optimized_plot
            self._export_plot_svg(plot_widget, "Export Spectrum SVG", hide_cursor=True)
        finally:
            self.noise_markers.setVisible(baseline_visible)
            self.peak_markers.setVisible(peaks_visible)

    def _export_spectrum_csv(self) -> None:
        if self.mode_view is None or self.mode_view.heatmap.size == 0:
            QMessageBox.warning(self, "Export Error", "No spectrum is currently available for CSV export.")
            return

        target = self._active_style_target()
        if target is None:
            QMessageBox.warning(self, "Export Error", "Open a spectrum tab before exporting CSV.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Spectrum CSV",
            str(Path.home() / "spectrum.csv"),
            "CSV Files (*.csv)",
        )
        if not out_path:
            return

        path = Path(out_path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        if target == "optimized" and self.mode_view.mode == OperationMode.STEPPED_VSIMS and self.mode_view.voltage_axis_kv is not None:
            voltage_axis, topt_values, optimized_trace = extract_optimized_trace(
                heatmap=self._display_heatmap_matrix(),
                x_time_ms=self.mode_view.x_axis,
                y_voltage_kv=self.mode_view.voltage_axis_kv,
                pulse_width_ms=self.loaded.config.pulse_width_ms,
                temperature_c=float(self.temperature_spin.value()),
                gate_multiplier=float(self.gate_mult_spin.value()),
                time_add_ms=float(self.time_add_spin.value()),
            )
            x = np.asarray(topt_values, dtype=np.float64)
            y = np.asarray(optimized_trace, dtype=np.float64)
        else:
            x = np.asarray(self.mode_view.x_axis, dtype=np.float64)
            y = np.asarray(self._display_row(self.current_row), dtype=np.float64)
        data = np.column_stack((x, y))
        x_label = ("Optimized Drift Time (ms)" if target == "optimized" else self.mode_view.x_label).replace(",", " ")
        np.savetxt(path, data, delimiter=",", header=f"{x_label},Signal", comments="")

    def _export_table_csv(self) -> None:
        table = self._active_peak_table()
        if table.rowCount() == 0:
            QMessageBox.warning(self, "Export Error", "Peak table is empty. Run peak detection first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Peak Table CSV",
            str(Path.home() / "peak_table.csv"),
            "CSV Files (*.csv)",
        )
        if not out_path:
            return

        path = Path(out_path)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        headers: list[str] = []
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item is not None else f"Column {col + 1}")

        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            for row in range(table.rowCount()):
                row_values: list[str] = []
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    row_values.append(item.text() if item is not None else "")
                writer.writerow(row_values)

    def _export_heatmap_svg(self) -> None:
        self._export_plot_svg(self.heat_plot, "Export Heatmap SVG")


def run_app() -> None:
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
