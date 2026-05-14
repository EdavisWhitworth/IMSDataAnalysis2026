from __future__ import annotations

import numpy as np

from ims_data_analysis.models import LoadedExperiment, ModeView, OperationMode


def _time_axis_ms(point_count: int, length_ms: float) -> np.ndarray:
    return np.linspace(0.0, float(length_ms), int(point_count), endpoint=True)


def average_heatmap_rows(matrix: np.ndarray, row_a: int, row_b: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.asarray([], dtype=np.float64)

    row_min = int(np.clip(min(row_a, row_b), 0, values.shape[0] - 1))
    row_max = int(np.clip(max(row_a, row_b), 0, values.shape[0] - 1))
    if row_min == row_max:
        return values[row_min, :].copy()
    return np.mean(values[row_min : row_max + 1, :], axis=0)


def _ftims_arrival_time_axis_ms(experiment: LoadedExperiment, displayed_point_count: int) -> np.ndarray:
    cfg = experiment.config
    if displayed_point_count <= 0:
        return np.asarray([], dtype=np.float64)

    if not cfg.ftims_config.enable_fft:
        return _time_axis_ms(displayed_point_count, cfg.experiment_length_ms)

    start_frequency_hz = max(1e-9, float(cfg.ftims_config.start_frequency_hz))
    frequency_step_hz = float(cfg.ftims_config.frequency_step_hz)
    if frequency_step_hz <= 0.0:
        return _time_axis_ms(displayed_point_count, cfg.experiment_length_ms)

    dwell_seconds = max(0.0, float(cfg.experiment_length_ms) / 1000.0)
    dwell_seconds += max(1, int(cfg.averages_per_iteration)) / start_frequency_hz
    if dwell_seconds <= 0.0:
        return _time_axis_ms(displayed_point_count, cfg.experiment_length_ms)

    fft_input_point_count = displayed_point_count * 2
    fft_bins_hz = np.fft.fftfreq(fft_input_point_count, dwell_seconds)[:displayed_point_count]
    sweep_rate_hz_per_s = frequency_step_hz / dwell_seconds
    return (fft_bins_hz / sweep_rate_hz_per_s) * 1000.0


def build_mode_view(experiment: LoadedExperiment, mode_override: OperationMode | None = None) -> ModeView:
    cfg = experiment.config
    mode = cfg.operation_mode if mode_override is None else mode_override
    matrix = np.asarray(experiment.matrix, dtype=np.float64)
    rows, cols = matrix.shape if matrix.ndim == 2 else (0, int(cfg.data_points))
    x_axis = _time_axis_ms(cols, cfg.experiment_length_ms)

    if mode == OperationMode.FTIMS:
        x_axis = _ftims_arrival_time_axis_ms(experiment, cols)
        freq = np.asarray(cfg.ftims_config.frequency_steps(), dtype=np.float64)
        y_axis = freq[:rows] if freq.size >= rows else np.arange(rows, dtype=np.float64)
        return ModeView(
            mode=mode,
            x_axis=x_axis,
            y_axis=y_axis,
            heatmap=matrix,
            x_label="Arrival Time (ms)",
            y_label="Stepped Frequency (Hz)",
        )

    if mode == OperationMode.STEPPED_VSIMS:
        voltages = np.asarray(cfg.vsims_config.voltage_steps_kv(), dtype=np.float64)
        if voltages.size == 0:
            y_axis = np.arange(rows, dtype=np.float64)
            mapped_voltages = None
        elif rows <= voltages.size:
            y_axis = voltages[:rows]
            mapped_voltages = y_axis
        else:
            y_axis = np.asarray([voltages[i % voltages.size] for i in range(rows)], dtype=np.float64)
            mapped_voltages = y_axis

        return ModeView(
            mode=mode,
            x_axis=x_axis,
            y_axis=y_axis,
            heatmap=matrix,
            x_label="Drift Time (ms)",
            y_label="Stepped Voltage (kV)",
            voltage_axis_kv=mapped_voltages,
        )

    if mode == OperationMode.SWEPT_VSIMS:
        y_axis = np.arange(1, rows + 1, dtype=np.float64)
        return ModeView(
            mode=mode,
            x_axis=x_axis,
            y_axis=y_axis,
            heatmap=matrix,
            x_label="Drift Time (ms)",
            y_label="Iteration",
        )

    if mode == OperationMode.SWEPT_FTIMS:
        y_axis = np.arange(1, rows + 1, dtype=np.float64)
        return ModeView(
            mode=mode,
            x_axis=x_axis,
            y_axis=y_axis,
            heatmap=matrix,
            x_label="Mobility / Time (ms)",
            y_label="Iteration",
        )

    y_axis = np.arange(1, rows + 1, dtype=np.float64)
    return ModeView(
        mode=OperationMode.DTIMS,
        x_axis=x_axis,
        y_axis=y_axis,
        heatmap=matrix,
        x_label="Drift Time (ms)",
        y_label="Iteration",
    )


def build_heatmap_display(view: ModeView) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    matrix = np.asarray(view.heatmap, dtype=np.float64)
    if matrix.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), matrix, view.x_label, view.y_label

    if view.mode in {OperationMode.DTIMS, OperationMode.FTIMS}:
        x_axis = np.arange(1, matrix.shape[0] + 1, dtype=np.float64)
        y_axis = np.asarray(view.x_axis, dtype=np.float64)
        return x_axis, y_axis, matrix.T, "Iteration", "Drift Time (ms)"

    if view.mode == OperationMode.STEPPED_VSIMS and view.voltage_axis_kv is not None:
        x_axis = np.asarray(view.voltage_axis_kv, dtype=np.float64)
        y_axis = np.asarray(view.x_axis, dtype=np.float64)
        return x_axis, y_axis, matrix.T, "Stepped Voltage (kV)", "Drift Time (ms)"

    return np.asarray(view.x_axis, dtype=np.float64), np.asarray(view.y_axis, dtype=np.float64), matrix, view.x_label, view.y_label
