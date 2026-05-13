from __future__ import annotations

import numpy as np

from ims_data_analysis.models import LoadedExperiment, ModeView, OperationMode


def _time_axis_ms(point_count: int, length_ms: float) -> np.ndarray:
    return np.linspace(0.0, float(length_ms), int(point_count), endpoint=True)


def build_mode_view(experiment: LoadedExperiment, mode_override: OperationMode | None = None) -> ModeView:
    cfg = experiment.config
    mode = cfg.operation_mode if mode_override is None else mode_override
    matrix = np.asarray(experiment.matrix, dtype=np.float64)
    rows, cols = matrix.shape if matrix.ndim == 2 else (0, int(cfg.data_points))
    x_axis = _time_axis_ms(cols, cfg.experiment_length_ms)

    if mode == OperationMode.FTIMS:
        freq = np.asarray(cfg.ftims_config.frequency_steps(), dtype=np.float64)
        y_axis = freq[:rows] if freq.size >= rows else np.arange(rows, dtype=np.float64)
        return ModeView(
            mode=mode,
            x_axis=x_axis,
            y_axis=y_axis,
            heatmap=matrix,
            x_label="Mobility / Time (ms)",
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

    if view.mode == OperationMode.DTIMS:
        x_axis = np.arange(1, matrix.shape[0] + 1, dtype=np.float64)
        y_axis = np.asarray(view.x_axis, dtype=np.float64)
        return x_axis, y_axis, matrix.T, "Iteration", "Drift Time (ms)"

    if view.mode == OperationMode.STEPPED_VSIMS and view.voltage_axis_kv is not None:
        x_axis = np.asarray(view.voltage_axis_kv, dtype=np.float64)
        y_axis = np.asarray(view.x_axis, dtype=np.float64)
        return x_axis, y_axis, matrix.T, "Stepped Voltage (kV)", "Drift Time (ms)"

    return np.asarray(view.x_axis, dtype=np.float64), np.asarray(view.y_axis, dtype=np.float64), matrix, view.x_label, view.y_label
