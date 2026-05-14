from __future__ import annotations

import numpy as np

from ims_data_analysis.analysis.mode_transform import average_heatmap_rows, build_heatmap_display, build_mode_view
from ims_data_analysis.models import ExperimentConfig, FTIMSConfig, LoadedExperiment, ModeView, OperationMode


def test_build_heatmap_display_transposes_dtims() -> None:
    view = ModeView(
        mode=OperationMode.DTIMS,
        x_axis=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        y_axis=np.array([1.0, 2.0], dtype=np.float64),
        heatmap=np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float64),
        x_label="Drift Time (ms)",
        y_label="Iteration",
    )

    x_axis, y_axis, heatmap, x_label, y_label = build_heatmap_display(view)

    assert x_label == "Iteration"
    assert y_label == "Drift Time (ms)"
    assert np.array_equal(x_axis, np.array([1.0, 2.0], dtype=np.float64))
    assert np.array_equal(y_axis, np.array([0.0, 1.0, 2.0], dtype=np.float64))
    assert np.array_equal(heatmap, np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], dtype=np.float64))


def test_build_mode_view_uses_fft_arrival_time_axis_for_stepped_ftims() -> None:
    experiment = LoadedExperiment(
        config=ExperimentConfig(
            operation_mode=OperationMode.FTIMS,
            experiment_length_ms=50.0,
            averages_per_iteration=1,
            ftims_config=FTIMSConfig(
                start_frequency_hz=10.0,
                frequency_step_hz=5.0,
                end_frequency_hz=20.0,
                enable_fft=True,
            ),
        ),
        matrix=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float64,
        ),
        created_at="",
        iteration_timestamps=[],
    )

    view = build_mode_view(experiment)

    assert view.x_label == "Arrival Time (ms)"
    assert np.allclose(view.x_axis, np.array([0.0, 25.0, 50.0, 75.0], dtype=np.float64))
    assert np.array_equal(view.y_axis, np.array([10.0, 15.0], dtype=np.float64))


def test_build_heatmap_display_transposes_stepped_ftims_to_iterations_vs_drift_time() -> None:
    view = ModeView(
        mode=OperationMode.FTIMS,
        x_axis=np.array([0.0, 25.0, 50.0], dtype=np.float64),
        y_axis=np.array([10.0, 15.0], dtype=np.float64),
        heatmap=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        x_label="Arrival Time (ms)",
        y_label="Stepped Frequency (Hz)",
    )

    x_axis, y_axis, heatmap, x_label, y_label = build_heatmap_display(view)

    assert x_label == "Iteration"
    assert y_label == "Drift Time (ms)"
    assert np.array_equal(x_axis, np.array([1.0, 2.0], dtype=np.float64))
    assert np.array_equal(y_axis, np.array([0.0, 25.0, 50.0], dtype=np.float64))
    assert np.array_equal(heatmap, np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float64))


def test_average_heatmap_rows_returns_mean_across_selected_range() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    averaged = average_heatmap_rows(matrix, 2, 0)

    assert np.allclose(averaged, np.array([4.0, 5.0, 6.0], dtype=np.float64))