from __future__ import annotations

import numpy as np

from ims_data_analysis.analysis.mode_transform import build_heatmap_display
from ims_data_analysis.models import ModeView, OperationMode


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