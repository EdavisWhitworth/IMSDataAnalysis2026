from __future__ import annotations

import numpy as np

from ims_data_analysis.analysis.vsims_optimizer import extract_optimized_trace, topt_ms


def test_topt_ms_increases_with_voltage() -> None:
    low = topt_ms(voltage_kv=4.0, pulse_width_ms=1.0, temperature_c=25.0, gate_multiplier=1.0, time_add_ms=0.0)
    high = topt_ms(voltage_kv=8.0, pulse_width_ms=1.0, temperature_c=25.0, gate_multiplier=1.0, time_add_ms=0.0)
    assert high > low


def test_extract_optimized_trace_shape_matches_voltage_axis() -> None:
    x_time = np.linspace(0.0, 50.0, 500)
    y_voltage = np.linspace(4.0, 8.0, 25)
    heatmap = np.asarray([np.sin(x_time / 8.0 + v) + 5.0 for v in y_voltage], dtype=np.float64)

    voltages, topt_values, trace = extract_optimized_trace(
        heatmap=heatmap,
        x_time_ms=x_time,
        y_voltage_kv=y_voltage,
        pulse_width_ms=1.0,
        temperature_c=25.0,
        gate_multiplier=1.0,
        time_add_ms=0.0,
    )

    assert voltages.shape == y_voltage.shape
    assert topt_values.shape == y_voltage.shape
    assert trace.shape == y_voltage.shape
