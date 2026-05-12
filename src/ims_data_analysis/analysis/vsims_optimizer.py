from __future__ import annotations

import numpy as np


def topt_ms(
    voltage_kv: float,
    pulse_width_ms: float,
    temperature_c: float,
    gate_multiplier: float,
    time_add_ms: float,
) -> float:
    temperature_k = float(temperature_c) + 273.15
    pulse_s = max(1e-9, float(pulse_width_ms) / 1000.0)
    voltage_v = max(1e-9, float(voltage_kv) * 1000.0 * max(1e-9, float(gate_multiplier)))
    time_add_s = float(time_add_ms) / 1000.0

    core = (pulse_s / (0.0395 * max(1e-9, temperature_k))) * np.sqrt((temperature_k * voltage_v) / 0.0395)
    topt_s = (273.15 / 760.0) * core + time_add_s
    return float(topt_s * 1000.0)


def extract_optimized_trace(
    heatmap: np.ndarray,
    x_time_ms: np.ndarray,
    y_voltage_kv: np.ndarray,
    pulse_width_ms: float,
    temperature_c: float,
    gate_multiplier: float,
    time_add_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if heatmap.size == 0 or x_time_ms.size == 0 or y_voltage_kv.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    topt_values = np.asarray(
        [
            topt_ms(v, pulse_width_ms, temperature_c, gate_multiplier, time_add_ms)
            for v in y_voltage_kv
        ],
        dtype=np.float64,
    )
    topt_values = np.clip(topt_values, float(np.min(x_time_ms)), float(np.max(x_time_ms)))

    trace = []
    max_time_ms = max(1e-9, float(np.max(x_time_ms)))
    for row_index, t_opt in enumerate(topt_values):
        row = np.asarray(heatmap[row_index, :], dtype=np.float64)
        point_count = int(row.shape[0])
        if point_count <= 0:
            trace.append(0.0)
            continue
        if point_count == 1:
            trace.append(float(row[0]))
            continue

        idx_float = (float(t_opt) / max_time_ms) * float(point_count - 1)
        idx_low = int(np.floor(idx_float))
        idx_high = int(np.ceil(idx_float))
        idx_low = int(np.clip(idx_low, 0, point_count - 1))
        idx_high = int(np.clip(idx_high, 0, point_count - 1))
        if idx_low == idx_high:
            trace.append(float(row[idx_low]))
            continue

        frac = float(idx_float - idx_low)
        trace.append(float((1.0 - frac) * row[idx_low] + frac * row[idx_high]))

    return y_voltage_kv.astype(np.float64), topt_values, np.asarray(trace, dtype=np.float64)
