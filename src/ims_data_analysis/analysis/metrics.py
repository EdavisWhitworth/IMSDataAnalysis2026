from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PeakMetrics:
    index: int
    x_position: float
    intensity: float
    fwhm: float
    resolving_power: float
    snr_linear: float
    snr_db: float
    ko: float | None


def _safe_bounds(low: float, high: float) -> tuple[float, float]:
    return (float(low), float(high)) if low <= high else (float(high), float(low))


def _interpolate_x(x1: float, y1: float, x2: float, y2: float, target_y: float) -> float:
    if y2 == y1:
        return x1
    return x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)


def _noise_stats(x: np.ndarray, y: np.ndarray, noise_start: float, noise_end: float) -> tuple[float, float]:
    low, high = _safe_bounds(noise_start, noise_end)
    mask = (x >= low) & (x <= high)
    noise_values = y[mask]
    if noise_values.size < 2:
        return 0.0, 0.0
    noise_mean = float(np.mean(noise_values))
    noise_rms = float(np.sqrt(np.mean((noise_values - noise_mean) ** 2)))
    return noise_mean, noise_rms


def _fwhm_from_peak(x: np.ndarray, y: np.ndarray, peak_idx: int) -> tuple[float, float, float] | None:
    peak_signal = float(y[peak_idx])
    if peak_signal <= 0:
        return None

    half_max = peak_signal * 0.5
    left_idx = int(peak_idx)
    right_idx = int(peak_idx)

    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1
    while right_idx < y.size - 1 and y[right_idx] > half_max:
        right_idx += 1

    if left_idx == peak_idx or right_idx == peak_idx:
        return None

    left_x = _interpolate_x(
        float(x[left_idx]),
        float(y[left_idx]),
        float(x[left_idx + 1]),
        float(y[left_idx + 1]),
        half_max,
    )
    right_x = _interpolate_x(
        float(x[right_idx - 1]),
        float(y[right_idx - 1]),
        float(x[right_idx]),
        float(y[right_idx]),
        half_max,
    )
    fwhm = right_x - left_x
    if fwhm <= 0:
        return None
    return left_x, right_x, fwhm


def compute_ko(
    drift_time_ms: float,
    pressure_torr: float,
    temperature_c: float,
    length_cm: float,
    voltage_kv: float,
    gate_multiplier: float,
) -> float | None:
    try:
        drift_time_sec = float(drift_time_ms) / 1000.0
        voltage_v = float(voltage_kv) * 1000.0
        numerator = float(length_cm) * float(length_cm)
        denominator = voltage_v * float(gate_multiplier) * drift_time_sec
        if denominator <= 0:
            return None
        pressure_factor = float(pressure_torr) / 760.0
        temperature_factor = 273.15 / (float(temperature_c) + 273.15)
        return (numerator / denominator) * pressure_factor * temperature_factor
    except (ValueError, ZeroDivisionError):
        return None


def _candidate_peaks(y: np.ndarray) -> list[int]:
    if y.size < 3:
        return []
    candidates = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    return [int(i) for i in candidates]


def detect_all_peaks(
    x: np.ndarray,
    y: np.ndarray,
    noise_start: float,
    noise_end: float,
    min_prominence: float,
    min_snr: float,
    pressure_torr: float,
    temperature_c: float,
    length_cm: float,
    voltage_kv: float,
    gate_multiplier: float,
) -> list[PeakMetrics]:
    if x.size == 0 or y.size == 0:
        return []

    noise_mean, noise_rms = _noise_stats(x, y, noise_start, noise_end)
    peaks: list[PeakMetrics] = []

    for idx in _candidate_peaks(y):
        peak_intensity = float(y[idx])
        if peak_intensity <= 0:
            continue

        window = max(10, int(0.02 * y.size))
        local_left = max(0, idx - window)
        local_right = min(y.size - 1, idx + window)
        local_floor = float(np.min(y[local_left : local_right + 1]))
        prominence = peak_intensity - local_floor
        if prominence < float(min_prominence):
            continue

        fwhm_result = _fwhm_from_peak(x, y, idx)
        if fwhm_result is None:
            continue
        _, _, fwhm = fwhm_result

        signal_height = max(peak_intensity - noise_mean, 0.0)
        if noise_rms > 0 and signal_height > 0:
            snr_linear = signal_height / noise_rms
            snr_db = 20.0 * np.log10(snr_linear)
        elif noise_rms == 0 and signal_height > 0:
            snr_linear = float("inf")
            snr_db = float("inf")
        else:
            snr_linear = 0.0
            snr_db = 0.0

        if snr_linear < float(min_snr):
            continue

        peak_x = float(x[idx])
        resolving_power = peak_x / fwhm
        ko = compute_ko(
            drift_time_ms=peak_x,
            pressure_torr=pressure_torr,
            temperature_c=temperature_c,
            length_cm=length_cm,
            voltage_kv=voltage_kv,
            gate_multiplier=gate_multiplier,
        )

        peaks.append(
            PeakMetrics(
                index=idx,
                x_position=peak_x,
                intensity=peak_intensity,
                fwhm=fwhm,
                resolving_power=resolving_power,
                snr_linear=snr_linear,
                snr_db=snr_db,
                ko=ko,
            )
        )

    return peaks


def detect_nearest_peak(
    x: np.ndarray,
    y: np.ndarray,
    cursor_x: float,
    noise_start: float,
    noise_end: float,
    pressure_torr: float,
    temperature_c: float,
    length_cm: float,
    voltage_kv: float,
    gate_multiplier: float,
) -> list[PeakMetrics]:
    if x.size == 0 or y.size == 0:
        return []

    peak_list = detect_all_peaks(
        x=x,
        y=y,
        noise_start=noise_start,
        noise_end=noise_end,
        min_prominence=0.0,
        min_snr=0.0,
        pressure_torr=pressure_torr,
        temperature_c=temperature_c,
        length_cm=length_cm,
        voltage_kv=voltage_kv,
        gate_multiplier=gate_multiplier,
    )
    if not peak_list:
        return []
    nearest_peak = min(peak_list, key=lambda peak: abs(float(peak.x_position) - float(cursor_x)))
    return [nearest_peak]
