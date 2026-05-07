from __future__ import annotations

import numpy as np

from ims_data_analysis.analysis.metrics import compute_ko, detect_all_peaks, detect_nearest_peak


def test_compute_ko_returns_positive_value() -> None:
    ko = compute_ko(
        drift_time_ms=12.0,
        pressure_torr=760.0,
        temperature_c=25.0,
        length_cm=10.0,
        voltage_kv=5.0,
        gate_multiplier=1.0,
    )
    assert ko is not None
    assert ko > 0


def test_detect_all_peaks_finds_two_gaussian_peaks() -> None:
    x = np.linspace(0.0, 40.0, 4000)
    peak1 = 100.0 * np.exp(-((x - 10.0) ** 2) / (2.0 * 0.5**2))
    peak2 = 80.0 * np.exp(-((x - 25.0) ** 2) / (2.0 * 0.6**2))
    y = peak1 + peak2 + 0.5

    peaks = detect_all_peaks(
        x=x,
        y=y,
        noise_start=0.0,
        noise_end=4.0,
        min_prominence=5.0,
        min_snr=1.0,
        pressure_torr=760.0,
        temperature_c=25.0,
        length_cm=10.0,
        voltage_kv=5.0,
        gate_multiplier=1.0,
    )

    assert len(peaks) >= 2
    positions = [p.x_position for p in peaks]
    assert any(abs(pos - 10.0) < 0.5 for pos in positions)
    assert any(abs(pos - 25.0) < 0.6 for pos in positions)


def test_detect_nearest_peak_returns_single_peak() -> None:
    x = np.linspace(0.0, 30.0, 3000)
    y = (
        120.0 * np.exp(-((x - 7.0) ** 2) / (2.0 * 0.4**2))
        + 90.0 * np.exp(-((x - 20.0) ** 2) / (2.0 * 0.5**2))
        + 1.0
    )

    peaks = detect_nearest_peak(
        x=x,
        y=y,
        cursor_x=19.8,
        noise_start=0.0,
        noise_end=3.0,
        pressure_torr=760.0,
        temperature_c=25.0,
        length_cm=10.0,
        voltage_kv=5.0,
        gate_multiplier=1.0,
    )

    assert len(peaks) == 1
    assert abs(peaks[0].x_position - 20.0) < 0.6


def test_detect_nearest_peak_prefers_closest_valid_peak() -> None:
    x = np.linspace(0.0, 30.0, 3000)
    y = (
        100.0 * np.exp(-((x - 8.0) ** 2) / (2.0 * 0.45**2))
        + 120.0 * np.exp(-((x - 21.0) ** 2) / (2.0 * 0.5**2))
        + 0.3
    )

    peaks = detect_nearest_peak(
        x=x,
        y=y,
        cursor_x=20.7,
        noise_start=0.0,
        noise_end=3.0,
        pressure_torr=760.0,
        temperature_c=25.0,
        length_cm=10.0,
        voltage_kv=5.0,
        gate_multiplier=1.0,
    )

    assert len(peaks) == 1
    assert abs(peaks[0].x_position - 21.0) < 0.6
