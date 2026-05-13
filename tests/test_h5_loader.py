from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ims_data_analysis.io.h5_loader import H5LoadError, load_h5_experiment


def _write_minimal_h5(
    path: Path,
    include_json: bool = True,
    bad_lengths: bool = False,
    config_gate_multiplier: float | None = None,
    gate_multiplier_key: str = "gate_v_multiplier",
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["created_at"] = "2026-05-07T10:00:00"
        cfg = handle.create_group("config")
        config_dict = {
            "operation_mode": "DTIMS",
            "pulse_width_ms": 1.0,
            "experiment_length_ms": 50.0,
            "data_points": 10,
            "averages_per_iteration": 2,
            "total_iterations": 2,
            "positive_mode": False,
        }
        if config_gate_multiplier is not None:
            config_dict["gate_multiplier"] = config_gate_multiplier
        if include_json:
            cfg.attrs["config_json"] = json.dumps(config_dict)
        else:
            for key, value in config_dict.items():
                cfg.attrs[key] = value

        # Metadata stored separately in user_params group (matching IMSControl2026 format)
        user_params = handle.create_group("user_params")
        user_params.attrs["pressure_torr"] = 740.0
        user_params.attrs["temperature_c"] = 30.0
        user_params.attrs["length_cm"] = 12.5
        user_params.attrs[gate_multiplier_key] = 0.85

        iters = handle.create_group("iterations")
        iters.create_dataset("iteration_1", data=np.linspace(0.0, 1.0, 10))
        if bad_lengths:
            iters.create_dataset("iteration_2", data=np.linspace(0.0, 1.0, 8))
        else:
            iters.create_dataset("iteration_2", data=np.linspace(1.0, 2.0, 10))


def test_load_h5_with_config_json(tmp_path: Path) -> None:
    target = tmp_path / "sample_json.h5"
    _write_minimal_h5(target, include_json=True)

    loaded = load_h5_experiment(str(target))
    assert loaded.config.operation_mode.value == "DTIMS"
    assert loaded.config.pressure_torr == pytest.approx(740.0)
    assert loaded.config.temperature_c == pytest.approx(30.0)
    assert loaded.config.length_cm == pytest.approx(12.5)
    assert loaded.config.gate_multiplier == pytest.approx(0.85)
    assert loaded.matrix.shape == (2, 10)


def test_load_h5_metadata_overrides_config_defaults(tmp_path: Path) -> None:
    target = tmp_path / "sample_override.h5"
    _write_minimal_h5(target, include_json=True, config_gate_multiplier=1.0)

    loaded = load_h5_experiment(str(target))
    assert loaded.config.gate_multiplier == pytest.approx(0.85)
    assert loaded.config.pressure_torr == pytest.approx(740.0)


def test_load_h5_accepts_spaced_gate_multiplier_key(tmp_path: Path) -> None:
    target = tmp_path / "sample_spaced_key.h5"
    _write_minimal_h5(target, include_json=True, config_gate_multiplier=1.0, gate_multiplier_key="gate voltage multiplier")

    loaded = load_h5_experiment(str(target))
    assert loaded.config.gate_multiplier == pytest.approx(0.85)


def test_load_h5_with_flat_attrs_fallback(tmp_path: Path) -> None:
    target = tmp_path / "sample_attrs.h5"
    _write_minimal_h5(target, include_json=False)

    loaded = load_h5_experiment(str(target))
    assert loaded.config.data_points == 10
    assert loaded.config.pressure_torr == pytest.approx(740.0)
    assert loaded.config.temperature_c == pytest.approx(30.0)
    assert loaded.config.length_cm == pytest.approx(12.5)
    assert loaded.config.gate_multiplier == pytest.approx(0.85)
    assert loaded.matrix.shape == (2, 10)


def test_load_h5_rejects_mismatched_iteration_lengths(tmp_path: Path) -> None:
    target = tmp_path / "bad_lengths.h5"
    _write_minimal_h5(target, include_json=True, bad_lengths=True)

    with pytest.raises(H5LoadError):
        load_h5_experiment(str(target))
