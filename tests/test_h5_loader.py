from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ims_data_analysis.io.h5_loader import H5LoadError, load_h5_experiment


def _write_minimal_h5(path: Path, include_json: bool = True, bad_lengths: bool = False) -> None:
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
        if include_json:
            cfg.attrs["config_json"] = json.dumps(config_dict)
        else:
            for key, value in config_dict.items():
                cfg.attrs[key] = value

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
    assert loaded.matrix.shape == (2, 10)


def test_load_h5_with_flat_attrs_fallback(tmp_path: Path) -> None:
    target = tmp_path / "sample_attrs.h5"
    _write_minimal_h5(target, include_json=False)

    loaded = load_h5_experiment(str(target))
    assert loaded.config.data_points == 10
    assert loaded.matrix.shape == (2, 10)


def test_load_h5_rejects_mismatched_iteration_lengths(tmp_path: Path) -> None:
    target = tmp_path / "bad_lengths.h5"
    _write_minimal_h5(target, include_json=True, bad_lengths=True)

    with pytest.raises(H5LoadError):
        load_h5_experiment(str(target))
