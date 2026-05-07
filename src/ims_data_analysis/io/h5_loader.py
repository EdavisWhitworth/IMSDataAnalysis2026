from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ims_data_analysis.models import ExperimentConfig, LoadedExperiment


class H5LoadError(RuntimeError):
    pass


def _parse_config(config_group: h5py.Group) -> ExperimentConfig:
    raw_json = config_group.attrs.get("config_json")
    if raw_json is not None:
        if isinstance(raw_json, bytes):
            raw_json = raw_json.decode("utf-8", errors="replace")
        return ExperimentConfig.from_dict(json.loads(str(raw_json)))

    raw_attrs: dict[str, object] = {}
    for key, value in dict(config_group.attrs).items():
        if isinstance(value, np.generic):
            raw_attrs[key] = value.item()
        else:
            raw_attrs[key] = value
    return ExperimentConfig.from_dict(raw_attrs)


def load_h5_experiment(file_path: str) -> LoadedExperiment:
    source = Path(file_path)
    if not source.exists():
        raise H5LoadError(f"File does not exist: {source}")

    with h5py.File(source, "r") as handle:
        if "config" not in handle:
            raise H5LoadError("H5 file is missing required /config group")
        if "iterations" not in handle:
            raise H5LoadError("H5 file is missing required /iterations group")

        config = _parse_config(handle["config"])
        created_at = str(handle.attrs.get("created_at", ""))

        datasets: list[np.ndarray] = []
        timestamps: list[str] = []
        iter_group = handle["iterations"]
        iteration_numbers: list[tuple[int, str]] = []
        for key in iter_group.keys():
            if not key.startswith("iteration_"):
                raise H5LoadError(f"Unsupported dataset name under /iterations: {key}")
            suffix = key.split("_")[-1]
            if not suffix.isdigit():
                raise H5LoadError(f"Invalid iteration dataset suffix: {key}")
            iteration_numbers.append((int(suffix), key))

        keys = [key for _, key in sorted(iteration_numbers, key=lambda item: item[0])]
        expected_len: int | None = None
        for key in keys:
            data = np.asarray(iter_group[key][:], dtype=np.float64)
            if data.ndim != 1:
                raise H5LoadError(f"Iteration dataset must be 1D: {key}")
            if expected_len is None:
                expected_len = int(data.shape[0])
            elif data.shape[0] != expected_len:
                raise H5LoadError(
                    f"Iteration length mismatch in {key}: expected {expected_len}, got {data.shape[0]}"
                )
            datasets.append(data)
            timestamps.append(str(iter_group[key].attrs.get("timestamp", "")))

    if not datasets:
        matrix = np.empty((0, config.data_points), dtype=np.float64)
    else:
        matrix = np.vstack(datasets)

    return LoadedExperiment(
        config=config,
        matrix=matrix,
        created_at=created_at,
        iteration_timestamps=timestamps,
    )
