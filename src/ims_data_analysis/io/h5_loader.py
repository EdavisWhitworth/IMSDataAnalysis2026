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

    # Load FTIMS stepped raw spectra (optional group)
    ftims_raw_spectrum_iterations: list[dict[float, float]] = []
    with h5py.File(source, "r") as handle:
        raw_group = handle.get("ftims_raw_spectra")
        if isinstance(raw_group, h5py.Group):
            for key in sorted(raw_group.keys(), key=lambda x: int(x.split("_")[-1])):
                ds = raw_group[key]
                rows = np.asarray(ds[:], dtype=np.float64)
                raw_points: dict[float, float] = {}
                if rows.ndim == 2 and rows.shape[1] == 2:
                    for row in rows:
                        raw_points[float(row[0])] = float(row[1])
                ftims_raw_spectrum_iterations.append(raw_points)

        # Load swept FTIMS raw time-domain and FFT bin data (optional groups)
        swept_raw_time_domain_iterations: list[np.ndarray] = []
        swept_fft_frequency_bins_iterations: list[np.ndarray] = []
        swept_raw_group = handle.get("swept_raw_iterations")
        swept_bins_group = handle.get("swept_fft_bins_hz")
        if isinstance(swept_raw_group, h5py.Group) or isinstance(swept_bins_group, h5py.Group):
            keys: set[str] = set()
            if isinstance(swept_raw_group, h5py.Group):
                keys.update(swept_raw_group.keys())
            if isinstance(swept_bins_group, h5py.Group):
                keys.update(swept_bins_group.keys())
            for key in sorted(keys, key=lambda x: int(x.split("_")[-1])):
                raw_arr = np.empty((0,), dtype=np.float64)
                bins_arr = np.empty((0,), dtype=np.float64)
                if isinstance(swept_raw_group, h5py.Group) and key in swept_raw_group:
                    raw_arr = np.asarray(swept_raw_group[key][:], dtype=np.float64)
                if isinstance(swept_bins_group, h5py.Group) and key in swept_bins_group:
                    bins_arr = np.asarray(swept_bins_group[key][:], dtype=np.float64)
                swept_raw_time_domain_iterations.append(raw_arr)
                swept_fft_frequency_bins_iterations.append(bins_arr)

    return LoadedExperiment(
        config=config,
        matrix=matrix,
        created_at=created_at,
        iteration_timestamps=timestamps,
        ftims_raw_spectrum_iterations=ftims_raw_spectrum_iterations,
        swept_raw_time_domain_iterations=swept_raw_time_domain_iterations,
        swept_fft_frequency_bins_iterations=swept_fft_frequency_bins_iterations,
    )
