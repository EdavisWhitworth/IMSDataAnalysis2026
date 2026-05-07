from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class OperationMode(Enum):
    DTIMS = "DTIMS"
    FTIMS = "FTIMS"
    SWEPT_FTIMS = "SWEPT_FTIMS"
    STEPPED_VSIMS = "STEPPED_VSIMS"
    SWEPT_VSIMS = "SWEPT_VSIMS"


@dataclass
class FTIMSConfig:
    start_frequency_hz: float = 10.0
    frequency_step_hz: float = 5.0
    end_frequency_hz: float = 4000.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "FTIMSConfig":
        raw = raw or {}
        return cls(
            start_frequency_hz=float(raw.get("start_frequency_hz", 10.0)),
            frequency_step_hz=float(raw.get("frequency_step_hz", 5.0)),
            end_frequency_hz=float(raw.get("end_frequency_hz", 4000.0)),
        )

    def frequency_steps(self) -> list[float]:
        step = float(self.frequency_step_hz)
        if step <= 0:
            return []
        values: list[float] = []
        current = float(self.start_frequency_hz)
        stop = float(self.end_frequency_hz)
        while current <= stop + 1e-9:
            values.append(current)
            current += step
        return values


@dataclass
class SteppedVSIMSConfig:
    initial_voltage_kv: float = 4.0
    final_voltage_kv: float = 8.0
    voltage_step_v: float = 100.0
    time_add_ms: float = 0.0
    ionization_bias_kv: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SteppedVSIMSConfig":
        raw = raw or {}
        return cls(
            initial_voltage_kv=float(raw.get("initial_voltage_kv", 4.0)),
            final_voltage_kv=float(raw.get("final_voltage_kv", 8.0)),
            voltage_step_v=float(raw.get("voltage_step_v", 100.0)),
            time_add_ms=float(raw.get("time_add_ms", 0.0)),
            ionization_bias_kv=float(raw.get("ionization_bias_kv", 0.0)),
        )

    def voltage_steps_kv(self) -> list[float]:
        step_kv = float(self.voltage_step_v) / 1000.0
        if step_kv <= 0:
            return []
        values: list[float] = []
        current = float(self.initial_voltage_kv)
        stop = float(self.final_voltage_kv)
        while current <= stop + 1e-9:
            values.append(current)
            current += step_kv
        return values


@dataclass
class SweptFTIMSConfig:
    initial_frequency_hz: float = 1.0
    final_frequency_hz: float = 8000.0
    sweep_time_seconds: float = 4.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SweptFTIMSConfig":
        raw = raw or {}
        return cls(
            initial_frequency_hz=float(raw.get("initial_frequency_hz", 1.0)),
            final_frequency_hz=float(raw.get("final_frequency_hz", 8000.0)),
            sweep_time_seconds=float(raw.get("sweep_time_seconds", 4.0)),
        )


@dataclass
class SweptVSIMSConfig:
    ionization_bias_kv: float = 0.0
    v_add_kv: float = 0.0
    gate_pulse_delay_ms: float = 10.0
    ims_max_output_kv: float = 20.0
    control_voltage_max_v: float = 10.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SweptVSIMSConfig":
        raw = raw or {}
        return cls(
            ionization_bias_kv=float(raw.get("ionization_bias_kv", 0.0)),
            v_add_kv=float(raw.get("v_add_kv", 0.0)),
            gate_pulse_delay_ms=float(raw.get("gate_pulse_delay_ms", 10.0)),
            ims_max_output_kv=float(raw.get("ims_max_output_kv", 20.0)),
            control_voltage_max_v=float(raw.get("control_voltage_max_v", 10.0)),
        )


@dataclass
class ExperimentConfig:
    operation_mode: OperationMode = OperationMode.DTIMS
    pulse_width_ms: float = 1.0
    experiment_length_ms: float = 50.0
    data_points: int = 4000
    averages_per_iteration: int = 10
    total_iterations: int = 50
    positive_mode: bool = False
    ftims_config: FTIMSConfig = field(default_factory=FTIMSConfig)
    swept_ftims_config: SweptFTIMSConfig = field(default_factory=SweptFTIMSConfig)
    vsims_config: SteppedVSIMSConfig = field(default_factory=SteppedVSIMSConfig)
    swept_vsims_config: SweptVSIMSConfig = field(default_factory=SweptVSIMSConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        mode_value = str(raw.get("operation_mode", "DTIMS"))
        try:
            mode = OperationMode(mode_value)
        except ValueError:
            mode = OperationMode.DTIMS

        return cls(
            operation_mode=mode,
            pulse_width_ms=float(raw.get("pulse_width_ms", 1.0)),
            experiment_length_ms=float(raw.get("experiment_length_ms", 50.0)),
            data_points=int(raw.get("data_points", 4000)),
            averages_per_iteration=int(raw.get("averages_per_iteration", 10)),
            total_iterations=int(raw.get("total_iterations", 50)),
            positive_mode=bool(raw.get("positive_mode", False)),
            ftims_config=FTIMSConfig.from_dict(raw.get("ftims_config")),
            swept_ftims_config=SweptFTIMSConfig.from_dict(raw.get("swept_ftims_config")),
            vsims_config=SteppedVSIMSConfig.from_dict(raw.get("vsims_config")),
            swept_vsims_config=SweptVSIMSConfig.from_dict(raw.get("swept_vsims_config")),
        )


@dataclass
class LoadedExperiment:
    config: ExperimentConfig
    matrix: np.ndarray
    created_at: str
    iteration_timestamps: list[str]

    @property
    def mode_label(self) -> str:
        labels = {
            OperationMode.DTIMS: "DTIMS",
            OperationMode.FTIMS: "Stepped FTIMS",
            OperationMode.SWEPT_FTIMS: "Sweep FTIMS",
            OperationMode.STEPPED_VSIMS: "Stepped VSIMS",
            OperationMode.SWEPT_VSIMS: "Sweep VSIMS",
        }
        return labels.get(self.config.operation_mode, self.config.operation_mode.value)


@dataclass
class ModeView:
    mode: OperationMode
    x_axis: np.ndarray
    y_axis: np.ndarray
    heatmap: np.ndarray
    x_label: str
    y_label: str
    voltage_axis_kv: np.ndarray | None = None
