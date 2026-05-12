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
    # Backward-compatibility field. UI no longer edits this directly.
    time_per_frequency_ms: float = 1000.0
    enable_fft: bool = True

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "FTIMSConfig":
        raw = raw or {}
        return cls(
            start_frequency_hz=float(raw.get("start_frequency_hz", 10.0)),
            frequency_step_hz=float(raw.get("frequency_step_hz", 5.0)),
            end_frequency_hz=float(raw.get("end_frequency_hz", 4000.0)),
            time_per_frequency_ms=float(raw.get("time_per_frequency_ms", 1000.0)),
            enable_fft=bool(raw.get("enable_fft", True)),
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

    def total_frequencies(self) -> int:
        return len(self.frequency_steps())

    def step_duration_seconds(self, averages_per_iteration: int) -> float:
        start_hz = max(1e-9, float(self.start_frequency_hz))
        avg_count = max(1, int(averages_per_iteration))
        return avg_count / start_hz

    def step_duration_ms(self, averages_per_iteration: int) -> float:
        return 1000.0 * self.step_duration_seconds(averages_per_iteration)

    def estimated_duration_seconds(self, averages_per_iteration: int) -> float:
        return self.total_frequencies() * self.step_duration_seconds(averages_per_iteration)


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

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def voltage_steps_kv(self) -> list[float]:
        step_kv = float(self.voltage_step_v) / 1000.0
        if step_kv <= 0.0:
            return []
        values: list[float] = []
        current = float(self.initial_voltage_kv)
        stop = float(self.final_voltage_kv)
        while current <= stop + 1e-9:
            values.append(current)
            current += step_kv
        return values

    def total_voltages(self) -> int:
        return len(self.voltage_steps_kv())

    def estimated_duration_seconds(
        self,
        experiment_length_ms: float,
        averages_per_iteration: int,
        total_iterations: int,
    ) -> float:
        per_point_s = max(0.0, float(experiment_length_ms)) / 1000.0
        return (
            float(self.total_voltages())
            * max(1, int(averages_per_iteration))
            * max(1, int(total_iterations))
            * per_point_s
        )


@dataclass
class SweptFTIMSConfig:
    initial_frequency_hz: float = 1.0
    final_frequency_hz: float = 8000.0
    sweep_time_seconds: float = 4.0

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

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

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

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

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        config_dict = asdict(self)
        config_dict["operation_mode"] = self.operation_mode.value
        if self.ftims_config:
            config_dict["ftims_config"] = self.ftims_config.to_dict()
        if self.swept_ftims_config:
            config_dict["swept_ftims_config"] = self.swept_ftims_config.to_dict()
        if self.vsims_config:
            config_dict["vsims_config"] = self.vsims_config.to_dict()
        if self.swept_vsims_config:
            config_dict["swept_vsims_config"] = self.swept_vsims_config.to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        mode_value = str(raw.get("operation_mode", "DTIMS"))
        try:
            mode = OperationMode(mode_value)
        except ValueError:
            mode = OperationMode.DTIMS

        ftims_config_dict = raw.get("ftims_config")
        ftims_config = FTIMSConfig.from_dict(ftims_config_dict) if ftims_config_dict else FTIMSConfig()
        swept_ftims_config_dict = raw.get("swept_ftims_config")
        swept_ftims_config = SweptFTIMSConfig.from_dict(swept_ftims_config_dict) if swept_ftims_config_dict else SweptFTIMSConfig()
        vsims_config_dict = raw.get("vsims_config")
        vsims_config = SteppedVSIMSConfig.from_dict(vsims_config_dict) if vsims_config_dict else SteppedVSIMSConfig()
        swept_vsims_config_dict = raw.get("swept_vsims_config")
        swept_vsims_config = SweptVSIMSConfig.from_dict(swept_vsims_config_dict) if swept_vsims_config_dict else SweptVSIMSConfig()

        return cls(
            operation_mode=mode,
            pulse_width_ms=float(raw.get("pulse_width_ms", 1.0)),
            experiment_length_ms=float(raw.get("experiment_length_ms", 50.0)),
            data_points=int(raw.get("data_points", 4000)),
            averages_per_iteration=int(raw.get("averages_per_iteration", 10)),
            total_iterations=int(raw.get("total_iterations", 50)),
            positive_mode=bool(raw.get("positive_mode", False)),
            ftims_config=ftims_config,
            swept_ftims_config=swept_ftims_config,
            vsims_config=vsims_config,
            swept_vsims_config=swept_vsims_config,
        )


@dataclass
class LoadedExperiment:
    config: ExperimentConfig
    matrix: np.ndarray
    created_at: str
    iteration_timestamps: list[str]
    frequency_domain_iterations: list[dict[float, np.ndarray]] = field(default_factory=list)
    frequency_bins: list[float] = field(default_factory=list)
    ftims_raw_spectrum_iterations: list[dict[float, float]] = field(default_factory=list)
    swept_raw_time_domain_iterations: list[np.ndarray] = field(default_factory=list)
    swept_fft_frequency_bins_iterations: list[np.ndarray] = field(default_factory=list)

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
