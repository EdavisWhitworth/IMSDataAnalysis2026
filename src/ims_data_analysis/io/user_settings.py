from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


SETTINGS_DIR = Path.home() / ".ims_data_analysis_2026"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


@dataclass
class SpectrumStyleSettings:
    curve_color: str = "#000000"
    baseline_color: str = "#4cacf7"
    peak_color: str = "#e67700"
    cursor_color: str = "#ff6b6b"
    background_color: str = "#ffffff"
    text_color: str = "#000000"
    font_family: str = "Arial"
    font_size: int = 11
    title_text: str = "Selected Spectrum"
    show_title: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, object] | None) -> "SpectrumStyleSettings":
        raw = raw or {}
        return cls(
            curve_color=str(raw.get("curve_color", "#000000")),
            baseline_color=str(raw.get("baseline_color", "#4cacf7")),
            peak_color=str(raw.get("peak_color", "#e67700")),
            cursor_color=str(raw.get("cursor_color", "#ff6b6b")),
            background_color=str(raw.get("background_color", "#ffffff")),
            text_color=str(raw.get("text_color", "#000000")),
            font_family=str(raw.get("font_family", "Arial")),
            font_size=int(raw.get("font_size", 11)),
            title_text=str(raw.get("title_text", "Selected Spectrum")),
            show_title=bool(raw.get("show_title", True)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class UserSettings:
    mode_override_enabled: bool = False
    mode_override_value: str = "DTIMS"
    voltage_override_enabled: bool = False
    voltage_override_kv: float = 0.0
    voltage_override_only_when_missing: bool = True
    vs_step_table_target: str = "selected"
    selected_spectrum_style: SpectrumStyleSettings = field(default_factory=SpectrumStyleSettings)
    optimized_spectrum_style: SpectrumStyleSettings = field(default_factory=SpectrumStyleSettings)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "UserSettings":
        return cls(
            mode_override_enabled=bool(raw.get("mode_override_enabled", False)),
            mode_override_value=str(raw.get("mode_override_value", "DTIMS")),
            voltage_override_enabled=bool(raw.get("voltage_override_enabled", False)),
            voltage_override_kv=float(raw.get("voltage_override_kv", 0.0)),
            voltage_override_only_when_missing=bool(raw.get("voltage_override_only_when_missing", True)),
            vs_step_table_target=str(raw.get("vs_step_table_target", "selected")),
            selected_spectrum_style=SpectrumStyleSettings.from_dict(raw.get("selected_spectrum_style")),
            optimized_spectrum_style=SpectrumStyleSettings.from_dict(raw.get("optimized_spectrum_style")),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["selected_spectrum_style"] = self.selected_spectrum_style.to_dict()
        data["optimized_spectrum_style"] = self.optimized_spectrum_style.to_dict()
        return data


def load_user_settings() -> UserSettings:
    if not SETTINGS_PATH.exists():
        return UserSettings()

    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return UserSettings()
        return UserSettings.from_dict(raw)
    except (json.JSONDecodeError, OSError, ValueError):
        return UserSettings()


def save_user_settings(settings: UserSettings) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
