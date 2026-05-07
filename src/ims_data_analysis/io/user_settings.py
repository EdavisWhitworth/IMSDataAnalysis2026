from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


SETTINGS_DIR = Path.home() / ".ims_data_analysis_2026"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


@dataclass
class UserSettings:
    mode_override_enabled: bool = False
    mode_override_value: str = "DTIMS"
    voltage_override_enabled: bool = False
    voltage_override_kv: float = 0.0
    voltage_override_only_when_missing: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "UserSettings":
        return cls(
            mode_override_enabled=bool(raw.get("mode_override_enabled", False)),
            mode_override_value=str(raw.get("mode_override_value", "DTIMS")),
            voltage_override_enabled=bool(raw.get("voltage_override_enabled", False)),
            voltage_override_kv=float(raw.get("voltage_override_kv", 0.0)),
            voltage_override_only_when_missing=bool(raw.get("voltage_override_only_when_missing", True)),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
