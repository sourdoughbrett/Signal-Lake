# src/config.py
from pathlib import Path
import os
import yaml
from pydantic import BaseModel, validator


class Settings(BaseModel):
    # ── API / auth ───────────────────────────
    API_KEY: str
    SECRET_KEY: str
    api_base_url: str

    # ── dates ────────────────────────────────
    start_time: str  # "YYYY-MM-DD"
    end_time: str

    # ── trading params ───────────────────────
    position_size: int
    symbols: list[str]

    # ── helpers ──────────────────────────────
    @property
    def primary_symbol(self) -> str:
        return self.symbols[0]

    @validator("symbols")
    def must_not_be_empty(cls, v):
        if not v:
            raise ValueError("config.settings.yml → symbols list cannot be empty.")
        return v


# -------------------------------------------------
# Locate YAML (../config/settings.yml relative to repo)
# -------------------------------------------------
CFG_FILE = (
    Path(__file__)
    .resolve()
    .parent.parent  # go up to project root
    / "config"
    / "settings.yml"
)

if not CFG_FILE.exists():
    raise FileNotFoundError(f"Settings file not found: {CFG_FILE}")


def load_settings() -> Settings:
    with CFG_FILE.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Allow secrets to be injected as env-vars (safer for commits)
    raw["API_KEY"] = os.getenv("ALPACA_KEY", raw.get("API_KEY", ""))
    raw["SECRET_KEY"] = os.getenv("ALPACA_SECRET", raw.get("SECRET_KEY", ""))

    return Settings(**raw)


# Singleton settings object importable everywhere
settings: Settings = load_settings()
