from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
