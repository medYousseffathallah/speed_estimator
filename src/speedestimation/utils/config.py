from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict at root of YAML: {path}")
    return data


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if base_dir is None:
        base_dir = os.getcwd()
    return str((Path(base_dir) / p).resolve())

