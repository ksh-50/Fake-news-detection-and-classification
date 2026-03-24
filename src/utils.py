from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union, Mapping

PathLike = Union[str, Path]


def ensure_outdir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def save_json(obj: Mapping[str, Any] | Any, path: PathLike, *, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    return p.resolve()


def load_json(path: PathLike) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
