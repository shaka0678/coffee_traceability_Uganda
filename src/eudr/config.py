"""Project configuration: find the project root, load config.yaml, build output folders."""
from __future__ import annotations

import re
from pathlib import Path

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: current folder) until config.yaml is found."""
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / "config.yaml").exists():
            return folder
    raise FileNotFoundError(
        "config.yaml not found. Open the project's root folder in VS Code and try again."
    )


def load_config(root: Path | None = None) -> dict:
    root = find_project_root(root)
    with open(root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = root
    return cfg


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def run_name(cfg: dict) -> str:
    """'auto' builds a name from the districts, e.g. buikwe_kyotera."""
    name = cfg.get("run_name", "auto")
    if name and str(name).lower() != "auto":
        return slugify(name)
    districts = cfg["study_area"]["districts"]
    return "_".join(slugify(d) for d in districts)[:80]


def run_dirs(cfg: dict) -> dict[str, Path]:
    """Create (if needed) and return the output folders for this run."""
    base = cfg["_root"] / "outputs" / run_name(cfg)
    dirs = {
        "base": base,
        "gis": base / "gis",
        "rasters": base / "rasters",
        "reports": base / "reports",
        "maps": base / "maps",
        "manifests": base / "manifests",
        "cache": cfg["_root"] / "data" / "cache" / run_name(cfg),
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_path(cfg: dict, p: str) -> Path:
    """Absolute paths are used as-is; relative paths are relative to the project root."""
    path = Path(p)
    return path if path.is_absolute() else cfg["_root"] / path

