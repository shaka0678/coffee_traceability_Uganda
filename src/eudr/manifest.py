"""Run manifests: record exactly what went into each output (reproducibility)."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sidecar_hashes(shp_path: Path) -> dict[str, str]:
    """For a shapefile, hash every sidecar (.shp/.dbf/.shx/.prj/.cpg) that exists."""
    out = {}
    for ext in (".shp", ".dbf", ".shx", ".prj", ".cpg"):
        f = shp_path.with_suffix(ext)
        if f.exists():
            out[f.name] = file_sha256(f)
    return out


def package_versions() -> dict[str, str]:
    import geopandas, pandas, pyogrio, pyproj, shapely

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "geopandas": geopandas.__version__,
        "shapely": shapely.__version__,
        "pyogrio": pyogrio.__version__,
        "pyproj": pyproj.__version__,
        "pandas": pandas.__version__,
    }


def write_manifest(folder: Path, step: str, info: dict) -> Path:
    record = {
        "step": step,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packages": package_versions(),
        **info,
    }
    out = Path(folder) / f"{step}.json"
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return out
