"""Administrative boundaries: load, standardise, match district names, select a study area.

Works for ANY Ugandan district: nothing here mentions a specific district.
"""
from __future__ import annotations

import difflib
import re
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.ops import unary_union

# Column names commonly used by UBOS / OCHA (COD-AB) / geoBoundaries style files.
# Matching is case-insensitive. Override in config.yaml if your file differs.
_CANDIDATES = {
    "district": ["adm2_en", "adm2_name", "adm2name", "admin2name", "dname_2020", "dname_2019",
                 "dname", "district", "district_name", "dist_name", "name_2"],
    "subcounty": ["adm3_en", "adm3_name", "adm3name", "admin3name", "sname_2020", "sname_2019",
                  "sname", "subcounty", "sub_county", "subcounty_name", "name_3"],
    "district_code": ["adm2_pcode", "adm2_code", "dcode_2020", "dcode", "district_id"],
    "subcounty_code": ["adm3_pcode", "adm3_code", "scode_2020", "scode", "subcounty_id"],
}


# ----------------------------------------------------------------------------- loading
def load_admin(path: str | Path) -> gpd.GeoDataFrame:
    """Read the boundary file. Boundary files are small, so reading locally is fine."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {path}\n"
            "Check boundaries.path in config.yaml (use forward slashes, e.g. D:/folder/file.shp)."
        )
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        warnings.warn("Boundary file has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs(4326)
    return gdf


def detect_columns(gdf: gpd.GeoDataFrame, overrides: dict | None = None) -> dict[str, str | None]:
    """Find district / subcounty name and code columns (case-insensitive)."""
    lookup = {c.lower(): c for c in gdf.columns}
    found: dict[str, str | None] = {}
    for key, candidates in _CANDIDATES.items():
        user = (overrides or {}).get(key)
        if user:
            if user not in gdf.columns:
                raise KeyError(f"Column '{user}' (config: columns.{key}) not found. "
                               f"Available columns: {list(gdf.columns)}")
            found[key] = user
            continue
        found[key] = next((lookup[c] for c in candidates if c in lookup), None)
    return found


def require_columns(cols: dict[str, str | None], gdf: gpd.GeoDataFrame) -> None:
    missing = [k for k in ("district", "subcounty") if not cols.get(k)]
    if missing:
        raise KeyError(
            f"Could not detect the {', '.join(missing)} column(s).\n"
            f"Columns in the file: {[c for c in gdf.columns if c != 'geometry']}\n"
            "Type the right names under boundaries.columns in config.yaml."
        )


# ----------------------------------------------------------------------------- geometry QC
def _polygonal(geom):
    """After make_valid a geometry can become a collection; keep only the polygon parts."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if parts:
            return unary_union(parts)
    return geom


def clean_geometries(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """Drop empty geometries and repair invalid ones. Returns (clean_gdf, report)."""
    report = {"features_in": int(len(gdf))}
    empty = gdf.geometry.isna() | gdf.geometry.is_empty
    report["dropped_empty"] = int(empty.sum())
    gdf = gdf.loc[~empty].copy()

    invalid = ~gdf.geometry.is_valid
    report["invalid_repaired"] = int(invalid.sum())
    if invalid.any():
        fixed = gdf.loc[invalid, "geometry"].apply(shapely.make_valid).apply(_polygonal)
        gdf.loc[invalid, "geometry"] = fixed
    report["features_out"] = int(len(gdf))
    return gdf, report


# ----------------------------------------------------------------------------- standardise
def normalise_name(x) -> str:
    """lower-case, strip the words 'district'/'sub-county', drop punctuation."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).lower().strip()
    s = re.sub(r"\b(district|sub[- ]?county|s/c)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def standardise(gdf: gpd.GeoDataFrame, cols: dict[str, str | None],
                storage_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Return a tidy layer with fixed column names, whatever the source file used."""
    out = gpd.GeoDataFrame(
        {
            "district": gdf[cols["district"]].astype(str).str.strip(),
            "subcounty": gdf[cols["subcounty"]].astype(str).str.strip(),
            "district_code": gdf[cols["district_code"]].astype(str) if cols.get("district_code") else None,
            "subcounty_code": gdf[cols["subcounty_code"]].astype(str) if cols.get("subcounty_code") else None,
        },
        geometry=gdf.geometry.values,
        crs=gdf.crs,
    ).to_crs(storage_crs)
    out["district_key"] = out["district"].map(normalise_name)
    out["subcounty_key"] = out["subcounty"].map(normalise_name)
    return out


# ----------------------------------------------------------------------------- matching
def resolve_districts(requested: list[str], available: list[str], cutoff: float = 0.8) -> dict[str, list[str]]:
    """Map each requested name to the official name(s) in the file.

    Exact match first (ignoring case, 'district', punctuation), then close spelling matches
    are only SUGGESTED, never silently used, so a typo cannot change the analysed area.
    """
    keys: dict[str, list[str]] = {}
    for name in sorted(set(available)):
        keys.setdefault(normalise_name(name), []).append(name)

    resolved, problems = {}, []
    for req in requested:
        k = normalise_name(req)
        if k in keys:
            resolved[req] = keys[k]
        else:
            close = difflib.get_close_matches(k, list(keys), n=3, cutoff=cutoff)
            hint = f" Did you mean: {', '.join(n for c in close for n in keys[c])}?" if close else ""
            problems.append(f"'{req}' was not found.{hint}")
    if problems:
        raise ValueError("District name problem(s):\n  - " + "\n  - ".join(problems)
                         + "\nCheck study_area.districts in config.yaml.")
    return resolved


def select_study_area(layer: gpd.GeoDataFrame, resolved: dict[str, list[str]]) -> gpd.GeoDataFrame:
    official = {n for names in resolved.values() for n in names}
    return layer[layer["district"].isin(official)].copy().reset_index(drop=True)


def district_layer(subcounties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve subcounties into one polygon per district."""
    d = subcounties.dissolve(by="district", as_index=False, aggfunc={"district_code": "first",
                                                                      "district_key": "first"})
    return d[["district", "district_code", "district_key", "geometry"]]


# ----------------------------------------------------------------------------- geometry helpers
def utm_epsg_for(gdf: gpd.GeoDataFrame) -> int:
    """Pick the right UTM zone (and hemisphere) from the area's centre, in metres for area maths.

    Uganda straddles the equator and spans UTM zones 35-36, so this must not be hard-coded.
    """
    g = gdf.to_crs(4326)
    c = shapely.union_all(g.geometry.values).centroid
    zone = int((c.x + 180) // 6) + 1
    return (32600 if c.y >= 0 else 32700) + zone


def add_area_ha(gdf: gpd.GeoDataFrame, epsg: int, col: str = "area_ha") -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf[col] = (gdf.to_crs(epsg).area / 10_000).round(2)
    return gdf


def write_gpkg(path: Path, layers: dict[str, gpd.GeoDataFrame]) -> None:
    """Write several layers into ONE GeoPackage (what QGIS will open). Re-running overwrites."""
    path = Path(path)
    if path.exists():
        path.unlink()
    for name, gdf in layers.items():
        gdf.to_file(path, layer=name, driver="GPKG")
