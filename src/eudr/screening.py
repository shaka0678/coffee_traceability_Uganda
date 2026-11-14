"""Plot-level forest-encroachment screening against the notebook 02 rasters (EUDR cut-off logic).

A flag is a screening result that triggers review, not proof of deforestation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from shapely.ops import transform

from .boundaries import _polygonal
from .rasterstream import pixel_area_ha, read_tif
from .validation import _utm_epsg

# plots with these critical problems cannot be placed reliably on the rasters
BLOCKING_FLAGS = ["missing_geometry", "coords_invalid", "outside_uganda",
                  "outside_study_area", "district_mismatch"]
FLAG_RANK = {"no loss": 0, "review": 1, "possible loss": 2}
PRIO_RANK = {"low": 0, "medium": 1, "high": 2}


# ------------------------------------------------------------------ geometry preparation
_TF: dict = {}


def _tf(src: int, dst: int):
    if (src, dst) not in _TF:
        _TF[(src, dst)] = Transformer.from_crs(src, dst, always_xy=True).transform
    return _TF[(src, dst)]


def buffer_m(geom, dist_m: float):
    """Buffer a lon/lat geometry by `dist_m` metres, measured in its own UTM zone."""
    epsg = int(_utm_epsg(np.array([geom], dtype=object))[0])
    g = transform(_tf(4326, epsg), geom).buffer(dist_m)
    return transform(_tf(epsg, 4326), g)


def _prep_one(r, cols, scfg) -> dict:
    out = {"geom_screen": None, "geom_buffered": None, "basis": "", "confidence": "",
           "not_screened_reason": ""}
    g = r.geometry
    if g is None or g.is_empty:
        out["not_screened_reason"] = "no geometry"
        return out
    blocked = [c for c in BLOCKING_FLAGS if c in cols and pd.notna(r[c]) and bool(r[c])]
    if blocked:
        out["not_screened_reason"] = "blocked by: " + ", ".join(blocked)
        return out

    acc = pd.to_numeric(r.get("gps_accuracy_m"), errors="coerce")
    acc = scfg["default_gps_m"] if pd.isna(acc) else float(acc)
    acc = max(acc, scfg["min_gps_buffer_m"])

    if g.geom_type == "Point":                                  # point plots become a circle of declared area
        a = pd.to_numeric(r.get("declared_area_ha"), errors="coerce")
        if pd.isna(a) or a <= 0:
            out["not_screened_reason"] = "point without declared area"
            return out
        geom, basis, conf = buffer_m(g, math.sqrt(a * 10_000 / math.pi)), "point_buffer", "low"
    else:
        repaired = not g.is_valid
        geom = _polygonal(shapely.make_valid(g)) if repaired else g
        area = pd.to_numeric(r.get("area_mapped_ha"), errors="coerce")
        basis = "polygon_repaired" if repaired else "polygon"
        if repaired:
            conf = "low"
        else:
            conf = "high" if (pd.notna(area) and area >= scfg["min_confident_area_ha"] and acc <= 15) else "medium"

    out.update(geom_screen=geom, geom_buffered=buffer_m(geom, acc), basis=basis, confidence=conf)
    return out


def prepare_geometries(plots: gpd.GeoDataFrame, scfg: dict) -> pd.DataFrame:
    cols = set(plots.columns)
    rows = [_prep_one(r, cols, scfg) for _, r in plots.iterrows()]
    return pd.DataFrame(rows, index=plots.index)


# ------------------------------------------------------------------ rasters
class Rasters:
    """The notebook 02 layers for one district, all on the same Hansen grid."""

    def __init__(self, forest, tc, ly, cls, tr, thr, pix_m):
        self.forest = forest == 1
        self.tree = tc >= thr
        self.ly = ly
        self.cls = cls
        self.tr = tr
        self.shape = forest.shape
        self.area_row = pixel_area_ha(tr, forest.shape)
        d = (ndimage.distance_transform_edt(~self.forest) * pix_m if self.forest.any()
             else np.full(forest.shape, np.inf))
        self.dist = np.where(self.forest, 0.0, d)               # metres to the nearest 2020 forest


def load_rasters(dirs: dict, slug: str, version: str, thr: float) -> Rasters:
    forest, tr, _ = read_tif(dirs["rasters"] / f"{slug}_forest_2020.tif")
    cls, _, _ = read_tif(dirs["rasters"] / f"{slug}_loss_class.tif")
    ly, _, _ = read_tif(dirs["cache"] / f"{slug}_hansen_lossyear_{version}.tif")
    tc, _, _ = read_tif(dirs["cache"] / f"{slug}_hansen_treecover2000_{version}.tif")
    if not (forest.shape == cls.shape == ly.shape == tc.shape):
        raise ValueError(f"{slug}: the raster layers do not share a grid. Re-run notebook 02 cell 4.")
    return Rasters(forest, tc, ly, cls, tr, thr, abs(tr.a) * 111_320)


def class_rgb(R: Rasters):
    rgb = np.full(R.cls.shape + (3,), 245, dtype="uint8")
    rgb[R.forest] = (120, 170, 120)            # forest in 2020
    rgb[R.cls == 3] = (240, 170, 80)           # loss 2001-2020
    rgb[R.cls == 2] = (250, 220, 60)           # loss after 2020, outside 2020 forest
    rgb[R.cls == 1] = (200, 30, 30)            # loss after 2020, inside 2020 forest
    rgb[R.cls == 255] = (255, 255, 255)        # outside the district
    return rgb


# ------------------------------------------------------------------ measuring one plot
def coverage(geom, tr, shape, s: int = 5):
    """Share of each raster pixel covered by `geom`, in the pixel window around it.

    The plot is drawn on a grid `s` times finer than the raster, so small plots are
    measured by fraction of pixel covered, not by whole pixels.
    """
    minx, miny, maxx, maxy = geom.bounds
    c0 = max(int(math.floor((minx - tr.c) / tr.a)) - 1, 0)
    c1 = min(int(math.floor((maxx - tr.c) / tr.a)) + 2, shape[1])
    r0 = max(int(math.floor((maxy - tr.f) / tr.e)) - 1, 0)
    r1 = min(int(math.floor((miny - tr.f) / tr.e)) + 2, shape[0])
    if c1 <= c0 or r1 <= r0:
        return None
    h, w = r1 - r0, c1 - c0
    sub = Affine(tr.a / s, 0, tr.c + c0 * tr.a, 0, tr.e / s, tr.f + r0 * tr.e)
    m = rasterize([(geom, 1)], out_shape=(h * s, w * s), transform=sub, fill=0, dtype="uint8")
    return (r0, r1, c0, c1), m.reshape(h, s, w, s).mean(axis=(1, 3))


def screen_geometry(geom, R: Rasters, cut_code: int):
    """Hectares of forest and tree loss inside `geom`. `cut_code` is the cut-off year minus 2000."""
    cov = coverage(geom, R.tr, R.shape)
    if cov is None:
        return None
    (r0, r1, c0, c1), w = cov
    wgt = w * R.area_row[r0:r1, None]                          # hectares of each pixel inside the plot
    forest = R.forest[r0:r1, c0:c1]
    ly = R.ly[r0:r1, c0:c1]
    loss = (ly > 0) & R.tree[r0:r1, c0:c1]
    post, pre = loss & (ly > cut_code), loss & (ly <= cut_code)

    out = {
        "mapped_area_ha": float(wgt.sum()),
        "forest2020_ha": float(wgt[forest].sum()),
        "loss_pre2020_ha": float(wgt[pre].sum()),
        "loss_post2020_ha": float(wgt[post].sum()),
        "loss_post_in_forest_ha": float(wgt[post & forest].sum()),
        "loss_post_outside_forest_ha": float(wgt[post & ~forest].sum()),
        "loss_year_first": np.nan,
        "loss_year_main": np.nan,
    }
    if out["loss_post2020_ha"] > 0:
        by_year = np.bincount(ly[post].astype(int), weights=wgt[post])
        years = np.nonzero(by_year > 0.001)[0]
        if len(years):
            out["loss_year_first"] = 2000 + int(years.min())
            out["loss_year_main"] = 2000 + int(by_year.argmax())

    c = geom.centroid                                           # distance to forest is context, not a rule
    rr, cc = int((c.y - R.tr.f) / R.tr.e), int((c.x - R.tr.c) / R.tr.a)
    inside = 0 <= rr < R.shape[0] and 0 <= cc < R.shape[1]
    out["dist_to_forest_m"] = round(float(R.dist[rr, cc]), 0) if inside else np.nan
    return out


# ------------------------------------------------------------------ protected areas (optional)
def load_reserves(study_geom, scfg: dict, resolver):
    """Protected-area polygons from a local file or OpenStreetMap. Returns None if neither is set."""
    path = scfg.get("protected_areas_path")
    try:
        if path:
            gdf = gpd.read_file(resolver(path)).to_crs(4326)
        elif scfg.get("use_osm_reserves"):
            import osmnx as ox
            tags = {"boundary": "protected_area", "leisure": "nature_reserve"}
            gdf = ox.features_from_polygon(study_geom.buffer(0.02), tags).reset_index(drop=True)
        else:
            return None
    except Exception as e:                                      # network, missing package, bad file
        print(f"Protected-area layer not loaded: {e}")
        return None
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        print("Protected-area layer loaded but holds no polygons in this area.")
        return None
    name_col = next((c for c in ("name", "NAME", "Name", "NAME_EN", "site_name") if c in gdf.columns), None)
    gdf["pa_name"] = gdf[name_col].fillna("unnamed").astype(str) if name_col else "unnamed"
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf[["pa_name", "geometry"]].reset_index(drop=True)


def reserve_share(geom, reserves):
    """(% of the plot inside protected areas, their names)."""
    if reserves is None:
        return 0.0, ""
    hit = reserves.iloc[list(reserves.sindex.query(geom, predicate="intersects"))]
    if hit.empty or geom.area == 0:
        return 0.0, ""
    inter = shapely.union_all(hit.geometry.values).intersection(geom)
    return inter.area / geom.area * 100, "; ".join(sorted(set(hit["pa_name"])))


# ---------------------- the rules
def _raise(cur, new):
    f = new[0] if FLAG_RANK[new[0]] > FLAG_RANK[cur[0]] else cur[0]
    p = new[1] if PRIO_RANK[new[1]] > PRIO_RANK[cur[1]] else cur[1]
    return f, p


def classify(m: dict, attrs, scfg: dict, cut_year: int, reserve_pct: float = 0.0, reserve_names: str = ""):
    """Return (flag, priority, reasons). Flags: no loss / review / possible loss."""
    area = max(m["mapped_area_ha"], 1e-9)
    pct = lambda ha: ha / area * 100
    big = lambda ha: ha >= scfg["min_loss_ha"] and pct(ha) >= scfg["min_loss_pct"]
    forest_pct = pct(m["forest2020_ha"])
    state, why = ("no loss", "low"), []

    if big(m["loss_post_in_forest_ha"]):
        state = _raise(state, ("possible loss", "high"))
        why.append(f"{m['loss_post_in_forest_ha']:.2f} ha tree loss after {cut_year} on land that was forest in {cut_year}")
    elif big(m["loss_post_outside_forest_ha"]):
        state = _raise(state, ("review", "medium"))
        why.append(f"{m['loss_post_outside_forest_ha']:.2f} ha tree loss after {cut_year} outside the {cut_year} forest map "
                   "(tree crop, shade tree or replanting?)")
    elif forest_pct >= scfg["forest_overlap_pct"]:
        state = _raise(state, ("review", "medium"))
        why.append(f"{forest_pct:.0f}% of the plot was forest in {cut_year}, with no loss since")

    year = pd.to_numeric(attrs.get("year_established"), errors="coerce")
    if pd.notna(year):
        if year > cut_year and forest_pct >= scfg["forest_overlap_pct"]:
            state = _raise(state, ("possible loss", "high"))
            why.append(f"declared established in {int(year)} (after the cut-off) on land that was forest in {cut_year}")
        if year <= cut_year and big(m["loss_post2020_ha"]):
            state = _raise(state, ("review", "medium"))
            why.append(f"declared established in {int(year)} but tree loss recorded after {cut_year}")

    prev = str(attrs.get("previous_land_use", "")).lower()
    if prev in {"cropland", "fallow", "bush"} and forest_pct >= 50:
        state = _raise(state, ("review", "medium"))
        why.append(f"declared previous use '{prev}' but {forest_pct:.0f}% was forest in {cut_year}")

    if reserve_pct >= scfg["reserve_overlap_pct"]:
        state = _raise(state, ("review", "high"))
        why.append(f"{reserve_pct:.0f}% of the plot lies inside a protected area ({reserve_names})")

    return state[0], state[1], "; ".join(why)