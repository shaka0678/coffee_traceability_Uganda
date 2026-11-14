"""EUDR-style plot validation: geometry, EUDR rules, duplicates and overlaps, admin consistency,
completeness. Works on any plot file that follows the plot template."""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.validation import explain_validity

from .boundaries import normalise_name

SEVERITY = {
    "missing_geometry": "critical", "coords_invalid": "critical", "outside_uganda": "critical",
    "outside_study_area": "critical", "invalid_geometry": "critical", "needs_polygon": "critical",
    "duplicate_plot_id": "critical", "duplicate_farmer_geometry": "critical", "district_mismatch": "critical",
    "overlap_other_farmer": "review", "area_tiny": "review", "area_huge": "review",
    "area_mismatch": "review", "low_precision": "review", "subcounty_mismatch": "review",
    "gps_poor": "review", "date_invalid": "review", "low_completeness": "review",
}

# which flag should catch each injected synthetic error (used to test the validator)
TRUTH_TO_FLAG = {
    "missing_coords": "missing_geometry", "coords_swapped": "outside_uganda",
    "point_over_4ha": "needs_polygon", "invalid_polygon": "invalid_geometry",
    "tiny_area": "area_tiny", "huge_area": "area_huge",
    "overlap_other_farmer": "overlap_other_farmer", "duplicate_farmer_plot": "duplicate_farmer_geometry",
    "outside_district": "outside_study_area", "area_mismatch": "area_mismatch",
    "low_precision": "low_precision", "admin_mismatch": "subcounty_mismatch",
    "future_date": "date_invalid",
}

MISSING_TOKENS = {"", "nan", "nat", "<na>", "unknown"}


def missing_template_columns(plots, tcfg):
    return [c for c in tcfg["core"] + tcfg["recommended"] if c not in plots.columns]


def decimals(x) -> int:
    if pd.isna(x):
        return 0
    s = f"{abs(float(x)):.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def _utm_epsg(arr):
    c = shapely.centroid(arr)
    zone = np.floor((shapely.get_x(c) + 180) / 6).astype(int) + 1
    return np.where(shapely.get_y(c) >= 0, 32600, 32700) + zone


def polygon_area_ha(gv: gpd.GeoSeries) -> pd.Series:
    """Area in hectares, each polygon measured in the UTM zone of its own location."""
    out = pd.Series(np.nan, index=gv.index)
    sel = gv.notna() & gv.geom_type.isin(["Polygon", "MultiPolygon", "GeometryCollection"])
    if not sel.any():
        return out
    arr = gv[sel].to_numpy()
    idx = gv[sel].index
    epsg = _utm_epsg(arr)
    for e in np.unique(epsg):
        m = epsg == e
        out.loc[idx[m]] = gpd.GeoSeries(list(arr[m]), index=idx[m], crs=4326).to_crs(int(e)).area.to_numpy() / 10_000
    return out


def _filled(s: pd.Series) -> pd.Series:
    return s.notna() & ~s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)


def completeness(p, tcfg) -> pd.Series:
    def share(cols):
        total = sum((_filled(p[c]).astype(int) for c in cols if c in p.columns),
                    start=pd.Series(0, index=p.index))
        return total / len(cols)          # a column missing from the file counts as empty
    w = tcfg["weights"]
    return (w["core"] * share(tcfg["core"]) + w["recommended"] * share(tcfg["recommended"])) * 100


def validate(plots, uganda, study, subcounties, vcfg, eudr_cfg, tcfg):
    """Return (plots with flags, status and completeness, long table of issues).

    uganda, study : shapely geometries in EPSG:4326 (national outline, union of study districts)
    """
    p = plots.reset_index(drop=True).copy()
    p["declared_area_ha"] = pd.to_numeric(p["declared_area_ha"], errors="coerce")
    glist = list(p.geometry)
    g = gpd.GeoSeries(glist, index=p.index, crs=4326)
    detail: dict[str, dict[int, str]] = {}

    def note(check, i, text):
        detail.setdefault(check, {})[int(i)] = text

    has = g.notna() & ~g.is_empty
    valid = pd.Series(True, index=p.index)
    valid[has] = g[has].is_valid
    for i in p.index[has & ~valid]:
        note("invalid_geometry", i, explain_validity(glist[i]))
        glist[i] = shapely.make_valid(glist[i])
    gv = gpd.GeoSeries(glist, index=p.index, crs=4326)          # repaired copy used for measuring

    lat = pd.to_numeric(p["lat"], errors="coerce")
    lon = pd.to_numeric(p["lon"], errors="coerce")
    f = pd.DataFrame(index=p.index)

    # 1. coordinates and geometry
    f["missing_geometry"] = ~has | lat.isna() | lon.isna()
    f["coords_invalid"] = lat.notna() & lon.notna() & ((lat.abs() > 90) | (lon.abs() > 180) | ((lat == 0) & (lon == 0)))
    f["invalid_geometry"] = has & ~valid
    dec = pd.concat([lat.map(decimals), lon.map(decimals)], axis=1).min(axis=1)
    f["low_precision"] = lat.notna() & lon.notna() & (dec < vcfg["min_decimals"])

    # 2. EUDR size rules
    gtype = gv.geom_type
    is_poly = gtype.isin(["Polygon", "MultiPolygon", "GeometryCollection"])
    is_point = gtype == "Point"
    area_c = polygon_area_ha(gv)
    declared = p["declared_area_ha"]
    area_rule = area_c.where(is_poly, declared)
    f["needs_polygon"] = is_point & (declared >= eudr_cfg["polygon_threshold_ha"])
    f["area_tiny"] = has & (area_rule < vcfg["min_area_ha"])
    f["area_huge"] = has & (area_rule > vcfg["max_area_ha"])
    gap = (area_c - declared).abs() / declared
    f["area_mismatch"] = is_poly & declared.gt(0) & (gap * 100 > vcfg["area_mismatch_pct"])
    for i in p.index[f["area_tiny"] | f["area_huge"]]:
        note("area_tiny" if f.at[i, "area_tiny"] else "area_huge", i, f"{area_rule[i]:.3f} ha")
    for i in p.index[f["area_mismatch"]]:
        note("area_mismatch", i, f"declared {declared[i]:.2f} ha, mapped {area_c[i]:.2f} ha")

    # 3. location: Uganda, study districts, declared vs mapped district and subcounty
    pts = shapely.point_on_surface(gv[has].to_numpy())
    shapely.prepare(uganda)
    shapely.prepare(study)
    in_ug = pd.Series(False, index=p.index)
    in_study = pd.Series(False, index=p.index)
    in_ug[has] = shapely.contains(uganda, pts)
    in_study[has] = shapely.contains(study, pts)
    f["outside_uganda"] = has & ~in_ug
    f["outside_study_area"] = has & in_ug & ~in_study

    gp = gpd.GeoDataFrame({"_i": p.index[has]}, geometry=pts, crs=4326)
    j = gpd.sjoin(gp, subcounties[["district", "subcounty", "geometry"]], how="left", predicate="within")
    j = j[~j["_i"].duplicated()].set_index("_i")
    p["district_spatial"] = j["district"].reindex(p.index)
    p["subcounty_spatial"] = j["subcounty"].reindex(p.index)
    nd, ns = p["district"].map(normalise_name), p["subcounty"].map(normalise_name)
    f["district_mismatch"] = p["district_spatial"].notna() & (nd != p["district_spatial"].map(normalise_name))
    f["subcounty_mismatch"] = (p["subcounty_spatial"].notna() & p["subcounty"].notna() & ~f["district_mismatch"]
                               & (ns != p["subcounty_spatial"].map(normalise_name)))
    for i in p.index[f["district_mismatch"]]:
        note("district_mismatch", i, f"declared {p.at[i, 'district']}, located in {p.at[i, 'district_spatial']}")
    for i in p.index[f["subcounty_mismatch"]]:
        note("subcounty_mismatch", i, f"declared {p.at[i, 'subcounty']}, located in {p.at[i, 'subcounty_spatial']}")

    # 4. collection quality
    acc = pd.to_numeric(p["gps_accuracy_m"], errors="coerce")
    f["gps_poor"] = acc > vcfg["max_gps_accuracy_m"]
    d = pd.to_datetime(p["collection_date"], errors="coerce")
    f["date_invalid"] = (d > pd.Timestamp.today().normalize()) | (p["collection_date"].notna() & d.isna())

    # 5. duplicates and overlaps
    f["duplicate_plot_id"] = p["plot_id"].notna() & p["plot_id"].duplicated(keep=False)
    f["duplicate_farmer_geometry"] = False
    f["overlap_other_farmer"] = False

    pol = gv[is_poly & has]
    if len(pol) > 1:
        left = gpd.GeoDataFrame({"pid": pol.index.to_numpy(), "farmer": p.loc[pol.index, "farmer_id"].to_numpy()},
                                geometry=pol.to_numpy(), crs=4326).reset_index(drop=True)
        pairs = gpd.sjoin(left, left[["geometry"]], predicate="intersects")
        pairs = pairs[pairs.index < pairs["index_right"]]
        if len(pairs):
            ia, ib = pairs.index.to_numpy(), pairs["index_right"].to_numpy()
            ga, gb = left.geometry.iloc[ia].to_numpy(), left.geometry.iloc[ib].to_numpy()
            inter = shapely.area(shapely.intersection(ga, gb))
            aa, ab = shapely.area(ga), shapely.area(gb)
            iou = inter / (aa + ab - inter)
            small = inter / np.minimum(aa, ab)
            pid_a, pid_b = left["pid"].to_numpy()[ia], left["pid"].to_numpy()[ib]
            same = left["farmer"].to_numpy()[ia] == left["farmer"].to_numpy()[ib]
            ov_ha = small * np.minimum(area_c.loc[pid_a].to_numpy(), area_c.loc[pid_b].to_numpy())
            dup = same & (iou >= vcfg["duplicate_iou"])
            ovl = (~same) & (small * 100 >= vcfg["overlap_min_pct"]) & (ov_ha >= vcfg["overlap_min_ha"])
            for k in np.nonzero(dup)[0]:
                for me, other in ((pid_a[k], pid_b[k]), (pid_b[k], pid_a[k])):
                    f.loc[me, "duplicate_farmer_geometry"] = True
                    note("duplicate_farmer_geometry", me, f"same farmer and geometry as {p.at[other, 'plot_id']}")
            for k in np.nonzero(ovl)[0]:
                for me, other in ((pid_a[k], pid_b[k]), (pid_b[k], pid_a[k])):
                    f.loc[me, "overlap_other_farmer"] = True
                    note("overlap_other_farmer", me,
                         f"overlaps {p.at[other, 'plot_id']} by {ov_ha[k]:.2f} ha ({small[k] * 100:.0f}% of smaller plot)")
    pm = is_point & has
    if pm.any():
        key = pd.DataFrame({"f": p.loc[pm, "farmer_id"], "a": lat[pm].round(4), "b": lon[pm].round(4)})
        dupe = key.duplicated(keep=False)
        f.loc[dupe[dupe].index, "duplicate_farmer_geometry"] = True

    # 6. completeness and status
    p["completeness_pct"] = completeness(p, tcfg).round(1)
    f["low_completeness"] = p["completeness_pct"] < vcfg.get("min_completeness_pct", 80)
    f = f.fillna(False).astype(bool)
    p["area_mapped_ha"] = area_c.round(3)

    crit = [c for c in f.columns if SEVERITY.get(c) == "critical"]
    rev = [c for c in f.columns if SEVERITY.get(c) == "review"]
    p["n_critical"] = f[crit].sum(axis=1)
    p["n_review"] = f[rev].sum(axis=1)
    p["status"] = np.select([p["n_critical"] > 0, p["n_review"] > 0], ["reject", "review"], "pass")

    rows = []
    for c in f.columns:
        for i in f.index[f[c]]:
            rows.append({"plot_id": p.at[i, "plot_id"], "district": p.at[i, "district"],
                         "subcounty": p.at[i, "subcounty"], "enumerator_id": p.at[i, "enumerator_id"],
                         "check": c, "severity": SEVERITY.get(c, "review"),
                         "detail": detail.get(c, {}).get(int(i), "")})
    issues = pd.DataFrame(rows, columns=["plot_id", "district", "subcounty", "enumerator_id",
                                         "check", "severity", "detail"])
    return p.join(f), issues


def summarise(v: pd.DataFrame, by: str) -> pd.DataFrame:
    out = v.groupby(by, dropna=False).agg(
        plots=("plot_id", "count"),
        passed=("status", lambda s: int((s == "pass").sum())),
        review=("status", lambda s: int((s == "review").sum())),
        rejected=("status", lambda s: int((s == "reject").sum())),
        mean_completeness_pct=("completeness_pct", "mean"),
        declared_area_ha=("declared_area_ha", "sum"))
    out["pass_rate_pct"] = out["passed"] / out["plots"] * 100
    return out.round(1).reset_index()