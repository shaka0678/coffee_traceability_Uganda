#note these are synthentic plot for our case study and they're fake
""" Every record is fake.

Plots are placed using the forest rasters from notebook 02, so each forest-context scenario
exists on the ground: clean cropland, near forest, inside forest, loss after 2020 inside
forest, loss after 2020 outside forest. Known errors are injected and recorded in
`_truth_issue`, so the validator can be tested against the right answers.
"""


from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy import ndimage
from shapely import affinity
from shapely.geometry import Point, Polygon

SCENARIOS = ["clean", "near_forest", "inside_forest", "loss_in_forest", "loss_outside"]
BLANKABLE = ["cooperative_or_group", "year_established", "previous_land_use", "land_tenure_type",
             "tenure_proof_available", "gps_accuracy_m", "collection_method", "variety_or_clone",
             "tree_count", "shade_type", "intercrops", "est_annual_yield_kg",
             "certification_scheme", "fertiliser_use", "hired_labour"]


# ------------------------------------------------------------------ where to put plots
def scenario_masks(forest, cls, wc, pix_m):
    inside = cls != 255
    ok = inside & (cls == 0)
    if forest.any():
        d_out = ndimage.distance_transform_edt(~forest) * pix_m   # non-forest pixel -> nearest forest
        d_in = ndimage.distance_transform_edt(forest) * pix_m     # forest pixel -> nearest non-forest
    else:
        d_out, d_in = np.full(forest.shape, np.inf), np.zeros(forest.shape)
    clean = ok & ~forest & (wc == 40) & (d_out > 500)             # cropland, far from forest
    if not clean.any():
        clean = ok & ~forest
    return {
        "clean": clean,
        "near_forest": ok & ~forest & (d_out > 30) & (d_out < 400),
        "inside_forest": ok & forest & (d_in > 90),
        "loss_in_forest": inside & (cls == 1),
        "loss_outside": inside & (cls == 2),
    }


def scenario_counts(n, mix, masks):
    counts = {s: int(round(n * mix.get(s, 0))) for s in SCENARIOS}
    counts["clean"] = max(0, counts["clean"] + n - sum(counts.values()))
    for s in SCENARIOS[1:]:
        if counts[s] > 0 and not masks[s].any():
            print(f"  note: no '{s}' pixels in this district; moving {counts[s]} plots to 'clean'")
            counts["clean"] += counts[s]
            counts[s] = 0
    return counts


def sample_centres(mask, tr, n, rng):
    rows, cols = np.nonzero(mask)
    if n == 0 or len(rows) == 0:
        return np.empty((0, 2))
    k = rng.choice(len(rows), size=n, replace=len(rows) < n)
    x = tr.c + (cols[k] + 0.5) * tr.a
    y = tr.f + (rows[k] + 0.5) * tr.e
    return np.column_stack([x, y])


def make_plot_polygon(cx, cy, area_ha, rng):
    """Irregular, rotated plot of roughly `area_ha`, centred on (cx, cy) in metres."""
    area = area_ha * 10_000
    aspect = rng.uniform(0.6, 1.8)
    w = (area * aspect) ** 0.5
    h = area / w
    pts = [(-w/2, -h/2), (0, -h/2), (w/2, -h/2), (w/2, 0), (w/2, h/2), (0, h/2), (-w/2, h/2), (-w/2, 0)]
    jit = rng.normal(0, 0.05, size=(len(pts), 2)) * np.array([w, h])
    poly = Polygon([(x + dx, y + dy) for (x, y), (dx, dy) in zip(pts, jit)])
    if not poly.is_valid:
        poly = poly.convex_hull
    poly = affinity.rotate(poly, rng.uniform(0, 180), origin=(0, 0))
    return affinity.translate(poly, cx, cy)


# ------------------------------------------------------------------ injected errors
def inject_errors(df, counts, rng, sub_names, district_utm):
    """Edit df in place. Geometries are in UTM metres. Every edited plot gets a `_truth_issue`."""
    free = set(df.index)

    def pick(n, polygon_only=False):
        pool = [i for i in free if not polygon_only or df.at[i, "geom"].geom_type == "Polygon"]
        if not pool or n <= 0:
            return []
        chosen = [int(i) for i in rng.choice(pool, size=min(n, len(pool)), replace=False)]
        free.difference_update(chosen)
        return chosen

    def mark(i, issue):
        df.at[i, "_truth_issue"] = issue

    def inherit(dst, src):
        for col in ("subcounty", "_truth_scenario"):
            df.at[dst, col] = df.at[src, col]

    for i in pick(counts.get("invalid_polygon", 0), True):                 # bow-tie polygon
        minx, miny, maxx, maxy = df.at[i, "geom"].bounds
        bow = Polygon([(minx, miny), (maxx, maxy), (maxx, miny), (minx, maxy)])
        df.at[i, "geom"] = bow
        df.at[i, "declared_area_ha"] = round(shapely.make_valid(bow).area / 10_000, 2)
        mark(i, "invalid_polygon")

    for i in pick(counts.get("tiny_area", 0), True):
        g = df.at[i, "geom"]
        target = rng.uniform(0.01, 0.04) * 10_000
        f = (target / g.area) ** 0.5
        df.at[i, "geom"] = affinity.scale(g, f, f, origin="centroid")
        df.at[i, "declared_area_ha"] = round(target / 10_000, 3)
        mark(i, "tiny_area")

    for i in pick(counts.get("huge_area", 0), True):
        g = df.at[i, "geom"]
        target = rng.uniform(60, 90) * 10_000
        f = (target / g.area) ** 0.5
        df.at[i, "geom"] = affinity.scale(g, f, f, origin="centroid")
        df.at[i, "declared_area_ha"] = round(target / 10_000, 1)
        mark(i, "huge_area")

    for i in pick(counts.get("point_over_4ha", 0), True):                  # large plot captured as a point
        df.at[i, "geom"] = df.at[i, "geom"].centroid
        df.at[i, "geometry_type"] = "point"
        df.at[i, "collection_method"] = "point_capture"
        df.at[i, "declared_area_ha"] = round(float(rng.uniform(4.5, 9.0)), 1)
        mark(i, "point_over_4ha")

    for _ in range(counts.get("overlap_other_farmer", 0)):                  # plot over another farmer's plot
        src, dst = pick(1, True), pick(1, True)
        if not src or not dst:
            break
        s, d = src[0], dst[0]
        g = df.at[s, "geom"]
        shift = 0.3 * g.area ** 0.5
        df.at[d, "geom"] = affinity.translate(g, shift, shift * 0.5)
        df.at[d, "declared_area_ha"] = df.at[s, "declared_area_ha"]
        if df.at[d, "farmer_id"] == df.at[s, "farmer_id"]:
            df.at[d, "farmer_id"] += "-B"
        inherit(d, s)
        mark(d, "overlap_other_farmer")

    for _ in range(counts.get("duplicate_farmer_plot", 0)):                 # same farmer, same plot, entered twice
        src, dst = pick(1, True), pick(1, True)
        if not src or not dst:
            break
        s, d = src[0], dst[0]
        df.at[d, "geom"] = affinity.translate(df.at[s, "geom"], 2.0, 2.0)
        for col in ("farmer_id", "farmer_name", "declared_area_ha"):
            df.at[d, col] = df.at[s, col]
        inherit(d, s)
        mark(d, "duplicate_farmer_plot")

    for i in pick(counts.get("outside_district", 0)):
        g = df.at[i, "geom"]
        for dx, dy in [(35_000, 0), (-35_000, 0), (0, 35_000), (0, -35_000), (60_000, 60_000)]:
            moved = affinity.translate(g, dx, dy)
            if not district_utm.intersects(moved):
                df.at[i, "geom"] = moved
                mark(i, "outside_district")
                break

    for i in pick(counts.get("area_mismatch", 0), True):
        df.at[i, "declared_area_ha"] = round(df.at[i, "declared_area_ha"] * float(rng.choice([0.4, 2.2])), 2)
        mark(i, "area_mismatch")

    if len(sub_names) > 1:
        for i in pick(counts.get("admin_mismatch", 0)):
            others = [s for s in sub_names if s != df.at[i, "subcounty"]]
            df.at[i, "subcounty"] = str(rng.choice(others))
            mark(i, "admin_mismatch")
    elif counts.get("admin_mismatch", 0):
        print("  note: only one subcounty polygon in this district, so admin_mismatch cases were skipped")

    for i in pick(counts.get("future_date", 0)):
        df.at[i, "collection_date"] = "2027-03-15"
        mark(i, "future_date")

    for op in ("missing_coords", "coords_swapped", "low_precision"):          # applied after conversion to lat/lon
        for i in pick(counts.get(op, 0)):
            df.at[i, "_coord_op"] = op
            mark(i, op)
    return df


# ------------------------------------------------------------------ one district
def _centroid(g):
    return (g if g.is_valid else shapely.make_valid(g)).centroid


def generate_district(dname, subs, forest, cls, wc, tr, utm_epsg, syn, rng, today_year=2026):
    prefix = dname[:3].upper()
    n_target = int(syn["plots_per_district"])
    pix_m = abs(tr.a) * 111_320
    masks = scenario_masks(forest, cls, wc, pix_m)
    counts = scenario_counts(n_target, syn["scenario_mix"], masks)

    xy, scen = [], []
    for s, k in counts.items():
        c = sample_centres(masks[s], tr, k, rng)
        xy.append(c)
        scen += [s] * len(c)
    xy = np.vstack(xy)
    order = rng.permutation(len(xy))
    xy, scen = xy[order], np.array(scen)[order]
    n = len(xy)

    cxy = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=4326).to_crs(utm_epsg)
    cx, cy = cxy.x.to_numpy(), cxy.y.to_numpy()

    area = np.clip(rng.lognormal(np.log(syn["size_median_ha"]), syn["size_sigma"], n), 0.1, 15)
    is_point = (area < 4) & (rng.random(n) > syn["share_polygon_if_small"])
    geoms = [Point(cx[i], cy[i]) if is_point[i] else make_plot_polygon(cx[i], cy[i], area[i], rng)
             for i in range(n)]
    true_area = np.array([area[i] if is_point[i] else geoms[i].area / 10_000 for i in range(n)])

    # subcounty (nearest polygon, so edge pixels never fall through)
    sub_utm = subs[["subcounty", "geometry"]].to_crs(utm_epsg).reset_index(drop=True)
    sub_utm["sub_idx"] = np.arange(len(sub_utm))
    pts = gpd.GeoDataFrame(geometry=cxy, crs=utm_epsg)
    j = gpd.sjoin_nearest(pts, sub_utm, how="left")
    j = j[~j.index.duplicated()].sort_index()
    subcounty, sub_idx = j["subcounty"].to_numpy(), j["sub_idx"].to_numpy().astype(int)

    # attributes
    loss_sc = np.isin(scen, ["loss_in_forest", "loss_outside"])
    forest_sc = np.isin(scen, ["inside_forest", "loss_in_forest"])
    year_est = rng.integers(2000, 2021, n)
    year_est = np.where(loss_sc & (rng.random(n) < 0.6), rng.integers(2021, today_year, n), year_est)
    prev = rng.choice(["cropland", "bush", "fallow", "forest", "wetland"], n, p=[0.6, 0.15, 0.15, 0.05, 0.05])
    prev = np.where(forest_sc & (rng.random(n) < 0.7), "cropland", prev)   # declared cropland on mapped forest
    tenure = rng.choice(["customary", "mailo", "freehold", "leasehold", "other"], n,
                        p=[0.45, 0.4, 0.05, 0.05, 0.05])
    proof = np.where(np.isin(tenure, ["mailo", "freehold", "leasehold"]), rng.random(n) < 0.85, rng.random(n) < 0.3)
    age = today_year - year_est
    scheme = rng.choice(["none", "Rainforest Alliance", "4C", "Fairtrade"], n, p=[0.7, 0.15, 0.1, 0.05])
    cert_status = np.where(scheme == "none", "n/a", rng.choice(["certified", "in_process"], n, p=[0.7, 0.3]))
    enum_ids = [f"E{k:02d}" for k in range(1, 7)]
    enum = np.array(enum_ids)[(sub_idx * 2 + rng.integers(0, 2, n)) % len(enum_ids)]
    farmer_idx = rng.integers(1, int(n * 0.85) + 1, size=n)
    cdate = np.array((pd.Timestamp("2026-06-01") + pd.to_timedelta(rng.integers(0, 150, n), unit="D"))
                     .strftime("%Y-%m-%d"))

    df = pd.DataFrame({
        "plot_id": [f"SYN-{prefix}-{i + 1:04d}" for i in range(n)],
        "farmer_id": [f"F-{prefix}-{k:04d}" for k in farmer_idx],
        "farmer_name": [f"Synthetic Farmer {k:04d}" for k in farmer_idx],
        "cooperative_or_group": rng.choice([f"Coop-{prefix}-{c}" for c in "ABCD"], n),
        "enumerator_id": enum,
        "collection_date": cdate,
        "survey_version": "v1",
        "district": dname,
        "subcounty": subcounty,
        "geometry_type": np.where(is_point, "point", "polygon"),
        "gps_accuracy_m": np.clip(rng.lognormal(np.log(6), 0.5, n), 2, 40).round(1),
        "collection_method": np.where(is_point, "point_capture",
                                      rng.choice(["walked_boundary", "drawn_on_map"], n, p=[0.7, 0.3])),
        "declared_area_ha": np.clip(np.round(true_area * rng.normal(1, 0.08, n), 2), 0.01, None),
        "year_established": year_est,
        "previous_land_use": prev,
        "land_tenure_type": tenure,
        "tenure_proof_available": np.where(proof, "yes", "no"),
        "land_dispute": np.where(rng.random(n) < 0.04, "yes", "no"),
        "consent_flag": np.where(rng.random(n) < 0.97, "yes", "no"),
        "coffee_type": "Robusta",
        "variety_or_clone": rng.choice(["improved clone", "local seedling", "unknown"], n, p=[0.35, 0.55, 0.10]),
        "crop_age_years": age,
        "tree_count": (true_area * rng.uniform(500, 1100, n)).round().astype(int),
        "shade_type": rng.choice(["full sun", "light shade", "dense shade", "agroforestry"], n,
                                 p=[0.2, 0.4, 0.2, 0.2]),
        "intercrops": rng.choice(["banana", "banana;beans", "maize", "none", "banana;cassava"], n),
        "stumped_or_rehabilitated": np.where(rng.random(n) < 0.15, "yes", "no"),
        "wilt_disease_observed": rng.choice(["none", "low", "high"], n, p=[0.6, 0.3, 0.1]),
        "est_annual_yield_kg": np.round(true_area * rng.uniform(300, 900, n) * np.where(age < 3, 0.2, 1.0)),
        "certification_scheme": scheme,
        "certification_status": cert_status,
        "fertiliser_use": np.where(rng.random(n) < 0.35, "yes", "no"),
        "hired_labour": np.where(rng.random(n) < 0.40, "yes", "no"),
        "synthetic_flag": True,
        "_truth_scenario": scen,
        "_truth_issue": "none",
        "_coord_op": "",
        "geom": geoms,
    })

    district_utm = shapely.union_all(sub_utm.geometry.values)
    df = inject_errors(df, syn["injected_errors"], rng, sorted(subs["subcounty"].unique()), district_utm)

    # convert to lat/lon, then apply coordinate-level errors
    cent = gpd.GeoSeries([_centroid(g) for g in df["geom"]], crs=utm_epsg).to_crs(4326)
    geoms4326 = gpd.GeoSeries(df["geom"].tolist(), crs=utm_epsg).to_crs(4326).tolist()
    df["lon"] = np.round(cent.x.to_numpy(), 6)
    df["lat"] = np.round(cent.y.to_numpy(), 6)
    for i, op in df["_coord_op"].items():
        if op == "missing_coords":
            geoms4326[i] = None
            df.loc[i, ["lat", "lon"]] = np.nan
        elif op == "coords_swapped":
            a, b = df.at[i, "lat"], df.at[i, "lon"]
            df.at[i, "lat"], df.at[i, "lon"] = b, a
            geoms4326[i] = Point(df.at[i, "lon"], df.at[i, "lat"])
        elif op == "low_precision":
            df.at[i, "lat"], df.at[i, "lon"] = round(df.at[i, "lat"], 3), round(df.at[i, "lon"], 3)

    # enumerators differ in data quality: some leave more blanks
    rates = dict(zip(enum_ids, rng.permutation(np.linspace(0.01, 0.15, len(enum_ids)))))
    p_blank = df["enumerator_id"].map(rates).astype(float).to_numpy()
    for col in BLANKABLE:
        mask = rng.random(n) < p_blank
        df.loc[mask, col] = np.nan

    out = df.drop(columns=["geom", "_coord_op"])
    return gpd.GeoDataFrame(out, geometry=gpd.GeoSeries(geoms4326, crs=4326), crs=4326)
    