"""Stream raster windows from the web, clip to a study area, cache, and compute zonal areas.

Only the pixels inside each district's bounding box are downloaded (windowed reads),
never the whole global tile.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.merge import merge
from rasterio.warp import reproject

# GDAL settings that make remote GeoTIFF reads faster and more robust
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="50000000",
)


# ------------------------------------------------------------------ tiles and sources
def padded_bounds(bounds, pad: float = 0.002):
    w, s, e, n = bounds
    return (w - pad, s - pad, e + pad, n + pad)


def hansen_tile_ids(bounds) -> list[str]:
    """Hansen tiles are 10x10 degrees, named by their top-left corner, e.g. 10N_030E, 00N_030E."""
    west, south, east, north = bounds
    lefts = range(math.floor(west / 10) * 10, math.floor(east / 10) * 10 + 1, 10)
    tops = range(math.ceil(south / 10) * 10, math.ceil(north / 10) * 10 + 1, 10)
    ids = []
    for t in tops:
        for l in lefts:
            ids.append(f"{abs(t):02d}{'N' if t >= 0 else 'S'}_{abs(l):03d}{'E' if l >= 0 else 'W'}")
    return ids


def hansen_urls(layer: str, bounds, base_url: str, version: str) -> list[str]:
    return [f"{base_url}/{version}/Hansen_{version}_{layer}_{t}.tif" for t in hansen_tile_ids(bounds)]


def worldcover_sources(bounds, year: int, stac_url: str) -> list[str]:
    """Signed COG links for ESA WorldCover tiles that touch `bounds` (Microsoft Planetary Computer)."""
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(stac_url, modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=["esa-worldcover"], bbox=bounds,
                            datetime=f"{year}-01-01/{year}-12-31")
    items = list(search.items())
    if not items:
        raise ValueError(f"No WorldCover tiles found for {year} in {bounds}")
    return [it.assets["map"].href for it in items]


# ------------------------------------------------------------------ streaming and caching
def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def stream_window(sources: list[str], bounds, nodata=0, resampling=Resampling.nearest):
    """Read ONLY the window `bounds` from one or more rasters, mosaicking if the area spans tiles.

    Returns (array, transform, crs).
    """
    with rasterio.Env(**GDAL_ENV):
        opened = []
        try:
            for s in sources:
                src = rasterio.open(s)
                if _overlaps(src.bounds, bounds):
                    opened.append(src)
                else:
                    src.close()
            if not opened:
                raise ValueError(f"None of the sources overlap {bounds}")
            arr, transform = merge(opened, bounds=bounds, nodata=nodata, indexes=[1],
                                   resampling=resampling)
            crs = opened[0].crs
        finally:
            for src in opened:
                src.close()
    return arr[0], transform, crs


def write_tif(path: Path, arr, transform, crs, nodata=None, dtype=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                   dtype=dtype or arr.dtype, crs=crs, transform=transform, nodata=nodata,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def read_tif(path: Path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs


def get_or_stream(cache_path: Path, sources, bounds, nodata=0,
                  resampling=Resampling.nearest, refresh: bool = False):
    """Stream once, then reuse the small clipped copy on later runs."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return read_tif(cache_path)
    arr, tr, crs = stream_window(sources, bounds, nodata=nodata, resampling=resampling)
    write_tif(cache_path, arr, tr, crs, nodata=nodata)
    return arr, tr, crs


def align_to(arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs,
             resampling=Resampling.nearest, dtype=None, dst_nodata=0):
    """Resample a raster onto another raster's grid (e.g. 10 m WorldCover onto the 30 m Hansen grid)."""
    dst = np.full(dst_shape, dst_nodata, dtype=dtype or arr.dtype)
    reproject(source=arr, destination=dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs, resampling=resampling,
              dst_nodata=dst_nodata)
    return dst


# ------------------------------------------------------------------ zonal statistics
def inside_mask(geom, shape, transform):
    """True for pixels inside the district polygon (the 'clip')."""
    return geometry_mask([geom], out_shape=shape, transform=transform, invert=True)


def label_grid(gdf, id_col: str, shape, transform):
    """Burn zone ids (1..n) onto the grid so statistics need no per-polygon loops."""
    pairs = ((g, int(i)) for g, i in zip(gdf.geometry, gdf[id_col]))
    return rasterize(pairs, out_shape=shape, transform=transform, fill=0, dtype="int32")


def pixel_area_ha(transform, shape) -> np.ndarray:
    """Area of each pixel row in hectares (pixels shrink slightly with latitude)."""
    R = 6371008.8
    dlon = math.radians(abs(transform.a))
    rows = np.arange(shape[0])
    lat_top = transform.f + rows * transform.e
    lat_bot = lat_top + transform.e
    area_m2 = R**2 * dlon * np.abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bot)))
    return area_m2 / 10_000.0          # one value per row


def weights_2d(area_row: np.ndarray, shape) -> np.ndarray:
    return np.broadcast_to(area_row[:, None], shape)


def zonal_area_ha(labels, mask, weights, n_zones: int) -> np.ndarray:
    sel = mask & (labels > 0)
    return np.bincount(labels[sel], weights=weights[sel], minlength=n_zones + 1)[1:]


def zonal_mean(labels, values, n_zones: int) -> np.ndarray:
    ok = (labels > 0) & np.isfinite(values)
    tot = np.bincount(labels[ok], weights=values[ok], minlength=n_zones + 1)[1:]
    cnt = np.bincount(labels[ok], minlength=n_zones + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return tot / cnt


# ------------------------------------------------------------------ optional: Sentinel-2 NDVI
def ndvi_composite(bounds, date_range: str, max_cloud: float, epsg: int,
                   resolution_m: float, stac_url: str, max_scenes: int = 40):
    """Stream Sentinel-2 L2A red (B04), NIR (B08) and the scene classification (SCL),
    mask clouds, compute NDVI per scene, and take the median.

    bounds : (west, south, east, north) in degrees
    epsg   : UTM zone to compute in (metres)
    Returns: ndvi (float32), n_obs (uint16), affine transform, list of scenes used
    """
    import dask
    import numpy as np
    import odc.stac
    import planetary_computer
    import pystac_client
    import xarray as xr
    from affine import Affine

    catalog = pystac_client.Client.open(stac_url, modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=["sentinel-2-l2a"], bbox=bounds, datetime=date_range,
                                query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    if not items:
        raise ValueError("No Sentinel-2 scenes found. Widen date_range or raise max_cloud_cover.")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    items = items[:max_scenes]
    print(f"Using {len(items)} Sentinel-2 scenes (cloud cover "
          f"{items[0].properties['eo:cloud_cover']:.0f}% to {items[-1].properties['eo:cloud_cover']:.0f}%)")

    ds = odc.stac.load(items, bands=["B04", "B08", "SCL"], bbox=bounds,
                       crs=f"EPSG:{epsg}", resolution=resolution_m,
                       groupby="solar_day", resampling="nearest",
                       chunks={"x": 512, "y": 512})

    # Scenes processed from about 25 Jan 2022 carry a +1000 offset in the digital numbers.
    offset = xr.where(ds["time"] >= np.datetime64("2022-01-25"), 1000.0, 0.0)
    red = (ds["B04"].astype("float32") - offset) / 10000.0
    nir = (ds["B08"].astype("float32") - offset) / 10000.0

    # keep clear pixels: SCL 4 vegetation, 5 bare soil, 6 water (drops cloud, shadow, cirrus)
    valid = (ds["B04"] > 0) & (ds["B08"] > 0) & ds["SCL"].isin([4, 5, 6]) & ((nir + red) > 0)
    ndvi = ((nir - red) / (nir + red)).where(valid)

    ndvi = ndvi.chunk({"time": -1})                 # median needs one chunk along time
    ndvi_med = ndvi.median("time", skipna=True)
    n_obs = ndvi.notnull().sum("time")
    ndvi_med, n_obs = dask.compute(ndvi_med, n_obs)  # one pass over the streamed data

    x, y = ds["x"].values, ds["y"].values
    rx, ry = float(x[1] - x[0]), float(y[1] - y[0])
    transform = Affine(rx, 0, float(x[0]) - rx / 2, 0, ry, float(y[0]) - ry / 2)

    scenes = [{"id": it.id, "date": str(it.datetime.date()),
               "cloud_cover": it.properties.get("eo:cloud_cover")} for it in items]
    return ndvi_med.values.astype("float32"), n_obs.values.astype("uint16"), transform, scenes