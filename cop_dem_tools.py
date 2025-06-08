from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import math
import itertools
import requests
import folium
import numpy as np
import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box, mapping
from rich.console import Console
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds, transform as window_transform

console = Console()

COP_DEM_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
ESRI_WORLD_IMAGERY = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

def cop_tile_url(lat: float, lon: float) -> str:
    lat_sw, lon_sw = math.floor(lat), math.floor(lon)
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    lat_s = f"{abs(lat_sw):02d}_00"
    lon_s = f"{abs(lon_sw):03d}_00"
    stem = f"Copernicus_DSM_COG_30_{ns}{lat_s}_{ew}{lon_s}_DEM"
    return f"{COP_DEM_BASE}/{stem}/{stem}.tif"

def fetch_cop_tiles(bbox: Tuple[float, float, float, float], out_dir: Path) -> List[Path]:
    """Download COP-DEM90 tiles intersecting ``bbox`` into ``out_dir``."""
    xmin, ymin, xmax, ymax = bbox
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_rng = range(math.floor(ymin), math.ceil(ymax) + 1)
    lon_rng = range(math.floor(xmin), math.ceil(xmax) + 1)

    tif_paths: List[Path] = []
    for lat, lon in itertools.product(lat_rng, lon_rng):
        url = cop_tile_url(lat, lon)
        local = out_dir / Path(url).name
        if not local.exists():
            console.log(f"Fetching {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local, "wb") as fp:
                    for chunk in r.iter_content(131_072):
                        fp.write(chunk)
        tif_paths.append(local)

    if not tif_paths:
        raise RuntimeError("No Copernicus DEM tiles fetched; check bbox.")
    return tif_paths

def mosaic_cop_tiles(tif_paths: List[Path], out_path: Path, bbox: Tuple[float, float, float, float]) -> Path:
    """Merge ``tif_paths`` into ``out_path``. The bounding box is stored as a tag."""
    if out_path.exists():
        console.log(f"[green]Using existing DEM mosaic → {out_path}")
        return out_path

    srcs = [rio.open(p) for p in tif_paths]
    mosaic, transform = merge(srcs, method="min")
    meta = srcs[0].meta.copy()
    meta.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        compress="deflate",
        tiled=True,
    )

    with rio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)
        dst.update_tags(bbox=",".join(map(str, bbox)))
    for s in srcs:
        s.close()

    console.log(f"[cyan]Wrote mosaic to {out_path}")
    return out_path

def crop_to_bbox(mosaic_path: Path, bbox: Tuple[float, float, float, float], out_path: Path) -> Path:
    """Crop ``mosaic_path`` to ``bbox`` and save to ``out_path``."""
    if out_path.exists():
        console.log(f"[green]Using existing cropped DEM → {out_path}")
        return out_path

    geom = [mapping(box(*bbox))]
    with rio.open(mosaic_path) as src:
        window = from_bounds(*bbox, transform=src.transform)
        out_img = src.read(window=window)
        out_transform = window_transform(window, src.transform)
        # out_img, out_transform = mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            height=out_img.shape[1],
            width=out_img.shape[2],
            transform=out_transform,
        )

    with rio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_img)

    console.log(f"[cyan]Wrote cropped DEM to {out_path}")
    return out_path

def _dem_to_overlay(src: rio.DatasetReader, cmap: str = "terrain") -> tuple[np.ndarray, list]:
    """Return (RGBA array, bounds) ready for Folium with ``mercator_project=True``."""
    arr = src.read(1).astype(float)
    nodata = src.nodata if src.nodata is not None else np.nan
    arr[arr == nodata] = np.nan

    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    norm = (arr - vmin) / max(vmax - vmin, 1)

    cm = plt.get_cmap(cmap)
    rgba = cm(norm)
    rgba[..., -1] = np.where(np.isnan(arr), 0, rgba[..., -1])

    bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
    return rgba, bounds


def save_dem_png(dem_path: Path, out_path: Path, cmap: str = "terrain") -> Path:
    """Save a DEM as a transparent PNG suitable for Folium overlays."""
    with rio.open(dem_path) as src:
        rgba, _ = _dem_to_overlay(src, cmap=cmap)
    img = (rgba * 255).round().astype(np.uint8)
    Image.fromarray(img).save(out_path)
    console.log(f"[cyan]Wrote {out_path}")
    return out_path


def save_surface_png(
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
    out_path: Path,
    *,
    cmap: str = "terrain",
) -> Path:
    """Save an interpolated surface as a PNG overlay.

    Parameters
    ----------
    xi, yi : 2-D arrays
        Longitude and latitude grids as returned by :func:`interpolate_bare_earth`.
    zi : 2-D array
        Interpolated elevations.
    out_path : Path
        Where to save the PNG file.
    cmap : str, optional
        Matplotlib colormap name for shading.
    """

    arr = zi.astype(float)
    arr[arr == -9999] = np.nan  # legacy nodata guard

    # ‼️  flip in the y–axis so north ends up at the top
    arr = np.flipud(arr)

    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    norm = (arr - vmin) / max(vmax - vmin, 1)

    cm = plt.get_cmap(cmap)
    rgba = cm(norm)
    rgba[..., -1] = np.where(np.isnan(arr), 0, rgba[..., -1])

    img = (rgba * 255).round().astype(np.uint8)
    Image.fromarray(img).save(out_path)
    console.log(f"[cyan]Wrote {out_path}")
    return out_path


def save_residual_png(
    rrm: np.ndarray,
    out_path: Path,
    *,
    cmap: str = "RdBu_r",
) -> Path:
    """Save a residual relief model as a PNG overlay.

    Colors are scaled symmetrically around zero so that elevations and
    depressions share the same intensity.
    """

    arr = rrm.astype(float)
    arr[arr == -9999] = np.nan

    # vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    # norm = (arr - vmin) / max(vmax - vmin, 1)
    vmax = np.nanmax(arr)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1
    norm = (arr + vmax) / (2 * vmax)

    cm = plt.get_cmap(cmap)
    rgba = cm(norm)
    rgba[..., -1] = np.where(np.isnan(arr), 0, rgba[..., -1])

    img = (rgba * 255).round().astype(np.uint8)
    Image.fromarray(img).save(out_path)
    console.log(f"[cyan]Wrote {out_path}")
    return out_path


def save_anomaly_points_png(
    anomalies,
    xi: np.ndarray,
    yi: np.ndarray,
    out_path: Path,
    *,
    color: tuple[int, int, int, int] = (255, 255, 0, 255),
    radius: int = 4,
) -> Path:
    """Save detected anomalies as small circle markers on a transparent PNG."""

    img = Image.new("RGBA", (xi.shape[1], yi.shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if anomalies is not None and not anomalies.empty:
        xmin, xmax = xi[0, 0], xi[0, -1]
        ymin, ymax = yi.min(), yi.max()

        for _, row in anomalies.iterrows():
            lon = row.geometry.x
            lat = row.geometry.y
            px = int(round((lon - xmin) / (xmax - xmin) * (xi.shape[1] - 1)))
            py = int(round((ymax - lat) / (ymax - ymin) * (yi.shape[0] - 1)))
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=color,
                outline=(0, 0, 0, 255),
            )

    img.save(out_path)
    console.log(f"[cyan]Wrote {out_path}")
    return out_path


def dem_bounds(dem_path: Path) -> tuple[float, float, float, float]:
    """Return the bounding box of a DEM as (xmin, ymin, xmax, ymax)."""
    with rio.open(dem_path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)

def dem_map(mosaic_path: Path, crop_path: Path, bbox: tuple[float, float, float, float], zoom_start: int = 9) -> folium.Map:
    lon_c = (bbox[0] + bbox[2]) / 2
    lat_c = (bbox[1] + bbox[3]) / 2

    m = folium.Map(location=[lat_c, lon_c], zoom_start=zoom_start, control_scale=True, tiles=None)

    folium.TileLayer(
        tiles=ESRI_WORLD_IMAGERY,
        name="Esri World Imagery",
        attr=(
            "Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
            "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        ),
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
    ).add_to(m)

    for tif, name, z in [
        (mosaic_path, "DEM mosaic", 1),
        (crop_path, "DEM crop", 2),
    ]:
        with rio.open(tif) as src:
            rgba, bounds = _dem_to_overlay(src)

        folium.raster_layers.ImageOverlay(
            name=name,
            image=rgba,
            bounds=bounds,
            mercator_project=True,
            opacity=1,
            interactive=False,
            zindex=1000 + z,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
