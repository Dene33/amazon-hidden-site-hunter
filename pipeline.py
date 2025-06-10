"""Refactored processing pipeline with configurable steps via YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import yaml
from rich.console import Console

from cop_dem_tools import (
    crop_to_bbox,
    fetch_cop_tiles,
    mosaic_cop_tiles,
    save_anomaly_points_png,
    save_dem_png,
    save_residual_png,
    save_surface_png,
)

# Reuse core functions from existing scripts
from detect_hidden_sites import (
    detect_anomalies,
    fetch_gedi_points,
    interpolate_bare_earth,
    krige_bare_earth,
    residual_relief,
)

# Reuse visualization helpers
from preview_pipeline import create_interactive_map, visualize_gedi_points
from sentinel_utils import (
    bounds,
    compute_kndvi,
    download_bands,
    read_band,
    save_index_png,
    save_true_color,
    resize_image,
    search_sentinel2_item,
)

console = Console()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------


def step_fetch_sentinel(
    cfg: Dict[str, Any], bbox: Tuple[float, float, float, float], base: Path
) -> Dict[str, Path]:
    """Fetch Sentinel-2 imagery, save bands and visualisations."""
    if not cfg.get("enabled", True):
        return {}
    console.rule("[bold green]Fetch Sentinel-2 imagery")
    item = search_sentinel2_item(
        bbox, cfg.get("time_start"), cfg.get("time_end"), cfg.get("max_cloud", 20)
    )
    if item is None:
        console.log("[red]No Sentinel-2 images found")
        return {}
    bands = cfg.get("bands", ["B02", "B03", "B04", "B08"])
    paths = download_bands(item, bands, ensure_dir(base / "sentinel2"))

    if paths:
        sb = bounds(next(iter(paths.values())))
        paths["bounds"] = sb

    if cfg.get("visualize", True) and {"B02", "B03", "B04"}.issubset(paths):
        dpi = cfg.get("dpi", 150)
        b02 = read_band(paths["B02"], bbox=sb)
        b03 = read_band(paths["B03"], bbox=sb)
        b04 = read_band(paths["B04"], bbox=sb)
        save_true_color(b02, b03, b04, base / "sentinel_true_color.jpg", dpi=dpi)
        console.log(f"[cyan]Wrote {base / 'sentinel_true_color.jpg'}")
        resize_image(base / "sentinel_true_color.jpg")

        b02_c = read_band(paths["B02"], bbox=bbox)
        b03_c = read_band(paths["B03"], bbox=bbox)
        b04_c = read_band(paths["B04"], bbox=bbox)
        save_true_color(
            b02_c, b03_c, b04_c, base / "sentinel_true_color_clean.png", dpi=dpi
        )
        console.log(f"[cyan]Wrote {base / 'sentinel_true_color_clean.png'}")

    if cfg.get("visualize", True) and {"B04", "B08"}.issubset(paths):
        red = read_band(paths["B04"], bbox=sb)
        nir = read_band(paths["B08"], bbox=sb)
        kndvi = compute_kndvi(red, nir)
        save_index_png(kndvi, base / "sentinel_kndvi.jpg", dpi=dpi)
        console.log(f"[cyan]Wrote {base / 'sentinel_kndvi.jpg'}")
        resize_image(base / "sentinel_kndvi.jpg")

        red_c = read_band(paths["B04"], bbox=bbox)
        nir_c = read_band(paths["B08"], bbox=bbox)
        kndvi_c = compute_kndvi(red_c, nir_c)
        save_index_png(kndvi_c, base / "sentinel_kndvi_clean.png", dpi=dpi)
        console.log(f"[cyan]Wrote {base / 'sentinel_kndvi_clean.png'}")

    return paths


def step_fetch_data(
    cfg: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    base: Path,
) -> Tuple[Path | None, Any | None]:
    """Fetch Copernicus tiles and GEDI points if enabled."""

    if not cfg.get("enabled", True):
        return None, None

    if len(bbox) != 4:
        raise ValueError("bbox must be provided with 4 coordinates")

    dem_path = None
    gedi = None

    if cfg.get("fetch_cop_tiles", {}).get("enabled", True):
        console.rule("[bold green]Fetch Copernicus DEM")
        tiles = fetch_cop_tiles(tuple(bbox), base)
        mosaic = mosaic_cop_tiles(tiles, base / "cop90_mosaic.tif", bbox)
        crop = crop_to_bbox(mosaic, bbox, base / "cop90_crop.tif")
        dem_path = crop
        if cfg.get("visualize", True):
            save_dem_png(mosaic, base / "1_copernicus_dem_mosaic_hillshade.png")
            save_dem_png(crop, base / "1_copernicus_dem_crop_hillshade.png")

    if cfg.get("fetch_gedi_points", {}).get("enabled", True):
        console.rule("[bold green]Fetch GEDI footprints")
        gedi_cfg = cfg.get("fetch_gedi_points", {})
        gedi = fetch_gedi_points(
            tuple(bbox),
            time_start=gedi_cfg.get("time_start"),
            time_end=gedi_cfg.get("time_end"),
            cache_dir=ensure_dir(base / "gedi_cache"),
        )
        if cfg.get("visualize", True) and gedi is not None:
            points: List[Tuple[float, float, float]] = [
                (geom.y, geom.x, elev)
                for geom, elev in zip(gedi.geometry, gedi["elev_lowestmode"])
            ]
            visualize_gedi_points(points, bbox, base)
            console.log(f"[cyan]Wrote {Path(base) / "2_gedi_points_clean.png"}")

    return dem_path, gedi


def step_bare_earth(
    cfg: Dict[str, Any], bbox: Tuple[float, float, float, float], gedi, base: Path
):
    if not cfg.get("enabled", True) or gedi is None:
        return None
    console.rule("[bold green]Bare-earth surface")
    # xi, yi, zi = interpolate_bare_earth(
    #     gedi,
    #     bbox,
    #     cfg.get("resolution", 0.0002695),
    # )
    xi, yi, zi = krige_bare_earth(
        gedi,
        bbox,
        res=0.0002695,  # 15 m grid, or keep 30 m if you like
        variogram_model="spherical",  # or "spherical", "gaussian"
        nlags=30,
        detrend=True,  # often improves short-range detail
    )
    if cfg.get("visualize", True):
        save_surface_png(
            xi,
            yi,
            zi,
            base / "3_bare_earth_surface_clean.png",
        )
    return xi, yi, zi


def step_residual_relief(cfg: Dict[str, Any], bearth, dem_path: Path, base: Path):
    if not cfg.get("enabled", True) or bearth is None or dem_path is None:
        return None
    console.rule("[bold green]Residual relief")
    xi, yi, zi = bearth
    rrm = residual_relief((xi, yi, zi), dem_path)
    if cfg.get("visualize", True):
        save_residual_png(
            rrm,
            base / "4_residual_relief_clean.png",
        )
    return rrm


def step_detect_anomalies(cfg: Dict[str, Any], rrm, xi, yi, base: Path):
    if not cfg.get("enabled", True) or rrm is None:
        return None
    console.rule("[bold green]Detect anomalies")
    anomalies = detect_anomalies(
        rrm,
        xi,
        yi,
        sigma=cfg.get("sigma", 2),
        amp_thresh=cfg.get("amp_thresh", 1.0),
        size_thresh_m=cfg.get("size_thresh_m", 200),
        debug_dir=(base / "debug") if cfg.get("debug", False) else None,
    )
    if cfg.get("visualize", True):
        save_anomaly_points_png(
            anomalies,
            xi,
            yi,
            base / "5_detected_anomalies_clean.png",
        )
    if cfg.get("save_json", True):
        out = base / "anomalies.geojson"
        anomalies.to_file(out, driver="GeoJSON")
        console.print(f"[bold cyan]Saved anomalies → {out}")
    return anomalies


def _write_obj_mesh(
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
    path: Path,
    *,
    cmap: str = "terrain",
    scale: float = 1.0,
) -> None:
    """Write a regular grid surface to ``path`` as an OBJ mesh with colors.

    Coordinates are converted from lon/lat degrees to metres and centred so the
    object imports nicely into Blender.
    """

    arr = zi.astype(float)
    arr[arr == -9999] = np.nan

    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    norm = (arr - vmin) / max(vmax - vmin, 1)

    cm = plt.get_cmap(cmap)
    colors = (cm(norm)[..., :3]).astype(float)

    nrows, ncols = arr.shape
    xi = np.asarray(xi).reshape(nrows, ncols)
    yi = np.asarray(yi).reshape(nrows, ncols)

    # lon/lat degrees -> metres (approximate)
    lat0 = float(np.nanmean(yi))
    lon_scale = 111_320 * np.cos(np.radians(lat0))
    lat_scale = 111_320
    x = (xi - np.nanmean(xi)) * lon_scale * scale
    y = (yi - np.nanmean(yi)) * lat_scale * scale
    z = (arr - np.nanmean(arr)) * scale

    idx_map = np.full((nrows, ncols), -1, dtype=int)
    verts: list[str] = []
    faces: list[str] = []
    idx = 1
    for i in range(nrows):
        for j in range(ncols):
            zv = z[i, j]
            if not np.isfinite(zv):
                continue
            r, g, b = colors[i, j]
            # OBJ format uses Y-up. Store height in Y so Blender imports with Z-up
            verts.append(
                f"v {x[i, j]:.3f} {zv:.3f} {-y[i, j]:.3f} {r:.3f} {g:.3f} {b:.3f}"
            )
            idx_map[i, j] = idx
            idx += 1

    for i in range(nrows - 1):
        for j in range(ncols - 1):
            v1 = idx_map[i, j]
            v2 = idx_map[i, j + 1]
            v3 = idx_map[i + 1, j + 1]
            v4 = idx_map[i + 1, j]
            if v1 > 0 and v2 > 0 and v3 > 0:
                faces.append(f"f {v1} {v2} {v3}")
            if v1 > 0 and v3 > 0 and v4 > 0:
                faces.append(f"f {v1} {v3} {v4}")

    with open(path, "w") as f:
        f.write("\n".join(verts + faces))


def _save_xyz_points(
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
    path: Path,
    *,
    scale: float = 1.0,
) -> None:
    """Save grid points as an XYZ text file in metres, centred around 0."""

    nrows, ncols = zi.shape
    xi = np.asarray(xi).reshape(nrows, ncols)
    yi = np.asarray(yi).reshape(nrows, ncols)

    lat0 = float(np.nanmean(yi))
    lon_scale = 111_320 * np.cos(np.radians(lat0))
    lat_scale = 111_320
    x = (xi - np.nanmean(xi)) * lon_scale * scale
    y = (yi - np.nanmean(yi)) * lat_scale * scale
    z = (zi.astype(float) - np.nanmean(zi)) * scale

    arr = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    mask = np.isfinite(arr[:, 2])
    np.savetxt(path, arr[mask], fmt="%.3f %.3f %.3f")


def step_export_obj(cfg: Dict[str, Any], bearth, dem_path: Path, base: Path):
    """Export bare-earth and DEM surfaces as OBJ meshes with vertex colors."""

    if not cfg.get("enabled", True) or bearth is None or dem_path is None:
        return

    console.rule("[bold green]Export surfaces as OBJ")
    cmap = cfg.get("cmap", "terrain")
    scale = cfg.get("scale", 1.0)

    xi, yi, zi = bearth
    out_be = base / cfg.get("bare_earth_file", "bare_earth.obj")
    _write_obj_mesh(xi, yi, zi, out_be, cmap=cmap, scale=scale)
    console.log(f"[cyan]Wrote {out_be}")

    with rio.open(dem_path) as src:
        arr = src.read(1)
        rows, cols = np.meshgrid(
            np.arange(src.height), np.arange(src.width), indexing="ij"
        )
        lon, lat = rio.transform.xy(src.transform, rows, cols, offset="center")
        lon = np.array(lon).reshape(arr.shape)
        lat = np.array(lat).reshape(arr.shape)

    out_dem = base / cfg.get("dem_file", "dem_crop.obj")
    _write_obj_mesh(lon, lat, arr, out_dem, cmap=cmap, scale=scale)
    console.log(f"[cyan]Wrote {out_dem}")


def step_export_xyz(cfg: Dict[str, Any], bearth, dem_path: Path, base: Path):
    """Export bare-earth and DEM surfaces as XYZ point clouds."""

    if not cfg.get("enabled", True) or bearth is None or dem_path is None:
        return

    console.rule("[bold green]Export surfaces as XYZ")
    scale = cfg.get("scale", 1.0)

    xi, yi, zi = bearth
    out_be = base / cfg.get("bare_earth_file", "bare_earth.xyz")
    _save_xyz_points(xi, yi, zi, out_be, scale=scale)
    console.log(f"[cyan]Wrote {out_be}")

    with rio.open(dem_path) as src:
        arr = src.read(1)
        rows, cols = np.meshgrid(
            np.arange(src.height), np.arange(src.width), indexing="ij"
        )
        lon, lat = rio.transform.xy(src.transform, rows, cols, offset="center")
        lon = np.array(lon).reshape(arr.shape)
        lat = np.array(lat).reshape(arr.shape)

    out_dem = base / cfg.get("dem_file", "dem_crop.xyz")
    _save_xyz_points(lon, lat, arr, out_dem, scale=scale)
    console.log(f"[cyan]Wrote {out_dem}")


def step_interactive_map(
    cfg: Dict[str, Any],
    points,
    anomalies,
    bbox,
    base: Path,
    sentinel_paths: Dict[str, Path] | None = None,
):
    if not cfg.get("enabled", True):
        return
    console.rule("[bold green]Create interactive map")
    include_data_vis = cfg.get("include_data_vis", False)
    create_interactive_map(
        points,
        anomalies,
        bbox,
        base,
        include_data_vis=include_data_vis,
        sentinel=sentinel_paths,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(config: Dict[str, Any]):
    base = ensure_dir(Path(config.get("out_dir", "pipeline_out")))
    bbox = tuple(config.get("bbox", []))
    if len(bbox) != 4:
        raise ValueError("bbox must be provided with 4 coordinates")
    # Step 1 – fetch data
    dem_path, gedi = step_fetch_data(config.get("step1", {}), bbox, base)

    # Step 1b – Sentinel-2 imagery
    sentinel_paths = step_fetch_sentinel(config.get("sentinel", {}), bbox, base)

    # Step 2 – bare-earth surface
    bearth = step_bare_earth(config.get("step2", {}), bbox, gedi, base)

    # Step 3 – residual relief
    if bearth is not None:
        rrm = step_residual_relief(config.get("step3", {}), bearth, dem_path, base)
    else:
        rrm = None

    # Step 4 – detect anomalies
    if bearth is not None:
        anomalies = step_detect_anomalies(
            config.get("step4", {}),
            rrm,
            bearth[0],
            bearth[1],
            base,
        )
    else:
        anomalies = None

    # Prepare points for interactive map
    points = None
    # if gedi is not None:
    #     points = [
    #         (geom.y, geom.x, elev)
    #         for geom, elev in zip(gedi.geometry, gedi["elev_lowestmode"])
    #     ]

    # Step 5 – interactive map
    step_interactive_map(
        config.get("step5", {}),
        points,
        anomalies,
        bbox,
        base,
        sentinel_paths,
    )

    # Step 6 – export surfaces for Blender
    step_export_obj(config.get("step6", {}), bearth, dem_path, base)

    # Step 7 – export XYZ point clouds
    step_export_xyz(config.get("step7", {}), bearth, dem_path, base)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run configurable hidden-site pipeline")
    ap.add_argument("config", help="Path to YAML config file")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_pipeline(cfg)
