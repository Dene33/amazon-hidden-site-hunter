"""Refactored processing pipeline with configurable steps via YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
import base64
from glob import glob
import os
import re
from chatgpt_parser import _parse_chatgpt_detections
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import yaml
from rich.console import Console

from cop_dem_tools import (
    crop_to_bbox,
    fetch_cop_tiles,
    fetch_srtm_tiles,
    fetch_aw3d_tiles,
    mosaic_cop_tiles,
    mosaic_srtm_tiles,
    mosaic_aw3d_tiles,
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
    compute_ndvi,
    compute_ndmi,
    compute_msi,
    download_bands,
    read_band,
    read_band_like,
    mask_clouds,
    apply_mask,
    cloud_mask,
    hollstein_cloud_mask,
    save_mask_png,
    save_index_png,
    save_true_color,
    resize_image,
    search_sentinel2_item,
)

console = Console()

# Default prompt for GPT analysis with bbox placeholders
ARCHAEO_PROMPT = (
    "You are Archaeo-GPT. Input: 1) bbox [$xmin, $ymin, $xmax, $ymax]"
    " (xmin,ymin,xmax,ymax); 2) possible rasters $rasters same grid;"
    " Workflow: check CRS; rescale layers; flag NDVI\u00b11.5\u03c3 with moisture;"
    " extract micro-relief & \u0394DEM; RX\u22653\u03c3; fuse masks, score clusters;"
    " return human readable description of findings with lat, lon coordinates of"
    " detections of interest. Output: Every 'header' (first line) of each"
    " detection should be formated like this: `ID 1  $coordinate S, $coordinate W"
    "   score = 9.4 `"
)

# Mapping from raster labels used in the prompt to image base names
RASTER_IMAGE_MAP = {
    "DEM_0": "1c_aw3d30_crop_hillshade",
    "DEM_1": "1b_srtm_crop_hillshade",
    "\u0394DEM": "4_residual_relief_clean",
    "NDVI": "sentinel_kndvi_high_clean",
    "SMI": "sentinel_msi_high_clean",
    "NDMI": "sentinel_ndmi_high_clean",
    "RX": "sentinel_ndvi_ratio_clean",
}


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
    cfg: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    base: Path,
) -> Dict[str, Any]:
    """
    Download Sentinel-2 imagery and (optionally) produce cloud masks,
    true-colour previews, kNDVI products and two-date comparison layers.

    Returns at least {"bounds": <scene-bounds>}.  Extra keys are harmless.
    """
    if not cfg.get("enabled", True):
        return {}

    console.rule("[bold green]Fetch Sentinel-2 imagery")

    # ----------------------------------------------------------------– set-up
    dpi        = cfg.get("dpi", 150)
    visualise  = cfg.get("visualize", True)
    out_dir    = (base / "sentinel2").resolve()
    source_dirs = [Path(p) for p in cfg.get("source_dirs", [])]
    dl_dir      = source_dirs[0] if source_dirs else out_dir
    resize_vis = cfg.get("resize_vis", False)
    save_full  = cfg.get("save_full", False)          # <-- now used!

    high_cfg, low_cfg = cfg.get("high_stress"), cfg.get("low_stress")
    has_two_periods   = bool(high_cfg and low_cfg)

    # ---------------------------------------------------------------- helpers
    def _wanted_bands(item) -> list[str]:
        core = ["B02", "B03", "B04", "B08", "B11"]
        return core + (["SCL"] if "scl" in item.get("assets", {}) else
                       ["B01", "B05", "B06", "B07", "B8A", "B09", "B11"])

    #  produce cloud mask + true colour + kNDVI for a single period ----------
    def _make_products(paths: Dict[str, Path], label: str, ref_b04: Path):
        """Return kNDVI and related products for a single period.

        ``paths`` must include at least the Sentinel-2 band paths used by the
        pipeline.  ``ref_b04`` is the B04 band from the *high* period and is
        used as the spatial reference for every band so all outputs across
        periods share the same grid.
        """

        sb = paths["bounds"]

        # ---- cloud mask
        def _mask_for(area_bbox):
            if "SCL" in paths:
                scl = read_band_like(paths["SCL"], ref_b04,
                                     bbox=area_bbox, scale=1.0)
                return cloud_mask(scl)
            # Hollstein fall-back
            needed = ["B01", "B02", "B03", "B05", "B06",
                      "B07", "B8A", "B09", "B11"]
            extras = [
                read_band_like(paths[b], ref_b04, bbox=area_bbox)
                for b in needed if b in paths
            ]
            return hollstein_cloud_mask(*extras)

        # We always need the mask & arrays for index maths,
        # but we only *save* the “full scene” visualisations if save_full=True
        mask = _mask_for(sb)

        if visualise and save_full:
            mask_path = base / f"sentinel_cloud_mask_{label}.png"
            save_mask_png(mask, mask_path, dpi=dpi, bbox=sb)
            console.log(
                f"[cyan]Wrote {mask_path} "
                f"({np.count_nonzero(mask):,} masked px)"
            )

        # ---- prepare RED/NIR + RGB
        def _masked(band: str, area_bbox):
            """Read ``band`` cropped to ``area_bbox`` and apply ``mask``.

            ``read_band`` can return arrays with slightly different shapes for
            the same bbox across bands due to rounding behaviour when
            reprojecting.  This can lead to shape mismatches when applying a
            mask derived from another band.  To guarantee consistent shapes we
            resample every band to match ``ref_b04`` using ``read_band_like``.
            """

            arr = read_band_like(paths[band], ref_b04, bbox=area_bbox)
            return apply_mask(mask, arr)[0] if visualise else arr

        red, nir = (_masked("B04", sb), _masked("B08", sb))
        b02, b03, b04 = (
            _masked("B02", sb),
            _masked("B03", sb),
            _masked("B04", sb),
        )

        if "B11" in paths:
            swir = read_band_like(paths["B11"], ref_b04, bbox=sb, scale=1.0)
            swir_mask = cloud_mask(swir)
            swir = apply_mask(swir_mask, swir)[0] if visualise else swir
        else:
            swir = None

        # ---- save true colour & kNDVI for whole scene
        kndvi = compute_kndvi(red, nir)
        kndvi_clean: np.ndarray | None = None
        ndmi: np.ndarray | None = None
        msi: np.ndarray | None = None
        ndmi_clean: np.ndarray | None = None
        msi_clean: np.ndarray | None = None

        if swir is not None:
            ndmi = compute_ndmi(nir, swir)
            msi = compute_msi(nir, swir)

        if visualise and save_full:
            tc_full = base / f"sentinel_true_color_{label}.jpg"
            save_true_color(b02, b03, b04, tc_full, dpi=dpi, bbox=sb)
            if resize_vis:
                resize_image(tc_full)
                console.log(f"[cyan]Resized {tc_full}")
            console.log(f"[cyan]Wrote {tc_full}")

            ndvi_full = base / f"sentinel_kndvi_{label}.png"
            save_index_png(kndvi, ndvi_full, dpi=dpi, bbox=sb)
            if resize_vis:
                resize_image(ndvi_full)
                console.log(f"[cyan]Resized {ndvi_full}")
            console.log(f"[cyan]Wrote {ndvi_full}")

            if swir is not None:
                msi_full = base / f"sentinel_msi_{label}.png"
                ndmi_full = base / f"sentinel_ndmi_{label}.png"
                save_index_png(msi, msi_full, dpi=dpi, bbox=sb)
                save_index_png(ndmi, ndmi_full, dpi=dpi, bbox=sb)
                if resize_vis:
                    resize_image(msi_full)
                    resize_image(ndmi_full)
                    console.log(f"[cyan]Resized {msi_full} and {ndmi_full}")
                console.log(f"[cyan]Wrote {msi_full} and {ndmi_full}")

        # ---- same products but strictly inside user bbox (“_clean”)
        if visualise:
            mask_c = _mask_for(bbox)
            b02_c, b03_c, b04_c = (
                apply_mask(mask_c, read_band_like(paths["B02"], ref_b04, bbox=bbox))[0],
                apply_mask(mask_c, read_band_like(paths["B03"], ref_b04, bbox=bbox))[0],
                apply_mask(mask_c, read_band_like(paths["B04"], ref_b04, bbox=bbox))[0],
            )
            tc_clean = base / f"sentinel_true_color_{label}_clean.jpg"
            save_true_color(b02_c, b03_c, b04_c, tc_clean, dpi=dpi, gain=5, bbox=bbox)
            console.log(f"[cyan]Wrote {tc_clean}")

            red_c = apply_mask(mask_c, read_band_like(paths["B04"], ref_b04, bbox=bbox))[0]
            nir_c = apply_mask(mask_c, read_band_like(paths["B08"], ref_b04, bbox=bbox))[0]
            kndvi_c = compute_kndvi(red_c, nir_c)
            kndvi_clean = kndvi_c
            ndvi_clean = base / f"sentinel_kndvi_{label}_clean.png"
            save_index_png(kndvi_c, ndvi_clean, dpi=dpi, bbox=bbox)
            console.log(f"[cyan]Wrote {ndvi_clean}")

            if "B11" in paths:
                swir_c = apply_mask(mask_c, read_band_like(paths["B11"], ref_b04, bbox=bbox, scale=1.0))[0]
                ndmi_c = compute_ndmi(nir_c, swir_c)
                msi_c = compute_msi(nir_c, swir_c)
                msi_clean_path = base / f"sentinel_msi_{label}_clean.png"
                ndmi_clean_path = base / f"sentinel_ndmi_{label}_clean.png"
                save_index_png(msi_c, msi_clean_path, dpi=dpi, bbox=bbox)
                save_index_png(ndmi_c, ndmi_clean_path, dpi=dpi, bbox=bbox)
                msi_clean = msi_c
                ndmi_clean = ndmi_c
                console.log(f"[cyan]Wrote {msi_clean_path} and {ndmi_clean_path}")

        return kndvi, kndvi_clean, ndmi, ndmi_clean, msi, msi_clean

    # ----------------------------------------------------------------– work
    if has_two_periods:
        item_hi = search_sentinel2_item(
            bbox,
            high_cfg.get("time_start"),
            high_cfg.get("time_end"),
            cfg.get("max_cloud", 20),
        )
        grid_code = item_hi.get("properties", {}).get("grid:code") if item_hi else None
        # Search the low-stress period using the same MGRS tile as the high
        # period to guarantee identical spatial coverage.  Use the high
        # period's bounding box as the search area rather than the user AOI to
        # avoid returning imagery from a neighbouring tile if ``bbox`` straddles
        # multiple tiles.
        hi_bbox = tuple(item_hi.get("bbox", bbox)) if item_hi else bbox
        item_lo = search_sentinel2_item(
            hi_bbox,
            low_cfg.get("time_start"),
            low_cfg.get("time_end"),
            cfg.get("max_cloud", 20),
            grid_code=grid_code,
        )
        if item_hi is None or item_lo is None:
            console.log("[red]No Sentinel‑2 images found for both periods")
            return {}

        bands_hi = _wanted_bands(item_hi)
        paths_hi = download_bands(
            item_hi,
            bands_hi,
            ensure_dir(out_dir),
            source_dirs=source_dirs,
            download_dir=dl_dir,
        )
        bands_lo = _wanted_bands(item_lo)
        paths_lo = download_bands(
            item_lo,
            bands_lo,
            ensure_dir(out_dir),
            source_dirs=source_dirs,
            download_dir=dl_dir,
        )
        
        if not (paths_hi and paths_lo):
            console.log("[red]No Sentinel‑2 bands downloaded for **high** and **low**")
            return {}
        
        sb = bounds(next(iter(paths_hi.values())))
        paths_hi["bounds"] = sb        # bounds identical by construction
        paths_lo["bounds"] = sb        # bounds identical by construction
        result: Dict[str, Any] = {"bounds": sb}

        # ---- products for each period
        (
            ndvi_hi,
            ndvi_hi_c,
            ndmi_hi,
            ndmi_hi_c,
            msi_hi,
            msi_hi_c,
        ) = _make_products(paths_hi, "high", paths_hi["B04"])

        (
            ndvi_lo,
            ndvi_lo_c,
            ndmi_lo,
            ndmi_lo_c,
            msi_lo,
            msi_lo_c,
        ) = _make_products(paths_lo, "low", paths_hi["B04"])

        # ---- two-date comparisons
        if visualise and save_full:
            diff   = ndvi_hi - ndvi_lo
            ratio  = ndvi_hi / (ndvi_lo + 1e-6)
            diff_p = base / "sentinel_ndvi_diff.png"
            ratio_p= base / "sentinel_ndvi_ratio.png"
            save_index_png(diff,  diff_p,  dpi=dpi, bbox=sb)
            save_index_png(ratio, ratio_p, dpi=dpi, bbox=sb)
            if ndmi_hi is not None and ndmi_lo is not None:
                ndmi_diff = ndmi_hi - ndmi_lo
                ndmi_diff_p = base / "sentinel_ndmi_diff.png"
                save_index_png(ndmi_diff, ndmi_diff_p, dpi=dpi, bbox=sb)
            if msi_hi is not None and msi_lo is not None:
                msi_diff = msi_hi - msi_lo
                msi_diff_p = base / "sentinel_msi_diff.png"
                save_index_png(msi_diff, msi_diff_p, dpi=dpi, bbox=sb)
            if resize_vis:
                resize_image(diff_p)
                resize_image(ratio_p)
                console.log(f"[cyan]Resized {diff_p} and {ratio_p}")
                if ndmi_hi is not None and ndmi_lo is not None:
                    resize_image(ndmi_diff_p)
                if msi_hi is not None and msi_lo is not None:
                    resize_image(msi_diff_p)
            console.log("[cyan]Wrote two-date NDVI diff / ratio")
        
        if visualise:
            diff_c = ndvi_hi_c - ndvi_lo_c
            ratio_c = ndvi_hi_c / (ndvi_lo_c + 1e-6)
            diff_cp = base / "sentinel_ndvi_diff_clean.png"
            ratio_cp = base / "sentinel_ndvi_ratio_clean.png"
            save_index_png(diff_c, diff_cp, dpi=dpi, bbox=bbox)
            save_index_png(ratio_c, ratio_cp, dpi=dpi, bbox=bbox)
            if ndmi_hi_c is not None and ndmi_lo_c is not None:
                ndmi_diff_c = ndmi_hi_c - ndmi_lo_c
                ndmi_diff_cp = base / "sentinel_ndmi_diff_clean.png"
                save_index_png(ndmi_diff_c, ndmi_diff_cp, dpi=dpi, bbox=bbox)
            if msi_hi_c is not None and msi_lo_c is not None:
                msi_diff_c = msi_hi_c - msi_lo_c
                msi_diff_cp = base / "sentinel_msi_diff_clean.png"
                save_index_png(msi_diff_c, msi_diff_cp, dpi=dpi, bbox=bbox)
            console.log("[cyan]Wrote two-date NDVI diff / ratio (clean)")

        return result

    # ------------------------------- single-period fall-back (unchanged API)
    item = search_sentinel2_item(cfg)
    if item is None:
        console.log("[red]No Sentinel-2 images found")
        return {}

    bands = _wanted_bands(item)
    paths = download_bands(
        item,
        bands,
        ensure_dir(out_dir),
        source_dirs=source_dirs,
        download_dir=dl_dir,
    )

    if not paths:
        console.log("[red]No Sentinel-2 bands downloaded for **single**")
        return {}

    sb = bounds(next(iter(paths.values())))     
    paths["bounds"] = sb

    _make_products(paths, "single", paths["B04"])             # obeys `visualize` + `save_full`
    return {"bounds": sb}


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
        cop_cfg = cfg.get("fetch_cop_tiles", {})
        src_dirs = [Path(p) for p in cop_cfg.get("source_dirs", [])]
        out_dir = base
        download_dir = src_dirs[0] if src_dirs else out_dir
        tiles = fetch_cop_tiles(
            tuple(bbox),
            ensure_dir(out_dir),
            source_dirs=src_dirs,
            download_dir=download_dir,
        )
        mosaic = mosaic_cop_tiles(tiles, base / "cop90_mosaic.tif", bbox)
        crop = crop_to_bbox(mosaic, bbox, base / "cop90_crop.tif")
        dem_path = crop
        if cfg.get("visualize", True):
            mosaic_png = base / "1_copernicus_dem_mosaic_hillshade.png"
            if not mosaic_png.exists():
                save_dem_png(mosaic, mosaic_png)
                console.log(f"[cyan]Wrote {mosaic_png}")
            else:
                console.log(f"[cyan]Using existing {mosaic_png}")
            save_dem_png(crop, base / "1_copernicus_dem_crop_hillshade.png")

    if cfg.get("fetch_gedi_points", {}).get("enabled", True):
        console.rule("[bold green]Fetch GEDI footprints")
        gedi_cfg = cfg.get("fetch_gedi_points", {})
        src_dirs = [Path(p) for p in gedi_cfg.get("source_dirs", [])]
        cache_dir = base / "gedi_cache"
        gedi = fetch_gedi_points(
            tuple(bbox),
            time_start=gedi_cfg.get("time_start"),
            time_end=gedi_cfg.get("time_end"),
            cache_dir=ensure_dir(cache_dir),
            source_dirs=src_dirs,
            force_download=gedi_cfg.get("force_download", False),
        )
        if cfg.get("visualize", True) and gedi is not None:
            points: List[Tuple[float, float, float]] = [
                (geom.y, geom.x, elev)
                for geom, elev in zip(gedi.geometry, gedi["elev_lowestmode"])
            ]
            visualize_gedi_points(points, bbox, base)
            console.log(f"[cyan]Wrote {Path(base) / '2_gedi_points_clean.png'}")

    return dem_path, gedi


def step_fetch_srtm(
    cfg: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    base: Path,
) -> Path | None:
    """Fetch SRTM GL1 DEM tiles if enabled."""

    if not cfg.get("enabled", False):
        return None

    console.rule("[bold green]Fetch SRTM DEM")
    src_dirs = [Path(p) for p in cfg.get("source_dirs", [])]
    out_dir = base
    download_dir = src_dirs[0] if src_dirs else out_dir
    tiles = fetch_srtm_tiles(
        tuple(bbox),
        ensure_dir(out_dir),
        source_dirs=src_dirs,
        download_dir=download_dir,
    )
    mosaic = mosaic_srtm_tiles(tiles, base / "srtm_mosaic.tif", bbox)
    crop = crop_to_bbox(mosaic, bbox, base / "srtm_crop.tif")

    if cfg.get("visualize", True):
        mosaic_png = base / "1b_srtm_mosaic_hillshade.png"
        if not mosaic_png.exists():
            save_dem_png(mosaic, mosaic_png)
            console.log(f"[cyan]Wrote {mosaic_png}")
        else:
            console.log(f"[cyan]Using existing {mosaic_png}")
        save_dem_png(crop, base / "1b_srtm_crop_hillshade.png")

    return crop


def step_fetch_aw3d(
    cfg: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    base: Path,
) -> Path | None:
    """Fetch AW3D30 DEM tiles if enabled."""

    if not cfg.get("enabled", False):
        return None

    console.rule("[bold green]Fetch AW3D30 DEM")
    src_dirs = [Path(p) for p in cfg.get("source_dirs", [])]
    out_dir = base
    download_dir = src_dirs[0] if src_dirs else out_dir
    tiles = fetch_aw3d_tiles(
        tuple(bbox),
        ensure_dir(out_dir),
        source_dirs=src_dirs,
        download_dir=download_dir,
    )
    mosaic = mosaic_aw3d_tiles(tiles, base / "aw3d30_mosaic.tif", bbox)
    crop = crop_to_bbox(mosaic, bbox, base / "aw3d30_crop.tif")

    if cfg.get("visualize", True):
        mosaic_png = base / "1c_aw3d30_mosaic_hillshade.png"
        if not mosaic_png.exists():
            save_dem_png(mosaic, mosaic_png)
            console.log(f"[cyan]Wrote {mosaic_png}")
        else:
            console.log(f"[cyan]Using existing {mosaic_png}")
        save_dem_png(crop, base / "1c_aw3d30_crop_hillshade.png")

    return crop


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
            bbox=(xi[0, 0], yi.min(), xi[0, -1], yi.max()),
        )
    return xi, yi, zi


def step_residual_relief(
    cfg: Dict[str, Any], bearth, dem_path: Path | None, base: Path
):
    if not cfg.get("enabled", True) or bearth is None:
        return None

    dem_list = cfg.get("dems")
    paths: List[Path] = []
    if dem_list:
        mapping = {
            "cop": base / "cop90_crop.tif",
            "srtm": base / "srtm_crop.tif",
            "aw3d": base / "aw3d30_crop.tif",
        }
        for name in dem_list:
            path = mapping.get(name)
            if path is not None:
                paths.append(path)
    elif dem_path is not None:
        paths = [dem_path]
    else:
        return None

    console.rule("[bold green]Residual relief")
    xi, yi, zi = bearth
    rrm = residual_relief((xi, yi, zi), paths if len(paths) > 1 else paths[0])
    if cfg.get("visualize", True):
        save_residual_png(
            rrm,
            base / "4_residual_relief_clean.png",
            bbox=(xi[0, 0], yi.min(), xi[0, -1], yi.max()),
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
            bbox=(xi[0, 0], yi.min(), xi[0, -1], yi.max()),
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




def step_chatgpt(
    cfg: Dict[str, Any], bbox: Tuple[float, float, float, float], base: Path
) -> List[Tuple[float, float, float]]:
    """Send images to OpenAI's model for analysis."""

    if not cfg.get("enabled", False):
        return

    console.rule("[bold green]Analyse images with ChatGPT")

    names = cfg.get("images", [])
    prompt = cfg.get("prompt", ARCHAEO_PROMPT)
    model = cfg.get("model", "o3")
    log_level = cfg.get("log_level")

    if not names:
        console.log("[red]No images specified for ChatGPT analysis")
        return

    # Look for images within this bbox folder and the global out_dir
    search_dirs = [
        base,
        base / "debug"
    ]
    candidates: List[Path] = []
    exts = (".png", ".jpg", ".jpeg")
    for name in names:
        found = False
        for sdir in search_dirs:
            for ext in exts:
                pattern = str(sdir / f"**/{name}{ext}")
                matches = glob(pattern, recursive=True)
                if matches:
                    candidates.append(Path(matches[0]))
                    found = True
                    break
            if found:
                break
        if not found:
            console.log(f"[yellow]Image {name} not found")

    if log_level:
        os.environ["OPENAI_LOG"] = str(log_level)

    try:
        import openai
        if log_level:
            from openai._utils._logs import setup_logging as _setup_logging
            _setup_logging()
    except Exception as exc:  # pragma: no cover - openai may not be installed in tests
        console.log(f"[red]Failed to import openai: {exc}")
        return

    active_rasters = [label for label, img in RASTER_IMAGE_MAP.items() if img in names]
    if "$rasters" in prompt:
        prompt = prompt.replace("$rasters", ", ".join(active_rasters))
    prompt = (
        prompt.replace("$xmin", str(bbox[0]))
        .replace("$ymin", str(bbox[1]))
        .replace("$xmax", str(bbox[2]))
        .replace("$ymax", str(bbox[3]))
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    for img in candidates:
        if img.exists():
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img.suffix.lstrip('.')};base64,{b64}"
                    },
                }
            )
        else:
            console.log(f"[yellow]Image {img} not found")

    try:
        response = openai.chat.completions.create(model=model, messages=messages)
        result = response.choices[0].message.content if response.choices else ""
    except Exception as exc:  # pragma: no cover - network issues
        console.log(f"[red]OpenAI request failed: {exc}")
        return []

    detections = _parse_chatgpt_detections(result)

    result_path = base / "chatgpt_analysis.txt"
    with open(result_path, "w") as f:
        f.write(result)

    console.log(f"[cyan]Wrote {result_path}")

    return detections

def step_interactive_map(
    cfg: Dict[str, Any],
    points,
    anomalies,
    bbox,
    base: Path,
    sentinel_paths: Dict[str, Path] | None = None,
    chatgpt_points: List[Tuple[float, float, float]] | None = None,
):
    if not cfg.get("enabled", True):
        return
    console.rule("[bold green]Create interactive map")
    include_data_vis = cfg.get("include_data_vis", False)
    include_full_sentinel = cfg.get("include_full_sentinel", False)
    include_full_srtm = cfg.get("include_full_srtm", False)
    include_full_aw3d = cfg.get("include_full_aw3d", False)
    create_interactive_map(
        points,
        anomalies,
        bbox,
        base,
        include_data_vis=include_data_vis,
        sentinel=sentinel_paths,
        include_full_sentinel=include_full_sentinel,
        include_full_srtm=include_full_srtm,
        include_full_aw3d=include_full_aw3d,
        chatgpt_points=chatgpt_points,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _run_pipeline_single(
    config: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    base: Path,
) -> None:
    # Step 1 – fetch data
    dem_path, gedi = step_fetch_data(config.get("fetch_data", {}), bbox, base)

    # Optional – fetch SRTM DEM
    srtm_path = step_fetch_srtm(config.get("srtm", {}), bbox, base)

    # Optional – fetch AW3D30 DEM
    aw3d_path = step_fetch_aw3d(config.get("aw3d", {}), bbox, base)

    # Step 1b – Sentinel-2 imagery
    sentinel_paths = step_fetch_sentinel(config.get("sentinel", {}), bbox, base)

    # Step 2 – bare-earth surface
    bearth = step_bare_earth(config.get("bare_earth", {}), bbox, gedi, base)

    # Step 3 – residual relief
    if bearth is not None:
        rrm = step_residual_relief(
            config.get("residual_relief", {}), bearth, dem_path, base
        )
    else:
        rrm = None

    # Step 4 – detect anomalies
    if bearth is not None:
        anomalies = step_detect_anomalies(
            config.get("detect_anomalies", {}),
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

    # Step 5 – export surfaces for Blender
    step_export_obj(config.get("export_obj", {}), bearth, dem_path, base)

    # Step 6 – export XYZ point clouds
    step_export_xyz(config.get("export_xyz", {}), bearth, dem_path, base)


    # Step 7 – analyse imagery with ChatGPT
    chatgpt_point = step_chatgpt(config.get("chatgpt", {}), bbox, base)

    # Step 8 – interactive map
    step_interactive_map(
        config.get("interactive_map", {}),
        points,
        anomalies,
        bbox,
        base,
        sentinel_paths,
        chatgpt_points=chatgpt_point,
    )


def run_pipeline(config: Dict[str, Any]) -> None:
    out_dir = Path(config.get("out_dir", "pipeline_out"))
    bbox_cfg = config.get("bbox", [])
    if not bbox_cfg:
        raise ValueError("bbox must be provided")

    # Normalize to list of bboxes
    if isinstance(bbox_cfg[0], (list, tuple)):
        bboxes = [tuple(b) for b in bbox_cfg]
    else:
        bboxes = [tuple(bbox_cfg)]

    for bbox in bboxes:
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 coordinates")
        name = "_".join(str(c) for c in bbox)
        base = ensure_dir(out_dir / name)
        _run_pipeline_single(config, bbox, base)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run configurable hidden-site pipeline")
    ap.add_argument("config", help="Path to YAML config file")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_pipeline(cfg)
