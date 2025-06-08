"""Refactored processing pipeline with configurable steps via YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, List

import yaml
from rich.console import Console

# Reuse core functions from existing scripts
from detect_hidden_sites import (
    fetch_gedi_points,
    interpolate_bare_earth,
    residual_relief,
    detect_anomalies,
    krige_bare_earth
)
from cop_dem_tools import (
    fetch_cop_tiles,
    mosaic_cop_tiles,
    crop_to_bbox,
    save_dem_png,
    save_surface_png,
    save_residual_png,
    save_anomaly_points_png,
)

# Reuse visualization helpers
from preview_pipeline import (
    visualize_gedi_points,
    create_interactive_map,
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


def step_bare_earth(cfg: Dict[str, Any], bbox: Tuple[float, float, float, float], gedi, base: Path):
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
        res=0.0002695,          # 15 m grid, or keep 30 m if you like
        variogram_model="spherical",  # or "spherical", "gaussian"
        nlags=30,
        detrend=True             # often improves short-range detail
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


def step_interactive_map(cfg: Dict[str, Any], points, anomalies, bbox, base: Path):
    if not cfg.get("enabled", True):
        return
    console.rule("[bold green]Create interactive map")
    include_data_vis = cfg.get("include_data_vis", False)
    create_interactive_map(points, anomalies, bbox, base, include_data_vis=include_data_vis)


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
    step_interactive_map(config.get("step5", {}), points, anomalies, bbox, base)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run configurable hidden-site pipeline")
    ap.add_argument("config", help="Path to YAML config file")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_pipeline(cfg)
