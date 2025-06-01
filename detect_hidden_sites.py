#!/usr/bin/env python3
"""
Amazon Hidden-Site Hunter — detect_hidden_sites.py
=================================================
DEM ( Copernicus GLO-90 )  ×  GEDI L2A ground returns  →  canopy-hidden earthworks.

v0.5  — persistent GEDI cache
-----------------------------
* `--cache DIR`  (default: <OUT>/gedi_cache)  
  Downloads go here once and are re-used on the next run.
* No temp folders: `earthaccess.download()` simply skips files that
  already exist in the cache, so repeated calls are fast and offline-safe.
* Same flexible time window as v0.4.
"""

from __future__ import annotations
import argparse, datetime as dt, math, os, shutil
from pathlib import Path
from typing import Any, List, Tuple
from getpass import getpass

import geopandas as gpd, numpy as np, pandas as pd, rasterio as rio
import rasterio.merge as merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rich.console import Console
from rich.progress import track
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point
from skimage.measure import label, regionprops
from rich.markup import escape
import matplotlib.pyplot as plt

import earthaccess, h5py, requests
from scipy.spatial import cKDTree
from rasterio import windows
from rasterio.merge import merge
from rasterio.transform import from_origin

console = Console()

# ─────────────────────────── constants ───────────────────────────
COP_DEM_BASE     = "https://copernicus-dem-90m.s3.amazonaws.com"
EARTHDATA_SHORT  = "GEDI02_A"
EARTHDATA_VERSION= "002"
# ──────────────────────────────────────────────────────────────────

def _nan_gaussian_filter(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter that ignores NaN values."""
    valid = np.isfinite(arr)
    arr_filled = np.where(valid, arr, 0.0)
    smoothed = gaussian_filter(arr_filled, sigma=sigma)
    weights = gaussian_filter(valid.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore"):
        smoothed = smoothed / weights
    smoothed[weights == 0] = np.nan
    return smoothed

def ensure_earthdata_login() -> None:
    """Login using ~/.netrc if available, otherwise prompt the user."""
    console.print("[yellow]Earthdata login required")
    try:
        earthaccess.login(strategy="netrc", persist=False)
        console.print("[green]Logged in successfully using ~/.netrc")
        return
    except Exception as e:
        console.print(f"[red]Failed to load credentials: {escape(str(e))}")        
    console.print("No ~/.netrc found or login failed, prompting for credentials")

    persist = input("Use persistent login (save credentials to ~/.netrc)? [y/N] ").strip().lower() == "y"
    
    earthaccess.login(persist=persist, strategy="interactive")


def cop_tile_url(lat: float, lon: float) -> str:
    lat_sw, lon_sw = math.floor(lat), math.floor(lon)
    ns, ew   = ("N" if lat_sw >= 0 else "S"), ("E" if lon_sw >= 0 else "W")
    lat_s, lon_s = f"{abs(lat_sw):02d}_00", f"{abs(lon_sw):03d}_00"
    stem = f"Copernicus_DSM_COG_30_{ns}{lat_s}_{ew}{lon_s}_DEM"
    return f"{COP_DEM_BASE}/{stem}/{stem}.tif"


def fetch_cop_tiles(bbox: tuple[float, float, float, float], out_dir: Path) -> Path:
    xmin, ymin, xmax, ymax = bbox
    dem_path = out_dir / "cop90_mosaic.tif"

    if dem_path.exists():
        try:
            with rio.open(dem_path) as src:
                tag = src.tags().get("bbox")
                if tag:
                    saved = tuple(map(float, tag.split(",")))
                    if all(abs(s - b) < 1e-6 for s, b in zip(saved, bbox)):
                        console.log(f"[green]Using existing DEM mosaic → {dem_path}")
                        return dem_path
                bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                if all(abs(s - b) < 1e-6 for s, b in zip(bounds, bbox)):
                    console.log(f"[green]Using existing DEM mosaic → {dem_path}")
                    return dem_path
        except Exception as e:
            console.log(f"[yellow]Failed to read existing DEM: {escape(str(e))}")           

    lat_rng = range(int(math.floor(ymin)), int(math.ceil(ymax)) + 1)
    lon_rng = range(int(math.floor(xmin)), int(math.ceil(xmax)) + 1)

    tif_paths: list[Path] = []
    for lat in lat_rng:
        for lon in lon_rng:
            url   = cop_tile_url(lat, lon)
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

    console.log("Merging + **cropping** DEM tiles")
    srcs = [rio.open(str(p)) for p in tif_paths]

    with rio.open(tif_paths[0]) as first:
        nodata_val = first.nodata or -9999     # fallback if tag is missing

    # 🔴  The magic line – merge **only** the pixels inside `bbox`
    mosaic, transform = merge(srcs, bounds=bbox, precision=30, nodata=nodata_val)

    meta = srcs[0].meta.copy()
    meta.update(
        driver     ="GTiff",
        height     =mosaic.shape[1],
        width      =mosaic.shape[2],
        transform  =transform,
        compress   ="lzw",
        nodata     =-9999,
    )

    dem_path = out_dir / "cop90_mosaic.tif"
    with rio.open(dem_path, "w", **meta) as dst:
        dst.write(mosaic)
        dst.update_tags(bbox=",".join(map(str, bbox)))

    for s in srcs:
        s.close()
    return dem_path

# ──────────────── GEDI helpers ─────────────────

def _size_mb_to_bytes(mb) -> int:
    return int(float(mb) * 1_048_576) if mb else 0


def _read_gedi_var(ds, coerce_float=True):
    """Return array with scale/offset applied and _FillValue set to NaN."""
    arr = ds[:].astype(np.float32 if coerce_float else ds.dtype)

    scale = ds.attrs.get("scale_factor", 1.0)
    offset = ds.attrs.get("add_offset", 0.0)
    if scale != 1 or offset != 0:
        arr = arr * scale + offset

    fill = ds.attrs.get("_FillValue")
    if fill is not None:
        arr = np.where(arr == (fill * scale + offset), np.nan, arr)

    return arr


def fetch_gedi_points(
    bbox: Tuple[float, float, float, float],
    *,
    time_start: str,
    time_end: str,
    cache_dir: Path,
    max_points: int = 50_000,
    threads: int = 4,
    verify_sizes: bool = True,  # Control size verification
    size_tolerance_pct: float = 0.5,  # Allow 0.5% difference in file size
) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = bbox

    # login
    ensure_earthdata_login()

    # search
    granules = earthaccess.search_data(
        short_name = EARTHDATA_SHORT,
        version    = EARTHDATA_VERSION,
        bounding_box = (xmin, ymin, xmax, ymax),
        temporal     = (time_start, time_end),
    )
    if not granules:
        raise RuntimeError("No GEDI granules intersect bbox + dates.")

    total_bytes = sum(_size_mb_to_bytes(g.size()) for g in granules)
    console.log(
        f"{len(granules)} granules to download "
        f"({total_bytes/1_048_576:,.1f} MiB)"
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"Downloading (or re-using) files → {cache_dir}")

    granules_to_download: List[Any] = []

    if verify_sizes:
        for g in granules:
            expected_size = _size_mb_to_bytes(g.size())
            local_path = cache_dir / Path(g.data_links()[0]).name

            needs_download = True
            if local_path.exists():
                actual_size = local_path.stat().st_size
                if expected_size > 0:
                    pct_diff = abs(actual_size - expected_size) / expected_size * 100
                    if pct_diff <= size_tolerance_pct:
                        console.log(f"[green]Verified cached file: {local_path.name}")
                        needs_download = False
                    else:
                        if actual_size < expected_size * 0.9:
                            console.log(
                                f"[yellow]File likely incomplete: {local_path.name}: "
                                f"expected {expected_size} bytes, got {actual_size} bytes "
                                f"({pct_diff:.2f}% difference)"
                            )
                            local_path.unlink()
                        else:
                            console.log(
                                f"[yellow]Size difference outside tolerance for {local_path.name}: "
                                f"{pct_diff:.2f}% difference, expected {expected_size} bytes, "
                                f"got {actual_size} bytes"
                            )
                            local_path.unlink()

            if needs_download:
                granules_to_download.append(g)
    else:
        for g in granules:
            local_path = cache_dir / Path(g.data_links()[0]).name
            if not local_path.exists():
                granules_to_download.append(g)

    if granules_to_download:
        to_dl_bytes = sum(_size_mb_to_bytes(g.size()) for g in granules_to_download)
        console.log(
            f"{len(granules_to_download)} files need download "
            f"({to_dl_bytes/1_048_576:,.1f} MiB)"
        )
        resp = input("Download missing files? [y/N] ").strip().lower()
        if resp == "y":
            earthaccess.download(granules_to_download, cache_dir, threads=threads)
        else:
            raise SystemExit("Cancelled by user.")
    else:
        console.log("All granules already downloaded")

    # Get list of all local files regardless of how they were obtained
    local_paths = [cache_dir / Path(g.data_links()[0]).name for g in granules]
    
    rows: List[pd.DataFrame] = []
    for p in track(local_paths, description="Parsing GEDI files"):
        if not p.exists():
            console.log(f"[red]Missing file: {p.name}")
            continue

        try:
            with h5py.File(p, "r") as h5:
                for beam in [k for k in h5 if k.startswith("BEAM")]:
                    lat  = _read_gedi_var(h5[f"{beam}/lat_lowestmode"])
                    lon  = _read_gedi_var(h5[f"{beam}/lon_lowestmode"])
                    elev = _read_gedi_var(h5[f"{beam}/elev_lowestmode"])
                    qflag = _read_gedi_var(
                        h5[f"{beam}/quality_flag"], coerce_float=False
                    )
                    m = (
                        (lon >= xmin)
                        & (lon <= xmax)
                        & (lat >= ymin)
                        & (lat <= ymax)
                        & (qflag == 1)
                    )
                    if m.any():
                        rows.append(
                            pd.DataFrame(
                                {
                                    "lat": lat[m].astype(float),
                                    "lon": lon[m].astype(float),
                                    "elev": elev[m].astype(float),
                                }
                            )
                        )
        except Exception as exc:
            console.log(f"[red]Skipped {p.name}: {escape(str(exc))}")
            # Mark file as corrupted by renaming
            if verify_sizes:
                corrupted_path = p.with_suffix(p.suffix + ".corrupted")
                p.rename(corrupted_path)
                console.log(f"[yellow]Marked {p.name} as corrupted for future re-download")

        if rows and sum(len(df) for df in rows) >= max_points:
            break

    if not rows:
        raise RuntimeError("No GEDI footprints inside bbox after filtering.")

    gdf = gpd.GeoDataFrame(
        pd.concat(rows, ignore_index=True),
        geometry=gpd.points_from_xy(pd.concat(rows)["lon"], pd.concat(rows)["lat"]),
        crs="EPSG:4326",
    )
    gdf.rename(columns={"elev": "elev_lowestmode"}, inplace=True)
    console.log(f"[green]{len(gdf):,} GEDI footprints kept[/green]")
    return gdf


# ───────── interpolation & analysis (unchanged) ─────────

# def interpolate_bare_earth(gdf, bbox, res):
#     xmin, ymin, xmax, ymax = bbox
#     xi = np.arange(xmin, xmax, res)
#     yi = np.arange(ymin, ymax, res)
#     xi_m, yi_m = np.meshgrid(xi, yi)
#     zi = griddata((gdf.geometry.x, gdf.geometry.y), gdf["elev_lowestmode"],
#                   (xi_m, yi_m), method="linear")
#     return xi_m, yi_m, zi


def interpolate_bare_earth(
        gdf,
        bbox: tuple[float, float, float, float],
        res: float = 0.2695,
        power: float = .1,
        k: int = 256,
        nodata: float = np.nan,
):
    """
    Inverse-distance weighted (IDW) interpolation of GEDI 'elev_lowestmode'.

    Parameters
    ----------
    gdf    : GeoDataFrame in EPSG:4326 with 'elev_lowestmode'
    bbox   : (xmin, ymin, xmax, ymax)
    res    : cell size in degrees  (0.0002695 ≈ 30 m)
    power  : IDW exponent (2 ≈ Shepard's; 1 = smoother, 3 = sharper)
    k      : use the k nearest GEDI points for each grid cell
    nodata : value to assign where no neighbours found (rare)

    Returns
    -------
    xi_m, yi_m, zi : 2-D lon grid, lat grid, interpolated surface
    """
    xmin, ymin, xmax, ymax = bbox
    xi = np.arange(xmin, xmax + res, res, dtype=np.float32)
    yi = np.arange(ymin, ymax + res, res, dtype=np.float32)
    xi_m, yi_m = np.meshgrid(xi, yi, indexing='xy')

    # Prepare point cloud
    pts  = np.column_stack((gdf.geometry.x.values,
                            gdf.geometry.y.values)).astype(np.float32)
    vals = gdf['elev_lowestmode'].values.astype(np.float32)

    # KD-tree for fast neighbour queries
    tree = cKDTree(pts)

    # Query k nearest neighbours for *all* grid nodes at once
    dists, idxs = tree.query(
        np.column_stack((xi_m.ravel(), yi_m.ravel())),
        k=k, workers=-1
    )                                       # shape = (n_cells, k)

    # Handle cells with <k neighbours inside the tree’s finite radius
    mask = np.isfinite(dists)
    w    = np.zeros_like(dists, dtype=np.float32)
    w[mask] = 1 / np.power(dists[mask], power)   # w ∝ 1 / d^power
    w_sum   = w.sum(axis=1)

    # Avoid division by zero
    safe    = w_sum > 0
    zi_flat = np.full_like(w_sum, nodata, dtype=np.float32)
    zi_flat[safe] = (w[safe] * vals[idxs[safe]]).sum(axis=1) / w_sum[safe]

    zi = zi_flat.reshape(xi_m.shape)
    return xi_m, yi_m, zi



# def residual_relief(bearth, dem_path: Path):
#     xi, yi, zi = bearth
#     with rio.open(dem_path) as src:
#         dest = np.empty_like(zi, dtype=np.float32)
#         transform, _, _ = calculate_default_transform(
#             src.crs, "EPSG:4326", src.width, src.height, *src.bounds
#         )
#         reproject(rio.band(src, 1), dest,
#                   src_transform=src.transform, src_crs=src.crs,
#                   dst_transform=transform, dst_crs="EPSG:4326",
#                   resampling=Resampling.bilinear)
#     return zi - dest

def residual_relief(bearth, dem_path: Path):
    """
    Subtract Copernicus DEM from the GEDI bare-earth surface, guaranteeing that
    any pixel touched by nodata / outside the DEM is returned as NaN.
    """
    xi, yi, zi = bearth
    res_x = xi[0, 1] - xi[0, 0]          # ° lon / pixel
    res_y = yi[1, 0] - yi[0, 0]          # ° lat / pixel (positive)

    # destination grid that matches (xi, yi)
    dst_transform = from_origin(
        xi[0, 0],                # xmin
        yi.max() + res_y,        # ymax (upper edge)
        res_x,                   # pixel width
        res_y                    # pixel height
    )

    with rio.open(dem_path) as src:
        nodata_val = src.nodata or -9999

        # start with an array of NaN so untouched cells stay invalid
        dest = np.full_like(zi, np.nan, dtype=np.float32)

        reproject(
            rio.band(src, 1), dest,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=nodata_val,
            dst_nodata=np.nan,           # any sample hit by nodata → NaN
            init_dest_nodata=True        # ignore nodata in the average
        )

    # final residual, but only where *both* surfaces are valid
    valid = (~np.isnan(zi)) & (~np.isnan(dest))
    rrm = np.full_like(zi, np.nan, dtype=np.float32)
    rrm[valid] = zi[valid] - dest[valid]

    return rrm


def detect_anomalies(
    rrm,
    xi,
    yi,
    sigma: float = 2,
    amp_thresh: float = 1.0,
    size_thresh_m: float = 200,
    debug_dir: Path | None = None,
):
    """Identify potential anomalies in the residual relief model."""

    rrm_smooth = _nan_gaussian_filter(rrm, sigma=sigma)
    mask = np.abs(rrm_smooth) >= amp_thresh
    lbl = label(mask)

    cell_deg2 = (xi[0, 1] - xi[0, 0]) * (yi[1, 0] - yi[0, 0])
    blobs = []
    extent = [xi.min(), xi.max(), yi.min(), yi.max()]

    for reg in regionprops(lbl):
        area_m2 = reg.area * (111_320**2) * cell_deg2
        if area_m2 < size_thresh_m**2:
            continue
        cy, cx = reg.centroid
        lon = float(np.interp(cx, np.arange(xi.shape[1]), xi[0]))
        lat = float(np.interp(cy, np.arange(yi.shape[0]), yi[:, 0]))
        score = float(np.nanmax(np.abs(rrm_smooth[reg.slice])))
        blobs.append({"geometry": Point(lon, lat), "score": score})

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        plt.imshow(
            rrm_smooth,
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
        )
        plt.colorbar(label="Smoothed residual (m)")
        plt.title("Smoothed Residual Relief")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(debug_dir / "rrm_smooth.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.imshow(mask, extent=extent, origin="upper", cmap="gray")
        plt.title("Threshold Mask")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(debug_dir / "threshold_mask.png", dpi=150, bbox_inches="tight")
        plt.close()

        if blobs:
            plt.figure(figsize=(8, 6))
            plt.imshow(
                rrm_smooth,
                extent=extent,
                origin="upper",
                cmap="RdBu_r",
            )
            xs = [b["geometry"].x for b in blobs]
            ys = [b["geometry"].y for b in blobs]
            plt.scatter(xs, ys, c="yellow", edgecolor="black", s=30)
            plt.title("Detected Anomalies")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.savefig(debug_dir / "anomalies.png", dpi=150, bbox_inches="tight")
            plt.close()

    return gpd.GeoDataFrame(blobs, crs="EPSG:4326")


# def detect_anomalies(rrm, xi, yi, sigma=2, amp_thresh=1.0, size_thresh_m=200):
#     rrm_smooth = gaussian_filter(rrm, sigma=sigma)
#     mask = np.abs(rrm_smooth) >= amp_thresh
#     lbl  = label(mask)
#     cell_deg2 = (xi[0,1]-xi[0,0]) * (yi[1,0]-yi[0,0])
#     blobs=[]
#     for reg in regionprops(lbl):
#         area_m2 = reg.area * (111_320**2) * cell_deg2
#         if area_m2 < size_thresh_m**2: continue
#         cy, cx = reg.centroid
#         lon = float(np.interp(cx, np.arange(xi.shape[1]), xi[0]))
#         lat = float(np.interp(cy, np.arange(yi.shape[0]), yi[:,0]))
#         score = float(np.nanmax(np.abs(rrm_smooth[reg.slice])))
#         blobs.append({"geometry": Point(lon, lat), "score": score})
#     return gpd.GeoDataFrame(blobs, crs="EPSG:4326")


# ───────── main CLI ─────────

def main():
    today = dt.date.today()
    p = argparse.ArgumentParser(description="Detect canopy-hidden earthworks")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("xmin","ymin","xmax","ymax"),
                   required=True, help="Longitude/Latitude bounds")
    p.add_argument("--out", required=True, help="Output folder")
    p.add_argument("--cache", help="Persistent GEDI cache directory "
                                   "(default: OUT/gedi_cache)")
    p.add_argument("--resolution", type=float, default=0.0002695,
                   help="Interpolation grid step in degrees (~30 m)")
    p.add_argument("--sigma", type=int, default=2,
                   help="Gaussian σ for residual-relief smoothing")
    # temporal
    p.add_argument("--years", type=int, default=8,
                   help="Most-recent N years (ignored if --start)")
    p.add_argument("--start", metavar="YYYY-MM-DD",
                   help="Start date (overrides --years)")
    p.add_argument("--end",   metavar="YYYY-MM-DD", default=today.isoformat(),
                   help="End date (default: today)")
    args = p.parse_args()

    # resolve dates
    if args.start:
        time_start = args.start
    else:
        time_start = (today - dt.timedelta(days=args.years*365)).isoformat()
    time_end = args.end

    bbox = tuple(args.bbox)
    out  = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache or (out / "gedi_cache")).expanduser()

    console.rule("[bold green]Step 1/4 – Copernicus DEM")
    dem_path = fetch_cop_tiles(bbox, out)

    console.rule("[bold green]Step 2/4 – GEDI footprints")
    gedi = fetch_gedi_points(
        bbox,
        time_start=time_start,
        time_end=time_end,
        cache_dir=cache_dir,
    )

    console.rule("[bold green]Step 3/4 – Bare-earth surface")
    xi, yi, zi = interpolate_bare_earth(gedi, bbox, args.resolution)

    console.rule("[bold green]Step 4/4 – Anomaly detection")
    rrm = residual_relief((xi, yi, zi), dem_path)
    anomalies = detect_anomalies(rrm, xi, yi, sigma=args.sigma)

    out_geojson = out / "anomalies.geojson"
    anomalies.to_file(out_geojson, driver="GeoJSON")
    console.print(f"[bold cyan]Done → {out_geojson}[/bold cyan]")


# if __name__ == "__main__":
#     main()
