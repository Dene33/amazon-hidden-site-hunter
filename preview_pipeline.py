#!/usr/bin/env python3
"""
Preview Analysis Pipeline — visualize each step of the hidden site detection
===========================================================================
Shows the data and intermediate results at each step of the analysis:
1. Copernicus DEM (input surface model)
2. GEDI points (ground penetrating measurements)
3. Interpolated bare-earth surface
4. Residual relief model
5. Detected anomalies
"""

import argparse
import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from affine import Affine
from folium.plugins import HeatMap, MarkerCluster
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from sentinel_utils import save_image_with_metadata
from pyproj import Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.interpolate import griddata
from shapely.geometry import Point, box
from shapely.ops import transform as shp_transform

# Import the interpolation and analysis functions to replicate the pipeline
from detect_hidden_sites import (
    detect_anomalies,
    interpolate_bare_earth,
    residual_relief,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview each step of the hidden site detection pipeline"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("xmin", "ymin", "xmax", "ymax"),
        required=False,
        help="Longitude/Latitude bounds",
    )
    parser.add_argument("--dem", help="Path to Copernicus DEM file (cop90_mosaic.tif)")
    parser.add_argument("--cache", help="GEDI cache directory")
    parser.add_argument(
        "--geojson", help="Path to output anomalies.geojson file (if available)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.0002695,
        help="Interpolation grid step in degrees (~30 m)",
    )
    parser.add_argument(
        "--sigma", type=int, default=2, help="Gaussian σ for residual-relief smoothing"
    )
    parser.add_argument(
        "--outdir",
        default="preview_outputs",
        help="Directory to save preview visualizations",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000000000000000,
        help="Maximum GEDI points to display",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Create interactive map visualizations too",
    )
    return parser.parse_args()


def _read_gedi_var(ds, coerce_float=True):
    """
    Return a numpy array with scale_factor / add_offset applied
    and _FillValue set to NaN.
    """
    arr = ds[:].astype(np.float32 if coerce_float else ds.dtype)

    scale = ds.attrs.get("scale_factor", 1.0)
    offset = ds.attrs.get("add_offset", 0.0)
    if scale != 1 or offset != 0:
        arr = arr * scale + offset

    fill = ds.attrs.get("_FillValue")
    if fill is not None:
        arr = np.where(arr == (fill * scale + offset), np.nan, arr)

    return arr


def load_gedi_points(cache_dir, bbox, max_points):
    """Load GEDI points from cache files"""
    xmin, ymin, xmax, ymax = bbox

    if not cache_dir:
        print("No GEDI cache directory provided")
        return None

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Cache directory {cache_path} not found")
        return None

    h5_files = list(cache_path.glob("*.h5"))
    if not h5_files:
        print("No GEDI files found in cache")
        return None

    print(f"Loading GEDI points from {len(h5_files)} files...")

    points = []
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, "r") as h5:
                for beam in [k for k in h5 if k.startswith("BEAM")]:
                    # lat = h5[f"{beam}/lat_lowestmode"][:]
                    # lon = h5[f"{beam}/lon_lowestmode"][:]
                    # elev = h5[f"{beam}/elev_lowestmode"][:]

                    # mask = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
                    # if mask.any():
                    #     lat_filtered = lat[mask]
                    #     lon_filtered = lon[mask]
                    #     elev_filtered = elev[mask]
                    lat = _read_gedi_var(h5[f"{beam}/lat_lowestmode"])
                    lon = _read_gedi_var(h5[f"{beam}/lon_lowestmode"])
                    elev = _read_gedi_var(h5[f"{beam}/elev_lowestmode"])

                    # optional, but HIGHLY recommended:
                    qflag = _read_gedi_var(
                        h5[f"{beam}/quality_flag"], coerce_float=False
                    )
                    mask = (
                        (lon >= xmin)
                        & (lon <= xmax)
                        & (lat >= ymin)
                        & (lat <= ymax)
                        & (qflag == 1)  # keep only good shots
                    )

                    if mask.any():
                        lat_filtered = lat[mask]
                        lon_filtered = lon[mask]
                        elev_filtered = elev[mask]

                        for lat_i, lon_i, elev_i in zip(
                            lat_filtered, lon_filtered, elev_filtered
                        ):
                            points.append((float(lat_i), float(lon_i), float(elev_i)))

                        if len(points) >= max_points:
                            break

                if len(points) >= max_points:
                    break
        except Exception as e:
            print(f"Error reading {h5_file}: {e}")

    if not points:
        print("No GEDI points found within the bounding box")
        return None

    print(f"Loaded {len(points)} GEDI points")
    return points


# def visualize_copernicus_dem(
#         dem_path,
#         bbox=None,                # (xmin, ymin, xmax, ymax) – lon/lat or EPSG:3857
#         outdir=".",
#         *,
#         project=True,             # << NEW: reproject to Web-Mercator?
#         bare=False,
#         dpi=150,
#         subsample=4,
#         cmap=plt.cm.terrain,
#         azdeg=315,
#         altdeg=45,
#         vert_exag=5,
#         max_elev=None,
#         min_elev=0,
# ):
#     """
#     Render a Copernicus DEM, optionally re-projecting to Web-Mercator, and
#     save exactly one PNG.

#     Parameters
#     ----------
#     dem_path : str or Path
#         Path to Copernicus DEM (in EPSG:4326, 1-arc-sec or 30 m version).
#     bbox : tuple | None
#         (xmin, ymin, xmax, ymax). If `project=True`, values may be either
#         lon/lat *or* EPSG : 3857 – the function will detect and convert.
#     project : bool, default True
#         If True, re-project the DEM to EPSG : 3857 before hillshading so that
#         horizontal and vertical scales are in metres (avoids the “squashed”
#         look you get when shading in lon/lat).
#     bare : bool, default False
#         If True, write a border-less, axis-free raster (one tile).
#     Other parameters are unchanged.
#     """
#     dem_path = Path(dem_path)
#     if not dem_path.exists():
#         raise FileNotFoundError(dem_path)

#     # ---------- read or reproject DEM ----------
#     with rio.open(dem_path) as src:
#         xmin, ymin, xmax, ymax = src.bounds
#         meta = src.meta.copy()
#         print(meta)
#         if project:
#             dst_crs = "EPSG:3857"
#             transform, width, height = calculate_default_transform(
#                 src.crs, dst_crs, src.width, src.height, *src.bounds)
#             # upscale/down-scale *before* hillshading for speed
#             width //= subsample
#             height //= subsample
#             transform = transform * Affine.scale(subsample)   # <— key line
#             dem = np.empty((height, width), dtype=src.dtypes[0])

#             reproject(
#                 source=rio.band(src, 1),
#                 destination=dem,
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=dst_crs,
#                 resampling=Resampling.bilinear,
#                 dst_width=width,
#                 dst_height=height,
#             )
#             bounds = rio.transform.array_bounds(height, width, transform)
#             # In Web-Mercator the transform is affine; pixel size ≈ meters
#             extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
#         else:
#             dem = src.read(1)[::subsample, ::subsample]
#             bounds = src.bounds
#             extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

#     ny, nx = dem.shape  # rows, cols for later

#     # ---------- optional clipping ----------
#     if max_elev is not None:
#         dem = np.clip(dem, a_min=None, a_max=max_elev)

#     # ---------- colour stretch ----------
#     vmin = min_elev
#     vmax = max_elev if max_elev is not None else float(np.nanmax(dem))
#     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

#     # ---------- hill-shade ----------
#     ls = LightSource(azdeg=azdeg, altdeg=altdeg)
#     rgb = ls.shade(dem, cmap=cmap, blend_mode="soft",
#                    vert_exag=vert_exag, vmin=vmin, vmax=vmax)   # ★ key line

#     # ---------- figure ----------
#     if bare:
#         fig = plt.figure(figsize=(nx / dpi, ny / dpi), dpi=dpi)
#         ax = fig.add_axes([0, 0, 1, 1])
#         ax.axis("off")
#         im_kwargs = {"aspect": "equal"}
#     else:
#         fig, ax = plt.subplots(figsize=(12, 10))
#         im_kwargs = {}

#     # draw DEM
#     ax.imshow(rgb, extent=extent, interpolation="nearest", **im_kwargs)

#     # ---------- overlays & decoration ----------
#     if not bare:
#         # colorbar in physical units (metres)
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="4%", pad=0.05)
#         fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
#                      cax=cax, label="Elevation (m)")

#         # optional bbox overlay

#         # print(float(meta["transform"][2]),)
#         # print(float(meta["transform"][5] + meta["transform"][4] * meta["height"]),)
#         # print(float(meta["transform"][2] + meta["transform"][0] * meta["width"]),)
#         # print(float(meta["transform"][5]),)

#         # xmin, ymin, xmax, ymax = (bounds.left, bounds.bottom,
#         #                           bounds.right, bounds.top)

#         # From meta transform:
#         # xmin, ymin, xmax, ymax = (
#         #     float(meta["transform"][2]),
#         #     float(meta["transform"][5] + meta["transform"][4] * meta["height"]),
#         #     float(meta["transform"][2] + meta["transform"][0] * meta["width"]),
#         #     float(meta["transform"][5])
#         # )

#         # detect CRS: heuristic – lon in (-180, 180) ⇒ need conversion
#         if project and (-180 <= xmin <= 180) and (-90 <= ymin <= 90):
#             transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857",
#                                                 always_xy=True)
#             geom_ll = box(xmin, ymin, xmax, ymax)
#             geom_merc = shp_transform(transformer.transform, geom_ll)
#             xmin, ymin, xmax, ymax = geom_merc.bounds
#         ax.plot([xmin, xmax, xmax, xmin, xmin],
#                 [ymin, ymin, ymax, ymax, ymin],
#                 color="red", lw=.5, label="AOI")
#         ax.legend()

#         ax.set_title("Copernicus Digital Elevation Model (Web-Mercator)"
#                      if project else
#                      "Copernicus Digital Elevation Model (native lon/lat)")
#         ax.set_xlabel("Easting (m)" if project else "Longitude")
#         ax.set_ylabel("Northing (m)" if project else "Latitude")

#     # ---------- save ----------
#     out_name = ("copernicus_dem_merc.png" if project else
#                 "copernicus_dem_native.png") if bare else \
#                ("copernicus_dem_map_merc.png" if project else
#                 "copernicus_dem_map.png")

#     out_path = Path(outdir) / out_name
#     fig.savefig(out_path, dpi=dpi,
#                 bbox_inches="tight" if bare else None,
#                 pad_inches=0 if bare else 0.1)
#     plt.close(fig)
#     print(f"✓ saved → {out_path}")

#     return dem

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject


def visualize_copernicus_dem(
    dem_path,
    bbox=None,
    outdir=".",
    *,
    project=True,
    bare=False,
    dpi=150,
    subsample=1,
    cmap=plt.cm.terrain,
    azdeg=315,
    altdeg=45,
    vert_exag=5,
    max_elev=None,
    min_elev=None,
    upscale=3,  # ← ADDED: blow-up tiny AOIs for print
):
    """Render a Copernicus DEM with hillshading.

    When ``min_elev`` or ``max_elev`` are not provided they are estimated from
    the DEM as mean ± one standard deviation to improve contrast.
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(dem_path)

    # ─── 1. read DEM ─────────────────────────────────────────────────────────
    with rio.open(dem_path) as src:
        src_nodata = src.nodata
        if project:
            dst_crs = "EPSG:3857"
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            width //= subsample
            height //= subsample
            transform = transform * Affine.scale(subsample)

            dem = np.full((height, width), np.nan, dtype=np.float32)  # ← FILL NaN
            # reproject(
            #     rio.band(src, 1), dem,
            #     src_transform=src.transform, src_crs=src.crs,
            #     dst_transform=transform, dst_crs=dst_crs,
            #     resampling=Resampling.bilinear,
            #     src_nodata=src_nodata, dst_nodata=np.nan                 # ← MASK nodata
            # )
            reproject(
                source=rio.band(src, 1),
                destination=dem,  # already an array of np.nan
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,  #  ← key
                init_dest_nodata=True,  #  ← ignore src nodata in mixing
            )
            bounds = rio.transform.array_bounds(height, width, transform)
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

        else:  # native lon/lat
            dem = src.read(1, masked=True).astype(np.float32)
            dem = dem.filled(np.nan)  # ← MASK nodata
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # estimate elevation range if not provided
    valid = dem[~np.isnan(dem)]
    if valid.size:
        mean_elev = float(np.nanmean(valid))
        std_elev = float(np.nanstd(valid))
        range_pad = 2 * std_elev
        if max_elev is None:
            max_elev = mean_elev + (range_pad * 2)
        if min_elev is None:
            min_elev = mean_elev - range_pad

    # ─── 2. colour stretch & hill-shade ──────────────────────────────────────
    if max_elev is not None:
        dem = np.clip(dem, None, max_elev)
    if min_elev is not None:
        dem = np.clip(dem, min_elev, None)

    vmin = min_elev if min_elev is not None else np.nanmin(dem)
    vmax = max_elev if max_elev is not None else np.nanmax(dem)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    # replace NaN by mean just for lighting; we'll zero-out alpha later
    dem_for_ls = np.where(np.isnan(dem), np.nanmean(dem), dem)
    rgb = ls.shade(
        dem_for_ls,
        cmap=cmap,
        blend_mode="soft",
        vert_exag=vert_exag,
        vmin=vmin,
        vmax=vmax,
    )

    # make nodata fully transparent
    alpha_mask = (~np.isnan(dem)).astype(np.float32)
    rgb[..., -1] = alpha_mask

    # ─── 3. figure ----------------------------------------------------------------
    ny, nx = dem.shape
    fig_size = (nx * upscale / dpi, ny * upscale / dpi) if bare else (12, 10)

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]) if bare else fig.add_subplot(111)
    ax.imshow(rgb, extent=extent, origin="upper")

    if bbox:
        xmin_b, ymin_b, xmax_b, ymax_b = bbox
        if project:
            t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xmin_b, ymin_b = t.transform(xmin_b, ymin_b)
            xmax_b, ymax_b = t.transform(xmax_b, ymax_b)
        ax.set_xlim(xmin_b, xmax_b)
        ax.set_ylim(ymin_b, ymax_b)

    if bare:
        ax.axis("off")
    else:
        # colour-bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Elevation (m)"
        )

        # AOI rectangle
        if bbox:
            xmin_b, ymin_b, xmax_b, ymax_b = bbox
            if project:
                t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                xmin_b, ymin_b = t.transform(xmin_b, ymin_b)
                xmax_b, ymax_b = t.transform(xmax_b, ymax_b)
            ax.plot(
                [xmin_b, xmax_b, xmax_b, xmin_b, xmin_b],
                [ymin_b, ymin_b, ymax_b, ymax_b, ymin_b],
                color="red",
                lw=0.7,
                label="AOI",
            )
            ax.legend()

        ax.set_xlabel("Easting (m)" if project else "Longitude")
        ax.set_ylabel("Northing (m)" if project else "Latitude")
        ax.set_title(
            "Copernicus DEM (Web-Mercator)" if project else "Copernicus DEM (WGS-84)"
        )

    # ─── 4. save PNG ---------------------------------------------------------------
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = "copernicus_dem_hillshade" if bare else "copernicus_dem_map"
    if project:
        base += "_3857"
    out_path = outdir / f"{base}.png"

    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight" if bare else None,
        pad_inches=0 if bare else 0.1,
    )
    plt.close(fig)
    print(f"✓ saved → {out_path}")

    return dem


def visualize_gedi_points(points, bbox, outdir):
    """Visualize the GEDI points.

    Two images are produced:

    ``2_gedi_points.png`` – standard version with legend and axes.
    ``2_gedi_points_clean.png`` – cropped version without legend for
    interactive map overlays.
    """
    if not points:
        print("No GEDI points to visualize")
        return None

    print(f"Visualizing {len(points)} GEDI points")

    # Extract coordinates and elevations
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    elevs = [p[2] for p in points]

    # Create figure for the regular version
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot points colored by elevation
    scatter = ax.scatter(
        lons, lats, c=elevs, cmap="viridis", s=2, alpha=0.7, edgecolors="none"
    )

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("Ground Elevation (m)")

    # Add bbox outline
    xmin, ymin, xmax, ymax = bbox
    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        "r-",
        linewidth=2,
        label="Area of Interest",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "2_gedi_points.png"

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    with Image.open(out_path) as img:
        save_image_with_metadata(img, out_path, bbox=bbox)

    # ----- clean version for map overlays -----
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(lons, lats, c=elevs, cmap="viridis", s=2, alpha=0.7, edgecolors="none")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.axis("off")
    out_path_clean = outdir / "2_gedi_points_clean.png"
    fig2.savefig(out_path_clean, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    with Image.open(out_path_clean) as img:
        save_image_with_metadata(img, out_path_clean, bbox=bbox)
    return True


def visualize_bare_earth(points, bbox, resolution, outdir):
    """Visualize the interpolated bare-earth surface.

    Creates ``3_bare_earth_surface.png`` with legend and
    ``3_bare_earth_surface_clean.png`` cropped for overlays.
    """
    if not points:
        print("No GEDI points for bare-earth interpolation")
        return None

    print("Creating and visualizing bare-earth model...")

    # Convert points to format needed for interpolation
    gedi_df = gpd.GeoDataFrame(
        data={"elev_lowestmode": [p[2] for p in points]},
        geometry=[Point(lon, lat) for lat, lon, _ in points],
        crs="EPSG:4326",
    )

    # Perform interpolation
    xi, yi, zi = interpolate_bare_earth(gedi_df, bbox, resolution)

    xmin, ymin, xmax, ymax = bbox

    # Create figure for the regular version
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create hillshade effect
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.terrain

    # Handle NaN values
    zi_valid = np.copy(zi)
    if np.isnan(zi).any():
        print(f"Warning: {np.isnan(zi).sum()} NaN values in interpolated surface")
        zi_mean = np.nanmean(zi)
        zi_valid[np.isnan(zi)] = zi_mean

    rgb = ls.shade(zi_valid, cmap=cmap, blend_mode="soft", vert_exag=5)

    # Plot the interpolated surface
    extent = [np.min(xi), np.max(xi), np.min(yi), np.max(yi)]
    im = ax.imshow(rgb, extent=extent, origin="upper")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    norm = mpl.colors.Normalize(vmin=np.nanmin(zi), vmax=np.nanmax(zi))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("Interpolated Ground Elevation (m)")

    ax.set_title("Interpolated Bare-Earth Surface from GEDI Points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    out_path = Path(outdir) / "3_bare_earth_surface.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bare-earth visualization to {out_path}")

    # ----- clean version for overlays -----
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(rgb, extent=extent, origin="upper")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.axis("off")
    out_path_clean = Path(outdir) / "3_bare_earth_surface_clean.png"
    plt.savefig(out_path_clean, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    print(f"Saved bare-earth overlay to {out_path_clean}")

    return xi, yi, zi


def visualize_residual_relief(xi, yi, zi, dem_path, outdir):
    """Visualize the residual relief model.

    Produces ``4_residual_relief.png`` and ``4_residual_relief_clean.png``
    cropped for overlay use.
    """
    if zi is None or dem_path is None:
        print("Missing data for residual relief model")
        return None

    dem_file = Path(dem_path)
    if not dem_file.exists():
        print(f"DEM file {dem_file} not found")
        return None

    print("Calculating and visualizing residual relief model...")

    # Calculate residual relief
    rrm = residual_relief((xi, yi, zi), dem_file)

    xmin, xmax = np.min(xi), np.max(xi)
    ymin, ymax = np.min(yi), np.max(yi)

    # Create figure for the regular version
    fig, ax = plt.subplots(figsize=(12, 10))

    # Handle NaN values
    rrm_valid = np.ma.masked_invalid(rrm)  # ← just mask them
    # rrm_valid = np.copy(rrm)
    # if np.isnan(rrm).any():
    #     print(f"Warning: {np.isnan(rrm).sum()} NaN values in residual relief model")
    #     rrm_valid[np.isnan(rrm)] = 0

    # Use a centered colormap
    vmax = max(abs(np.nanmin(rrm)), abs(np.nanmax(rrm)))
    vmin = -vmax

    # Plot the residual relief model
    extent = [np.min(xi), np.max(xi), np.min(yi), np.max(yi)]
    im = ax.imshow(
        rrm_valid, extent=extent, origin="upper", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Elevation Difference (m)")

    ax.set_title("Residual Relief Model (Bare Earth - DEM)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    out_path = Path(outdir) / "4_residual_relief.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved residual relief visualization to {out_path}")

    # ----- clean version for overlays -----
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(
        rrm_valid, extent=extent, origin="upper", cmap="RdBu_r", vmin=vmin, vmax=vmax
    )
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.axis("off")
    out_path_clean = Path(outdir) / "4_residual_relief_clean.png"
    plt.savefig(out_path_clean, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    print(f"Saved residual relief overlay to {out_path_clean}")

    return rrm


def visualize_anomalies(anomalies, rrm, xi, yi, sigma, geojson_path, outdir):
    """Visualize the detected anomalies"""
    if rrm is None:
        print("No residual relief model for anomaly detection")
        return None

    print("Detecting and visualizing anomalies...")

    # Detect anomalies
    # anomalies = detect_anomalies(rrm, xi, yi, sigma=sigma)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot residual relief as background
    extent = [np.min(xi), np.max(xi), np.min(yi), np.max(yi)]
    vmax = max(abs(np.nanmin(rrm)), abs(np.nanmax(rrm)))
    vmin = -vmax

    im = ax.imshow(
        rrm,
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
    )

    # Plot detected anomalies
    if not anomalies.empty:
        # Color by score
        scores = anomalies["score"].values
        anomaly_points = ax.scatter(
            anomalies.geometry.x,
            anomalies.geometry.y,
            c=scores,
            cmap="plasma",
            s=50,
            edgecolor="k",
            linewidth=0.5,
            zorder=10,
        )

        # Add colorbar for anomaly scores
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(anomaly_points, cax=cax1)
        cbar1.set_label("Anomaly Score")
    else:
        print("No anomalies detected")

    # Add colorbar for background
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar2 = fig.colorbar(im, cax=cax2, orientation="horizontal")
    cbar2.set_label("Elevation Difference (m)")

    # Load and plot ground truth if available
    if geojson_path:
        try:
            truth_gdf = gpd.read_file(geojson_path)
            truth_gdf.plot(
                ax=ax,
                color="none",
                edgecolor="lime",
                linewidth=2,
                zorder=5,
                label="Known Sites",
            )
            ax.legend()
        except Exception as e:
            print(f"Error loading ground truth: {e}")

    ax.set_title("Detected Anomalies")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Save the figure
    out_path = Path(outdir) / "5_detected_anomalies.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save the anomalies to GeoJSON
    if not anomalies.empty:
        anomalies_path = Path(outdir) / "detected_anomalies.geojson"
        anomalies.to_file(anomalies_path, driver="GeoJSON")
        print(f"Saved {len(anomalies)} anomalies to {anomalies_path}")

    print(f"Saved anomaly visualization to {out_path}")
    return anomalies


def create_interactive_map(
    points,
    anomalies,
    bbox,
    outdir,
    include_data_vis=False,
    sentinel=None,
    *,
    include_full_sentinel=False,
    include_full_srtm=False,
    include_full_aw3d=False,
    chatgpt_points: list | None = None,
):
    """Create an interactive map with pipeline results.

    Parameters
    ----------
    points : list | None
        Ignored but kept for backwards compatibility.
    anomalies : GeoDataFrame | None
        Detected anomalies to plot.
    bbox : tuple
        (xmin, ymin, xmax, ymax) in lon/lat.
    outdir : str or Path
        Directory containing the generated PNG images.
    include_data_vis : bool, default False
        If True, include the additional reference layers from ``data_vis``.
        Debug overlay images from ``outdir/debug`` are added automatically when
        present.
    sentinel : dict, optional
        Sentinel image paths and bounds returned by ``step_fetch_sentinel``.
    include_full_sentinel : bool, default False
        If ``True``, add the full Sentinel overlays (or their ``_web``
        versions). If ``False``, only the cropped versions are added.
    include_full_srtm : bool, default False
        If ``True``, include SRTM hillshade overlays when available.
    include_full_aw3d : bool, default False
        If ``True``, include AW3D30 hillshade overlays when available.
    chatgpt_points : list, optional
        Detections parsed from ChatGPT output.
    """
    if anomalies is None and not Path(outdir).exists():
        print("No data for interactive map")
        return

    from data_vis import create_combined_map, load_reference_datasets

    arch_dataframes = []
    lidar_df = pd.DataFrame()
    outdir = Path(outdir)
    # Use only the clean images for overlays and include the DEM hillshade
    image_files = []

    # Copernicus DEM overlays
    mosaic_png = list(outdir.glob("1_copernicus_dem_mosaic_hillshade*.png"))
    if mosaic_png:
        image_files.append(str(mosaic_png[0].resolve()))
    hillshade = list(outdir.glob("1_copernicus_dem_crop_hillshade*.png"))
    if hillshade:
        image_files.append(str(hillshade[0].resolve()))

    srtm_mosaic = list(outdir.glob("1b_srtm_mosaic_hillshade*.png"))
    srtm_crop = list(outdir.glob("1b_srtm_crop_hillshade*.png"))
    aw3d_mosaic = list(outdir.glob("1c_aw3d30_mosaic_hillshade*.png"))
    aw3d_crop = list(outdir.glob("1c_aw3d30_crop_hillshade*.png"))

    if include_full_srtm and srtm_mosaic:
        image_files.append(str(srtm_mosaic[0].resolve()))
    if include_full_aw3d and aw3d_mosaic:
        image_files.append(str(aw3d_mosaic[0].resolve()))
    if srtm_crop:
        image_files.append(str(srtm_crop[0].resolve()))
    if aw3d_crop:
        image_files.append(str(aw3d_crop[0].resolve()))
    image_files.extend(str(p.resolve()) for p in sorted(outdir.glob("*_clean.png")))
    image_files.extend(str(p.resolve()) for p in sorted(outdir.glob("*_clean.jpg")))

    if include_full_sentinel:
        image_files.extend(str(p.resolve()) for p in sorted(outdir.glob("sentinel_*.png")))
        image_files.extend(str(p.resolve()) for p in sorted(outdir.glob("sentinel_*.jpg")))

    # Prefer downsampled versions when available
    if include_full_sentinel:
        def _swap_for_web(name: str) -> None:
            web_jpg = outdir / f"{name}_web.jpg"
            web_png = outdir / f"{name}_web.png"
            target = None
            if web_jpg.exists():
                target = web_jpg
            elif web_png.exists():
                target = web_png
            if target:
                for ext in (".jpg", ".png"):
                    p = str((outdir / f"{name}{ext}").resolve())
                    if p in image_files:
                        image_files.remove(p)
                if str(target.resolve()) not in image_files:
                    image_files.append(str(target.resolve()))

        _swap_for_web("sentinel_true_color")
        _swap_for_web("sentinel_kndvi")
        _swap_for_web("sentinel_ndvi_diff")
        _swap_for_web("sentinel_ndvi_ratio")

    debug_dir = outdir / "debug"
    if debug_dir.exists():
        image_files.extend(
            str(p.resolve()) for p in sorted(debug_dir.glob("*_clean.png"))
        )

    if include_data_vis:
        from data_vis import load_reference_datasets

        ref_arch, ref_lidar, ref_images = load_reference_datasets()
        arch_dataframes = ref_arch
        lidar_df = ref_lidar
        image_files.extend(ref_images)

    # Determine custom bounds for the DEM mosaic if present
    image_files = list(dict.fromkeys(image_files))
    image_bounds = {}
    if mosaic_png:
        mosaic_tif = outdir / "cop90_mosaic.tif"
        if mosaic_tif.exists():
            with rio.open(mosaic_tif) as src:
                b = src.bounds
                image_bounds[str(Path(mosaic_png[0]).resolve())] = [
                    [b.bottom, b.left],
                    [b.top, b.right],
                ]


    # Sentinel bounds for full and cropped images
    if include_full_sentinel and sentinel and "bounds" in sentinel:
        sb = sentinel["bounds"]
        full_bounds = [[sb[1], sb[0]], [sb[3], sb[2]]]

        def _add_bound(name: str) -> None:
            for suffix in ["_web.jpg", "_web.png", ".jpg", ".png"]:
                f = outdir / f"{name}{suffix}"
                if f.exists():
                    image_bounds[str(f.resolve())] = full_bounds
                    break

        _add_bound("sentinel_true_color")
        _add_bound("sentinel_kndvi")
        _add_bound("sentinel_ndvi_diff")
        _add_bound("sentinel_ndvi_ratio")
    crop_bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
    if (outdir / "sentinel_true_color_high_clean.jpg").exists():
        image_bounds[str((outdir / "sentinel_true_color_high_clean.jpg").resolve())] = crop_bounds
    if (outdir / "sentinel_true_color_low_clean.jpg").exists():
        image_bounds[str((outdir / "sentinel_true_color_low_clean.jpg").resolve())] = crop_bounds
    if (outdir / "sentinel_kndvi_clean.png").exists():
        image_bounds[str((outdir / "sentinel_kndvi_clean.png").resolve())] = crop_bounds
    if (outdir / "sentinel_ndvi_diff_clean.png").exists():
        image_bounds[str((outdir / "sentinel_ndvi_diff_clean.png").resolve())] = crop_bounds
    if (outdir / "sentinel_ndvi_ratio_clean.png").exists():
        image_bounds[str((outdir / "sentinel_ndvi_ratio_clean.png").resolve())] = crop_bounds

    if include_full_srtm and srtm_crop and srtm_crop[0].exists():
        image_bounds[str(srtm_crop[0].resolve())] = crop_bounds
    if include_full_aw3d and aw3d_crop and aw3d_crop[0].exists():
        image_bounds[str(aw3d_crop[0].resolve())] = crop_bounds

    map_obj = create_combined_map(
        arch_dataframes,
        lidar_df,
        image_files,
        points=points,
        anomalies=anomalies,
        bbox=bbox,
        image_bounds=image_bounds,
        chatgpt_points=chatgpt_points,
    )

    output_path = outdir / "interactive_map.html"
    map_obj.save(output_path)
    print(f"Saved interactive map to {output_path}")


def main():
    args = parse_args()

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load GEDI points
    points = load_gedi_points(args.cache, args.bbox, args.max_points)

    # Visualize each step

    dem = visualize_copernicus_dem(
        args.dem, args.bbox, outdir, bare=False, dpi=1200, project=True, subsample=1
    )
    dem = visualize_copernicus_dem(
        args.dem, args.bbox, outdir, bare=True, dpi=1200, project=True, subsample=1
    )
    visualize_gedi_points(points, args.bbox, outdir)
    xi, yi, zi = visualize_bare_earth(points, args.bbox, args.resolution, outdir)
    rrm = visualize_residual_relief(xi, yi, zi, args.dem, outdir)
    anomalies = visualize_anomalies(rrm, xi, yi, args.sigma, args.geojson, outdir)

    # Create interactive map if requested
    if args.interactive:
        create_interactive_map(points, anomalies, args.bbox, outdir, sentinel=None)

    print("\nAll visualizations complete!")
    print(f"Check the output directory: {outdir.absolute()}")


if __name__ == "__main__":
    main()
