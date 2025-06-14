from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.ndimage import binary_dilation
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import requests
from PIL import Image
from rich.console import Console

SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"

console = Console()


def _to_rfc3339(date: str, end: bool = False) -> str:
    """Convert ``YYYY-MM-DD`` strings to RFC3339 format accepted by Earth Search."""
    if "T" in date:
        return date
    return f"{date}T23:59:59Z" if end else f"{date}T00:00:00Z"


def search_sentinel2_item(
    bbox: Tuple[float, float, float, float],
    time_start: str,
    time_end: str,
    cloud_cover: int = 20,
) -> dict | None:
    """Search for a Sentinel-2 L2A item intersecting ``bbox``."""
    query = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{_to_rfc3339(time_start)}/{_to_rfc3339(time_end, end=True)}",
        "query": {"eo:cloud_cover": {"lt": cloud_cover}},
        "limit": 1,
    }
    try:
        r = requests.post(SEARCH_URL, json=query, timeout=60)
        r.raise_for_status()
    except requests.HTTPError:
        return None
    features = r.json().get("features", [])
    return features[0] if features else None


BAND_MAP = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    # Scene classification layer used for cloud masking
    "SCL": "SCL",
}


def download_bands(
    feature: dict,
    bands: List[str],
    out_dir: Path,
    *,
    source_dirs: Optional[List[Path]] = None,
    download_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Download selected ``bands`` from a STAC feature.

    The resulting files are named using the pattern
    ``<item_id>_<xmin>_<ymin>_<xmax>_<ymax>_<YYYYMMDD>_<band>.tif`` so they are
    unique and reusable across experiments.

    Parameters
    ----------
    out_dir : Path
        Directory checked first for existing files. This matches the original
        pipeline output folder.
    source_dirs : list of Path, optional
        Additional directories checked for previously downloaded data.
    download_dir : Path, optional
        Directory where new downloads are stored. Defaults to ``out_dir``.
    """

    out_dir = Path(out_dir)
    download_dir = Path(download_dir or out_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = [out_dir]
    for d in source_dirs or []:
        p = Path(d)
        if p not in search_dirs:
            search_dirs.append(p)

    paths: Dict[str, Path] = {}
    item_id = feature.get("id", "item")
    bbox = feature.get("bbox", [])
    bbox_str = (
        f"_{bbox[0]:.5f}_{bbox[1]:.5f}_{bbox[2]:.5f}_{bbox[3]:.5f}"
        if len(bbox) == 4
        else ""
    )
    datetime_str = feature.get("properties", {}).get("datetime")
    if datetime_str:
        try:
            dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
            date_part = f"_{dt.strftime('%Y%m%d')}"
        except ValueError:
            date_part = ""
    else:
        date_part = ""
    for band in bands:
        asset = BAND_MAP.get(band)
        if asset is None or asset not in feature["assets"]:
            if band == "SCL":
                console.log("[yellow]SCL band not available; skipping cloud mask")
            continue

        filename = f"{item_id}{bbox_str}{date_part}_{band}.tif"
        found: Optional[Path] = None
        for d in search_dirs:
            p = Path(d) / filename
            if p.exists():
                found = p
                break

        if found is not None:
            console.log(f"[green]Using existing band file → {found}")
            paths[band] = found
            continue

        url = feature["assets"][asset]["href"]
        local = download_dir / filename
        console.log(f"Fetching {url} → {local}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for chunk in r.iter_content(1_048_576):
                    f.write(chunk)
        paths[band] = local

    return paths


def read_band(
    path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    *,
    scale: float = 1 / 10000.0,
) -> np.ndarray:
    """Read a band and optionally crop it to a WGS84 ``bbox``.

    Parameters
    ----------
    path : Path
        Raster path.
    bbox : tuple, optional
        (xmin, ymin, xmax, ymax) to crop to in WGS84.
    scale : float, default 1/10000.0
        Multiplicative scale applied to the data. Set to 1 for integer
        classification bands like ``SCL``.
    """

    with rio.open(path) as src:
        if bbox is None:
            arr = src.read(1).astype(np.float32)
        else:
            with rio.vrt.WarpedVRT(src, crs="EPSG:4326") as vrt:
                window = rio.windows.from_bounds(*bbox, transform=vrt.transform)
                arr = vrt.read(1, window=window).astype(np.float32)
    return arr * scale


def bounds(path: Path) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) of ``path`` in WGS84."""
    with rio.open(path) as src:
        with rio.vrt.WarpedVRT(src, crs="EPSG:4326") as vrt:
            b = vrt.bounds
    return (b.left, b.bottom, b.right, b.top)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute the normalized difference vegetation index."""
    return (nir - red) / (nir + red + 1e-6)


def compute_kndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    ndvi = compute_ndvi(red, nir)
    return np.tanh(np.square(ndvi))


# ---------------------------------------------------------------------------
# Cloud masking utilities
# ---------------------------------------------------------------------------

# Pixel values in the Sentinel-2 scene classification layer corresponding
# to clouds or their shadows. These will be masked out before further
# processing.
CLOUD_CLASSES = {3, 8, 9, 10, 11}


def cloud_mask(scl: np.ndarray, dilation: int = 2) -> np.ndarray:
    """Return a boolean mask where ``True`` indicates cloud or shadow pixels."""

    mask = np.isin(scl, list(CLOUD_CLASSES))
    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)
    return mask


# Pixel values in the Sentinel-2 scene classification layer corresponding
# to clouds or their shadows. These will be masked out before further
# processing.
CLOUD_CLASSES = {3, 8, 9, 10, 11}


def mask_clouds(
    scl: np.ndarray,
    *bands: np.ndarray,
    fill_value: float = np.nan,
    dilation: int = 2,
) -> Tuple[np.ndarray, ...]:
    """Mask cloud and shadow pixels in ``bands`` using the ``scl`` array.

    Parameters
    ----------
    scl : np.ndarray
        Scene classification layer where cloud/shadow pixels have the
        values defined in ``CLOUD_CLASSES``.
    bands : np.ndarray
        One or more arrays to mask in-place.
    fill_value : float, default ``np.nan``
        Value assigned to masked pixels.
    dilation : int, default 2
        Number of dilations applied to the mask to remove cloud edges.

    Returns
    -------
    tuple of np.ndarray
        The masked arrays in the same order as provided.
    """

    mask = cloud_mask(scl, dilation=dilation)
    masked = []
    for arr in bands:
        m = arr.astype(np.float32, copy=True)
        m[mask] = fill_value
        masked.append(m)
    return tuple(masked)


def save_mask_png(mask: np.ndarray, path: Path, dpi: int = 150) -> None:
    """Save a binary cloud mask as an image."""

    plt.imsave(path, mask.astype(float), cmap="gray", dpi=dpi)



def save_true_color(
    b02: np.ndarray,
    b03: np.ndarray,
    b04: np.ndarray,
    path: Path,
    gain: float = 2.5,
    quality: int = 95,
    dpi: int = 150,
) -> None:
    """Save a true color RGB image to ``path``.

    Parameters
    ----------
    b02, b03, b04
        Reflectance bands scaled between 0 and 1.
    path
        Destination file. ``.jpg`` or ``.png`` are supported.
    gain
        Multiplicative factor applied before clipping.
    quality
        JPEG quality if saving to that format.
    dpi
        Resolution metadata stored in the output image.
    """

    rgb = np.stack([b04, b03, b02], axis=-1) * gain
    rgb = np.clip(rgb, 0, 1)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        img.save(path, quality=quality, dpi=(dpi, dpi))
    else:
        plt.imsave(path, rgb, dpi=dpi)


def save_index_png(
    arr: np.ndarray,
    path: Path,
    cmap: str = "RdYlGn",
    quality: int = 95,
    dpi: int = 150,
) -> None:
    """Save an index array as an image.

    Parameters
    ----------
    arr
        Array with values scaled between 0 and 1.
    path
        Destination ``.png`` or ``.jpg`` file.
    cmap
        Matplotlib colormap name.
    quality
        JPEG quality if saving to that format.
    dpi
        Resolution metadata stored in the output image.
    """

    arr = np.clip(arr, np.nanmin(arr), np.nanmax(arr))
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        norm = plt.Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))
        cm = plt.get_cmap(cmap)
        rgba = cm(norm(arr))
        img = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
        img.save(path, quality=quality, dpi=(dpi, dpi))
    else:
        plt.imsave(path, arr, cmap=cmap, dpi=dpi)


def resize_image(
    src: Path,
    dest: Path | None = None,
    factor: float = 0.5,
    resample: int = Image.Resampling.BICUBIC,
) -> Path:
    """Save a resized copy of ``src``.

    Parameters
    ----------
    src : Path
        Source image path.
    dest : Path, optional
        Destination path. If ``None``, ``_web`` is appended to the filename.
    factor : float, default 0.5
        Scale factor for width and height.
    resample : int, default ``Image.Resampling.BICUBIC``
        Pillow resampling filter.

    Returns
    -------
    Path
        Path to the resized image.
    """

    dest = dest or src.with_name(src.stem + "_web" + src.suffix)
    with Image.open(src) as img:
        w, h = img.size
        new_size = (max(1, int(w * factor)), max(1, int(h * factor)))
        resized = img.resize(new_size, resample=resample)
        resized.save(dest)
    return dest
