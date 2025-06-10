from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import requests
from PIL import Image

SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"


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
}


def download_bands(feature: dict, bands: List[str], out_dir: Path) -> Dict[str, Path]:
    """Download selected ``bands`` from a STAC feature into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for band in bands:
        asset = BAND_MAP.get(band)
        if asset is None or asset not in feature["assets"]:
            continue
        url = feature["assets"][asset]["href"]
        local = out_dir / f"{band}.tif"
        if not local.exists():
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(1_048_576):
                        f.write(chunk)
        paths[band] = local
    return paths


def read_band(
    path: Path, bbox: Optional[Tuple[float, float, float, float]] = None
) -> np.ndarray:
    """Read a band and optionally crop it to a WGS84 ``bbox``."""
    with rio.open(path) as src:
        if bbox is None:
            arr = src.read(1).astype(np.float32) / 10000.0
        else:
            with rio.vrt.WarpedVRT(src, crs="EPSG:4326") as vrt:
                window = rio.windows.from_bounds(*bbox, transform=vrt.transform)
                arr = vrt.read(1, window=window).astype(np.float32) / 10000.0
    return arr


def bounds(path: Path) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) of ``path`` in WGS84."""
    with rio.open(path) as src:
        with rio.vrt.WarpedVRT(src, crs="EPSG:4326") as vrt:
            b = vrt.bounds
    return (b.left, b.bottom, b.right, b.top)


def compute_kndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.tanh(np.square(ndvi))


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
