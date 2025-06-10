from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import requests
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt

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


def read_band(path: Path) -> np.ndarray:
    with rio.open(path) as src:
        arr = src.read(1).astype(np.float32) / 10000.0
    return arr


def compute_kndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    ndvi = (nir - red) / (nir + red + 1e-6)
    return np.tanh(np.square(ndvi))


def save_true_color(b02: np.ndarray, b03: np.ndarray, b04: np.ndarray, path: Path, gain: float = 2.5) -> None:
    rgb = np.stack([b04, b03, b02], axis=-1) * gain
    rgb = np.clip(rgb, 0, 1)
    plt.imsave(path, rgb)


def save_index_png(arr: np.ndarray, path: Path, cmap: str = "RdYlGn") -> None:
    arr = np.clip(arr, np.nanmin(arr), np.nanmax(arr))
    plt.imsave(path, arr, cmap=cmap)
