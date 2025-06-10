import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch

import numpy as np
import rasterio as rio
import requests

from sentinel_utils import (bounds, compute_kndvi, read_band,
                            search_sentinel2_item)


def test_compute_kndvi_simple():
    red = np.array([[0.2, 0.3], [0.2, 0.1]])
    nir = np.array([[0.6, 0.5], [0.4, 0.2]])
    kndvi = compute_kndvi(red, nir)
    assert kndvi.shape == red.shape
    # Ensure values are between -1 and 1
    assert np.all(kndvi >= -1) and np.all(kndvi <= 1)


def test_search_sentinel_rfc3339():
    bbox = (-1, -1, 1, 1)
    with patch("sentinel_utils.requests.post") as post:
        post.return_value.json.return_value = {"features": []}
        post.return_value.raise_for_status.return_value = None
        search_sentinel2_item(bbox, "2024-01-01", "2024-12-31")
        args, kwargs = post.call_args
        assert kwargs["json"]["datetime"] == "2024-01-01T00:00:00Z/2024-12-31T23:59:59Z"


def test_search_sentinel_http_error():
    bbox = (-1, -1, 1, 1)
    with patch("sentinel_utils.requests.post") as post:
        post.return_value.raise_for_status.side_effect = requests.HTTPError()
        result = search_sentinel2_item(bbox, "2024-01-01", "2024-12-31")
        assert result is None


def _create_raster(
    path: Path, data: np.ndarray, bounds: tuple[float, float, float, float], crs: str = "EPSG:4326"
) -> None:
    from rasterio.transform import from_bounds

    transform = from_bounds(*bounds, data.shape[1], data.shape[0])
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_read_band_crop_and_bounds(tmp_path: Path):
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    tif = tmp_path / "test.tif"
    _create_raster(tif, arr, (0, 0, 10, 10))
    full = read_band(tif)
    assert full.shape == (10, 10)
    cropped = read_band(tif, bbox=(2, 2, 8, 8))
    assert cropped.shape[0] < 10 and cropped.shape[1] < 10
    b = bounds(tif)
    assert b == (0, 0, 10, 10)
