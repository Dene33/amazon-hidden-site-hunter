import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch

import numpy as np
import rasterio as rio
import requests
from PIL import Image

from sentinel_utils import (
    bounds,
    compute_ndvi,
    compute_kndvi,
    read_band,
    search_sentinel2_item,
    search_sentinel2_items,
    download_bands,
    composite_cloud_free,
)


def test_compute_kndvi_simple():
    red = np.array([[0.2, 0.3], [0.2, 0.1]])
    nir = np.array([[0.6, 0.5], [0.4, 0.2]])
    kndvi = compute_kndvi(red, nir)
    assert kndvi.shape == red.shape
    # Ensure values are between -1 and 1
    assert np.all(kndvi >= -1) and np.all(kndvi <= 1)


def test_compute_ndvi_simple():
    red = np.array([[0.2, 0.3], [0.2, 0.1]])
    nir = np.array([[0.6, 0.5], [0.4, 0.2]])
    ndvi = compute_ndvi(red, nir)
    assert ndvi.shape == red.shape
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1)


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


def test_search_sentinel_items_limit():
    bbox = (-1, -1, 1, 1)
    with patch("sentinel_utils.requests.post") as post:
        post.return_value.json.return_value = {"features": [1, 2, 3]}
        post.return_value.raise_for_status.return_value = None
        items = search_sentinel2_items(bbox, "2024-01-01", "2024-12-31", limit=5)
        args, kwargs = post.call_args
        assert kwargs["json"]["limit"] == 5
        assert items == [1, 2, 3]


def _create_raster(
    path: Path,
    data: np.ndarray,
    bounds: tuple[float, float, float, float],
    crs: str = "EPSG:4326",
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


def test_save_with_dpi(tmp_path: Path):
    from sentinel_utils import save_true_color, save_index_png

    b = np.ones((5, 5), dtype=float)
    tc = tmp_path / "tc.jpg"
    save_true_color(b, b, b, tc, dpi=222)
    with Image.open(tc) as img:
        assert img.info.get("dpi") == (222, 222)

    idx = tmp_path / "idx.jpg"
    save_index_png(b, idx, dpi=333)
    with Image.open(idx) as img:
        assert img.info.get("dpi") == (333, 333)


def test_resize_image(tmp_path: Path):
    from sentinel_utils import resize_image

    src = tmp_path / "orig.jpg"
    Image.new("RGB", (100, 40), color="red").save(src)

    resized = resize_image(src, factor=0.5)

    assert resized.name == "orig_web.jpg"
    with Image.open(resized) as img:
        assert img.size == (50, 20)


def test_download_bands_unique_names(tmp_path: Path) -> None:
    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"123"

    feature = {
        "id": "item123",
        "bbox": [0, 1, 2, 3],
        "properties": {"datetime": "2024-04-30T10:00:00Z"},
        "assets": {"blue": {"href": "dummy"}},
    }

    with patch("sentinel_utils.requests.get", return_value=DummyResponse()):
        paths = download_bands(feature, ["B02"], tmp_path)

    expected = "item123_0.00000_1.00000_2.00000_3.00000_20240430_B02.tif"
    assert paths["B02"].name == expected
    assert paths["B02"].exists()


def test_composite_cloud_free(tmp_path: Path) -> None:
    arr1 = np.full((5, 5), 2000, dtype=np.float32)
    arr2 = np.full((5, 5), 4000, dtype=np.float32)
    scl1 = np.zeros((5, 5), dtype=np.uint8)
    scl1[:, 2:] = 9  # cloud
    scl2 = np.zeros((5, 5), dtype=np.uint8)

    b1 = tmp_path / "b1.tif"
    b2 = tmp_path / "b2.tif"
    s1 = tmp_path / "s1.tif"
    s2 = tmp_path / "s2.tif"

    _create_raster(b1, arr1, (0, 0, 5, 5))
    _create_raster(b2, arr2, (0, 0, 5, 5))
    _create_raster(s1, scl1, (0, 0, 5, 5))
    _create_raster(s2, scl2, (0, 0, 5, 5))

    items = [
        {"B04": b1, "SCL": s1},
        {"B04": b2, "SCL": s2},
    ]

    comp = composite_cloud_free(items, ["B04"], bbox=(0, 0, 5, 5))
    assert comp["B04"].shape == (5, 5)
    assert np.isclose(comp["B04"][0, 0], 0.3, atol=1e-6)
    assert np.isclose(comp["B04"][0, 4], 0.4, atol=1e-6)
