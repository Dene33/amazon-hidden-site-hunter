import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch
import pytest

import numpy as np
import rasterio as rio
import requests
from PIL import Image

from sentinel_utils import (
    bounds,
    compute_ndvi,
    compute_kndvi,
    read_band,
    read_band_like,
    mask_clouds,
    apply_mask,
    hollstein_cloud_mask,
    cloud_mask,
    save_mask_png,
    save_index_png,
    read_bbox_metadata,
    search_sentinel2_item,
    download_bands,
    FILL_VALUE,
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


def test_compute_ndvi_ignore_fill():
    red = np.array([[0.2, FILL_VALUE]])
    nir = np.array([[0.6, 0.6]])
    ndvi = compute_ndvi(red, nir)
    assert np.isnan(ndvi[0, 1])


def test_compute_kndvi_ignore_fill():
    red = np.array([[0.2, FILL_VALUE]])
    nir = np.array([[0.6, 0.6]])
    kndvi = compute_kndvi(red, nir)
    assert np.isnan(kndvi[0, 1])


def test_compute_ndmi_simple():
    nir = np.array([[0.6, 0.5], [0.4, 0.2]])
    swir = np.array([[0.3, 0.2], [0.4, 0.1]])
    from sentinel_utils import compute_ndmi

    ndmi = compute_ndmi(nir, swir)
    assert ndmi.shape == nir.shape
    assert np.all(ndmi >= -1) and np.all(ndmi <= 1)


def test_compute_msi_simple():
    nir = np.array([[0.6, 0.5], [0.4, 0.2]])
    swir = np.array([[0.3, 0.2], [0.4, 0.1]])
    from sentinel_utils import compute_msi

    msi = compute_msi(nir, swir)
    assert msi.shape == nir.shape
    assert np.all(msi >= 0)


def test_compute_ndmi_ignore_fill():
    nir = np.array([[0.6, FILL_VALUE]])
    swir = np.array([[0.3, 0.2]])
    from sentinel_utils import compute_ndmi

    ndmi = compute_ndmi(nir, swir)
    assert np.isnan(ndmi[0, 1])


def test_compute_msi_ignore_fill():
    nir = np.array([[0.6, FILL_VALUE]])
    swir = np.array([[0.3, 0.2]])
    from sentinel_utils import compute_msi

    msi = compute_msi(nir, swir)
    assert np.isnan(msi[0, 1])


def test_search_sentinel_rfc3339():
    bbox = (-1, -1, 1, 1)
    with patch("sentinel_utils.requests.post") as post:
        post.return_value.json.return_value = {"features": []}
        post.return_value.raise_for_status.return_value = None
        search_sentinel2_item(bbox, "2024-01-01", "2024-12-31")
        args, kwargs = post.call_args
        assert kwargs["json"]["datetime"] == "2024-01-01T00:00:00Z/2024-12-31T23:59:59Z"


def test_search_sentinel_grid_code_filter():
    bbox = (-1, -1, 1, 1)
    with patch("sentinel_utils.requests.post") as post:
        post.return_value.json.return_value = {"features": []}
        post.return_value.raise_for_status.return_value = None
        search_sentinel2_item(bbox, "2024-01-01", "2024-12-31", grid_code="MGRS-20LLL")
        args, kwargs = post.call_args
        assert kwargs["json"]["query"]["grid:code"] == {"eq": "MGRS-20LLL"}


def test_search_sentinel_http_error():
    bbox = (-1, -1, 1, 1)
    with patch("sentinel_utils.requests.post") as post:
        post.return_value.raise_for_status.side_effect = requests.HTTPError()
        result = search_sentinel2_item(bbox, "2024-01-01", "2024-12-31")
        assert result is None


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
    save_true_color(b, b, b, tc, dpi=222, bbox=(0, 0, 1, 1))
    with Image.open(tc) as img:
        assert img.info.get("dpi") == (222, 222)
    assert read_bbox_metadata(tc) == (0.0, 0.0, 1.0, 1.0)

    idx = tmp_path / "idx.png"
    save_index_png(b, idx, dpi=333, bbox=(0, 0, 1, 1))
    with Image.open(idx) as img:
        dpi = img.info.get("dpi")
        assert round(dpi[0]) == 333 and round(dpi[1]) == 333
    assert read_bbox_metadata(idx) == (0.0, 0.0, 1.0, 1.0)


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


def test_mask_clouds_basic():
    scl = np.array([[4, 9], [3, 0]], dtype=np.float32)
    red = np.ones_like(scl)
    nir = np.ones_like(scl) * 2
    red_m, nir_m = mask_clouds(scl, red, nir, dilation=0)
    assert np.isnan(red_m[0, 1]) and np.isnan(nir_m[0, 1])
    assert np.isnan(red_m[1, 0]) and np.isnan(nir_m[1, 0])
    assert red_m[0, 0] == 1 and nir_m[0, 0] == 2


def test_mask_clouds_dilation():
    scl = np.array([[4, 9], [4, 4]], dtype=np.float32)
    band = np.ones_like(scl)
    masked, = mask_clouds(scl, band, fill_value=-9999, dilation=1)
    assert masked[0, 0] == -9999


def test_read_band_scale(tmp_path: Path):
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    tif = tmp_path / "scl.tif"
    _create_raster(tif, arr, (0, 0, 2, 2))
    raw = read_band(tif, scale=1.0)
    assert np.array_equal(raw, arr)


def test_read_band_like(tmp_path: Path):
    arr_ref = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    ref = tmp_path / "ref.tif"
    other = tmp_path / "other.tif"
    _create_raster(ref, arr_ref, (0, 0, 4, 4))
    _create_raster(other, arr, (0, 0, 4, 4))
    resampled = read_band_like(other, ref, scale=1.0)
    assert resampled.shape == arr_ref.shape


def test_read_band_like_bbox(tmp_path: Path):
    arr_ref = np.arange(16, dtype=np.float32).reshape(4, 4)
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    ref = tmp_path / "ref.tif"
    other = tmp_path / "other.tif"
    _create_raster(ref, arr_ref, (0, 0, 4, 4))
    _create_raster(other, arr, (0, 0, 4, 4))
    bbox = (1, 1, 3, 3)
    cropped = read_band_like(other, ref, bbox=bbox, scale=1.0)
    expected = read_band(ref, bbox=bbox, scale=1.0)
    assert cropped.shape == expected.shape


def test_cloud_mask_and_save(tmp_path: Path):
    scl = np.array([[0, 9], [4, 1]], dtype=np.float32)
    mask = cloud_mask(scl, dilation=0)
    assert mask.dtype == bool
    assert mask[0, 1] and not mask[1, 0]
    png = tmp_path / "mask.png"
    save_mask_png(mask, png, dpi=123, bbox=(0, 0, 1, 1))
    with Image.open(png) as img:
        dpi_info = img.info.get("dpi")
        assert dpi_info and round(dpi_info[0]) == 123 and round(dpi_info[1]) == 123
    assert read_bbox_metadata(png) == (0.0, 0.0, 1.0, 1.0)


def test_hollstein_cloud_mask_basic():
    mask = hollstein_cloud_mask(
        b01=np.array([[0.4, 0.2]], dtype=np.float32),
        b02=np.array([[0.1, 0.1]], dtype=np.float32),
        b03=np.array([[0.35, 0.1]], dtype=np.float32),
        b05=np.array([[0.2, 0.1]], dtype=np.float32),
        b06=np.array([[0.1, 0.1]], dtype=np.float32),
        b07=np.array([[0.0, 0.0]], dtype=np.float32),
        b8a=np.array([[0.2, 0.05]], dtype=np.float32),
        b09=np.array([[0.1, 0.1]], dtype=np.float32),
        b11=np.array([[0.2, 0.1]], dtype=np.float32),
        dilation=0,
    )
    assert mask.shape == (1, 2)
    assert mask[0, 0]


def test_apply_mask():
    mask = np.array([[True, False]])
    arr = np.array([[1.0, 2.0]])
    masked, = apply_mask(mask, arr, fill_value=-9999)
    assert masked[0, 0] == -9999 and masked[0, 1] == 2.0


def test_save_index_png_all_nan(tmp_path: Path):
    arr = np.full((2, 2), np.nan, dtype=np.float32)
    out = tmp_path / "nan.png"
    save_index_png(arr, out, dpi=120, bbox=(0, 0, 1, 1))
    assert out.exists()
    with Image.open(out) as img:
        dpi_info = img.info.get("dpi")
        assert dpi_info and round(dpi_info[0]) == 120
    assert read_bbox_metadata(out) == (0.0, 0.0, 1.0, 1.0)


def test_save_index_png_constant(tmp_path: Path):
    arr = np.ones((2, 2), dtype=np.float32) * 0.5
    out = tmp_path / "const.png"
    save_index_png(arr, out, dpi=180, bbox=(0, 0, 1, 1))
    with Image.open(out) as img:
        dpi_info = img.info.get("dpi")
        assert dpi_info and round(dpi_info[0]) == 180
        pix = np.array(img)
        assert pix.var() >= 0
    assert read_bbox_metadata(out) == (0.0, 0.0, 1.0, 1.0)
