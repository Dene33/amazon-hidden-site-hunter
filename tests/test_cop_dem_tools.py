import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
import pytest

import geopandas as gpd
from shapely.geometry import Point

from cop_dem_tools import (
    cop_tile_url,
    srtm_tile_url,
    aw3d_tile_url,
    mosaic_cop_tiles,
    crop_to_bbox,
    _dem_to_overlay,
    save_dem_png,
    dem_bounds,
    save_surface_png,
    save_residual_png,
    save_anomaly_points_png,
)


def _create_tile(path: Path, value: float, bounds: tuple[float, float, float, float], size: int = 100) -> None:
    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, size, size)
    data = np.full((size, size), value, dtype=np.float32)
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_cop_tile_url():
    url = cop_tile_url(5.2, -3.7)
    assert "N05_00_W004_00" in url


def test_srtm_tile_url():
    url = srtm_tile_url(-1.3, 12.8)
    assert url.endswith("S02E012.tif")


def test_aw3d_tile_url():
    url = aw3d_tile_url(-1.3, 12.8)
    assert url.endswith("S02E012_DSM.tif")


def test_mosaic_and_crop(tmp_path: Path):
    t1 = tmp_path / "tile1.tif"
    t2 = tmp_path / "tile2.tif"
    _create_tile(t1, 10, (0, 0, 1, 1))
    _create_tile(t2, 20, (1, 0, 2, 1))

    mosaic = mosaic_cop_tiles([t1, t2], tmp_path / "mosaic.tif", (0, 0, 2, 1))
    assert mosaic.exists()

    crop = crop_to_bbox(mosaic, (0.5, 0, 1.5, 1), tmp_path / "crop.tif")
    with rio.open(crop) as src:
        assert src.bounds.left == pytest.approx(0.5)
        assert src.bounds.right == pytest.approx(1.5)
        data = src.read(1)
        assert np.isfinite(data).all()

    # Save mosaic as PNG and ensure it exists
    png = tmp_path / "mosaic.png"
    save_dem_png(mosaic, png)
    assert png.exists()


def test_dem_to_overlay(tmp_path: Path):
    t = tmp_path / "tile.tif"
    _create_tile(t, 5, (0, 0, 1, 1))
    with rio.open(t) as src:
        rgba, bounds = _dem_to_overlay(src)
    assert rgba.shape[2] == 4
    assert bounds[0][0] == pytest.approx(0)
    assert bounds[1][1] == pytest.approx(1)


def test_save_dem_png(tmp_path: Path):
    t = tmp_path / "tile.tif"
    out = tmp_path / "out.png"
    _create_tile(t, 7, (0, 0, 1, 1))
    save_dem_png(t, out)
    assert out.exists()
    from PIL import Image
    with Image.open(out) as img:
        assert img.size == (100, 100)
        assert img.mode == "RGBA"


def test_dem_bounds(tmp_path: Path):
    t = tmp_path / "tile.tif"
    _create_tile(t, 9, (1, 2, 3, 4))
    bounds = dem_bounds(t)
    assert bounds == pytest.approx((1, 2, 3, 4))


def test_save_surface_png(tmp_path: Path):
    xi = np.linspace(0, 1, 10)
    yi = np.linspace(0, 1, 20)
    xi_m, yi_m = np.meshgrid(xi, yi)
    zi = xi_m + yi_m
    out = tmp_path / "surf.png"
    save_surface_png(xi_m, yi_m, zi, out)
    assert out.exists()
    from PIL import Image
    with Image.open(out) as img:
        assert img.size == (xi_m.shape[1], yi_m.shape[0])
        assert img.mode == "RGBA"


def test_save_residual_png(tmp_path: Path):
    rrm = np.array([[1, -1], [0, 2]], dtype=float)
    out = tmp_path / "rrm.png"
    save_residual_png(rrm, out)
    assert out.exists()
    from PIL import Image
    with Image.open(out) as img:
        assert img.size == (rrm.shape[1], rrm.shape[0])
        assert img.mode == "RGBA"


def test_save_anomaly_points_png(tmp_path: Path):
    xi = np.linspace(0, 1, 5)
    yi = np.linspace(0, 1, 5)
    xi_m, yi_m = np.meshgrid(xi, yi)
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [Point(0.5, 0.0), Point(0.5, 1.0)],
            "score": [1, 2],
        },
        crs="EPSG:4326",
    )
    out = tmp_path / "anom.png"
    save_anomaly_points_png(gdf, xi_m, yi_m, out)
    assert out.exists()
    from PIL import Image
    with Image.open(out) as img:
        arr = np.array(img)
        assert img.size == (xi_m.shape[1], yi_m.shape[0])
        assert img.mode == "RGBA"
        # north anomaly at top row, south anomaly at bottom row
        assert arr[0, 2, 3] > 0
        assert arr[-1, 2, 3] > 0
