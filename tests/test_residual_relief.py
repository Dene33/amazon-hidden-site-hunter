from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds

from detect_hidden_sites import residual_relief


def _create_raster(path: Path, value: float, bounds: tuple[float, float, float, float], size: int = 4) -> None:
    transform = from_bounds(*bounds, size, size)
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


def _bearth_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = np.linspace(0, 1, 4)
    yi = np.linspace(0, 1, 4)
    xi_m, yi_m = np.meshgrid(xi, yi)
    zi = np.full_like(xi_m, 10.0, dtype=np.float32)
    return xi_m, yi_m, zi


def test_residual_relief_single(tmp_path: Path) -> None:
    xi, yi, zi = _bearth_grid()
    dem = tmp_path / "dem.tif"
    _create_raster(dem, 3, (0, 0, 1, 1))

    rrm = residual_relief((xi, yi, zi), dem)

    assert np.allclose(rrm[~np.isnan(rrm)], 7)


def test_residual_relief_average(tmp_path: Path) -> None:
    xi, yi, zi = _bearth_grid()
    d1 = tmp_path / "d1.tif"
    d2 = tmp_path / "d2.tif"
    _create_raster(d1, 2, (0, 0, 1, 1))
    _create_raster(d2, 4, (0, 0, 1, 1))

    rrm = residual_relief((xi, yi, zi), [d1, d2])

    # average dem value is 3
    assert np.allclose(rrm[~np.isnan(rrm)], 7)
