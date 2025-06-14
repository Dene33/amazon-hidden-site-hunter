import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds

from pipeline import step_residual_relief


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


def test_step_residual_relief_name_mapping(tmp_path: Path) -> None:
    xi, yi, zi = _bearth_grid()
    cop = tmp_path / "cop90_crop.tif"
    srtm = tmp_path / "srtm_crop.tif"
    _create_raster(cop, 3, (0, 0, 1, 1))
    _create_raster(srtm, 5, (0, 0, 1, 1))

    cfg = {"enabled": True, "dems": ["cop", "srtm"], "visualize": False}

    rrm = step_residual_relief(cfg, (xi, yi, zi), cop, tmp_path)

    # average DEM value is 4
    assert np.allclose(rrm[~np.isnan(rrm)], 6)


