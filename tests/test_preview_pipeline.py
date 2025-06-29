from pathlib import Path

import pandas as pd
from PIL import Image

import sys
import types

# Stub heavy optional dependencies before importing the module
sys.modules.setdefault("earthaccess", types.ModuleType("earthaccess"))
ok_mod = types.ModuleType("pykrige.ok")
ok_mod.OrdinaryKriging = object
pykrige_mod = types.ModuleType("pykrige")
pykrige_mod.ok = ok_mod
sys.modules.setdefault("pykrige", pykrige_mod)
sys.modules.setdefault("pykrige.ok", ok_mod)

from preview_pipeline import create_interactive_map, visualize_gedi_points
from sentinel_utils import read_bbox_metadata


def _make_img(path: Path) -> None:
    Image.new("RGB", (2, 2), color="white").save(path)


def test_create_interactive_map_full_option(tmp_path: Path) -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)
    (_make_img(tmp_path / "sentinel_true_color_web.jpg"))
    (_make_img(tmp_path / "sentinel_true_color_clean.png"))
    (_make_img(tmp_path / "sentinel_ndvi_diff_web.jpg"))
    (_make_img(tmp_path / "sentinel_ndvi_diff_clean.png"))

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=False,
        include_full_srtm=True,
        include_full_aw3d=True,
    )
    html = (tmp_path / "interactive_map.html").read_text()
    assert "sentinel_true_color_web" not in html
    assert "sentinel_ndvi_diff_web" not in html

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=True,
        include_full_srtm=True,
        include_full_aw3d=True,
    )
    html = (tmp_path / "interactive_map.html").read_text()
    assert "sentinel_true_color_web" in html
    assert "sentinel_ndvi_diff_web" in html


def test_create_interactive_map_dem_overlays(tmp_path: Path) -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)

    (_make_img(tmp_path / "1b_srtm_crop_hillshade.png"))
    (_make_img(tmp_path / "1c_aw3d30_crop_hillshade.png"))
    (_make_img(tmp_path / "1b_srtm_mosaic_hillshade.png"))
    (_make_img(tmp_path / "1c_aw3d30_mosaic_hillshade.png"))

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=False,
        include_full_srtm=True,
        include_full_aw3d=True,
    )

    html = (tmp_path / "interactive_map.html").read_text()
    assert "1b_srtm_crop_hillshade" in html
    assert "1c_aw3d30_crop_hillshade" in html
    assert "1b_srtm_mosaic_hillshade" in html
    assert "1c_aw3d30_mosaic_hillshade" in html


def test_create_interactive_map_dem_optional(tmp_path: Path) -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)

    (_make_img(tmp_path / "1b_srtm_crop_hillshade.png"))
    (_make_img(tmp_path / "1c_aw3d30_crop_hillshade.png"))

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=False,
        include_full_srtm=False,
        include_full_aw3d=False,
    )

    html = (tmp_path / "interactive_map.html").read_text()
    assert "1b_srtm_crop_hillshade" in html
    assert "1c_aw3d30_crop_hillshade" in html
    assert "1b_srtm_mosaic_hillshade" not in html
    assert "1c_aw3d30_mosaic_hillshade" not in html


def test_create_interactive_map_ndvi_diff_clean(tmp_path: Path) -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)

    (_make_img(tmp_path / "sentinel_ndvi_diff_clean.png"))

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=False,
        include_full_srtm=False,
        include_full_aw3d=False,
    )

    html = (tmp_path / "interactive_map.html").read_text()
    assert "sentinel_ndvi_diff_clean" in html


def test_visualize_gedi_points_basic(tmp_path: Path) -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)
    points = [
        (0.25, 0.25, 10.0),
        (0.75, 0.75, 20.0),
    ]

    visualize_gedi_points(points, bbox, tmp_path)

    for name in ["2_gedi_points.png", "2_gedi_points_clean.png"]:
        p = tmp_path / name
        assert p.exists()
        assert read_bbox_metadata(p) == bbox


def test_create_interactive_map_chatgpt_fallback(tmp_path: Path) -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)
    (tmp_path / "chatgpt_analysis.txt").write_text("ID 1 0 S, 0 W score = 1.0")

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=False,
        include_full_srtm=False,
        include_full_aw3d=False,
    )

    html = (tmp_path / "interactive_map.html").read_text()
    assert "ChatGPT Detections" in html

