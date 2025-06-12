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

from preview_pipeline import create_interactive_map


def _make_img(path: Path) -> None:
    Image.new("RGB", (2, 2), color="white").save(path)


def test_create_interactive_map_full_option(tmp_path: Path) -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)
    (_make_img(tmp_path / "sentinel_true_color_web.jpg"))
    (_make_img(tmp_path / "sentinel_true_color_clean.png"))

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=False,
    )
    html = (tmp_path / "interactive_map.html").read_text()
    assert "sentinel_true_color_web" not in html

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=True,
    )
    html = (tmp_path / "interactive_map.html").read_text()
    assert "sentinel_true_color_web" in html


def test_create_interactive_map_dem_overlays(tmp_path: Path) -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)

    (_make_img(tmp_path / "1b_srtm_crop_hillshade.png"))
    (_make_img(tmp_path / "1c_aw3d30_crop_hillshade.png"))

    create_interactive_map(
        None,
        pd.DataFrame(),
        bbox,
        tmp_path,
        include_full_sentinel=False,
    )

    html = (tmp_path / "interactive_map.html").read_text()
    assert "1b_srtm_crop_hillshade" in html
    assert "1c_aw3d30_crop_hillshade" in html
