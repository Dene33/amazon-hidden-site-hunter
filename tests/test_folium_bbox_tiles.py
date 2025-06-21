from pathlib import Path
from PIL import Image

from folium_bbox_tiles import build_map
from sentinel_utils import save_image_with_metadata


def test_build_map_draw(tmp_path: Path) -> None:
    bdir = tmp_path / "bbox1"
    bdir.mkdir()
    img = Image.new("RGB", (2, 2), color="red")
    save_image_with_metadata(img, bdir / "img.png", bbox=(0, 0, 1, 1))

    html = tmp_path / "map.html"
    build_map(tmp_path, html, draw_file=tmp_path / "boxes.json")
    out = html.read_text()
    assert "Esri World Imagery" in out
    assert "save-bbox-button" in out
