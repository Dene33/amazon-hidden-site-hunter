from pathlib import Path
import json
from preview_bbox_json import build_map


def test_build_map(tmp_path: Path) -> None:
    d = tmp_path / "bboxes"
    d.mkdir()
    (d / "a.json").write_text(json.dumps([[0, 0, 1, 1]]))
    (d / "b.json").write_text(json.dumps([[1, 0, 2, 1]]))

    html = tmp_path / "map.html"
    build_map(d, html, output_json=tmp_path / "out.json")
    out = html.read_text()
    assert "save-bbox-button" in out
    assert "setupAltDelete" in out
