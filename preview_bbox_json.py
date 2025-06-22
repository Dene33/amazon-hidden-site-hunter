#!/usr/bin/env python3
"""Interactively review bounding boxes saved as JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import webbrowser

import folium


def build_map(json_dir: Path, html_out: Path, *, output_json: Path) -> None:
    """Display all bounding boxes from ``json_dir`` on a Folium map."""
    if not json_dir.exists():
        raise FileNotFoundError(json_dir)

    bboxes: list[list[float]] = []
    for p in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            for b in data:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    bboxes.append([float(x) for x in b])

    if not bboxes:
        raise ValueError("no bounding boxes found")

    xmin = min(b[0] for b in bboxes)
    ymin = min(b[1] for b in bboxes)
    xmax = max(b[2] for b in bboxes)
    ymax = max(b[3] for b in bboxes)

    center = [(ymin + ymax) / 2, (xmin + xmax) / 2]
    m = folium.Map(location=center, zoom_start=8, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Esri World Imagery",
        attr="Esri",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    group = folium.FeatureGroup(name="BBoxes").add_to(m)
    for b in bboxes:
        rect = folium.Rectangle(
            bounds=[[b[1], b[0]], [b[3], b[2]]],
            color="red",
            weight=2,
            fill=False,
        )
        rect.add_to(group)

    map_id = m.get_name()
    group_id = group.get_name()
    js = f"""
    setTimeout(function() {{
        function setupAltDelete(layer) {{
            layer.on('click', function(e) {{
                if (e.originalEvent && e.originalEvent.altKey) {{
                    {group_id}.removeLayer(layer);
                }}
            }});
        }}
        {group_id}.eachLayer(setupAltDelete);
        var SaveControl = L.Control.extend({{
            options: {{position: 'topleft'}},
            onAdd: function() {{
                var btn = L.DomUtil.create('button', 'save-bbox-button');
                btn.innerHTML = 'Save';
                L.DomEvent.on(btn, 'click', function() {{
                    var bboxes = [];
                    {group_id}.eachLayer(function(l) {{
                        if (l instanceof L.Rectangle) {{
                            var b = l.getBounds();
                            bboxes.push([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
                        }}
                    }});
                    var data = JSON.stringify(bboxes, null, 2);
                    var a = document.createElement('a');
                    a.href = URL.createObjectURL(new Blob([data], {{type: 'application/json'}}));
                    a.download = '{output_json.name}';
                    a.click();
                    URL.revokeObjectURL(a.href);
                }});
                return btn;
            }}
        }});
        new SaveControl().addTo({map_id});
    }}, 0);
    """
    m.get_root().script.add_child(folium.Element(js))

    folium.LayerControl().add_to(m)
    m.save(html_out)
    print(f"Map saved to {html_out.resolve()}")
    try:
        webbrowser.open(html_out.resolve().as_uri())
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Review bbox JSONs in an interactive map")
    ap.add_argument('json_dir', help='Directory containing bbox JSON files')
    ap.add_argument('-o', '--output-html', default='bbox_review.html', help='HTML output')
    ap.add_argument('-j', '--output-json', default='bboxes_clean.json', help='Filename for exported JSON')
    args = ap.parse_args()
    build_map(Path(args.json_dir), Path(args.output_html), output_json=Path(args.output_json))


if __name__ == '__main__':
    main()
