#!/usr/bin/env python3
"""Interactive grid preview of a bounding box.

This script displays a bounding box on an interactive Folium map and splits
it into a grid of smaller bounding boxes. The grid can be edited directly on
 the map using the Leaflet draw controls. When finished, use the "Export"
button on the map to download the boxes as a GeoJSON file.
"""

import argparse
from pathlib import Path

import folium
from folium.plugins import Draw
from shapely.geometry import box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a bounding box and divide it into a grid"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("xmin", "ymin", "xmax", "ymax"),
        required=True,
        help="Longitude/Latitude bounds",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=0.5,
        help="Size of the sub‑boxes in degrees (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="preview_bbox_multi.html",
        help="Output HTML file",
    )
    parser.add_argument(
        "--json-name",
        default="bboxes.json",
        help="Suggested filename for the exported GeoJSON",
    )
    return parser.parse_args()


def create_grid(bbox: tuple[float, float, float, float], step: float):
    """Return a list of shapely boxes covering *bbox* with given *step*."""
    xmin, ymin, xmax, ymax = bbox
    cells = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            x2 = min(x + step, xmax)
            y2 = min(y + step, ymax)
            cells.append(box(x, y, x2, y2))
            y += step
        x += step
    return cells


def main() -> None:
    args = parse_args()

    xmin, ymin, xmax, ymax = args.bbox
    center = [(ymin + ymax) / 2, (xmin + xmax) / 2]

    m = folium.Map(location=center, zoom_start=10)

    drawn_group = folium.FeatureGroup(name="BBoxes").add_to(m)

    # initial grid
    for cell in create_grid(args.bbox, args.grid_size):
        folium.GeoJson(
            cell.__geo_interface__,
            style_function=lambda _: {
                "color": "blue",
                "weight": 2,
                "fill": True,
                "fillOpacity": 0.05,
            },
        ).add_to(drawn_group)

    # add draw controls so the user can tweak the boxes and export them
    Draw(
        export=True,
        filename=args.json_name,
        feature_group=drawn_group,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
    ).add_to(m)

    folium.LayerControl().add_to(m)

    out = Path(args.output)
    m.save(out)
    print(f"Map saved to {out.resolve()}")
    print("Open the HTML file and use the Export button to save the boxes.")


if __name__ == "__main__":
    main()
