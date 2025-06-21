import argparse
from pathlib import Path
import tempfile
import webbrowser

import folium
import numpy as np
from PIL import Image

from sentinel_utils import mosaic_images, read_bbox_metadata


def build_map(out_dir: Path, output: Path) -> None:
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)

    bbox_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    images_by_name = {}
    for bdir in bbox_dirs:
        for p in bdir.glob('*.[jp][pn]g'):
            images_by_name.setdefault(p.name, []).append(p)

    bounds_global = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    overlays = []

    for name, paths in images_by_name.items():
        if len(paths) == 1:
            img_path = paths[0]
        else:
            tmp = Path(tempfile.gettempdir()) / f"mosaic_{name}"
            mosaic_images(paths, tmp)
            img_path = tmp
        bbox = read_bbox_metadata(img_path)
        if bbox is None:
            continue
        with Image.open(img_path) as img:
            img_arr = np.array(img)
        overlay_bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
        overlays.append((name, img_arr, overlay_bounds))

        bounds_global[0] = min(bounds_global[0], bbox[0])
        bounds_global[1] = min(bounds_global[1], bbox[1])
        bounds_global[2] = max(bounds_global[2], bbox[2])
        bounds_global[3] = max(bounds_global[3], bbox[3])

    center = [(bounds_global[1] + bounds_global[3]) / 2,
              (bounds_global[0] + bounds_global[2]) / 2]
    m = folium.Map(location=center, zoom_start=8, tiles="OpenStreetMap")

    for name, img_arr, overlay_bounds in overlays:
        folium.raster_layers.ImageOverlay(
            image=img_arr,
            bounds=overlay_bounds,
            name=name,
            mercator_project=True,
        ).add_to(m)

    folium.LayerControl().add_to(m)

    output.write_text(folium.Figure().add_child(m).render())
    print(f"Map saved to {output.resolve()}")
    try:
        webbrowser.open(output.resolve().as_uri())
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview mosaiced bbox images on a Folium map")
    ap.add_argument('out_dir', help='Directory containing bbox folders')
    ap.add_argument('-o', '--output', default='bbox_tiles.html', help='HTML output file')
    args = ap.parse_args()
    build_map(Path(args.out_dir), Path(args.output))


if __name__ == '__main__':
    main()
