import argparse
from pathlib import Path
import tempfile
import webbrowser

import folium
import numpy as np
from PIL import Image

from sentinel_utils import mosaic_images, read_bbox_metadata


def build_map(out_dir: Path, output: Path, *, mosaic: bool = False) -> None:
    """Create an interactive map from georeferenced images.

    ``out_dir`` may either contain multiple bbox subdirectories or a single
    directory of pre-combined images. When ``mosaic`` is True, images with the
    same filename across bbox folders are merged before being displayed.
    """
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)

    bbox_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    images_by_name: dict[str, list[Path]] = {}
    if bbox_dirs:
        for bdir in bbox_dirs:
            for p in bdir.glob('*.[jp][pn]g'):
                images_by_name.setdefault(p.name, []).append(p)
    else:
        for p in out_dir.glob('*.[jp][pn]g'):
            images_by_name.setdefault(p.name, []).append(p)

    bounds_global = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    overlays = []

    for name, paths in images_by_name.items():
        if mosaic and len(paths) > 1:
            tmp = Path(tempfile.gettempdir()) / f"mosaic_{name}"
            mosaic_images(paths, tmp)
            paths = [tmp]

        for idx, img_path in enumerate(paths, start=1):
            bbox = read_bbox_metadata(img_path)
            if bbox is None:
                continue
            with Image.open(img_path) as img:
                img_arr = np.array(img)
            overlay_bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
            layer_name = name if len(paths) == 1 else f"{name} {idx}"
            overlays.append((layer_name, img_arr, overlay_bounds))

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
    ap = argparse.ArgumentParser(
        description="Preview bbox images on a Folium map"
    )
    ap.add_argument('out_dir', help='Directory with bbox folders or combined images')
    ap.add_argument('-o', '--output', default='bbox_tiles.html', help='HTML output file')
    ap.add_argument('--mosaic', action='store_true', help='Combine tiles with the same name before displaying')
    args = ap.parse_args()
    build_map(Path(args.out_dir), Path(args.output), mosaic=args.mosaic)


if __name__ == '__main__':
    main()
