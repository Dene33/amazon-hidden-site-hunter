import argparse
from pathlib import Path
import tempfile
import webbrowser

import folium
from folium import plugins
import numpy as np
from PIL import Image

from sentinel_utils import mosaic_images, read_bbox_metadata

# Disable Pillow's decompression bomb check for large images
Image.MAX_IMAGE_PIXELS = None


def build_map(
    out_dir: Path,
    output: Path,
    *,
    mosaic: bool = False,
    sentinel_scale: float = 1.0,
    draw_file: Path | None = None,
) -> None:
    """Create an interactive map from georeferenced images.

    ``out_dir`` may either contain multiple bbox subdirectories or a single
    directory of pre-combined images. When ``mosaic`` is True, images with the
    same filename across bbox folders are merged before being displayed.

    ``sentinel_scale`` can be used to downscale PNG images whose filenames
    contain ``"sentinel"`` before adding them to the map.  JPEG images are left
    untouched.

    When ``draw_file`` is provided, a drawing toolbar is added so rectangles can
    be drawn and removed with ``Alt``-click. A Save button downloads the
    resulting bounding boxes to ``draw_file`` in JSON format.
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
                if (
                    sentinel_scale != 1.0
                    and "sentinel" in img_path.name.lower()
                    and img_path.suffix.lower() == ".png"
                ):
                    w, h = img.size
                    new_size = (
                        max(1, int(w * sentinel_scale)),
                        max(1, int(h * sentinel_scale)),
                    )
                    img = img.resize(new_size, resample=Image.Resampling.BILINEAR)
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
    m = folium.Map(location=center, zoom_start=8, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Esri World Imagery",
        attr="Esri",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    for name, img_arr, overlay_bounds in overlays:
        folium.raster_layers.ImageOverlay(
            image=img_arr,
            bounds=overlay_bounds,
            name=name,
            mercator_project=True,
        ).add_to(m)

    if draw_file:
        draw = plugins.Draw(
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "rectangle": True,
            },
            edit_options={"edit": False, "remove": False},
        )
        draw.add_to(m)
        map_id = m.get_name()
        control_id = draw.get_name()
        feature_group = f"drawnItems_{control_id}"
        js = f"""
        setTimeout(function() {{
            function setupAltDelete(layer) {{
                layer.on('click', function(e) {{
                    if (e.originalEvent && e.originalEvent.altKey) {{
                        {feature_group}.removeLayer(layer);
                    }}
                }});
            }}
            {feature_group}.eachLayer(setupAltDelete);
            {map_id}.on('draw:created', function(e) {{
                var layer = e.layer;
                setupAltDelete(layer);
            }});
            var SaveControl = L.Control.extend({{
                options: {{position: 'topleft'}},
                onAdd: function() {{
                    var btn = L.DomUtil.create('button', 'save-bbox-button');
                    btn.innerHTML = 'Save';
                    L.DomEvent.on(btn, 'click', function() {{
                        var bboxes = [];
                        {feature_group}.eachLayer(function(l) {{
                            if (l instanceof L.Rectangle) {{
                                var b = l.getBounds();
                                bboxes.push([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
                            }}
                        }});
                        var data = JSON.stringify(bboxes, null, 2);
                        var a = document.createElement('a');
                        a.href = URL.createObjectURL(new Blob([data], {{type: 'application/json'}}));
                        a.download = '{draw_file.name}';
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

    # Save the map to ``output`` and open it in a browser. ``folium.Map.save``
    # writes a fully self-contained HTML file with all required resources.
    m.save(output)
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
    ap.add_argument(
        '--sentinel-scale',
        type=float,
        default=1.0,
        help='Scale factor for PNG images containing "sentinel" in their name',
    )
    ap.add_argument(
        '--draw-bboxes',
        metavar='FILE',
        help='Enable drawing rectangles that can be saved to FILE with a Save button',
    )
    args = ap.parse_args()
    build_map(
        Path(args.out_dir),
        Path(args.output),
        mosaic=args.mosaic,
        sentinel_scale=args.sentinel_scale,
        draw_file=Path(args.draw_bboxes) if args.draw_bboxes else None,
    )


if __name__ == '__main__':
    main()
