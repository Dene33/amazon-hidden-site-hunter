import argparse
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins, Element
import json
import os
from sentinel_utils import read_bbox_metadata
from data_vis import create_combined_map, load_reference_datasets

DEFAULT_DIRS = [Path("pipeline_out")]

def find_bbox_folders(base_dirs):
    pattern = re.compile(r"^-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?$")
    for base in base_dirs:
        base = Path(base)
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.is_dir() and pattern.match(p.name):
                try:
                    bbox = tuple(float(x) for x in p.name.split("_"))
                    yield p, bbox
                except ValueError:
                    continue

def main():
    ap = argparse.ArgumentParser(description="Visualize results from multiple bbox folders")
    ap.add_argument("dirs", nargs="*", type=Path, default=DEFAULT_DIRS, help="Base directories to search")
    ap.add_argument("--output", default="all_bboxes_map.html", help="Output HTML file")
    ap.add_argument("--include-data-vis", action="store_true", default=True, help="Include reference datasets")
    ap.add_argument(
        "--draw-bboxes",
        action="store_true",
        help="Allow drawing additional bounding boxes on the map",
    )
    ap.add_argument(
        "--yaml-output",
        default="drawn_bboxes.yaml",
        help="Filename for exported drawn bounding boxes",
    )
    args = ap.parse_args()

    bbox_dirs = list(find_bbox_folders(args.dirs))

    bbox_infos: list[tuple[str, tuple[float, float, float, float]]] = []
    bbox_images: list[dict[str, list[dict]]] = []
    all_image_types: set[str] = set()

    anomalies_fg = folium.FeatureGroup(name="Anomalies", show=True, control=True)
    gpt_fg = folium.FeatureGroup(name="ChatGPT", show=True, control=True)

    for idx, (folder, bbox) in enumerate(bbox_dirs):
        bbox_infos.append((folder.name, bbox))

        anom_path = folder / "anomalies.geojson"
        if anom_path.exists():
            try:
                gdf = gpd.read_file(anom_path)
                if not gdf.empty:
                    def _color(s):
                        if s > 5:
                            return "#FF0000"
                        if s > 3:
                            return "#FFA500"
                        return "#FFFF00"
                    for _, row in gdf.iterrows():
                        pt = row.geometry
                        sc = row.get("score", 0)
                        folium.CircleMarker(
                            [pt.y, pt.x],
                            radius=6,
                            color=_color(sc),
                            fill=True,
                            fill_color=_color(sc),
                            fill_opacity=0.7,
                            tooltip=f"Anomaly {sc:.2f}",
                        ).add_to(anomalies_fg)
            except Exception as exc:
                print(f"Could not read {anom_path}: {exc}")

        gpt_path = folder / "chatgpt_points.csv"
        if gpt_path.exists():
            try:
                df = pd.read_csv(gpt_path)
                if not df.empty:
                    for _, row in df.iterrows():
                        popup = f"Score: {row.get('score', '')}"
                        if isinstance(row.get("description", None), str):
                            popup += f"<br>{row['description']}"
                        folium.Marker(
                            location=[row["latitude"], row["longitude"]],
                            icon=folium.Icon(color="blue", icon="flag"),
                            popup=folium.Popup(popup, max_width=400),
                        ).add_to(gpt_fg)
            except Exception as exc:
                print(f"Could not read {gpt_path}: {exc}")

        imgs: dict[str, list[dict]] = {}
        for p in sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")):
            if p.name == "interactive_map.html" or p.name.startswith("2_gedi_points"):
                continue
            bbox_meta = read_bbox_metadata(p)
            if bbox_meta is None:
                continue
            bounds = [[bbox_meta[1], bbox_meta[0]], [bbox_meta[3], bbox_meta[2]]]
            rel_path = os.path.relpath(p, Path(args.output).resolve().parent)
            img_type = p.stem
            imgs.setdefault(img_type, []).append({"path": rel_path, "bounds": bounds})
            all_image_types.add(img_type)

        bbox_images.append(imgs)

    if args.include_data_vis:
        arch_dfs, lidar_df, image_files = load_reference_datasets()
    else:
        arch_dfs, lidar_df, image_files = [], None, []

    m = create_combined_map(arch_dfs, lidar_df, image_files, bbox=None)

    highlight_js = """
    window._ctrlDown = false;
    window._altDown = false;
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Control') _ctrlDown = true;
        if (e.key === 'Alt') _altDown = true;
    });
    document.addEventListener('keyup', function(e) {
        if (e.key === 'Control') _ctrlDown = false;
        if (e.key === 'Alt') _altDown = false;
    });
    """
    m.get_root().script.add_child(Element(highlight_js))

    # Remove the initial LayerControl so newly added layers show up
    for key, child in list(m._children.items()):
        if isinstance(child, folium.map.LayerControl):
            del m._children[key]

    if bbox_infos:
        all_bboxes = [b for _, b in bbox_infos]
        xmin = min(b[0] for b in all_bboxes)
        ymin = min(b[1] for b in all_bboxes)
        xmax = max(b[2] for b in all_bboxes)
        ymax = max(b[3] for b in all_bboxes)
        center = [(ymin + ymax) / 2, (xmin + xmax) / 2]
        m.location = center
        bbox_group = folium.FeatureGroup(name="Bounding Boxes", show=True, control=True)
        bbox_group.add_to(m)
        anomalies_fg.add_to(m)
        gpt_fg.add_to(m)

        img_type_groups: dict[str, folium.FeatureGroup] = {}
        for img_type in sorted(all_image_types):
            fg = folium.FeatureGroup(name=f"Image: {img_type}", show=False, control=True)
            fg.add_to(m)
            img_type_groups[img_type] = fg

        for idx, ((name, b), imgs) in enumerate(zip(bbox_infos, bbox_images)):
            rect = folium.Rectangle(
                [[b[1], b[0]], [b[3], b[2]]],
                color="red",
                weight=2,
                fill=True,
                fill_opacity=0,
            )
            rect.add_to(bbox_group)

            groups_js = "{" + ", ".join(
                f"'{k}': {img_type_groups[k].get_name()}" for k in imgs
            ) + "}"

            js = f"""
            setTimeout(function() {{
                var rect_{idx} = {rect.get_name()};
                var imgs_{idx} = {json.dumps(imgs)};
                var groups_{idx} = {groups_js};
                var overlays_{idx} = {{}};
                var tooltip_{idx} = L.tooltip({{className: 'bbox-label'}}).setContent({json.dumps(name)});
                var defaultStyle_{idx} = {{color: 'red', weight: 2, fillOpacity: 0}};
                var highlightStyle_{idx} = {{color: 'yellow', weight: 3, fillOpacity: 0.3}};
                rect_{idx}.isHovered = false;

                function updateStyle_{idx}() {{
                    if (rect_{idx}.isHovered && (window._ctrlDown || window._altDown)) {{
                        rect_{idx}.setStyle(highlightStyle_{idx});
                    }} else {{
                        rect_{idx}.setStyle(defaultStyle_{idx});
                    }}
                }}

                document.addEventListener('keydown', updateStyle_{idx});
                document.addEventListener('keyup', updateStyle_{idx});

                rect_{idx}.on('mouseover', function(e) {{
                    rect_{idx}.isHovered = true;
                    tooltip_{idx}.setLatLng([{b[3]}, {b[0]}]).addTo({m.get_name()});
                    updateStyle_{idx}();
                }});
                rect_{idx}.on('mouseout', function(e) {{
                    rect_{idx}.isHovered = false;
                    {m.get_name()}.removeLayer(tooltip_{idx});
                    updateStyle_{idx}();
                }});

                rect_{idx}.on('click', function(e) {{
                    if (e.originalEvent.altKey) {{
                        for (var t in overlays_{idx}) {{
                            overlays_{idx}[t].forEach(function(o) {{ groups_{idx}[t].removeLayer(o); }});
                        }}
                    }} else if (e.originalEvent.ctrlKey) {{
                        for (var t in imgs_{idx}) {{
                            var grp = groups_{idx}[t];
                            if (!overlays_{idx}[t]) overlays_{idx}[t] = [];
                            if (overlays_{idx}[t].length === 0) {{
                                imgs_{idx}[t].forEach(function(info) {{
                                    var o = L.imageOverlay(info.path, info.bounds);
                                    overlays_{idx}[t].push(o);
                                    o.addTo(grp);
                                }});
                            }} else {{
                                overlays_{idx}[t].forEach(function(o) {{ if (!grp.hasLayer(o)) o.addTo(grp); }});
                            }}
                        }}
                    }}
                }});
            }}, 0);
            """
            m.get_root().script.add_child(Element(js))

        m.fit_bounds([[ymin, xmin], [ymax, xmax]])

    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(primary_length_unit='kilometers').add_to(m)
    plugins.MiniMap().add_to(m)

    if args.draw_bboxes:
        drawn = folium.FeatureGroup(name="DrawnBBoxes", show=True, control=False)
        drawn.add_to(m)
        plugins.Draw(
            export=False,
            position="topleft",
            feature_group=drawn,
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "rectangle": {"shapeOptions": {"color": "green"}},
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        save_js = f"""
        setTimeout(function() {{
            var grp = {drawn.get_name()};
            var SaveCtrl = L.Control.extend({{
                options: {{ position: 'topleft' }},
                onAdd: function() {{
                    var btn = L.DomUtil.create('button', 'save-drawn-bbox');
                    btn.innerHTML = 'Save BBoxes';
                    L.DomEvent.on(btn, 'click', function() {{
                        var bboxes = [];
                        grp.eachLayer(function(l) {{
                            if (l instanceof L.Rectangle) {{
                                var b = l.getBounds();
                                bboxes.push([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
                            }}
                        }});
                        var yaml = 'bbox:\n';
                        bboxes.forEach(function(b) {{
                            yaml += '  - [' + b[0] + ', ' + b[1] + ', ' + b[2] + ', ' + b[3] + ']\n';
                        }});
                        var a = document.createElement('a');
                        a.href = URL.createObjectURL(new Blob([yaml], {{type: 'text/yaml'}}));
                        a.download = {json.dumps(args.yaml_output)};
                        a.click();
                        URL.revokeObjectURL(a.href);
                    }});
                    return btn;
                }}
            }});
            new SaveCtrl().addTo({m.get_name()});
        }}, 0);
        """
        m.get_root().script.add_child(Element(save_js))

    m.save(args.output)
    print(f"Map saved to {args.output}")

if __name__ == "__main__":
    main()
