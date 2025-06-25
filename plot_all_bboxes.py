import argparse
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins, Element
import json
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
    ap.add_argument("--include-data-vis", action="store_true", help="Include reference datasets")
    args = ap.parse_args()

    bbox_dirs = list(find_bbox_folders(args.dirs))

    bboxes = []
    bbox_infos = []
    anomaly_frames = []
    gpt_frames = []
    for folder, bbox in bbox_dirs:
        bboxes.append(bbox)
        bbox_infos.append((folder.name, bbox))
        anom_path = folder / "anomalies.geojson"
        if anom_path.exists():
            try:
                gdf = gpd.read_file(anom_path)
                if not gdf.empty:
                    anomaly_frames.append(gdf)
            except Exception as exc:
                print(f"Could not read {anom_path}: {exc}")
        gpt_path = folder / "chatgpt_points.csv"
        if gpt_path.exists():
            try:
                df = pd.read_csv(gpt_path)
                if not df.empty:
                    gpt_frames.append(df)
            except Exception as exc:
                print(f"Could not read {gpt_path}: {exc}")

    anomalies = gpd.GeoDataFrame(pd.concat(anomaly_frames, ignore_index=True)) if anomaly_frames else gpd.GeoDataFrame()
    chatgpt = pd.concat(gpt_frames, ignore_index=True) if gpt_frames else pd.DataFrame(columns=["latitude", "longitude", "score", "description"])

    if args.include_data_vis:
        arch_dfs, lidar_df, image_files = load_reference_datasets()
    else:
        arch_dfs, lidar_df, image_files = [], None, []

    m = create_combined_map(arch_dfs, lidar_df, image_files, bbox=None)

    # Remove the initial LayerControl so newly added layers show up
    for key, child in list(m._children.items()):
        if isinstance(child, folium.map.LayerControl):
            del m._children[key]

    if bbox_infos:
        xmin = min(b[0] for b in bboxes)
        ymin = min(b[1] for b in bboxes)
        xmax = max(b[2] for b in bboxes)
        ymax = max(b[3] for b in bboxes)
        center = [(ymin + ymax) / 2, (xmin + xmax) / 2]
        m.location = center
        bbox_group = folium.FeatureGroup(name="Bounding Boxes", show=True, control=True)
        for idx, (name, b) in enumerate(bbox_infos):
            rect = folium.Rectangle(
                [[b[1], b[0]], [b[3], b[2]]],
                color="red",
                fill=False,
                weight=2,
            )
            rect.add_to(bbox_group)
            js = f"""
            setTimeout(function() {{
                var rect_{idx} = {rect.get_name()};
                var tooltip_{idx} = L.tooltip({{className: 'bbox-label'}}).setContent({json.dumps(name)});
                rect_{idx}.on('mouseover', function(e) {{
                    tooltip_{idx}.setLatLng([{b[3]}, {b[0]}]).addTo({m.get_name()});
                }});
                rect_{idx}.on('mouseout', function(e) {{
                    {m.get_name()}.removeLayer(tooltip_{idx});
                }});
            }}, 0);
            """
            m.get_root().script.add_child(Element(js))
        bbox_group.add_to(m)
        m.fit_bounds([[ymin, xmin], [ymax, xmax]])

    if not anomalies.empty:
        anom_group = folium.FeatureGroup(name="Anomalies", show=True, control=True)
        def _color(s):
            if s > 5:
                return "#FF0000"
            if s > 3:
                return "#FFA500"
            return "#FFFF00"
        for _, row in anomalies.iterrows():
            pt = row.geometry
            sc = row.get("score", 0)
            folium.CircleMarker([pt.y, pt.x], radius=6, color=_color(sc), fill=True,
                                fill_color=_color(sc), fill_opacity=0.7,
                                tooltip=f"Anomaly {sc:.2f}").add_to(anom_group)
        anom_group.add_to(m)

    if not chatgpt.empty:
        gpt_group = folium.FeatureGroup(name="ChatGPT Detections", show=True, control=True)
        for _, row in chatgpt.iterrows():
            popup = f"Score: {row.get('score', '')}"
            if isinstance(row.get("description", None), str):
                popup += f"<br>{row['description']}"
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.Icon(color="blue", icon="flag"),
                popup=folium.Popup(popup, max_width=400),
            ).add_to(gpt_group)
        gpt_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(primary_length_unit='kilometers').add_to(m)
    plugins.MiniMap().add_to(m)

    m.save(args.output)
    print(f"Map saved to {args.output}")

if __name__ == "__main__":
    main()
