"""
Amazon Archaeological and LiDAR Data Visualization
=================================================

This script provides visualization of archaeological sites, LiDAR coverage,
and image overlays in the Amazon region on a single interactive map.

Requirements:
- pandas
- numpy
- folium
- pyproj
- openpyxl
- shapely
- branca (for custom slider controls)

Install with: pip install pandas numpy folium pyproj openpyxl shapely branca
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
from folium import JavascriptLink, Element
import itertools
# import re
from pyproj import Transformer
# import openpyxl
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os
# from IPython.display import display
# import branca.colormap as cm
# import base64
from folium.raster_layers import ImageOverlay
from PIL import Image

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
MAP_CENTER = [-5.0, -62.5]  # Center of the Amazon region
MAP_ZOOM = 4

# Image overlay bounds
OVERLAY_BOUNDS = [
    [10.15, -80],   # Upper left  (lat, lon)   +0.45°
    [-19.85, -80],  # Lower left               +0.45°
    [-19.85, -45],  # Lower right              +0.45°
    [10.15,  -45],  # Upper right              +0.45°
]

OVERLAY_BOUNDS_CLEAN = [
    [10, -80],   # Upper left  (lat, lon)   +0.45°
    [-20, -80],  # Lower left               +0.45°
    [-20, -45],  # Lower right              +0.45°
    [10, -45],  # Upper right              +0.45°
]

OVERLAY_BOUNDS_COPERNICUS = [[-19.999583333333337, -80.00041666666667], [11.000416666666666, -44.000416666666666]]

OVERLAY_BOUNDS_COPERNICUS_2 = [[-15.999583333333334, -65.00041666666667], [-12.999583333333334, -63.000416666666666]]

# OVERLAY_BOUNDS_COPERNICUS = OVERLAY_BOUNDS

def utm_to_latlon(utm_x, utm_y, utm_zone=19, hemisphere='south'):
    """
    Convert UTM coordinates to latitude/longitude.
    
    Parameters:
    utm_x, utm_y: UTM coordinates
    utm_zone: UTM zone (default 19 for western Brazil)
    hemisphere: 'north' or 'south'
    """
    # Define the coordinate systems
    utm_crs = f"EPSG:326{utm_zone}" if hemisphere == 'north' else f"EPSG:327{utm_zone}"
    wgs84_crs = "EPSG:4326"
    
    # Create transformer
    transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
    
    # Transform coordinates (returns lon, lat)
    lon, lat = transformer.transform(utm_x, utm_y)
    return lat, lon

def row_to_poly(r):
    """Create a polygon from the row data."""
    return Polygon([
        (r["min_lon"], r["min_lat"]),
        (r["min_lon"], r["max_lat"]),
        (r["max_lon"], r["max_lat"]),
        (r["max_lon"], r["min_lat"]),
    ])

def read_mound_villages_data(filepath):
    """Read the mound villages data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        
        # Convert UTM to lat/lon (using UTM zone 19S as indicated in original data)
        lats, lons = [], []
        for _, row in df.iterrows():
            if pd.notna(row['UTM X (Easting)']) and pd.notna(row['UTM Y (Northing)']):
                lat, lon = utm_to_latlon(row['UTM X (Easting)'], row['UTM Y (Northing)'], utm_zone=19)
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        
        df['latitude'] = lats
        df['longitude'] = lons
        df['source'] = 'Mound Villages'
        
        return df
        
    except Exception as e:
        print(f"Error reading mound villages data: {e}")
        return pd.DataFrame()

def read_casarabe_sites_data(filepath):
    """Read the Casarabe sites data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        
        # Convert UTM to lat/lon
        lats, lons = [], []
        for _, row in df.iterrows():
            if pd.notna(row['UTM X (Easting)']) and pd.notna(row['UTM Y (Northing)']):
                # Assuming UTM zone 20S for Casarabe sites (Bolivia region)
                lat, lon = utm_to_latlon(row['UTM X (Easting)'], row['UTM Y (Northing)'], utm_zone=20)
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        
        df['latitude'] = lats
        df['longitude'] = lons
        df['source'] = 'Casarabe Sites'
        
        return df
        
    except Exception as e:
        print(f"Error reading Casarabe sites data: {e}")
        return pd.DataFrame()

def read_geoglyphs_data(filepath):
    """Read the Amazon geoglyphs data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        
        # Convert latitude to numeric (it might be a string)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        df['source'] = 'Amazon Geoglyphs'
        
        return df
        
    except Exception as e:
        print(f"Error reading geoglyphs data: {e}")
        return pd.DataFrame()

def read_submit_data(filepath):
    """Read the submit.csv data."""
    try:
        df = pd.read_csv(filepath)
        
        # Check if coordinates are already in lat/lon format
        sample_x = df['x'].iloc[0] if len(df) > 0 else None
        sample_y = df['y'].iloc[0] if len(df) > 0 else None
        
        if sample_x and -180 <= sample_x <= 180 and -90 <= sample_y <= 90:
            # Coordinates are likely already in lat/lon
            df['latitude'] = df['y']
            df['longitude'] = df['x']
        else:
            # Coordinates might be in UTM, convert them
            df['latitude'] = np.nan
            df['longitude'] = np.nan
            
            # Try to determine appropriate UTM zone based on coordinate ranges
            for idx, row in df.iterrows():
                try:
                    # Try different UTM zones (18-21 are common for Amazon region)
                    for zone in [18, 19, 20, 21]:
                        try:
                            lat, lon = utm_to_latlon(row['x'], row['y'], utm_zone=zone)
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                df.at[idx, 'latitude'] = lat
                                df.at[idx, 'longitude'] = lon
                                break
                        except:
                            continue
                except:
                    continue
        
        df['source'] = 'Archaeological Survey Data'
        
        return df
        
    except Exception as e:
        print(f"Error reading submit data: {e}")
        return pd.DataFrame()

def read_science_data(filepath):
    """Read the science data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        
        # The data already has Latitude and Longitude columns
        df['latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        df['source'] = 'Science Data'
        
        return df
        
    except Exception as e:
        print(f"Error reading science data: {e}")
        return pd.DataFrame()
    
# ---------------------------------------------------------------------
# NEW – reader for the “Lomas …” points that arrived as UTM 20 S
def read_lomas_sites_data(filepath):
    """
    Read the new 'Lomas …' CSV (UTM Zone 20 S) and return a
    dataframe compatible with the visualisation pipeline.
    Expected columns in the raw file
        - Name
        - WGS 1984 UTM Zone 20S (x)
        - WGS 1984 UTM Zone 20S (y)
        - Tier           (→ Classification)
        - Observations
    """
    try:
        df = pd.read_csv(filepath)

        # Harmonise column names so the common popup routine “just works”
        df.rename(
            columns={
                "WGS 1984 UTM Zone 20S (x)": "UTM X (Easting)",
                "WGS 1984 UTM Zone 20S (y)": "UTM Y (Northing)",
                "Name": "Site Name",
                "Tier": "Classification",
            },
            inplace=True,
        )

        # UTM → lat/lon (zone 20 S)
        lats, lons = [], []
        for _, r in df.iterrows():
            if pd.notna(r["UTM X (Easting)"]) and pd.notna(r["UTM Y (Northing)"]):
                lat, lon = utm_to_latlon(
                    r["UTM X (Easting)"], r["UTM Y (Northing)"], utm_zone=20
                )
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)

        df["latitude"] = lats
        df["longitude"] = lons
        df["source"] = "Lomas Sites"          # layer name shown in the map legend
        return df

    except Exception as e:
        print(f"Error reading Lomas sites data: {e}")
        return pd.DataFrame()
# ---------------------------------------------------------------------


def read_my_sites_data(filepath):
    """Read data from .csv of the format:
    latitude,longitude
    -15.896633,-64.570459
    -15.752482,-63.78057
    -15.959823,-63.514838
    -15.950954,-63.35323
    -15.959899,-63.349485

    Args:
        filepath (str): Path to the CSV file containing latitude and longitude data.
    """

    try:
        df = pd.read_csv(filepath)
        
        # Ensure latitude and longitude are numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Add a source column for identification
        df['source'] = 'My Sites'
        
        return df
        
    except Exception as e:
        print(f"Error reading my sites data: {e}")
        return pd.DataFrame()

def load_reference_datasets(base_dir: str = BASE_DIR):
    """Load reference archaeological and LiDAR datasets used for the map.

    Parameters
    ----------
    base_dir : str
        Base directory containing the ``data_vis`` assets.

    Returns
    -------
    tuple
        ``(arch_dataframes, lidar_df, image_files)`` where ``arch_dataframes``
        is a list of ``(name, DataFrame)`` pairs, ``lidar_df`` is a DataFrame of
        LiDAR coverage polygons and ``image_files`` is a list of overlay image
        paths.
    """

    # Find image files in the base directory
    image_files = []
    dv_dir = os.path.join(base_dir, "data_vis")
    if os.path.isdir(dv_dir):
        for filename in os.listdir(dv_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.isfile(
                os.path.join(dv_dir, filename)
            ):
                image_files.append(os.path.join(dv_dir, filename))

    arch_dataframes = []

    # 1. Mound villages
    try:
        mound_df = read_mound_villages_data(
            os.path.join(base_dir, "data_vis/mound-villages-acre/mound_villages_acre.csv")
        )
        if not mound_df.empty:
            arch_dataframes.append(("Mound Villages", mound_df))
    except Exception as e:  # noqa: BLE001
        print(f"  - Mound villages file not found or cannot be read: {e}")

    # 2. Casarabe sites
    try:
        casarabe_df = read_casarabe_sites_data(
            os.path.join(base_dir, "data_vis/casarabe-sites-utm/casarabe_sites_utm.csv")
        )
        if not casarabe_df.empty:
            arch_dataframes.append(("Casarabe Sites", casarabe_df))
    except Exception as e:  # noqa: BLE001
        print(f"  - Casarabe sites file not found or cannot be read: {e}")

    # 3. Amazon geoglyphs
    try:
        geoglyphs_df = read_geoglyphs_data(
            os.path.join(base_dir, "data_vis/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv")
        )
        if not geoglyphs_df.empty:
            if len(geoglyphs_df) > 2000:
                geoglyphs_df = geoglyphs_df.sample(n=2000, random_state=42)
            arch_dataframes.append(("Amazon Geoglyphs", geoglyphs_df))
    except Exception as e:  # noqa: BLE001
        print(f"  - Amazon geoglyphs file not found or cannot be read: {e}")

    # 4. Archaeological survey data
    try:
        submit_df = read_submit_data(
            os.path.join(base_dir, "data_vis/archaeological-survey-data/submit.csv")
        )
        if not submit_df.empty:
            if len(submit_df) > 1000:
                submit_df = submit_df.sample(n=1000, random_state=42)
            arch_dataframes.append(("Archaeological Survey Data", submit_df))
    except Exception as e:  # noqa: BLE001
        print(f"  - Submit data file not found or cannot be read: {e}")

    # 5. Science data
    try:
        science_df = read_science_data(
            os.path.join(base_dir, "data_vis/science_data/science.ade2541_data_s2.csv")
        )
        if not science_df.empty:
            arch_dataframes.append(("Science Data", science_df))
    except Exception as e:  # noqa: BLE001
        print(f"  - Science data file not found or cannot be read: {e}")

    # 6. LiDAR inventory
    try:
        lidar_file = os.path.join(
            base_dir,
            "data_vis/cms_brazil_lidar_tile_inventory/cms_brazil_lidar_tile_inventory.csv",
        )
        lidar_df = pd.read_csv(lidar_file)
        lidar_df["geometry"] = lidar_df.apply(row_to_poly, axis=1)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading LiDAR inventory data: {e}")
        lidar_df = pd.DataFrame()

    # 7. Lomas sites
    try:
        lomas_df = read_lomas_sites_data(os.path.join(base_dir, "data_vis/lomas-sites/lomas_sites.csv"))
        if not lomas_df.empty:
            arch_dataframes.append(("Lomas Sites", lomas_df))
    except Exception as e:  # noqa: BLE001
        print(f"  - Lomas sites file not found or cannot be read: {e}")

    # 8. Custom user sites
    try:
        my_sites_df = read_my_sites_data(os.path.join(base_dir, "data_vis/my-sites/my_sites.csv"))
        if not my_sites_df.empty:
            arch_dataframes.append(("My Sites", my_sites_df))
    except Exception as e:  # noqa: BLE001
        print(f"  - My sites file not found or cannot be read: {e}")

    return arch_dataframes, lidar_df, image_files


def create_combined_map(
    arch_dataframes=None,
    lidar_df=None,
    image_files=None,
    *,
    points=None,
    anomalies=None,
    bbox=None,
):
    arch_dataframes = arch_dataframes or []
    lidar_df = lidar_df if lidar_df is not None else pd.DataFrame()
    image_files = image_files or []

    # Determine center from bbox if provided
    center = MAP_CENTER
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        center = [(ymin + ymax) / 2, (xmin + xmax) / 2]

    # 1️⃣  Build the Map **without** a default tile set
    m = folium.Map(location=center, zoom_start=MAP_ZOOM, tiles=None)

    # Add bounding box if provided
    if bbox:
        folium.Rectangle(
            bounds=[[ymin, xmin], [ymax, xmax]],
            color="red",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.1,
            tooltip="Area of Interest",
        ).add_to(m)

    # 1️⃣ Load the JS for the graticule plugin
    JavascriptLink(
        "https://unpkg.com/leaflet.latlng-graticule@1.0.0/dist/Leaflet.Graticule.min.js"
    ).add_to(m)

    # 2️⃣ Add the graticule after Leaflet has initialised
    graticule_js = f"""
    <script>
        L.latlngGraticule({{
            showLabel: true,
            opacity: 0.6,
            weight: 0.8,
            color: '#ffff00',
            zoomInterval: [
                {{start: 2, end: 3, interval: 30}},
                {{start: 4, end: 4, interval: 10}},
                {{start: 5, end: 7, interval: 5}},
                {{start: 8, end:10, interval: 1}}
            ]
        }}).addTo({m.get_name()});
    </script>
    """
    m.get_root().html.add_child(Element(graticule_js))

    # 2️⃣  Add Esri World Imagery      (satellite)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, "
            "CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community"
        ),
        name="Esri World Imagery",
        overlay=False,      # base-map, not an overlay
        control=True,       # list it in the LayerControl
        show=True           # make it the initial visible layer
    ).add_to(m)

    # (Optional) keep your other base maps as alternatives
    folium.TileLayer(
        "CartoDB positron",
        name="Carto Light",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
        attr="Map tiles by Stamen",
        name="Stamen Terrain",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # OpenStreetMap as an additional base map option
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # """Create a single interactive map with archaeological sites, LiDAR coverage, and image overlays."""
    # # Create base map
    # m = folium.Map(
    #     location=MAP_CENTER,
    #     zoom_start=MAP_ZOOM,
    #     tiles='CartoDB positron'
    # )
    
    # # Add different tile layers with proper attributions
    # folium.TileLayer(
    #     tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
    #     attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    #     name='Stamen Terrain',
    #     overlay=False,
    #     control=True
    # ).add_to(m)
    
    # folium.TileLayer(
    #     tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}{r}.png',
    #     attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    #     name='Stamen Toner',
    #     overlay=False,
    #     control=True
    # ).add_to(m)
    
    
    # Add image overlays with transparency controls
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_name_simple = os.path.splitext(img_name)[0]
        
        # Create a feature group for this image
        img_group = folium.FeatureGroup(
            name=f"Image: {img_name_simple}", show=False, control=True
        )
        
        # Create the image overlay
        bounds = OVERLAY_BOUNDS

        if img_name_simple.startswith("copernicus"):
            # Use a different set of bounds for cleaned images
            bounds = OVERLAY_BOUNDS_COPERNICUS_2
        #     img_url = np.asarray(Image.open(img_path))
        # else:
        #     # Get image URL
        img_url = f"file://{img_path}"
        
        # Create a unique ID for this image's control
        img_id = f"img_{img_name_simple.replace(' ', '_').replace('.', '_')}"
        
        
        
        # Add the image overlay to the feature group
        image_overlay = folium.raster_layers.ImageOverlay(
            image=img_url,
            bounds=bounds,
            opacity=0.7,
            name=img_name_simple,
            cross_origin=False,
            zindex=1,
            interactive=True,
            alt=img_name_simple,
            # mercator_project=True if img_name_simple.startswith("copernicus") else False
        )
        image_overlay.add_to(img_group)
        
        # Create HTML for the opacity slider
                # --- NEW slider_html -------------------------------------------------
        slider_html = f"""
        <div id="slider_{img_id}"
             style="padding:10px;background:#fff;border-radius:6px;margin:10px;width:240px;box-shadow:0 1px 4px rgba(0,0,0,0.3);">
            <strong style="display:block;margin-bottom:4px;">{img_name_simple}</strong>
            <input id="opacity_{img_id}" type="range" min="0" max="100" value="70" style="width:100%;">
            <span id="value_{img_id}">Opacity: 70%</span>
        </div>

        <script>
        (function () {{

            const slider   = document.getElementById("opacity_{img_id}");
            const readout  = document.getElementById("value_{img_id}");
            const imgAlt   = "{img_name_simple}";
            const vShift   = 0;        // <----- change this number if you need a different offset

            function setOpacity () {{
                /* (re-)locate our <img>, adjust opacity and nudge if needed */
                const imgs = document.getElementsByClassName("leaflet-image-layer");
                for (let i = 0; i < imgs.length; i++) {{
                    if (imgs[i].getAttribute("alt") === imgAlt) {{
                        imgs[i].style.opacity = slider.value / 100;                        
                        break;
                    }}
                }}
                readout.textContent = "Opacity: " + slider.value + "%";
            }}

            /* Update while the user drags … */
            slider.addEventListener("input",  setOpacity);

            /* … and after they release (helps some mobile browsers) */
            slider.addEventListener("change", setOpacity);

            /* … and every time Leaflet (re)adds the overlay to the map */
            document.addEventListener("overlayadd", setOpacity);

            /* One initial pass after everything has loaded */
            window.addEventListener("load", () => setTimeout(setOpacity, 0));

        }})();
        </script>
        """
        # ---------------------------------------------------------------------

        
        # Add the slider as a custom control
        slider_control = folium.Element(slider_html)
        m.get_root().html.add_child(slider_control)
        
        img_group.add_to(m)
    
    # Add LiDAR coverage data
    # Create group for LiDAR data
    lidar_group = folium.FeatureGroup(name="LiDAR Coverage", show=False, control=True)
    
    # Dark mask of the total surveyed corridors
    if not lidar_df.empty:
        all_union = unary_union(lidar_df["geometry"].tolist())
        folium.GeoJson(
            data=all_union.__geo_interface__,
            name="Total LiDAR swath",
            style_function=lambda x: {
                "fillColor": "orange",
                "fillOpacity": 0.15,
                "color": "orange",
                "weight": 0,
            },
        ).add_to(lidar_group)
    
        # Add each tile as a thin red outline, grouping by flight-year
        for yr, grp in lidar_df.groupby(lidar_df["created"].str[-4:]):      # pulls "2017","2018" from 214/2017
            year_layer = folium.FeatureGroup(
                name=f"LiDAR Tiles {yr}", show=False, control=True
            )
            for _, r in grp.iterrows():
                folium.GeoJson(
                    data=r["geometry"].__geo_interface__,
                    style_function=lambda x: {
                        "fill": False,
                        "color": "red",
                        "weight": 1,
                    },
                    tooltip=r["filename"],
                ).add_to(year_layer)
            year_layer.add_to(m)
    
    lidar_group.add_to(m)
    
    # Add archaeological sites
    color_cycle = itertools.cycle(
        [
            "red",
            "blue",
            "orange",
            "green",
            "purple",
            "cyan",
            "magenta",
            "yellow",
            "brown",
            "pink",
        ]
    )

    colors = {}
    feature_groups = {}
    for df_name, _ in arch_dataframes:
        if df_name not in colors:
            colors[df_name] = next(color_cycle)
        feature_groups[df_name] = folium.FeatureGroup(name=df_name)
    
    # Add points for each dataset
    total_points = 0
    for df_name, df in arch_dataframes:
        if df.empty:
            continue

        source = df.get('source', [df_name])[0] if isinstance(df, pd.DataFrame) else df_name
        color = colors.get(df_name, 'black')
        
        # Filter valid coordinates
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        
        for idx, row in valid_coords.iterrows():
            # Create popup text
            popup_text = f"<b>Source:</b> {source}<br>"
            
            # Site name handling for different datasets
            if 'Site Name' in row and pd.notna(row['Site Name']):
                popup_text += f"<b>Site:</b> {row['Site Name']}<br>"
            elif 'Site' in row and pd.notna(row['Site']):
                popup_text += f"<b>Site:</b> {row['Site']}<br>"
            elif 'name' in row and pd.notna(row['name']):
                popup_text += f"<b>Site:</b> {row['name']}<br>"
            
            # Classification and type information
            if 'Classification' in row and pd.notna(row['Classification']):
                popup_text += f"<b>Type:</b> {row['Classification']}<br>"
            if 'PlotType' in row and pd.notna(row['PlotType']):
                popup_text += f"<b>Plot Type:</b> {row['PlotType']}<br>"
            if 'type' in row and pd.notna(row['type']):
                popup_text += f"<b>Type:</b> {row['type']}<br>"
            
            # Location information
            if 'Country' in row and pd.notna(row['Country']):
                popup_text += f"<b>Country:</b> {row['Country']}<br>"
            if 'Subdivision' in row and pd.notna(row['Subdivision']):
                popup_text += f"<b>Region:</b> {row['Subdivision']}<br>"
            
            # Numerical data
            if 'Number of mounds' in row and pd.notna(row['Number of mounds']):
                popup_text += f"<b>Mounds:</b> {int(row['Number of mounds'])}<br>"
            if 'Diameter (m)' in row and pd.notna(row['Diameter (m)']):
                popup_text += f"<b>Diameter:</b> {row['Diameter (m)']} m<br>"
            if 'Elevation (m)' in row and pd.notna(row['Elevation (m)']):
                popup_text += f"<b>Elevation:</b> {row['Elevation (m)']} m<br>"
            if 'Altitude' in row and pd.notna(row['Altitude']):
                popup_text += f"<b>Altitude:</b> {row['Altitude']} m<br>"
            if 'PlotSize' in row and pd.notna(row['PlotSize']):
                popup_text += f"<b>Plot Size:</b> {row['PlotSize']}<br>"
            
            # Additional features
            if 'LIDAR' in row and pd.notna(row['LIDAR']):
                popup_text += f"<b>LIDAR Coverage:</b> {row['LIDAR']}<br>"
            if 'Associated features' in row and pd.notna(row['Associated features']):
                popup_text += f"<b>Features:</b> {row['Associated features']}<br>"
            
            popup_text += f"<b>Coordinates:</b> {row['latitude']:.6f}, {row['longitude']:.6f}"
            
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(feature_groups[source])
            
            total_points += 1
    
    # Add feature groups to map
    for fg in feature_groups.values():
        fg.add_to(m)

    # Add GEDI points if provided
    if points:
        gedi_fg = folium.FeatureGroup(name="GEDI Points", show=False, control=True)
        for lat, lon, *_ in points:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color="black",
                fill=True,
                fill_color="black",
                fill_opacity=0.5,
            ).add_to(gedi_fg)
        gedi_fg.add_to(m)

    # Add anomalies if provided
    if anomalies is not None and not anomalies.empty:
        anomalies_layer = folium.FeatureGroup(name="Detected Anomalies")

        def get_color(score):
            if score > 5:
                return "#FF0000"
            elif score > 3:
                return "#FFA500"
            else:
                return "#FFFF00"

        for idx, row in anomalies.iterrows():
            point = row.geometry
            score = row.get("score", 0)
            folium.CircleMarker(
                location=[point.y, point.x],
                radius=8,
                color=get_color(score),
                fill=True,
                fill_color=get_color(score),
                fill_opacity=0.7,
                tooltip=f"Anomaly Score: {score:.2f}",
                popup=(
                    f"<b>Anomaly #{idx+1}</b><br>Score: {score:.2f}<br>Location: {point.y:.6f}, {point.x:.6f}"
                ),
            ).add_to(anomalies_layer)

        anomalies_layer.add_to(m)

    # ――― ➊  add coordinate reference lines  ―――
    # Grab the limits once so it works for either BOUNDS variant you switch to
    upper_lat  = OVERLAY_BOUNDS_CLEAN[0][0]   #  10 °  N
    lower_lat  = OVERLAY_BOUNDS_CLEAN[1][0]   # −20 °  S
    left_lon   = OVERLAY_BOUNDS_CLEAN[0][1]   # −80 °  W
    right_lon  = OVERLAY_BOUNDS_CLEAN[2][1]   # −45 °  W

    coord_fg = folium.FeatureGroup(
        name="Coordinate lines", show=True, control=True
    )

    # Left (west) meridian
    folium.PolyLine(
        locations=[[upper_lat, left_lon], [lower_lat, left_lon]],
        color="yellow", weight=2, dash_array="6,4",
        tooltip="80° W"
    ).add_to(coord_fg)

    # Bottom (south) parallel
    folium.PolyLine(
        locations=[[lower_lat, left_lon], [lower_lat, right_lon]],
        color="yellow", weight=2, dash_array="6,4",
        tooltip="20° S"
    ).add_to(coord_fg)

    coord_fg.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap()
    m.add_child(minimap)
    
    # Add measure control
    plugins.MeasureControl(primary_length_unit='kilometers').add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add some JavaScript to handle the opacity controls
    js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Wait for map layers to be fully loaded
        setTimeout(function() {
            // Find all our slider elements
            var sliders = document.querySelectorAll('input[type="range"][id^="opacity_img_"]');
            
            // Trigger the input event on each slider to initialize
            sliders.forEach(function(slider) {
                var event = new Event('input');
                slider.dispatchEvent(event);
            });
        }, 2000);
    });
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))
    
    print(f"Created combined map with {total_points} archaeological sites, LiDAR coverage, and {len(image_files)} image overlays")
    return m

def main():
    """Read reference data and produce the combined map."""
    arch_dataframes, lidar_df, image_files = load_reference_datasets()

    print("Creating combined interactive map...")
    map_obj = create_combined_map(arch_dataframes, lidar_df, image_files)
    
    # Save the map
    output_path = os.path.join(BASE_DIR, 'data_vis/amazon_combined_map.html')
    map_obj.save(output_path)
    print(f"Map saved as '{output_path}'")
    print("Open this file in a web browser to view the interactive map")
    
    # Display basic statistics
    print("\n=== Summary ===")
    total_sites = sum(len(df) for _, df in arch_dataframes)
    print(f"Total archaeological sites plotted: {total_sites}")

    for name, df in arch_dataframes:
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        print(f"  - {name}: {len(valid_coords)} sites with valid coordinates")
    
    if not lidar_df.empty:
        print(f"Total LiDAR tiles: {len(lidar_df)}")
    
    print(f"Total image overlays: {len(image_files)}")
    
    return map_obj

if __name__ == "__main__":
    main()