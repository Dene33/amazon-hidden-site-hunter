import folium
from folium.plugins import FloatImage

# Approximate territorial polygons for each tribe circa 1925
tribe_polygons = {
    "Kalapalo (1925)": [
        [-12.7, -53.7],
        [-12.7, -53.3],
        [-13.5, -53.3],
        [-13.5, -53.7]
    ],
    "Arumá (1925)": [
        [-10.8, -52.0],
        [-10.8, -51.5],
        [-12.2, -51.5],
        [-12.2, -52.0]
    ],
    "Suyá / Kisêdjê (1925)": [
        [-10.0, -52.7],
        [-10.0, -53.4],
        [-10.8, -53.4],
        [-10.8, -52.7]
    ],
    "Xavante (1925)": [
        [-12.5, -53.0],
        [-12.5, -51.0],
        [-15.0, -51.0],
        [-15.0, -53.0]
    ]
}

# Center the map roughly on Mato Grosso, Brazil
m = folium.Map(location=[-12.0, -52.5], zoom_start=6, tiles=None)

# Add Esri World Imagery basemap
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

# Color list for polygons
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Add polygons for each tribe
for (tribe, coords), color in zip(tribe_polygons.items(), colors):
    folium.Polygon(
        locations=coords,
        color=color,
        fill=True,
        fill_opacity=0.4,
        weight=2,
        popup=tribe,
        tooltip=tribe
    ).add_to(m)

# Add a legend image (simple color boxes) for quick reference
legend_html = """
<div style="position: fixed; 
     bottom: 30px; left: 30px; width: 180px; height: 140px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color: white; opacity: 0.9; padding: 10px;">
<b>Tribes (c. 1925)</b><br>
<span style="color:#1f77b4;">&#9632;</span> Kalapalo<br>
<span style="color:#ff7f0e;">&#9632;</span> Arumá<br>
<span style="color:#2ca02c;">&#9632;</span> Suyá<br>
<span style="color:#d62728;">&#9632;</span> Xavante
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)

# Save map
file_path = "tribes_1925_map.html"
m.save(file_path)

file_path
