#!/usr/bin/env python3
"""Interactive generation of multiple bounding boxes.

Open a Folium map and draw a bounding box interactively. The chosen area
is divided into a grid of smaller boxes which can be tweaked in the
browser. Hold the Alt key while clicking a grid cell to remove it, and
Ctrl-click to add it back. Use the export button to download a JSON file
containing ``xmin``, ``ymin``, ``xmax`` and ``ymax`` for each selected
sub-bounding box.
"""

from pathlib import Path
import webbrowser

import folium
from folium import MacroElement
from folium.elements import Figure
from folium.plugins import Draw
from jinja2 import Template


DEFAULT_OUTPUT = "preview_bbox_multi.html"
DEFAULT_GRID_SIZE = 0.05  # degrees


class GridPlugin(MacroElement):
    """Add interactive grid controls to a Folium map."""

    def __init__(self, grid_group, draw_group, grid_size):
        super().__init__()
        self._name = "GridPlugin"
        self.grid_group = grid_group
        self.draw_group = draw_group
        self.grid_size = grid_size
        self._template = Template(
            """
            {% macro script(this, kwargs) %}
            var map = {{this._parent.get_name()}};
            var drawn = {{this.draw_group.get_name()}};
            var grid = {{this.grid_group.get_name()}};
            var gridSize = {{this.grid_size}};

            function buildGrid(bounds) {
                grid.clearLayers();
                var sw = bounds.getSouthWest();
                var ne = bounds.getNorthEast();
                function addRect(x1, y1, x2, y2) {
                    var rect = L.rectangle([[y1, x1], [y2, x2]], {
                        color: 'blue',
                        weight: 1,
                        fillOpacity: 0.05
                    });
                    rect.selected = true;
                    rect.on('click', function(e) {
                        if (e.originalEvent.altKey) {
                            rect.selected = false;
                            rect.setStyle({color: 'gray', fillOpacity: 0});
                        } else if (e.originalEvent.ctrlKey) {
                            rect.selected = true;
                            rect.setStyle({color: 'blue', fillOpacity: 0.05});
                        }
                    });
                    rect.addTo(grid);
                }
                for (var x = sw.lng; x < ne.lng; x += gridSize) {
                    for (var y = sw.lat; y < ne.lat; y += gridSize) {
                        var x2 = Math.min(x + gridSize, ne.lng);
                        var y2 = Math.min(y + gridSize, ne.lat);
                        addRect(x, y, x2, y2);
                    }
                }
            }

            map.on('draw:created', function(e) {
                drawn.clearLayers();
                drawn.addLayer(e.layer);
                buildGrid(e.layer.getBounds());
            });

            var control = L.control({position: 'topright'});
            control.onAdd = function() {
                var div = L.DomUtil.create('div', 'grid-control');
                div.style.background = 'white';
                div.style.padding = '6px';
                div.style.font = '14px/1 Arial';
                div.innerHTML =
                    'Grid size (deg): <input id="gs" type="number" step="0.01" value="' + gridSize + '" style="width:4em"> ' +
                    '<button id="exgrid">Export</button>';
                return div;
            };
            control.addTo(map);

            function refreshGrid() {
                var layers = drawn.getLayers();
                if (layers.length > 0) {
                    buildGrid(layers[0].getBounds());
                }
            }

            document.getElementById('gs').addEventListener('input', function() {
                gridSize = parseFloat(this.value);
                refreshGrid();
            });

            document.getElementById('exgrid').addEventListener('click', function() {
                var cells = [];
                grid.eachLayer(function(l) {
                    if (l.selected) {
                        var b = l.getBounds();
                        cells.push({
                            xmin: b.getWest(),
                            ymin: b.getSouth(),
                            xmax: b.getEast(),
                            ymax: b.getNorth()
                        });
                    }
                });
                var url = URL.createObjectURL(new Blob([JSON.stringify(cells)], {type: 'application/json'}));
                var a = document.createElement('a');
                a.href = url;
                a.download = 'bboxes.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });
            {% endmacro %}
            """
        )

    def render(self, **kwargs):
        super().render(**kwargs)


def main() -> None:
    output = DEFAULT_OUTPUT
    grid_size = DEFAULT_GRID_SIZE

    m = folium.Map(location=[0, 0], zoom_start=2)
    drawn = folium.FeatureGroup(name="DrawnBBox").add_to(m)
    grid = folium.FeatureGroup(name="Grid").add_to(m)

    Draw(
        export=False,
        filename="bboxes.json",
        feature_group=drawn,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
    ).add_to(m)

    GridPlugin(grid, drawn, grid_size).add_to(m)
    folium.LayerControl().add_to(m)

    output_path = Path(output)
    output_path.write_text(Figure().add_child(m).render())
    print(f"Map saved to {output_path.resolve()}")
    print("Open the HTML file, draw a rectangle, adjust the grid size, and click Export.")

    try:
        webbrowser.open(output_path.resolve().as_uri())
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}")


if __name__ == "__main__":
    main()
