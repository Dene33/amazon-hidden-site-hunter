#!/usr/bin/env python3
"""
multi_bbox.py – draw rectangles in a browser, each one prints
`--bbox xmin ymin xmax ymax` to the terminal.

• No Jupyter, no Flask, no extra packages – stdlib + Leaflet from CDN.
• Draw another rectangle: just click SW corner, then NE corner again.
• Each new box replaces the old outline and prints fresh coords.
• Quit anytime with Ctrl‑C in the terminal or closing the tab.
• **NEW:** Added Esri World Imagery basemap option alongside OpenStreetMap.
"""

import http.server, json, socket, threading, webbrowser, sys

HTML = """<!DOCTYPE html><html><head>
<meta charset=\"utf-8\"/>
<link rel=\"stylesheet\"
 href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"/>
<style>html,body,#map{height:100%;margin:0}</style></head><body>
<div id=\"map\"></div>
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
<script>
// --- Basemaps -------------------------------------------------------------
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'});

const esri = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution: 'Tiles © Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR‑EGP, and the GIS User Community'});

// --- Map ------------------------------------------------------------------
const map = L.map('map', {scrollWheelZoom: true, layers: [osm]}).setView([0, 0], 2);

L.control.layers({"OpenStreetMap": osm, "Esri World Imagery": esri}).addTo(map);

// --- Rectangle drawing ----------------------------------------------------
let first = null, rect = null, marker = null;
function startCapture() {           // 1st click
  map.once('click', e => {
    first = e.latlng;
    if (marker) map.removeLayer(marker);
    marker = L.marker(first).addTo(map);
    endCapture();                  // wait for 2nd click
  });
}
function endCapture() {             // 2nd click
  map.once('click', e => {
    const second = e.latlng;
    const bounds = L.latLngBounds(first, second);
    if (rect) map.removeLayer(rect);
    rect = L.rectangle(bounds, {color: '#ff7800', weight: 2}).addTo(map);

    const xmin = bounds.getWest(),  xmax = bounds.getEast();
    const ymin = bounds.getSouth(), ymax = bounds.getNorth();
    fetch('/bbox', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({xmin, xmax, ymin, ymax})})
      .then(() => console.log('bbox sent'));
    startCapture();                // arm for next rectangle
  });
}
startCapture();
</script></body></html>"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass  # silence default logging
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(HTML.encode())
    def do_POST(self):
        if self.path != "/bbox":
            return self.send_error(404)
        length = int(self.headers['Content-Length'])
        data = json.loads(self.rfile.read(length))
        print(f"--bbox {data['xmin']:.6f} {data['ymin']:.6f} "
              f"{data['xmax']:.6f} {data['ymax']:.6f}")
        self.send_response(200)
        self.end_headers()

def free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    if sys.version_info < (3, 7):
        sys.exit("Python ≥3.7 required.")
    port = free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("Browser opening…   Draw rectangles: 1st click SW, 2nd click NE.\n"
          "Each box prints a --bbox line.  Press Ctrl-C to quit.\n")
    webbrowser.open(f"http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGoodbye.")
        server.server_close()
