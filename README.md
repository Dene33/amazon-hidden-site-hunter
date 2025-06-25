# amazon-hidden-site-hunter

This project downloads and processes Sentinel-2 imagery and various DEM products
for user-defined bounding boxes.

If a bbox spans multiple Sentinel-2 tiles only the portion that lies inside the
available tile will be downloaded. The pipeline logs a message in this case and
continues processing with the partial data. Try adjusting `max_cloud` or the
`time_start`/`time_end` values if full coverage is required.

Detected anomaly points and ChatGPT detections are saved as CSV files in each
output directory. Coordinates are stored as `latitude` and `longitude` columns
to make downstream analysis easier.
