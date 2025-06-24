# amazon-hidden-site-hunter

This project downloads and processes Sentinel-2 imagery and various DEM products
for user-defined bounding boxes. The pipeline assumes each bbox falls entirely
within a single Sentinel-2 MGRS tile.

If a bbox spans multiple tiles, only a portion of the imagery would be fetched
leading to cropped results. The pipeline now checks for this condition and stops
with an error when the AOI extends beyond the bounds of the selected tile.
Split large AOIs into smaller bboxes that match individual Sentinel-2 tiles to
ensure complete coverage.
