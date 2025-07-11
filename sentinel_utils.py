from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.ndimage import binary_dilation
from rasterio.enums import Resampling
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import requests
from PIL import Image, PngImagePlugin
import json
from rich.console import Console

SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"

console = Console()

FILL_VALUE = np.nan


def _is_valid_tif(path: Path) -> bool:
    """Return True if ``path`` can be read entirely using rasterio."""

    try:
        with rio.open(path) as src:
            for _, window in src.block_windows(1):
                src.read(1, window=window)
        return True
    except Exception:
        return False


def save_image_with_metadata(
    img: Image.Image,
    path: Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> None:
    """Save ``img`` to ``path`` embedding ``bbox`` as metadata if provided."""

    if bbox is not None:
        bbox_json = json.dumps([float(x) for x in bbox])
        suffix = path.suffix.lower()
        if suffix == ".png":
            info = PngImagePlugin.PngInfo()
            info.add_text("bounds", bbox_json)
            img.save(path, pnginfo=info, **kwargs)
            return
        elif suffix in {".jpg", ".jpeg"}:
            exif = img.getexif()
            exif[270] = bbox_json  # ImageDescription
            img.save(path, exif=exif.tobytes(), **kwargs)
            return

    img.save(path, **kwargs)


def read_bbox_metadata(path: Path) -> tuple[float, float, float, float] | None:
    """Return the bounding box stored in ``path`` if present."""

    with Image.open(path) as img:
        suffix = path.suffix.lower()
        if suffix == ".png":
            bbox_str = img.info.get("bounds")
        else:
            exif = img.getexif()
            bbox_str = exif.get(270)

    if bbox_str:
        try:
            data = json.loads(bbox_str)
            if isinstance(data, (list, tuple)) and len(data) == 4:
                return tuple(float(x) for x in data)
        except Exception:
            return None
    return None


def _to_rfc3339(date: str, end: bool = False) -> str:
    """Convert ``YYYY-MM-DD`` strings to RFC3339 format accepted by Earth Search."""
    if "T" in date:
        return date
    return f"{date}T23:59:59Z" if end else f"{date}T00:00:00Z"


def search_sentinel2_item(
    bbox: Tuple[float, float, float, float],
    time_start: str,
    time_end: str,
    cloud_cover: int = 20,
    grid_code: str | None = None,
) -> dict | None:
    """Search for a Sentinel-2 L2A item intersecting ``bbox``.

    Parameters
    ----------
    bbox : tuple
        Bounding box in WGS84.
    time_start, time_end : str
        Start and end date of the search range.
    cloud_cover : int, default 20
        Maximum allowed cloud cover percentage.
    grid_code : str, optional
        MGRS tile code (``grid:code``) to restrict the search to.
    """
    query = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": f"{_to_rfc3339(time_start)}/{_to_rfc3339(time_end, end=True)}",
        "query": {"eo:cloud_cover": {"lt": cloud_cover}},
        "limit": 1,
    }
    if grid_code is not None:
        query["query"]["grid:code"] = {"eq": grid_code}
    try:
        r = requests.post(SEARCH_URL, json=query, timeout=60)
        r.raise_for_status()
    except requests.HTTPError:
        return None
    features = r.json().get("features", [])
    return features[0] if features else None


BAND_MAP = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
    # Scene classification layer used for cloud masking
    "SCL": "scl",
}


def download_bands(
    feature: dict,
    bands: List[str],
    out_dir: Path,
    *,
    source_dirs: Optional[List[Path]] = None,
    download_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Download selected ``bands`` from a STAC feature.

    The resulting files are named using the pattern
    ``<item_id>_<xmin>_<ymin>_<xmax>_<ymax>_<YYYYMMDD>_<band>.tif`` so they are
    unique and reusable across experiments.

    Parameters
    ----------
    out_dir : Path
        Directory checked first for existing files. This matches the original
        pipeline output folder.
    source_dirs : list of Path, optional
        Additional directories checked for previously downloaded data.
    download_dir : Path, optional
        Directory where new downloads are stored. Defaults to ``out_dir``.
    """

    out_dir = Path(out_dir)
    download_dir = Path(download_dir or out_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = [out_dir]
    for d in source_dirs or []:
        p = Path(d)
        if p not in search_dirs:
            search_dirs.append(p)

    paths: Dict[str, Path] = {}
    item_id = feature.get("id", "item")
    bbox = feature.get("bbox", [])
    bbox_str = (
        f"_{bbox[0]:.5f}_{bbox[1]:.5f}_{bbox[2]:.5f}_{bbox[3]:.5f}"
        if len(bbox) == 4
        else ""
    )
    datetime_str = feature.get("properties", {}).get("datetime")
    if datetime_str:
        try:
            dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
            date_part = f"_{dt.strftime('%Y%m%d')}"
        except ValueError:
            date_part = ""
    else:
        date_part = ""
    for band in bands:
        asset = BAND_MAP.get(band)
        if asset is None or asset not in feature["assets"]:
            if band == "SCL":
                console.log("[yellow]SCL band not available; skipping cloud mask")
            continue

        filename = f"{item_id}{bbox_str}{date_part}_{band}.tif"
        found: Optional[Path] = None
        for d in search_dirs:
            p = Path(d) / filename
            if p.exists():
                if _is_valid_tif(p):
                    found = p
                    break
                console.log(f"[yellow]Corrupt band file {p}; re-downloading")
                try:
                    p.unlink()
                except Exception:
                    pass

        if found is not None:
            console.log(f"[green]Using existing band file → {found}")
            paths[band] = found
            continue

        url = feature["assets"][asset]["href"]
        local = download_dir / filename
        for attempt in range(2):
            console.log(f"Fetching {url} → {local}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(1_048_576):
                        f.write(chunk)
            if _is_valid_tif(local):
                break
            console.log(f"[red]Validation failed for {local}; retrying")
            try:
                local.unlink()
            except Exception:
                pass
        else:
            raise RuntimeError(f"Failed to download valid band file: {local}")

        paths[band] = local

    return paths


def read_band(
    path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    *,
    scale: float = 1 / 10000.0,
) -> np.ndarray:
    """Read a band and optionally crop it to a WGS84 ``bbox``.

    Parameters
    ----------
    path : Path
        Raster path.
    bbox : tuple, optional
        (xmin, ymin, xmax, ymax) to crop to in WGS84.
    scale : float, default 1/10000.0
        Multiplicative scale applied to the data. Set to 1 for integer
        classification bands like ``SCL``.
    """

    with rio.open(path) as src:
        if bbox is None:
            arr = src.read(1).astype(np.float32)
        else:
            with rio.vrt.WarpedVRT(src, crs="EPSG:4326") as vrt:
                window = rio.windows.from_bounds(*bbox, transform=vrt.transform)
                arr = vrt.read(1, window=window).astype(np.float32)
    return arr * scale


def read_band_like(
    path: Path,
    reference: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    *,
    scale: float = 1 / 10000.0,
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Read ``path`` resampled to match ``reference``.

    This ensures the output array has the same shape and spatial resolution as
    ``reference`` for the given ``bbox``.
    """

    with rio.open(reference) as ref:
        with rio.vrt.WarpedVRT(ref, crs="EPSG:4326") as vrt_ref:
            if bbox is None:
                window = rio.windows.Window(0, 0, vrt_ref.width, vrt_ref.height)
                transform = vrt_ref.transform
            else:
                window = rio.windows.from_bounds(*bbox, transform=vrt_ref.transform)
                transform = vrt_ref.window_transform(window)
            width = round(window.width)
            height = round(window.height)

    with rio.open(path) as src:
        with rio.vrt.WarpedVRT(
            src,
            crs="EPSG:4326",
            transform=transform,
            width=width,
            height=height,
            resampling=resampling,
        ) as vrt:
            arr = vrt.read(1).astype(np.float32)

    return arr * scale


def bounds(path: Path) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) of ``path`` in WGS84."""
    with rio.open(path) as src:
        with rio.vrt.WarpedVRT(src, crs="EPSG:4326") as vrt:
            b = vrt.bounds
    return (b.left, b.bottom, b.right, b.top)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI while honouring a nodata fill value."""
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)

    # Build a mask of bad pixels
    nodata = (red == FILL_VALUE) | (nir == FILL_VALUE)

    denom = nir + red
    zero_denom = denom == 0

    # Combine the two masks and prepare a result output
    mask = nodata | zero_denom
    ndvi = np.full_like(red, np.nan, dtype=np.float32)

    # Safe division only on valid pixels
    valid = ~mask
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]

    return ndvi


def compute_kndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    ndvi = compute_ndvi(red, nir)
    return np.tanh(np.square(ndvi))


def compute_ndmi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute NDMI while honouring a nodata fill value."""
    nir = nir.astype(np.float32)
    swir = swir.astype(np.float32)

    nodata = (nir == FILL_VALUE) | (swir == FILL_VALUE)
    denom = nir + swir
    zero_denom = denom == 0

    mask = nodata | zero_denom
    ndmi = np.full_like(nir, np.nan, dtype=np.float32)

    valid = ~mask
    ndmi[valid] = (nir[valid] - swir[valid]) / denom[valid]

    return ndmi


def compute_msi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute MSI while honouring a nodata fill value."""
    nir = nir.astype(np.float32)
    swir = swir.astype(np.float32)

    nodata = (nir == FILL_VALUE) | (swir == FILL_VALUE)
    zero_denom = nir == 0

    mask = nodata | zero_denom
    msi = np.full_like(nir, np.nan, dtype=np.float32)

    valid = ~mask
    msi[valid] = swir[valid] / nir[valid]

    return msi


# ---------------------------------------------------------------------------
# Cloud masking utilities
# ---------------------------------------------------------------------------

# Pixel values in the Sentinel-2 scene classification layer corresponding
# to clouds or their shadows. These will be masked out before further
# processing.
CLOUD_CLASSES = {3, 8, 9, 10, 11}


def cloud_mask(scl: np.ndarray, dilation: int = 0) -> np.ndarray:
    """Return a boolean mask where ``True`` indicates cloud or shadow pixels."""

    mask = np.isin(scl, list(CLOUD_CLASSES))
    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)
    return mask


def hollstein_cloud_mask(
    b01: np.ndarray,
    b02: np.ndarray,
    b03: np.ndarray,
    b05: np.ndarray,
    b06: np.ndarray,
    b07: np.ndarray,
    b8a: np.ndarray,
    b09: np.ndarray,
    b11: np.ndarray,
    b10: Optional[np.ndarray] = None,
    *,
    dilation: int = 2,
) -> np.ndarray:
    """Approximate cloud mask using the Hollstein algorithm.

    Parameters
    ----------
    b01, b02, b03, b05, b06, b07, b8a, b09, b11 : np.ndarray
        Reflectance bands scaled between 0 and 1.
    b10 : np.ndarray, optional
        Cirrus band. If ``None``, zeros are used instead.
    dilation : int, default 2
        Number of dilations applied to the mask.
    """

    if b10 is None:
        b10 = np.zeros_like(b01)

    cond_a = b03 < 0.319
    cond_b = b8a < 0.166
    cond_c = b03 - b07 < 0.027
    cond_d = b09 - b11 < -0.097

    shadow1 = cond_a & cond_b & cond_c & ~cond_d
    shadow2 = cond_a & cond_b & ~cond_c & (b09 - b11 >= 0.021)

    cond_e = cond_a & ~cond_b & (b02 / (b10 + 1e-6) < 14.689)
    cirrus1 = cond_e & ~(b02 / (b09 + 1e-6) < 0.788)

    cond_f = ~cond_a
    cond_g = b05 / (b11 + 1e-6) < 4.33
    cond_h = b11 - b10 < 0.255
    cond_i = b06 - b07 < -0.016

    cloud1 = cond_f & cond_g & cond_h & cond_i
    cirrus2 = cond_f & cond_g & cond_h & ~cond_i
    cloud2 = cond_f & cond_g & ~cond_h & ~(b01 < 0.3)

    shadow3 = cond_f & ~cond_g & (b03 < 0.525) & ~((b01 / (b05 + 1e-6)) < 1.184)

    mask = shadow1 | shadow2 | cirrus1 | cloud1 | cirrus2 | cloud2 | shadow3

    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)

    return mask


def apply_mask(mask: np.ndarray, *bands: np.ndarray, fill_value: float = np.nan) -> Tuple[np.ndarray, ...]:
    """Apply ``mask`` to ``bands``, returning masked copies."""

    masked = []
    for arr in bands:
        m = arr.astype(np.float32, copy=True)
        m[mask] = fill_value
        masked.append(m)
    return tuple(masked)


def mask_clouds(
    scl: np.ndarray,
    *bands: np.ndarray,
    fill_value: float = np.nan,
    dilation: int = 2,
) -> Tuple[np.ndarray, ...]:
    """Mask cloud and shadow pixels in ``bands`` using the ``scl`` array.

    Parameters
    ----------
    scl : np.ndarray
        Scene classification layer where cloud/shadow pixels have the
        values defined in ``CLOUD_CLASSES``.
    bands : np.ndarray
        One or more arrays to mask in-place.
    fill_value : float, default ``np.nan``
        Value assigned to masked pixels.
    dilation : int, default 2
        Number of dilations applied to the mask to remove cloud edges.

    Returns
    -------
    tuple of np.ndarray
        The masked arrays in the same order as provided.
    """

    mask = cloud_mask(scl, dilation=dilation)
    return apply_mask(mask, *bands, fill_value=fill_value)


def save_mask_png(
    mask: np.ndarray,
    path: Path,
    dpi: int = 150,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Save a binary cloud mask as an image with optional ``bbox`` metadata."""

    img = Image.fromarray((mask.astype(float) * 255).astype(np.uint8))
    save_image_with_metadata(img, path, bbox=bbox, dpi=(dpi, dpi))



def save_true_color(
    b02: np.ndarray,
    b03: np.ndarray,
    b04: np.ndarray,
    path: Path,
    *,
    gain: float = 2.5,
    quality: int = 95,
    dpi: int = 150,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Save a true-colour image, painting all no-data / cloudy pixels neon-purple.

    Parameters
    ----------
    b02, b03, b04 : np.ndarray
        Reflectance bands, already on 0‒1 scale except for FILL_VALUE.
    path : Path
        Destination file (.jpg or .png supported).
    gain : float, default 2.5
        Contrast multiplier before clipping.
    quality : int, default 95
        JPEG quality when saving as JPEG.
    dpi : int, default 150
        Resolution metadata written to the file.
    bbox : tuple of float, optional
        Bounding box ``(xmin, ymin, xmax, ymax)`` written as metadata.
    """
    # --- stack channels (R=B04, G=B03, B=B02) and apply gain
    rgb_raw = np.stack([b04, b03, b02], axis=-1)

    # --- mask: any band is non-finite **or** equals the fill value
    # mask_no_data = (
    #     ~np.isfinite(rgb_raw).all(axis=-1)
    #     | (rgb_raw == FILL_VALUE).any(axis=-1)
    # )

    mask_no_data = (rgb_raw == FILL_VALUE).any(axis=-1)

    # --- convert to display domain
    rgb = np.clip(rgb_raw * gain, 0.0, 1.0)

    # --- paint masked pixels neon-purple (R=1, G=0, B=1)
    rgb[mask_no_data] = (1.0, 0.0, 1.0)

    # --- ensure no NaNs/Infs sneak through
    rgb = np.nan_to_num(rgb, nan=1.0, posinf=1.0, neginf=0.0)

    # --- write with Pillow (PNG comes out *without* gAMA/sRGB chunks)
    img = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")

    path_jpg = path.with_suffix(".jpg")

    save_image_with_metadata(
        img,
        path_jpg,
        bbox=bbox,
        quality=quality,
        dpi=(dpi, dpi),
    )


def save_index_png(
    arr: np.ndarray,
    path: Path,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
    nodata_rgba: tuple[int, int, int, int] = (0, 0, 0, 0),   # transparent
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Save a float array (any range) using a Matplotlib colormap.

    Parameters
    ----------
    bbox : tuple of float, optional
        Bounding box ``(xmin, ymin, xmax, ymax)`` written as metadata.
    """

    # If caller did not provide limits, deduce them (ignoring NaNs)
    if vmin is None or vmax is None:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(finite.min()), float(finite.max())
            if vmin == vmax:
                vmax = vmin + 1e-6

    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cm   = plt.get_cmap(cmap)
    rgba = cm(norm(arr))             # shape (H,W,4), float 0‑1

    # Paint nodata
    nodata_mask = ~np.isfinite(arr)
    rgba[nodata_mask] = np.array(nodata_rgba) / 255.0

    img = Image.fromarray((rgba * 255).astype(np.uint8))
    save_image_with_metadata(img, path, bbox=bbox, dpi=(dpi, dpi))


def resize_image(
    src: Path,
    dest: Path | None = None,
    factor: float = 0.5,
    resample: int = Image.Resampling.BICUBIC,
) -> Path:
    """Save a resized copy of ``src``.

    Parameters
    ----------
    src : Path
        Source image path.
    dest : Path, optional
        Destination path. If ``None``, ``_web`` is appended to the filename.
    factor : float, default 0.5
        Scale factor for width and height.
    resample : int, default ``Image.Resampling.BICUBIC``
        Pillow resampling filter.

    Returns
    -------
    Path
        Path to the resized image.
    """

    dest = dest or src.with_name(src.stem + "_web" + src.suffix)
    with Image.open(src) as img:
        w, h = img.size
        new_size = (max(1, int(w * factor)), max(1, int(h * factor)))
        resized = img.resize(new_size, resample=resample)
        resized.save(dest)
    return dest


def mosaic_images(paths: List[Path], out: Path) -> Path:
    """Combine georeferenced images into a single mosaic.

    ``paths`` must contain images saved via :func:`save_image_with_metadata` so
    that bounding boxes are embedded in the files. The output image is written
    to ``out`` with bounding box metadata covering the union of the inputs.
    """

    if not paths:
        raise ValueError("no images provided")

    # Load images and their bounds
    imgs: List[Image.Image] = []
    bounds = []
    for p in paths:
        bbox = read_bbox_metadata(p)
        if bbox is None:
            raise ValueError(f"missing bbox metadata for {p}")
        img = Image.open(p).convert("RGBA")
        imgs.append(img)
        bounds.append(bbox)

    # Determine overall bounding box
    xmin = min(b[0] for b in bounds)
    ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds)
    ymax = max(b[3] for b in bounds)

    # Use pixel size from the first image
    pw = (bounds[0][2] - bounds[0][0]) / imgs[0].width
    ph = (bounds[0][3] - bounds[0][1]) / imgs[0].height

    width = max(1, round((xmax - xmin) / pw))
    height = max(1, round((ymax - ymin) / ph))
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    for img, bbox in zip(imgs, bounds):
        w = max(1, round((bbox[2] - bbox[0]) / pw))
        h = max(1, round((bbox[3] - bbox[1]) / ph))
        if img.size != (w, h):
            img = img.resize((w, h), resample=Image.Resampling.BILINEAR)
        dx = int(round((bbox[0] - xmin) / pw))
        dy = int(round((ymax - bbox[3]) / ph))
        canvas.alpha_composite(img, dest=(dx, dy))

    img_out: Image.Image = canvas
    if out.suffix.lower() in {".jpg", ".jpeg"}:
        img_out = canvas.convert("RGB")

    save_image_with_metadata(img_out, out, bbox=(xmin, ymin, xmax, ymax))
    return out
