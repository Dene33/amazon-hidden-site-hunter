import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from detect_hidden_sites import detect_anomalies
from sentinel_utils import read_bbox_metadata


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 4)
    xi, yi = np.meshgrid(x, y)
    rrm = np.zeros_like(xi, dtype=float)
    rrm[1, 1] = 2.0
    return xi, yi, rrm


def test_debug_metadata(tmp_path: Path) -> None:
    xi, yi, rrm = _grid()
    debug = tmp_path / "debug"
    detect_anomalies(rrm, xi, yi, sigma=0, amp_thresh=1.0, debug_dir=debug)
    bbox = (xi[0, 0], yi.min(), xi[0, -1], yi.max())
    files = [
        "rrm_smooth.png",
        "rrm_smooth_clean.png",
        "threshold_mask.png",
        "threshold_mask_clean.png",
        "anomalies.png",
        "anomalies_clean.png",
    ]
    for name in files:
        p = debug / name
        assert p.exists()
        assert read_bbox_metadata(p) == bbox

