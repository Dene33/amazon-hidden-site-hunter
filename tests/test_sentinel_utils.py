import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sentinel_utils import compute_kndvi


def test_compute_kndvi_simple():
    red = np.array([[0.2, 0.3], [0.2, 0.1]])
    nir = np.array([[0.6, 0.5], [0.4, 0.2]])
    kndvi = compute_kndvi(red, nir)
    assert kndvi.shape == red.shape
    # Ensure values are between -1 and 1
    assert np.all(kndvi >= -1) and np.all(kndvi <= 1)
