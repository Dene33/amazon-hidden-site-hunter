import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from unittest.mock import patch
import requests

from sentinel_utils import compute_kndvi, search_sentinel2_item


def test_compute_kndvi_simple():
    red = np.array([[0.2, 0.3], [0.2, 0.1]])
    nir = np.array([[0.6, 0.5], [0.4, 0.2]])
    kndvi = compute_kndvi(red, nir)
    assert kndvi.shape == red.shape
    # Ensure values are between -1 and 1
    assert np.all(kndvi >= -1) and np.all(kndvi <= 1)


def test_search_sentinel_rfc3339():
    bbox = (-1, -1, 1, 1)
    with patch('sentinel_utils.requests.post') as post:
        post.return_value.json.return_value = {"features": []}
        post.return_value.raise_for_status.return_value = None
        search_sentinel2_item(bbox, '2024-01-01', '2024-12-31')
        args, kwargs = post.call_args
        assert kwargs['json']['datetime'] == '2024-01-01T00:00:00Z/2024-12-31T23:59:59Z'


def test_search_sentinel_http_error():
    bbox = (-1, -1, 1, 1)
    with patch('sentinel_utils.requests.post') as post:
        post.return_value.raise_for_status.side_effect = requests.HTTPError()
        result = search_sentinel2_item(bbox, '2024-01-01', '2024-12-31')
        assert result is None
