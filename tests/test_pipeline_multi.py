import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import run_pipeline


def test_run_pipeline_multiple(tmp_path):
    cfg = {
        "bbox": [
            [0, 0, 1, 1],
            [1, 1, 2, 2],
        ],
        "out_dir": str(tmp_path),
        "fetch_data": {"enabled": False},
        "sentinel": {"enabled": False},
        "srtm": {"enabled": False},
        "aw3d": {"enabled": False},
        "bare_earth": {"enabled": False},
        "residual_relief": {"enabled": False},
        "detect_anomalies": {"enabled": False},
        "interactive_map": {"enabled": False},
        "export_obj": {"enabled": False},
        "export_xyz": {"enabled": False},
    }
    run_pipeline(cfg)
    assert (tmp_path / "0_0_1_1").exists()
    assert (tmp_path / "1_1_2_2").exists()
