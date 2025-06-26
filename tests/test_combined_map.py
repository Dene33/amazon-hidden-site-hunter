from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import run_pipeline


def _base_cfg(tmp_path: Path, combined: bool):
    return {
        "bbox": [[0, 0, 1, 1], [1, 1, 2, 2]],
        "out_dir": str(tmp_path),
        "fetch_data": {"enabled": False},
        "sentinel": {"enabled": False},
        "srtm": {"enabled": False},
        "aw3d": {"enabled": False},
        "bare_earth": {"enabled": False},
        "residual_relief": {"enabled": False},
        "detect_anomalies": {"enabled": False},
        "chatgpt": {"enabled": False},
        "interactive_map": {
            "enabled": True,
            "include_data_vis": False,
            "combined_bboxes_map": combined,
        },
        "export_obj": {"enabled": False},
        "export_xyz": {"enabled": False},
    }


def test_combined_map_created(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, True)
    run_pipeline(cfg)
    assert (tmp_path / "all_bboxes_map.html").exists()


def test_combined_map_disabled(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, False)
    run_pipeline(cfg)
    assert not (tmp_path / "all_bboxes_map.html").exists()
