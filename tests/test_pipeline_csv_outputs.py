import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
import numpy as np
import pandas as pd
import pytest

from pipeline import step_detect_anomalies, step_chatgpt


def _grid():
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 4)
    xi, yi = np.meshgrid(x, y)
    rrm = np.zeros_like(xi, dtype=float)
    rrm[1, 1] = 2.0
    return xi, yi, rrm


def _stub_openai(text: str) -> None:
    openai_mod = types.ModuleType("openai")
    openai_mod.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(
            create=lambda model=None, messages=None: types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
            )
        )
    )
    sys.modules["openai"] = openai_mod
    utils = types.ModuleType("openai._utils._logs")
    utils.setup_logging = lambda: None
    sys.modules["openai._utils._logs"] = utils


def test_step_detect_anomalies_csv(tmp_path: Path) -> None:
    xi, yi, rrm = _grid()
    step_detect_anomalies(
        {"enabled": True, "visualize": False, "save_json": False, "size_thresh_m": 1, "sigma": 0},
        rrm,
        xi,
        yi,
        tmp_path,
    )
    csv = tmp_path / "anomalies.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert list(df.columns) == ["latitude", "longitude", "score"]
    assert len(df) == 1
    assert df["latitude"].iloc[0] == pytest.approx(2 / 3)
    assert df["longitude"].iloc[0] == pytest.approx(1 / 3)


def test_step_chatgpt_csv(tmp_path: Path) -> None:
    _stub_openai("ID 1 10 S, 20 W score = 1.0")
    step_chatgpt(
        {"enabled": True, "images": ["dummy"], "prompt": "p", "model": "o3"},
        (0.0, 0.0, 1.0, 1.0),
        tmp_path,
    )
    csv = tmp_path / "chatgpt_points.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert list(df.columns) == ["latitude", "longitude", "score", "description"]
    assert df.iloc[0]["latitude"] == -10.0
    assert df.iloc[0]["longitude"] == -20.0
    assert df.iloc[0]["score"] == 1.0
