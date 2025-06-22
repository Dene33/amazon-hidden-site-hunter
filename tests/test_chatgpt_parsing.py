import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from chatgpt_parser import _parse_chatgpt_detections


def test_parse_chatgpt_various_formats():
    text = (
        "ID 1 12.681 S, 63.872 W score = 1.0\n"
        "ID 2 12\u00b040'55.2\" S, 63\u00b052'33.6\" W score = 2.0\n"
        "ID 3 12.6803\u00b0 S, 63.8727\u00b0 W score = 3.0\n"
        "ID 4 \u201312.6832, \u201363.8735 score = 4.0\n"
    )
    detections = _parse_chatgpt_detections(text)
    assert detections[0] == pytest.approx((-12.681, -63.872, 1.0))
    assert detections[1][0] == pytest.approx(-12.682, rel=1e-3)
    assert detections[1][1] == pytest.approx(-63.876, rel=1e-3)
    assert detections[1][2] == pytest.approx(2.0)
    assert detections[2] == pytest.approx((-12.6803, -63.8727, 3.0))
    assert detections[3] == pytest.approx((-12.6832, -63.8735, 4.0))
