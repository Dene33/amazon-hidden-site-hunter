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
    lat, lon, score, desc = detections[0]
    assert (lat, lon, score) == pytest.approx((-12.681, -63.872, 1.0))
    assert desc == ""
    lat, lon, score, _ = detections[1]
    assert lat == pytest.approx(-12.682, rel=1e-3)
    assert lon == pytest.approx(-63.876, rel=1e-3)
    assert score == pytest.approx(2.0)
    assert detections[2][:3] == pytest.approx((-12.6803, -63.8727, 3.0))
    assert detections[3][:3] == pytest.approx((-12.6832, -63.8735, 4.0))


def test_parse_chatgpt_description():
    text = (
        "ID 1 10 S, 10 W score = 1\n"
        "First detection description.\n"
        "ID 2 11 S, 11 W score = 2\n"
        "Second detection.\n"
    )
    detections = _parse_chatgpt_detections(text)
    assert detections[0][3] == "First detection description."
    assert detections[1][3] == "Second detection."


def test_parse_chatgpt_spacing_and_symbols():
    text = (
        "ID 1 12\u00b0 41 \u2032 35 \u2033 S, 63\u00b0 52 \u2032 03 \u2033 W score = 1\n"
        "ID 2 12.6842 \u00b0 S, 63.8756 \u00b0 W score = 2\n"
        "ID 3 12\u00b0 41\u2032 00.9\u2033 S, 63\u00b0 52\u2032 34.3\u2033 W score = 3\n"
    )
    detections = _parse_chatgpt_detections(text)
    assert detections[0][:3] == pytest.approx((-12.693056, -63.8675, 1.0))
    assert detections[1][:3] == pytest.approx((-12.6842, -63.8756, 2.0))
    assert detections[2][0] == pytest.approx(-12.683583, rel=1e-6)
    assert detections[2][1] == pytest.approx(-63.876194, rel=1e-6)
    assert detections[2][2] == pytest.approx(3.0)

