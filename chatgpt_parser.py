from __future__ import annotations

import re
from typing import List, Tuple

_DET_RE = re.compile(r"ID\s*\d+.*?score\s*=\s*\d+(?:\.\d+)?", re.IGNORECASE)
_SCORE_RE = re.compile(r"score\s*=\s*([\d.]+)", re.IGNORECASE)
_COORD_RE = re.compile(
    r"[-−–]?\d+(?:\.\d+)?"  # degrees with optional sign
    r"(?:\s*[°º]\s*(?:\d+(?:\.\d+)?\s*(?:[′'’]\s*\d+(?:\.\d+)?\s*(?:[\"″”])?)?)?)?"  # optional minutes/seconds
    r"\s*[NSEW]?",
    re.UNICODE,
)


def _parse_coordinate(token: str) -> float | None:
    token = token.strip().replace("−", "-").replace("–", "-")
    hemi = ""
    if token and token[-1] in "NSEWnsew":
        hemi = token[-1].upper()
        token = token[:-1].strip()

    token = (
        token.replace("°", " ")
        .replace("º", " ")
        .replace("'", " ")
        .replace("’", " ")
        .replace("′", " ")
        .replace('"', " ")
        .replace("”", " ")
        .replace("″", " ")
    )
    parts = [p for p in token.split() if p]
    if not parts:
        return None

    deg = float(parts[0])
    minutes = float(parts[1]) if len(parts) > 1 else 0.0
    seconds = float(parts[2]) if len(parts) > 2 else 0.0

    value = abs(deg) + minutes / 60 + seconds / 3600
    if deg < 0:
        value = -value
    if hemi in {"S", "W"}:
        value = -abs(value)
    elif hemi in {"N", "E"}:
        value = abs(value)
    return value


def _parse_chatgpt_detections(text: str) -> List[Tuple[float, float, float, str]]:
    """Parse detections from a ChatGPT response.

    Each detection starts with a header like ``ID 1 ... score = X`` followed by an
    optional description. The description continues until the next header or end
    of text.
    """

    detections: List[Tuple[float, float, float, str]] = []

    matches = list(_DET_RE.finditer(text))
    for idx, m in enumerate(matches):
        header = m.group(0)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        description = text[start:end].strip()

        body = re.sub(r"^ID\s*\d+\s*", "", header)
        score_match = _SCORE_RE.search(header)
        if not score_match:
            continue
        score = float(score_match.group(1))
        body = _SCORE_RE.sub("", body)
        tokens = _COORD_RE.findall(body)
        if len(tokens) >= 2:
            lat = _parse_coordinate(tokens[0])
            lon = _parse_coordinate(tokens[1])
            if lat is not None and lon is not None:
                detections.append((lat, lon, score, description))

    return detections
