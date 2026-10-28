"""I/O helpers: nested/flat packet conversion and JSONL read/write."""
from __future__ import annotations

import json
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, Mapping, Optional,
                    Union)

PathLike = Union[str, Path]

FLAT_KEYS = (
    "frame_id", "timestamp_ms", "face_detected",
    "left_open_prob", "right_open_prob", "ear_left", "ear_right",
    "eyes_closed", "eyes_valid",
    "yawn_prob", "cue_flags", "mouth_valid",
)


def _block(nested: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = nested.get(name)
    return value if isinstance(value, Mapping) else {}


def is_nested(record: Mapping[str, Any]) -> bool:
    """True if the record has ``face`` / ``eyes`` / ``mouth`` sub-blocks."""
    return any(isinstance(record.get(k), Mapping) for k in ("face", "eyes", "mouth"))


def flatten_packet(nested: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the flat SG-4 input dict from a nested per-frame record.

    ``frame_id`` and ``timestamp_ms`` are read from the top level;
    ``face_detected`` comes from the ``face`` block. Missing blocks give
    ``None`` cues and ``False`` validity flags (never a made-up 0).
    """
    face, eyes, mouth = _block(nested, "face"), _block(nested, "eyes"), _block(nested, "mouth")
    return {
        "frame_id": nested.get("frame_id"),
        "timestamp_ms": nested.get("timestamp_ms"),
        "face_detected": face.get("face_detected", False),
        "left_open_prob": eyes.get("left_open_prob"),
        "right_open_prob": eyes.get("right_open_prob"),
        "ear_left": eyes.get("ear_left"),
        "ear_right": eyes.get("ear_right"),
        "eyes_closed": eyes.get("eyes_closed"),
        "eyes_valid": eyes.get("eyes_valid", False),
        "yawn_prob": mouth.get("yawn_prob"),
        "cue_flags": mouth.get("cue_flags"),
        "mouth_valid": mouth.get("mouth_valid", False),
    }


def nest_packet(flat: Mapping[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`flatten_packet` (used to write nested test data)."""
    return {
        "frame_id": flat.get("frame_id"),
        "timestamp_ms": flat.get("timestamp_ms"),
        "face": {"face_detected": flat.get("face_detected", False)},
        "eyes": {
            "left_open_prob": flat.get("left_open_prob"),
            "right_open_prob": flat.get("right_open_prob"),
            "ear_left": flat.get("ear_left"),
            "ear_right": flat.get("ear_right"),
            "eyes_closed": flat.get("eyes_closed"),
            "eyes_valid": flat.get("eyes_valid", False),
        },
        "mouth": {
            "yawn_prob": flat.get("yawn_prob"),
            "cue_flags": flat.get("cue_flags"),
            "mouth_valid": flat.get("mouth_valid", False),
        },
    }


def normalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a flat record, flattening it first if it is nested."""
    if is_nested(record) and "face_detected" not in record:
        return flatten_packet(record)
    return dict(record)


ErrorHandler = Callable[[int, str, Exception], None]


def read_jsonl(path: PathLike,
               on_error: Optional[ErrorHandler] = None) -> Iterator[Dict[str, Any]]:
    """Yield one dict per non-empty line of a JSONL file.

    A malformed line raises ``ValueError`` unless ``on_error`` is given, in
    which case ``on_error(line_number, line, exception)`` is called and the
    line is skipped.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("line is not a JSON object")
            except ValueError as exc:
                if on_error is None:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
                on_error(lineno, line, exc)
                continue
            yield obj


def write_jsonl(path: PathLike, records: Iterable[Mapping[str, Any]]) -> int:
    """Write records as JSONL (``None`` becomes ``null``). Returns the count."""
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, allow_nan=False) + "\n")
            n += 1
    return n
