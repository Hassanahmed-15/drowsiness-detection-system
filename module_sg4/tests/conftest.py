"""Shared pytest helpers: stream builder with hand-computable timing."""
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sg4 import SG4Config, TemporalAnalyzer  # noqa: E402

Segment = Union[Tuple[int, str], Tuple[int, str, float]]


def rec(t: int, fid: int, kind: str = "open", yawn: float = 0.0) -> Dict[str, Any]:
    """One flat record. kind: open | closed | lost | eyes_invalid."""
    face = kind != "lost"
    eyes_ok = kind in ("open", "closed")
    return {
        "frame_id": fid, "timestamp_ms": t, "face_detected": face,
        "left_open_prob": None, "right_open_prob": None,
        "ear_left": None, "ear_right": None,
        "eyes_closed": (kind == "closed") if eyes_ok else None,
        "eyes_valid": eyes_ok,
        "yawn_prob": yawn if face else None, "cue_flags": None,
        "mouth_valid": face,
    }


def stream(step: int, segments: Sequence[Segment], t0: int = 0) -> List[Dict[str, Any]]:
    """Build records at a fixed step from (duration_ms, kind[, yawn]) segments."""
    out: List[Dict[str, Any]] = []
    t, fid = t0, 0
    for seg in segments:
        dur, kind = seg[0], seg[1]
        yawn = seg[2] if len(seg) > 2 else 0.0
        for _ in range(dur // step):
            out.append(rec(t, fid, kind, yawn))
            t += step
            fid += 1
    return out


def cfg(**kw: Any) -> SG4Config:
    """Test config: short warm-up, everything else default."""
    base = {"warmup_ms": 1000}
    base.update(kw)
    return SG4Config(**base)


def run(config: SG4Config, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    analyzer = TemporalAnalyzer(config)
    return [analyzer.update(r) for r in records]
