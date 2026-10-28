"""SG-4 temporal analyzer: per-frame cues in, sliding-window indicators out.

Deterministic and replayable: only ``timestamp_ms`` from the records is used;
the wall clock is never read.
"""
from __future__ import annotations

import math
import numbers
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from .closure import ClosureTracker
from .config import SG4Config
from .io_utils import is_nested, flatten_packet
from .yawn import YawnTracker

OUTPUT_KEYS: Tuple[str, ...] = (
    "frame_id", "timestamp_ms", "window_ms", "valid", "window_valid_fraction",
    "perclos", "blink_rate_per_min", "mean_blink_duration_ms",
    "longest_closure_ms", "current_closure_ms", "yawn_count",
)
MEASUREMENT_KEYS: Tuple[str, ...] = OUTPUT_KEYS[5:]

_EPS = 1e-9


# --------------------------------------------------------------------- parsing
def _as_int(value: Any) -> Optional[int]:
    """Integer-like value -> int, else None (bool and NaN are rejected)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        return int(value)
    return None


def _as_bool(value: Any) -> Optional[bool]:
    """Real bool (or numpy bool) -> bool, anything else (0/1, str) -> None."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item") and type(value).__name__ == "bool_":
        return bool(value.item())
    return None


def _as_prob(value: Any) -> Optional[float]:
    """Number in [0, 1] -> float, else None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    x = float(value)
    if not math.isfinite(x) or x < 0.0 or x > 1.0:
        return None
    return x


@dataclass(frozen=True)
class _Sample:
    t: int
    frame_id: Optional[int]
    eye_valid: bool
    closed: bool
    mouth_valid: bool
    yawn_prob: Optional[float]


def _parse(record: Any) -> Optional[_Sample]:
    """Parse a record into a sample; None if it has no usable timestamp."""
    if not isinstance(record, Mapping):
        return None
    if is_nested(record) and "face_detected" not in record:
        record = flatten_packet(record)
    t = _as_int(record.get("timestamp_ms"))
    if t is None:
        return None
    face = _as_bool(record.get("face_detected")) is True
    eyes_ok = _as_bool(record.get("eyes_valid")) is True
    closed = _as_bool(record.get("eyes_closed"))
    mouth_ok = _as_bool(record.get("mouth_valid")) is True
    prob = _as_prob(record.get("yawn_prob"))
    return _Sample(
        t=t,
        frame_id=_as_int(record.get("frame_id")),
        eye_valid=face and eyes_ok and closed is not None,
        closed=bool(closed),
        mouth_valid=face and mouth_ok and prob is not None,
        yawn_prob=prob,
    )


# ---------------------------------------------------------------- window store
class _HoldWindow:
    """Sliding store of held-time segments with O(1) running totals.

    Segments are appended in time order and never overlap, so only the oldest
    one can straddle the window start; it is clipped when totals are read.
    """

    def __init__(self) -> None:
        self._segs: Deque[Tuple[int, int, bool]] = deque()
        self._valid = 0
        self._closed = 0

    def clear(self) -> None:
        self._segs.clear()
        self._valid = 0
        self._closed = 0

    def add(self, start: int, end: int, closed: bool) -> None:
        if end <= start:
            return
        self._segs.append((start, end, closed))
        self._valid += end - start
        if closed:
            self._closed += end - start

    def expire(self, window_start: int) -> None:
        while self._segs and self._segs[0][1] <= window_start:
            s, e, c = self._segs.popleft()
            self._valid -= e - s
            if c:
                self._closed -= e - s

    def totals(self, window_start: int) -> Tuple[int, int]:
        """Return (valid_time, closed_time) clipped to the window."""
        valid, closed = self._valid, self._closed
        if self._segs:
            s, _e, c = self._segs[0]
            if s < window_start:
                clip = window_start - s
                valid -= clip
                if c:
                    closed -= clip
        return valid, closed


# -------------------------------------------------------------------- analyzer
class TemporalAnalyzer:
    """Stateful per-frame temporal analyzer (SG-4).

    Usage::

        analyzer = TemporalAnalyzer(SG4Config())
        for record in stream:
            out = analyzer.update(record)   # dict with the keys in OUTPUT_KEYS

    Diagnostics (not part of the output dict): :attr:`dropped_samples`,
    :attr:`mouth_valid_fraction`, :attr:`closure` (tracker with
    ``completed`` runs) and :attr:`yawn`.
    """

    def __init__(self, config: Optional[SG4Config] = None) -> None:
        self.config = config if config is not None else SG4Config()
        self.closure = ClosureTracker(self.config)
        self.yawn = YawnTracker(self.config)
        self._eye_win = _HoldWindow()
        self._mouth_win = _HoldWindow()
        self.reset()

    def reset(self) -> None:
        """Start a new session: forget all state and counters."""
        self.closure.reset()
        self.yawn.reset()
        self._eye_win.clear()
        self._mouth_win.clear()
        self._prev: Optional[_Sample] = None
        self._first_t: Optional[int] = None
        self._last_fid: Optional[int] = None
        self._last_output: Optional[Dict[str, Any]] = None
        self.dropped_samples = 0
        self.mouth_valid_fraction = 0.0

    # ------------------------------------------------------------------ update
    def update(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Process one frame and return the ``temporal`` output dict.

        A record with a missing/non-numeric timestamp, or a timestamp not
        greater than the previous accepted one, is dropped: it is counted in
        :attr:`dropped_samples`, state is untouched, and the previous output
        is returned again (or an invalid placeholder if there is none yet).
        """
        sample = _parse(record)
        if sample is None or (self._prev is not None and sample.t <= self._prev.t):
            self.dropped_samples += 1
            return self._dropped_output(record, sample)

        cfg = self.config
        t = sample.t
        prev = self._prev
        if prev is None:
            self._first_t = t
        else:
            hold = min(t - prev.t, cfg.max_gap_ms)
            if prev.eye_valid:
                self._eye_win.add(prev.t, prev.t + hold, prev.closed)
            if prev.mouth_valid:
                self._mouth_win.add(prev.t, prev.t + hold, False)

        self.closure.update(t, sample.eye_valid, sample.closed)
        self.yawn.update(t, sample.mouth_valid, sample.yawn_prob)
        self._prev = sample
        fid = sample.frame_id
        if fid is None:
            fid = (self._last_fid + 1) if self._last_fid is not None else 0
        self._last_fid = fid

        output = self._build_output(fid, t)
        self._last_output = output
        return dict(output)

    # ----------------------------------------------------------------- outputs
    def _build_output(self, frame_id: int, t: int) -> Dict[str, Any]:
        cfg = self.config
        assert self._first_t is not None
        win_start = t - cfg.window_ms
        self._eye_win.expire(win_start)
        self._mouth_win.expire(win_start)
        self.closure.expire(win_start)

        elapsed = min(cfg.window_ms, t - self._first_t)
        valid_time, closed_time = self._eye_win.totals(win_start)
        mouth_time, _ = self._mouth_win.totals(win_start)
        if elapsed > 0:
            fraction = min(1.0, valid_time / elapsed)
            self.mouth_valid_fraction = min(1.0, mouth_time / elapsed)
        else:
            fraction = 0.0
            self.mouth_valid_fraction = 0.0

        out: Dict[str, Any] = {
            "frame_id": frame_id,
            "timestamp_ms": t,
            "window_ms": int(cfg.window_ms),
            "valid": False,
            "window_valid_fraction": float(fraction),
        }
        for key in MEASUREMENT_KEYS:
            out[key] = None

        is_valid = (elapsed >= cfg.warmup_ms
                    and fraction >= cfg.min_valid_fraction - _EPS
                    and valid_time > 0)
        if not is_valid:
            return out

        blinks = self.closure.blink_durations()
        out["valid"] = True
        out["perclos"] = float(closed_time / valid_time)
        out["blink_rate_per_min"] = float(len(blinks) / (valid_time / 60000.0))
        out["mean_blink_duration_ms"] = (float(sum(blinks) / len(blinks))
                                         if blinks else None)
        out["longest_closure_ms"] = self.closure.longest_ms(t)
        out["current_closure_ms"] = self.closure.current_ms(t)
        out["yawn_count"] = int(self.yawn.count(win_start))
        return out

    def _dropped_output(self, record: Any,
                        sample: Optional[_Sample]) -> Dict[str, Any]:
        if self._last_output is not None:
            return dict(self._last_output)
        fid = None
        if isinstance(record, Mapping):
            fid = _as_int(record.get("frame_id"))
        out: Dict[str, Any] = {
            "frame_id": fid if fid is not None else -1,
            "timestamp_ms": sample.t if sample is not None else -1,
            "window_ms": int(self.config.window_ms),
            "valid": False,
            "window_valid_fraction": 0.0,
        }
        for key in MEASUREMENT_KEYS:
            out[key] = None
        return out


def run_stream(config: SG4Config,
               records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Run a fresh analyzer over a record stream and return all outputs."""
    analyzer = TemporalAnalyzer(config)
    return [analyzer.update(r) for r in records]
