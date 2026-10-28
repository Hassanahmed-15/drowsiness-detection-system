"""Eye-closure run tracker (state machine) and blink classification."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from .config import SG4Config


@dataclass(frozen=True)
class ClosureRun:
    """One completed continuous eye closure."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class ClosureTracker:
    """Tracks closure runs from a stream of (time, valid, closed) samples.

    * A run starts at the first valid closed sample.
    * It ends when a valid *open* sample arrives (end = that timestamp), or
      when invalid/missing data lasts longer than ``max_gap_ms`` (end = last
      closed sample time + capped hold).
    * Invalid stretches up to ``max_gap_ms`` are bridged (PROPOSED).

    The "invalid stretch" is measured from the end of the last closed
    sample's capped hold to the current time, so it is the same notion of
    invalid time the window statistics use.
    """

    def __init__(self, config: SG4Config) -> None:
        self._max_gap = config.max_gap_ms
        self._blink_min = config.t_blink_min_ms
        self._blink_max = config.t_blink_max_ms
        self.completed: Deque[ClosureRun] = deque()
        self.reset()

    def reset(self) -> None:
        self.completed.clear()
        self._run_start: Optional[int] = None
        self._last_closed_t: Optional[int] = None
        self._covered_end: Optional[int] = None

    # ------------------------------------------------------------------ update
    def update(self, t: int, valid: bool, closed: bool) -> Optional[ClosureRun]:
        """Feed one sample (timestamps strictly increasing).

        Returns the run that completed on this sample, if any.
        """
        finished: Optional[ClosureRun] = None
        if self._run_start is not None:
            assert self._last_closed_t is not None
            if self._covered_end is None:
                hold = min(t - self._last_closed_t, self._max_gap)
                self._covered_end = self._last_closed_t + hold
            if t - self._covered_end > self._max_gap:
                finished = self._finish(self._covered_end)
            elif valid and not closed:
                finished = self._finish(t)
        if valid and closed:
            if self._run_start is None:
                self._run_start = t
            self._last_closed_t = t
            self._covered_end = None
        return finished

    def _finish(self, end_ms: int) -> ClosureRun:
        assert self._run_start is not None
        run = ClosureRun(self._run_start, end_ms)
        self.completed.append(run)
        self._run_start = None
        self._last_closed_t = None
        self._covered_end = None
        return run

    # ------------------------------------------------------------------ queries
    def expire(self, window_start_ms: int) -> None:
        """Forget completed runs whose end lies before the window start."""
        while self.completed and self.completed[0].end_ms < window_start_ms:
            self.completed.popleft()

    def current_ms(self, now_ms: int) -> float:
        """Time the eyes have been closed right now (0.0 if open)."""
        if self._run_start is None:
            return 0.0
        return float(now_ms - self._run_start)

    def longest_ms(self, now_ms: int) -> float:
        """Longest closure in the window, including one in progress."""
        best = max((r.duration_ms for r in self.completed), default=0)
        return float(max(best, self.current_ms(now_ms)))

    def blink_durations(self) -> List[int]:
        """Durations of completed runs classified as blinks (PROPOSED)."""
        return [r.duration_ms for r in self.completed
                if self._blink_min <= r.duration_ms <= self._blink_max]
