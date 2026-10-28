"""Yawn tracker: smoothing, hysteresis, minimum duration, refractory period."""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from .config import SG4Config

_EPS = 1e-9


class YawnTracker:
    """Counts distinct yawns from a stream of (time, valid, yawn_prob).

    PROPOSED behaviour (all thresholds from :class:`SG4Config`):

    * moving average of the last ``yawn_smooth_n`` valid mouth samples;
    * a *candidate* starts when the smoothed value is >= ``yawn_on_threshold``
      (and the refractory period is over) and is *confirmed* once it has
      stayed >= the on threshold for ``yawn_min_ms``;
    * an unconfirmed candidate is cancelled if the smoothed value drops below
      the on threshold; a confirmed event ends below ``yawn_off_threshold``;
    * after a confirmed event ends, new candidates are ignored for
      ``yawn_refractory_ms``;
    * a mouth-data gap longer than ``max_gap_ms`` cancels a candidate and ends
      a confirmed event.
    """

    def __init__(self, config: SG4Config) -> None:
        self._cfg = config
        self.confirmed_starts: Deque[int] = deque()
        self.reset()

    def reset(self) -> None:
        self.confirmed_starts.clear()
        self._buf: Deque[float] = deque(maxlen=int(self._cfg.yawn_smooth_n))
        self._cand_start: Optional[int] = None
        self._confirmed = False
        self._refractory_until: Optional[int] = None
        self._last_valid_t: Optional[int] = None
        self._covered_end: Optional[int] = None

    @property
    def in_event(self) -> bool:
        """True while a confirmed yawn is in progress."""
        return self._confirmed

    # ------------------------------------------------------------------ update
    def update(self, t: int, valid: bool, prob: Optional[float]) -> None:
        """Feed one sample (every frame, valid or not)."""
        cfg = self._cfg
        if self._last_valid_t is not None:
            if self._covered_end is None:
                hold = min(t - self._last_valid_t, cfg.max_gap_ms)
                self._covered_end = self._last_valid_t + hold
            if t - self._covered_end > cfg.max_gap_ms:
                self._break(self._covered_end)
        if not valid or prob is None:
            return

        self._buf.append(prob)
        smoothed = sum(self._buf) / len(self._buf)
        self._last_valid_t = t
        self._covered_end = None
        above_on = smoothed >= cfg.yawn_on_threshold - _EPS

        if self._confirmed:
            if smoothed < cfg.yawn_off_threshold - _EPS:
                self._end_event(t)
            return

        if self._cand_start is None:
            in_refractory = (self._refractory_until is not None
                             and t < self._refractory_until)
            if above_on and not in_refractory:
                self._cand_start = t
        elif not above_on:
            self._cand_start = None
            return

        if self._cand_start is not None and t - self._cand_start >= cfg.yawn_min_ms:
            self.confirmed_starts.append(self._cand_start)
            self._confirmed = True

    def _end_event(self, at_ms: int) -> None:
        self._confirmed = False
        self._cand_start = None
        self._refractory_until = at_ms + self._cfg.yawn_refractory_ms

    def _break(self, at_ms: int) -> None:
        """Mouth-data gap too long: end/cancel the event and drop stale state."""
        if self._confirmed:
            self._end_event(at_ms)
        self._cand_start = None
        self._buf.clear()
        self._last_valid_t = None
        self._covered_end = None

    # ------------------------------------------------------------------ queries
    def count(self, window_start_ms: int) -> int:
        """Confirmed yawns whose start lies inside the window."""
        while self.confirmed_starts and self.confirmed_starts[0] < window_start_ms:
            self.confirmed_starts.popleft()
        return len(self.confirmed_starts)
