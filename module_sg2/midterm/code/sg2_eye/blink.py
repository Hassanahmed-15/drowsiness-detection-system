"""Blink / eye-closure event extraction from the per-frame eye state.

A closure run starts on the first "closed" frame and ends on the first
"open" frame after it. When it ends it is reported as:
    blink         - min_closed_frames <= length and duration <= max_blink_ms
    long_closure  - duration > max_blink_ms (candidate micro-sleep; SG4 decides)
Short "unknown" gaps (face lost for <= max_gap_frames) do not break a run.

Note: SG2 only reports eye-level events. Windows, PERCLOS and drowsiness
logic belong to SG4/SG5.
"""


class BlinkDetector:
    def __init__(self, min_closed_frames=1, max_blink_ms=500, max_gap_frames=2, **_):
        self.min_closed_frames = int(min_closed_frames)
        self.max_blink_ms = float(max_blink_ms)
        self.max_gap_frames = int(max_gap_frames)
        self.reset()

    def reset(self):
        self.run_start = None  # (frame_id, timestamp_ms) of first closed frame
        self.run_last = None   # (frame_id, timestamp_ms) of last closed frame
        self.run_frames = 0
        self.gap = 0
        self.blink_count = 0
        self.long_closure_count = 0

    def closed_duration_ms(self, timestamp_ms):
        """How long the eyes have been closed so far (0 if currently open)."""
        if self.run_start is None:
            return 0.0
        return float(timestamp_ms - self.run_start[1])

    def _close_run(self):
        (f0, t0), (f1, t1) = self.run_start, self.run_last
        frames = self.run_frames
        self.run_start = self.run_last = None
        self.run_frames = 0
        self.gap = 0
        if frames < self.min_closed_frames:
            return None
        # duration includes the last closed frame (one frame period ~ t1 - t0 / (n-1))
        frame_ms = (t1 - t0) / (frames - 1) if frames > 1 else 33.3
        duration = float(t1 - t0 + frame_ms)
        kind = "blink" if duration <= self.max_blink_ms else "long_closure"
        if kind == "blink":
            self.blink_count += 1
        else:
            self.long_closure_count += 1
        return {"type": kind, "start_frame": f0, "end_frame": f1,
                "start_ms": t0, "end_ms": t1, "duration_ms": round(duration, 1)}

    def update(self, frame_id, timestamp_ms, state):
        """state: 'open' | 'closed' | 'unknown'. Returns a finished event or None."""
        if state == "closed":
            if self.run_start is None:
                self.run_start = (frame_id, timestamp_ms)
            self.run_last = (frame_id, timestamp_ms)
            self.run_frames += 1
            self.gap = 0
            return None
        if self.run_start is None:
            return None
        if state == "unknown":
            self.gap += 1
            if self.gap > self.max_gap_frames:
                return self._close_run()
            return None
        return self._close_run()  # reopened
