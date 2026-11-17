"""Basic and borderline SG-5 test cases with expected states."""

from mock_inputs import (
    normal_case,
    warning_case,
    drowsy_case,
    long_closure_case,
    multiple_moderate_cues,
    invalid_case,
)


def _case(frame_id, ts, frac, perclos, blink_rate, blink_dur, longest, current, yawns):
    return {
        "frame_id": frame_id,
        "timestamp_ms": ts,
        "window_ms": 10000,
        "valid": True,
        "window_valid_fraction": frac,
        "perclos": perclos,
        "blink_rate_per_min": blink_rate,
        "mean_blink_duration_ms": blink_dur,
        "longest_closure_ms": longest,
        "current_closure_ms": current,
        "yawn_count": yawns,
    }


basic_cases = {
    "Normal": (normal_case, "NORMAL"),
    "Warning": (warning_case, "WARNING"),
    "Drowsy": (drowsy_case, "DROWSY"),
    "Long Closure": (long_closure_case, "DROWSY"),
    "Multiple Moderate Cues": (multiple_moderate_cues, "WARNING"),
    "Invalid": (invalid_case, None),
}

# Cases sitting just either side of the thresholds
borderline_cases = {
    "PERCLOS Below Warning": (_case(200, 11000, 0.95, 0.19, 16.0, 190.0, 500.0, 0.0, 0), "NORMAL"),
    "PERCLOS Above Warning": (_case(201, 12000, 0.95, 0.21, 17.0, 200.0, 500.0, 0.0, 0), "WARNING"),
    "Closure Below Drowsy": (_case(202, 13000, 0.94, 0.15, 14.0, 220.0, 1450.0, 0.0, 0), "WARNING"),
    "Closure Above Drowsy": (_case(203, 14000, 0.94, 0.15, 14.0, 220.0, 1550.0, 0.0, 0), "DROWSY"),
    "Yawns Only": (_case(204, 15000, 0.93, 0.10, 15.0, 180.0, 400.0, 0.0, 2), "WARNING"),
    "Low Valid Fraction": (_case(205, 16000, 0.60, 0.40, 10.0, 350.0, 1700.0, 1000.0, 2), None),
    "Moderate Combination": (_case(206, 17000, 0.92, 0.22, 18.0, 250.0, 850.0, 0.0, 1), "WARNING"),
    "High PERCLOS Only": (_case(207, 18000, 0.96, 0.38, 13.0, 300.0, 700.0, 0.0, 0), "DROWSY"),
}
