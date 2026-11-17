"""Mock SG-4 temporal outputs used to test SG-5 independently of SG-4."""

# CASE 1: NORMAL DRIVER - low PERCLOS, short closures, no yawns -> NORMAL
normal_case = {
    "frame_id": 100,
    "timestamp_ms": 5000,
    "window_ms": 10000,
    "valid": True,
    "window_valid_fraction": 0.95,
    "perclos": 0.10,
    "blink_rate_per_min": 15.0,
    "mean_blink_duration_ms": 180.0,
    "longest_closure_ms": 300.0,
    "current_closure_ms": 0.0,
    "yawn_count": 0,
}

# CASE 2: WARNING-LEVEL DRIVER - moderate PERCLOS and closure -> WARNING
warning_case = {
    "frame_id": 101,
    "timestamp_ms": 6000,
    "window_ms": 10000,
    "valid": True,
    "window_valid_fraction": 0.92,
    "perclos": 0.24,
    "blink_rate_per_min": 18.0,
    "mean_blink_duration_ms": 240.0,
    "longest_closure_ms": 900.0,
    "current_closure_ms": 0.0,
    "yawn_count": 1,
}

# CASE 3: CLEARLY DROWSY - high PERCLOS, long closure, repeated yawns -> DROWSY
drowsy_case = {
    "frame_id": 102,
    "timestamp_ms": 7000,
    "window_ms": 10000,
    "valid": True,
    "window_valid_fraction": 0.90,
    "perclos": 0.40,
    "blink_rate_per_min": 12.0,
    "mean_blink_duration_ms": 400.0,
    "longest_closure_ms": 1800.0,
    "current_closure_ms": 1500.0,
    "yawn_count": 3,
}

# CASE 4: PROLONGED EYE CLOSURE - PERCLOS not high, but long closure -> DROWSY
long_closure_case = {
    "frame_id": 103,
    "timestamp_ms": 8000,
    "window_ms": 10000,
    "valid": True,
    "window_valid_fraction": 0.94,
    "perclos": 0.18,
    "blink_rate_per_min": 14.0,
    "mean_blink_duration_ms": 200.0,
    "longest_closure_ms": 2200.0,
    "current_closure_ms": 2200.0,
    "yawn_count": 0,
}

# CASE 5: MULTIPLE MODERATE CUES -> WARNING (or DROWSY depending on method)
multiple_moderate_cues = {
    "frame_id": 104,
    "timestamp_ms": 9000,
    "window_ms": 10000,
    "valid": True,
    "window_valid_fraction": 0.91,
    "perclos": 0.28,
    "blink_rate_per_min": 19.0,
    "mean_blink_duration_ms": 300.0,
    "longest_closure_ms": 1000.0,
    "current_closure_ms": 0.0,
    "yawn_count": 2,
}

# CASE 6: INVALID TEMPORAL DATA -> decision_valid = False
invalid_case = {
    "frame_id": 105,
    "timestamp_ms": 10000,
    "window_ms": 10000,
    "valid": False,
    "window_valid_fraction": 0.30,
    "perclos": None,
    "blink_rate_per_min": None,
    "mean_blink_duration_ms": None,
    "longest_closure_ms": None,
    "current_closure_ms": None,
    "yawn_count": None,
}
