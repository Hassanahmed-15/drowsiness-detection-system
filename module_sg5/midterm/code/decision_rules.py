"""SG-5 BACKUP method: rule-based thresholds.

Strong cues produce DROWSY. Moderate cues produce WARNING.
"""


def _invalid(data):
    return {
        "frame_id": data["frame_id"],
        "timestamp_ms": data["timestamp_ms"],
        "decision_valid": False,
        "state": None,
        "drowsiness_score": None,
        "alarm_flag": False,
        "alarm_type": None,
    }


def make_decision(data):
    # STEP 1: validate SG-4 input
    if not data["valid"] or data["window_valid_fraction"] < 0.70:
        return _invalid(data)

    # STEP 2: extract features
    perclos = data["perclos"]
    longest = data["longest_closure_ms"]
    current = data["current_closure_ms"]
    yawns = data["yawn_count"]

    # STEP 3: strong drowsiness conditions
    if perclos >= 0.35 or longest >= 1500 or current >= 2000 or yawns >= 3:
        state, score = "DROWSY", 0.80
    else:
        # STEP 4: moderate warning conditions
        warning_count = 0
        if perclos >= 0.20:
            warning_count += 1
        if longest >= 800:
            warning_count += 1
        if yawns >= 2:
            warning_count += 1

        if warning_count >= 1:
            state, score = "WARNING", 0.50
        else:
            state, score = "NORMAL", 0.20

    # STEP 5: return SG-5 output
    return {
        "frame_id": data["frame_id"],
        "timestamp_ms": data["timestamp_ms"],
        "decision_valid": True,
        "state": state,
        "drowsiness_score": score,
        "alarm_flag": state == "DROWSY",
        "alarm_type": "DROWSINESS" if state == "DROWSY" else None,
    }
