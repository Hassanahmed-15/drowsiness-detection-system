"""SG-5 comparison method: pure weighted fusion (rejected, see README).

score < 0.30 -> NORMAL, 0.30 to < 0.60 -> WARNING, >= 0.60 -> DROWSY
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


def weighted_decision(data):
    # STEP 1: validate SG-4 input
    if not data["valid"] or data["window_valid_fraction"] < 0.70:
        return _invalid(data)

    # STEP 2: extract features
    perclos = data["perclos"]
    longest = data["longest_closure_ms"]
    current = data["current_closure_ms"]
    yawns = data["yawn_count"]

    # STEP 3: normalise features to 0-1
    perclos_score = min(max(perclos, 0.0), 1.0)
    closure_score = min(max(longest / 2000.0, 0.0), 1.0)
    yawn_score = min(max(yawns / 3.0, 0.0), 1.0)

    # STEP 4: safety override
    if current >= 2000:
        score, state = 1.0, "DROWSY"
    else:
        # STEP 5: weighted score
        score = 0.50 * perclos_score + 0.30 * closure_score + 0.20 * yawn_score
        score = min(max(score, 0.0), 1.0)

        # STEP 6: score -> state
        if score >= 0.60:
            state = "DROWSY"
        elif score >= 0.30:
            state = "WARNING"
        else:
            state = "NORMAL"

    # STEP 7: return SG-5 output
    return {
        "frame_id": data["frame_id"],
        "timestamp_ms": data["timestamp_ms"],
        "decision_valid": True,
        "state": state,
        "drowsiness_score": round(score, 3),
        "alarm_flag": state == "DROWSY",
        "alarm_type": "DROWSINESS" if state == "DROWSY" else None,
    }
