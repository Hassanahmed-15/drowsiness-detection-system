"""SG-5 PRIMARY method: hybrid weighted fusion.

Combines:
1. Direct thresholds for important individual cues (strong cue -> DROWSY)
2. Weighted fusion for multiple moderate cues

A strong single cue cannot be diluted by averaging, while several weak
cues can still add up to a WARNING.
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


def hybrid_decision(data):
    # STEP 1: validate SG-4 input
    if not data["valid"] or data["window_valid_fraction"] < 0.70:
        return _invalid(data)

    # STEP 2: extract features
    perclos = data["perclos"]
    longest = data["longest_closure_ms"]
    current = data["current_closure_ms"]
    yawns = data["yawn_count"]

    # STEP 3: strong safety conditions -> DROWSY directly
    if perclos >= 0.35 or longest >= 1500 or current >= 2000 or yawns >= 3:
        state, score = "DROWSY", 0.80
    else:
        # STEP 4: normalise features to 0-1
        perclos_score = min(max(perclos, 0.0), 1.0)
        closure_score = min(max(longest / 2000.0, 0.0), 1.0)
        yawn_score = min(max(yawns / 3.0, 0.0), 1.0)

        # STEP 5: weighted score
        score = 0.50 * perclos_score + 0.30 * closure_score + 0.20 * yawn_score
        score = min(max(score, 0.0), 1.0)

        # STEP 6: moderate individual cues
        moderate_cue = perclos >= 0.20 or longest >= 800 or yawns >= 2

        # STEP 7: assign state
        state = "WARNING" if (moderate_cue or score >= 0.30) else "NORMAL"

    # STEP 8: return SG-5 output
    return {
        "frame_id": data["frame_id"],
        "timestamp_ms": data["timestamp_ms"],
        "decision_valid": True,
        "state": state,
        "drowsiness_score": round(score, 3),
        "alarm_flag": state == "DROWSY",
        "alarm_type": "DROWSINESS" if state == "DROWSY" else None,
    }
