#Weight Based

def weighted_decision(data):

    if not data["valid"]:
        return {
            "frame_id": data["frame_id"],
            "timestamp_ms": data["timestamp_ms"],
            "decision_valid": False,
            "state": None,
            "drowsiness_score": None,
            "alarm_flag": False,
            "alarm_type": None
        }

    if data["window_valid_fraction"] < 0.70:
        return {
            "frame_id": data["frame_id"],
            "timestamp_ms": data["timestamp_ms"],
            "decision_valid": False,
            "state": None,
            "drowsiness_score": None,
            "alarm_flag": False,
            "alarm_type": None
        }

    perclos = data["perclos"]
    longest = data["longest_closure_ms"]
    current = data["current_closure_ms"]
    yawns = data["yawn_count"]

    # Normalize values between 0 and 1
    perclos_score = min(perclos, 1.0)
    closure_score = min(longest / 2000, 1.0)
    yawn_score = min(yawns / 3, 1.0)

    # Safety override for very long current eye closure
    if current >= 2000:
        score = 1.0
        state = "DROWSY"

    else:
        score = (
            0.50 * perclos_score
            + 0.30 * closure_score
            + 0.20 * yawn_score
        )

        score = min(max(score, 0.0), 1.0)

        if score >= 0.70:
            state = "DROWSY"

        elif score >= 0.40:
            state = "WARNING"

        else:
            state = "NORMAL"

    return {
        "frame_id": data["frame_id"],
        "timestamp_ms": data["timestamp_ms"],
        "decision_valid": True,
        "state": state,
        "drowsiness_score": round(score, 3),
        "alarm_flag": state == "DROWSY",
        "alarm_type": "DROWSINESS" if state == "DROWSY" else None
    }
