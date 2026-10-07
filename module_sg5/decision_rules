#Rule Based

def make_decision(data):

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

    if (
        perclos >= 0.35
        or longest >= 1500
        or current >= 2000
        or yawns >= 3
    ):
        state = "DROWSY"
        score = 0.80

    else:
        warning_count = 0

        if perclos >= 0.20:
            warning_count += 1

        if longest >= 800:
            warning_count += 1

        if yawns >= 2:
            warning_count += 1

        if warning_count >= 2:
            state = "DROWSY"
            score = 0.70

        elif warning_count == 1:
            state = "WARNING"
            score = 0.50

        else:
            state = "NORMAL"
            score = 0.20

    return {
        "frame_id": data["frame_id"],
        "timestamp_ms": data["timestamp_ms"],
        "decision_valid": True,
        "state": state,
        "drowsiness_score": score,
        "alarm_flag": state == "DROWSY",
        "alarm_type": "DROWSINESS" if state == "DROWSY" else None
    }
