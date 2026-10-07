# SG1 Landmark Map

This document defines the **MediaPipe Face Mesh landmark indices** used by **SG1: Face/Landmark Detection**.

> **Coordinate rule:** All landmark coordinates are converted to **full-frame pixel coordinates** before being passed to downstream modules.

## 1. Eye Landmark Ordering

The SG1 interface outputs **6 ordered landmark points for each eye**. The required ordering is:

| Point | Description   |
|:-----:|---------------|
| `p1`  | Outer eye corner |
| `p2`  | Upper eyelid  |
| `p3`  | Upper eyelid  |
| `p4`  | Inner eye corner |
| `p5`  | Lower eyelid  |
| `p6`  | Lower eyelid  |

This ordering is important because **SG2** uses these points to calculate the **Eye Aspect Ratio (EAR)**.

The expected EAR point layout:

```text
          p2        p3
           ●--------●
         /            \
      p1 ●              ● p4
         \            /
           ●--------●
          p6        p5
```

---

## 2. Driver's Left Eye

**MediaPipe indices:**

```python
LEFT_EYE = [263, 387, 385, 362, 380, 373]
```

| Point | MediaPipe Index | Meaning       |
|:-----:|:---------------:|---------------|
| `p1`  | `263`           | Outer corner  |
| `p2`  | `387`           | Upper eyelid  |
| `p3`  | `385`           | Upper eyelid  |
| `p4`  | `362`           | Inner corner  |
| `p5`  | `380`           | Lower eyelid  |
| `p6`  | `373`           | Lower eyelid  |

---

## 3. Driver's Right Eye

**MediaPipe indices:**

```python
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
```

| Point | MediaPipe Index | Meaning       |
|:-----:|:---------------:|---------------|
| `p1`  | `33`            | Outer corner  |
| `p2`  | `160`           | Upper eyelid  |
| `p3`  | `158`           | Upper eyelid  |
| `p4`  | `133`           | Inner corner  |
| `p5`  | `153`           | Lower eyelid  |
| `p6`  | `144`           | Lower eyelid  |

---

## 4. Mouth Landmarks

The current V1 mouth representation uses four landmarks:

```python
MOUTH = [61, 13, 291, 14]
```

| MediaPipe Index | Meaning            |
|:---------------:|--------------------|
| `61`            | Left mouth corner  |
| `13`            | Upper lip          |
| `291`           | Right mouth corner |
| `14`            | Lower lip          |

The `mouth_pts` field is **optional** in the SG1 V1 interface.

---

## 5. Head Pose Landmarks

The following landmarks are used for approximate **head-pose estimation**:

| MediaPipe Index | Facial Point       |
|:---------------:|--------------------|
| `1`             | Nose               |
| `152`           | Chin               |
| `33`            | Right eye corner   |
| `263`           | Left eye corner    |
| `61`            | Left mouth corner  |
| `291`           | Right mouth corner |

### Head Pose Output

SG1 may provide the following head-pose angles:

```text
yaw_deg
pitch_deg
roll_deg
```

- **Yaw:** head turning left/right
- **Pitch:** head looking up/down
- **Roll:** head tilting sideways

## 6. Coordinate Convention

All SG1 landmark coordinates follow the same coordinate system:

```text
Origin (0,0)
    ┌──────────────────────────► +x
    │
    │
    │
    ▼
   +y
```

**Rules:**

- Coordinates are given in **full-frame pixels**.
- The origin `(0, 0)` is at the **top-left corner** of the frame.
- `x` increases toward the **right**.
- `y` increases toward the **bottom**.
- **Left** and **right** refer to the **driver's own left and right**, not the viewer's perspective.

---

## 7. SG1 Landmark Summary

```python
LEFT_EYE = [
    263,  # p1 - outer corner
    387,  # p2 - upper eyelid
    385,  # p3 - upper eyelid
    362,  # p4 - inner corner
    380,  # p5 - lower eyelid
    373,  # p6 - lower eyelid
]

RIGHT_EYE = [
    33,   # p1 - outer corner
    160,  # p2 - upper eyelid
    158,  # p3 - upper eyelid
    133,  # p4 - inner corner
    153,  # p5 - lower eyelid
    144,  # p6 - lower eyelid
]

MOUTH = [
    61,   # left mouth corner
    13,   # upper lip
    291,  # right mouth corner
    14,   # lower lip
]

HEAD_POSE = [
    1,    # nose
    152,  # chin
    33,   # right eye corner
    263,  # left eye corner
    61,   # left mouth corner
    291,  # right mouth corner
]

## 8. Interface Notes

- Eye landmarks **must preserve the defined six-point ordering**.
- SG2 depends on this ordering for correct **EAR calculation**.
- Landmark coordinates must be converted from MediaPipe's normalized coordinates to **full-frame pixel coordinates** before transmission.
- `mouth_pts` is optional in the **V1 SG1 interface**.
- Head-pose values are expressed in **degrees**.
- Driver left/right terminology must remain consistent across all project modules.
