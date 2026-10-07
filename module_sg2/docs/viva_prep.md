# SG2 viva prep — questions the TA may ask either member

Short, plain-English answers. Both members should be able to say these in their own words.

### 1. What does SG2 do?
SG1 gives us the eye points of the driver in each video frame. **SG2 decides whether the eyes are open or closed in that frame.** It also finds **blinks**, which are short closures (≤ 500 ms), and **long closures** (> 500 ms, a possible micro-sleep). It sends one small record per frame to SG4. SG2 does **not** decide whether the driver is drowsy. That is SG4/SG5's job.

### 2. Input → processing → output?
- **Input:** SG1 V1 block: `left_eye_pts`, `right_eye_pts` (6 points each, full-frame pixels), `face_valid`, `frame_id`, `timestamp_ms`. The video frame is used only by the CNN method.
- **Processing:** compute EAR for each eye → open/closed decision → blink detector.
- **Output:** `sg2-v1` record: EAR values, `eye_state`, `closed_duration_ms`, `blink_event`, counters. See `interfaces/sg2/README.md`.

### 3. What is EAR?
Eye Aspect Ratio = (two vertical eyelid distances) ÷ (2 × horizontal corner distance).
`EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)`. An open eye gives about 0.25–0.35. A closed eye drops towards 0.05. Because it is a ratio, it does not change when the face is nearer or further from the camera.

### 4. Why did the 6-point order from SG1 matter?
EAR pairs p2 with p6 and p3 with p5. If SG1 sent the points in a different order, we would measure the wrong distances and the EAR would be meaningless. That is why the order is frozen in the V1 interface.

### 5. What was the Week 4 baseline?
Config A: EAR with the textbook threshold **0.20** (eyes closed if EAR < 0.20), averaged over both eyes.

### 6. What alternatives did we test in Week 5?
- **B** EAR with the threshold **tuned** on DEV videos.
- **C** EAR with an **adaptive per-driver threshold**: k × that driver's normal open-eye EAR (80th percentile of the last 30 s).
- **D** A **learned CNN** that looks at a 32×32 grey crop of the eye. We trained it on the MRL + CEW eye datasets.
- **E** MediaPipe's built-in **eyeBlink blendshape** score. It needs a new field from SG1, so it is only a proposal.

### 7. How did you make the comparison fair?
- Same test videos for every config (Eyeblink8 videos 4, 8, 9, 10, 11).
- Same landmarks for every config. They were extracted once and cached.
- Same metrics for every config.
- Thresholds were chosen on **different** videos (DEV), then frozen. We never tuned on the test set.

### 8. Which metrics and why?
Closed frames are rare (~3% of frames), so plain accuracy is misleading. A model that always says "open" scores 97%. So we use:
- **precision, recall and F1** of the *closed* class
- **false-closed rate:** open eye called closed, which leads to false alarms later
- **false-open rate:** closed eye missed, which leads to missed alarms later
- **blink precision/recall:** does each real blink get detected?

### 9. What is the difference between a frame label and a blink?
A frame label is one image: open or closed. A blink is an *event* made of several closed frames in a row. We count a blink as detected when our closure overlaps the annotated blink (±2 frames).

### 10. Why can't the CNN just be used everywhere?
See `week5_report.md` for the numbers. In short: it was trained on other datasets (infrared MRL, CEW photos). In the webcam test videos it also calls eyes that are only *narrowed* "closed", for example when looking down. That is **domain shift**. It is also slower than EAR, but still tiny.

### 11. What are SG2's failure cases?
- Looking down / large pitch: the eyelid looks closed from the camera.
- Glasses reflections.
- Strong head turn: the far eye is squashed.
- Very dark frames: SG1 landmarks become less accurate.
- Half-closed eyes: hard by definition.

### 12. What does `closed_duration_ms` give SG4?
How long the eyes have been closed **so far**. SG4 can react to a 2-second closure while it is still happening. It does not have to wait for the eyes to open again.

### 13. How do you run it? (show live)
```bash
cd module_sg2
python tests/test_sg2.py                                  # 8 self-checks, no data needed
python tools/run_sg2.py --sg1-jsonl ../interfaces/sg2/mock/mock_sg1_drowsy.jsonl \
       --config configs/w6_selected.yaml --out /tmp/out.jsonl
```

### 14. What did each member do?
Fill in honestly before the review (see `docs/w5_board_tasks.md`, "Owner" column).
