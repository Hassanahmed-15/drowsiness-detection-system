# SG2 viva prep — midterm questions (updated 25 Oct 2026)

Short, plain-English answers. Say them in your own words. The midterm brief says either student may be asked about the **algorithm, interface, code, test results or contribution**.

## The module

**1. What does SG2 do?**
For every frame, SG2 takes the 6 eye points that SG1 found and decides whether the eyes are **open or closed**. It also reports **blinks** (closed ≤ 0.5 s) and **long closures** (> 0.5 s). It sends one record per frame to SG4. SG2 does **not** decide drowsiness; that is SG4/SG5.

**2. Input → processing → output?**
- **Input:** SG1 V1 block (`left_eye_pts`, `right_eye_pts` with 6 points each, `face_valid`, `frame_id`, `timestamp_ms`) + the video frame.
- **Processing:** EAR → adaptive threshold → CNN check on the eye picture → blink detector.
- **Output:** `sg2-v1` record (`eye_state`, EAR, `closed_duration_ms`, `blink_event`, counters).

**3. What is EAR?**
Eye height ÷ eye width = `(|p2−p6| + |p3−p5|) / (2·|p1−p4|)`. It is about 0.25–0.30 when open and about 0.05 when closed. It is a ratio, so it does not change with distance to the camera. It is a formula from Soukupová & Čech (2016), not an OpenCV function. We compute it with NumPy.

## The algorithm choice

**4. What is your primary method?** — **Config G.**
- Step 1: the **adaptive EAR** says "closed" when EAR < 0.425 × this driver's normal open-eye EAR. The normal level is the 80th percentile of the last 30 s.
- Step 2: before we believe "closed", a small **CNN** (72k parameters) looks at a 32×32 picture of each eye and must agree (P(closed) > 0.525).

**5. What is your backup?** — **Config C**, the adaptive EAR alone. It needs no model file, only NumPy + OpenCV. If the CNN model or onnxruntime is missing (for example on the Jetson), G switches to C by itself.

**6. Why an *adaptive* threshold?**
People's eyes are different sizes. In the test videos, normal open EAR goes from 0.225 to 0.31. The textbook fixed 0.20 called thousands of open frames "closed" (F1 0.36). Comparing each driver with their own normal fixes this (F1 0.92).

**7. Why add the CNN check?**
EAR only sees the outline of the eye. When the head tilts down or the driver smiles, the outline narrows and EAR thinks "closed". The CNN looks at the actual eye picture and vetoes those cases. False alarms fell from 80 to 21. With the head tilted ≥ 10° from normal they fell from 70 to 12.

**8. Why not use the CNN on its own?**
On its own the new CNN called "closed" far too often for people it had never seen (F1 0.70, 718 false alarms). It is good at saying "no, this eye is open" when EAR is unsure, so it works as a second opinion.

**9. What else did you try?**
A fixed 0.20 (A), a tuned 0.13 (B), CNN v1 only (D), CNN v2 only (D2), MediaPipe's built-in blink score (E), and EAR + blink-score fusion (F). E and F are as good as G, but they need SG1 to send an extra field that is not in V1, so we did not adopt them.

## Testing

**10. What data did you test on?**
Public videos with **hand-made labels for every frame**: Eyeblink8 and Talking Face. We split them by video:
- **DEV** (tuning only): Talking Face + Eyeblink8 videos 1, 2, 3.
- **TEST** (results only): Eyeblink8 videos 4, 8, 9, 10, 11. That is 35,174 frames, 852 closed frames and 217 blinks.

We never tuned anything on TEST.

**11. Which metrics, and why not accuracy?**
Only 2.5% of frames are closed, so saying "always open" would score 97.5% accuracy. We use:
- **F1 of the closed class**
- **false-closed** (false alarms) and **false-open** (missed closures)
- **blink F1**: does each real blink get detected?

**12. What does the confidence interval mean?**
We resampled the test videos 2,000 times in 10-second chunks. G's F1 stays between 0.928 and 0.962 in 95% of the resamples. G beat C in 79% of the resamples, so G is *probably* better but not proven better. That is why C stays as the backup.

**13. How do you know the output is correct in the demo?**
The demo compares every frame with the hand label and shows "correct" or "wrong" plus a running agreement score. On the real sample in the repo: 8/8 blinks, F1 0.98. On the hardest full test video (eb8_08): 99.8% agreement.

**14. Where does SG2 fail?**
- Half-closed frames at the start and end of a blink: G misses about 8% of closed frames.
- Strong head tilt (improved by the CNN, but still the hardest case).
- Glasses reflections.
- Very dark video (synthetic low light: F1 0.89).
- The first second, while the adaptive threshold calibrates.

**14b. Does it work for a driver who wears a niqab?**
SG2 only needs the eyes, but SG1's face detector must first find the face. With only the eyes visible, MediaPipe at confidence 0.5 lost the face in 28% of frames on our glasses + niqab test video. At 0.2 it lost 1.7%, and the eye points were just as accurate. With the face found, our primary still gets F1 0.91 and blink F1 0.93. Our live tools use 0.2, and we asked SG1 to consider it. The test used a simulated niqab, so real fabrics and lighting still need testing.

## Interface, code, Jetson

**15. What is the V1 contract and how do you check it?**
- **Input:** SG1's V1 fields, unchanged, using SG1's own MediaPipe indices and 6-point order.
- **Output:** `sg2-v1`, defined in `interfaces/sg2/README.md` with a JSON schema.
- `tests/test_contract.py` checks every input block and validates every output record against the schema (24 tests pass).

**16. How can someone else reproduce your results?**
`pip install -r requirements.txt`, then `bash models/download_models.sh` (checksums verified), then `pytest`, then `demo.py` on the real sample in the repo. The full experiment can be re-run with the dataset script and `experiments/*.py`.

**17. Does it run on the Jetson?**
Not run on the board yet, because SG2 had no access. We tested it with the **same Python (3.10) and onnxruntime (1.23.2)** the Jetson will use: all tests pass. We wrote `check_env.py` and `camera_check.py` for SG6 to run on the board. We also found an install trap: MediaPipe pulls in a pip OpenCV without GStreamer, which breaks the CSI camera. SG2 itself needs only 0.06 ms per frame on a laptop CPU.

**18. What does `closed_duration_ms` give SG4?**
How long the eyes have been closed **so far**. SG4 can flag a 2-second closure while it is still happening. SG5 already expects a `current_closure_ms` value from SG4.

**19. What did you (each member) do?**
Fill in honestly before the review, using the "Owner" field on the project board and `git log --author`.
