# SG2 datasets (Eye State & Blink Analysis)

Raw data is **not** committed. Run `bash datasets/sg2/download_sg2_data.sh`. It puts everything in `datasets/raw/sg2/`.

| Dataset | Content | SG2 use | Licence / citation |
|---|---|---|---|
| **Eyeblink8** ([blinkingmatters.com](https://www.blinkingmatters.com/research)) | 8 webcam videos, 640×480 @ 30 fps, 4 people (one wears glasses). Frame-level blink + "fully closed" annotation (`.tag`) | Week 5 **DEV** (videos 1, 2, 3) and **TEST** (videos 4, 8, 9, 10, 11) | Research use. Cite Drutarovsky & Fogelton, *Eye blink detection using variance of motion vectors*, ECCV-W 2014 |
| **Talking Face** (video: Cootes/FGNet. Blink tags: blinkingmatters.com) | 5000 frames, 720×576, 61 annotated blinks | Week 5 **DEV** | Research use |
| **MRL Eye Dataset 2018** ([mrl.cs.vsb.cz](https://mrl.cs.vsb.cz/eyedataset.html)) | 84,898 infrared eye crops, 37 people. Labels: eye state, glasses, reflections, lighting, sensor | CNN training. 6 held-out people give the glasses/reflection/lighting breakdown | Research use. Cite Fusek, *Pupil localization using geodesic distance*, ISVC 2018 |
| **CEW eye patches** ([NUAA](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html)) | 4,846 visible-light 24×24 eye crops, open/closed, in the wild | CNN training (85/15 split by face) | Research only, no commercial use. Cite Song et al., *Pattern Recognition* 2014 |

## Label rules (Eyeblink8 / Talking Face, `module_sg2/experiments/common.py`)
- **closed** = either eye is annotated *fully closed* (`C`)
- **open** = frame is outside every annotated blink and both eyes are visible
- **ignored** = half-closed frames inside a blink, frames where an eye is not visible, and frames with no annotation

Synthetic **low-light** test condition (applied to the TEST videos only): `pixel = 255 · 0.6 · (pixel/255)^2.0 + N(0, 3)`. Seeded and reproducible.
