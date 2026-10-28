"""Niqab (lower-face + head covering) robustness of SG1-compatible landmarks and SG2.

Uses the synthetic niqab copies of the TEST videos (prepare_videos.py --variants niqab)
and compares them with the normal copies, at SG1's face-detection confidence (0.5) and
at a lower one (0.2; prepare_videos.py --variants normal niqab --test-only --min-confidence 0.2):
  * landmark quality: SG1-compatible eye corners vs the hand-annotated eye corners in
    the .tag files, as normalised error (NME = mean corner error / inter-ocular distance)
  * SG2 quality: closed-frame F1 and blink F1 for each configuration

Run from module_sg2/:  python experiments/evaluate_occlusion.py
Outputs: results/occlusion/*.csv, results/occlusion/figures/*.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_ROOT, MODULE_DIR, RESULTS_DIR, TEST, VIDEOS  # noqa: E402
from evaluate import CNN_PATH, load_video  # noqa: E402
from evaluate_midterm import block_counts, load_config, run, summarise  # noqa: E402
from occlusion import parse_geometry  # noqa: E402
from sg2_eye.classifiers import CNNEyeState  # noqa: E402

OUT = RESULTS_DIR / "occlusion"
CFG_DIR = MODULE_DIR / "configs"
CONFIGS = {"A": "baseline_w4_ear_fixed.yaml", "C (backup)": "backup.yaml", "G (primary)": "primary.yaml",
           "D": "w5_D_cnn.yaml", "D2": "midterm_D2_cnn_v2.yaml"}


def landmark_error(v, name):
    """Normalised eye-corner error of the cached SG1-compatible points vs the .tag annotation."""
    stem = DATA_ROOT / VIDEOS[name]
    _, gt = parse_geometry(str(stem) + ".tag", len(v["valid"]))
    mp = np.concatenate([v["right_pts"][:, [0, 3]], v["left_pts"][:, [0, 3]]], axis=1)  # 4 corners
    ok = v["valid"] & ~np.isnan(gt).any(axis=(1, 2))
    nme = np.full(len(ok), np.nan)
    for i in np.where(ok)[0]:
        a = mp[i][np.argsort(mp[i][:, 0])]
        b = gt[i][np.argsort(gt[i][:, 0])]
        iod = np.linalg.norm(b[:2].mean(0) - b[2:].mean(0))
        nme[i] = np.linalg.norm(a - b, axis=1).mean() / max(iod, 1e-6)
    return nme


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    cnn = CNNEyeState(CNN_PATH)
    rows, lm_rows = [], []
    variants = {"normal, conf 0.5": "", "niqab, conf 0.5": "_niqab",
                "normal, conf 0.2": "_c0.2", "niqab, conf 0.2": "_niqab_c0.2"}
    for variant, suffix in variants.items():
        vids = [load_video(n + suffix, cnn) for n in TEST]
        for n, v in zip(TEST, vids):
            nme = landmark_error(v, n)
            lab = v["label"]
            lm_rows.append({"variant": variant, "video": n, "face_found": float(v["valid"].mean()),
                            "nme_median": float(np.nanmedian(nme)),
                            "nme_good_lt_0.10": float(np.nanmean(nme < 0.10)),
                            "open_EAR_median": float(np.nanmedian(v["ear"][lab == 0])),
                            "closed_EAR_median": float(np.nanmedian(v["ear"][lab == 1]))})
        for key, f in CONFIGS.items():
            cfg = load_config(CFG_DIR / f)
            per = [run(cfg, v) for v in vids]
            cnt = np.concatenate([block_counts(v, c, e) for v, (c, e, _) in zip(vids, per)])
            tot = cnt.sum(axis=0)
            # frames where SG2 gives no answer (no face) but the annotation says closed
            unknown_closed = int(sum(((~v["valid"]) & (v["label"] == 1)).sum() for v in vids))
            rows.append({"variant": variant, "config": key, "TP": int(tot[0]), "FP": int(tot[1]), "FN": int(tot[2]),
                         "closed_frames_with_no_face": unknown_closed, **summarise(cnt)})
            print(variant, key, f"F1 {rows[-1]['f1']:.3f} blink F1 {rows[-1]['blink_f1']:.3f}", flush=True)
    lm = pd.DataFrame(lm_rows)
    res = pd.DataFrame(rows)
    lm.round(4).to_csv(OUT / "landmarks.csv", index=False)
    res.round(4).to_csv(OUT / "comparison.csv", index=False)
    print(lm.round(3).to_string(index=False))
    print(res.round(3).to_string(index=False))

    piv = res.pivot(index="config", columns="variant", values="f1").loc[list(CONFIGS), list(variants)]
    ax = piv.plot.bar(figsize=(9, 4), rot=0)
    ax.set_ylabel("closed-frame F1 (TEST)")
    ax.set_title("SG2 with a synthetic niqab, at face-detection confidence 0.5 vs 0.2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "niqab_f1.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
