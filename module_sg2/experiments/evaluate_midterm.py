"""Midterm (Stage 1) SG2 evaluation: primary vs backup with statistical evidence.

Extends the Week 5 protocol (experiments/evaluate.py, same DEV/TEST split,
same labels, same cached SG1-compatible landmarks) with:
  * D2  - CNN v2 (MRL + CEW + webcam crops of DEV videos talking/eb8_01/eb8_02)
  * F   - adaptive EAR + eyeBlink blendshape score fusion (needs non-V1 field)
  * G   - adaptive EAR, confirmed by CNN v2 (V1 inputs only)
  * 95% confidence intervals by block bootstrap (10 s blocks, 2000 resamples)
    and paired differences against the primary config C
  * difficult cases with head pitch measured RELATIVE to each driver's median

Tuning rules (no TEST data is used for any choice):
  * F: weight + threshold tuned on all DEV videos
  * D2, G: tuned only on DEV video eb8_03, the one video held out of CNN v2
    training (tuning on the CNN's own training videos would be biased)

Run from module_sg2/ after prepare_videos.py, train_cnn.py and
train_cnn.py --with-dev-video:
    python experiments/evaluate_midterm.py
Outputs: results/midterm/*.csv, results/midterm/figures/*.png, configs/midterm_*.yaml
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DEV, MODULE_DIR, RESULTS_DIR, TEST, blink_intervals  # noqa: E402
from evaluate import BLINK_CFG, BLINK_TOL, CNN_PATH, frame_metrics, load_video  # noqa: E402
from sg2_eye.classifiers import CNNEyeState  # noqa: E402
from sg2_eye.module import EyeStateModule, load_config  # noqa: E402

OUT = RESULTS_DIR / "midterm"
FIG = OUT / "figures"
CFG_DIR = MODULE_DIR / "configs"
CNN2_PATH = MODULE_DIR / "models" / "eye_state_cnn_v2.onnx"
HELD_OUT = ["eb8_03"]          # DEV video never seen by CNN v2 training
BLOCK = 300                    # bootstrap block = 10 s at 30 fps
N_BOOT = 2000
K_C = 0.425                    # Config C (frozen in Week 5)
FALLBACK = 0.13
PROB_GRID = np.concatenate([np.arange(0.005, 0.05, 0.005), np.arange(0.05, 0.95, 0.025),
                            np.arange(0.95, 0.9951, 0.005)])


# ------------------------------------------------------------------ helpers
def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def c_pred(v):
    """Config C decision, vectorised exactly like the module (fallback in warm-up)."""
    thr = np.clip(K_C * v["ear_base"], 0.10, 0.35)
    thr = np.where(np.isnan(v["ear_base"]), FALLBACK, thr)
    return (v["ear"] < thr) & v["valid"]


def labelled(videos):
    return np.concatenate([np.where(v["valid"], v["label"], -1) for v in videos])


def best_threshold(videos, score_fn, grid):
    y = labelled(videos)
    s = np.concatenate([score_fn(v) for v in videos])
    rows = []
    for t in grid:
        m = frame_metrics(y, np.nan_to_num(s, nan=-1) > t)
        rows.append({"threshold": round(float(t), 4), **m})
    df = pd.DataFrame(rows)
    return df.loc[df.f1.idxmax()], df


def make_cfg(name, method, **params):
    return {"name": name, "classifier": {"method": method, **params}, "blink": dict(BLINK_CFG),
            "crop": {"size": 32, "scale": 1.6}}


def save_cfg(cfg, filename, header):
    (CFG_DIR / filename).write_text(header + yaml.safe_dump(json.loads(json.dumps(cfg)), sort_keys=False))


# ------------------------------------------------------- run the real module
def run(cfg, v):
    """Replay one video through EyeStateModule; per-frame closed + events + ms/frame."""
    module = EyeStateModule(load_config_dict(cfg))
    module.reset()
    ts = v["timestamp_ms"]
    closed = np.zeros(len(ts), bool)
    events, t_total = [], 0.0
    for i in range(len(ts)):
        ok = bool(v["valid"][i])
        block = {"frame_id": i, "timestamp_ms": float(ts[i]), "face_valid": ok,
                 "left_eye_pts": v["left_pts"][i].tolist() if ok else None,
                 "right_eye_pts": v["right_pts"][i].tolist() if ok else None,
                 "head_pose": None,
                 "blendshapes": {"eyeBlinkLeft": float(v["blink_bs"][i, 0]),
                                 "eyeBlinkRight": float(v["blink_bs"][i, 1])} if ok else None}
        t0 = time.perf_counter()
        rec = module.process(block, crops=(v["left_crop"][i], v["right_crop"][i]))
        t_total += time.perf_counter() - t0
        closed[i] = rec["eye_state"] == "closed"
        if rec["blink_event"]:
            events.append((rec["blink_event"]["start_frame"], rec["blink_event"]["end_frame"]))
    return closed, events, 1000 * t_total / len(ts)


def load_config_dict(cfg):
    cfg = json.loads(json.dumps(cfg))
    mp = cfg["classifier"].get("model_path")
    if mp and not Path(mp).is_absolute():
        cfg["classifier"]["model_path"] = str(MODULE_DIR / mp)
    return cfg


def block_counts(v, closed, events, mask=None):
    """Per 10-s block: frame TP/FP/FN and blink matching counts (for bootstrap)."""
    y = np.where(v["valid"], v["label"], -1)
    if mask is not None:
        y = np.where(mask, y, -1)
    n = len(y)
    nb = int(np.ceil(n / BLOCK))
    blk = np.arange(n) // BLOCK
    out = np.zeros((nb, 7))  # TP FP FN | gt_blinks gt_matched det det_matched
    m = y >= 0
    np.add.at(out[:, 0], blk[m & closed & (y == 1)], 1)
    np.add.at(out[:, 1], blk[m & closed & (y == 0)], 1)
    np.add.at(out[:, 2], blk[m & ~closed & (y == 1)], 1)
    gt = blink_intervals(v["blink_id"])
    used = set()
    det_matched = np.zeros(len(events), bool)
    for k, (s, e) in enumerate(events):
        for j, (gs, ge) in enumerate(gt):
            if j not in used and s - BLINK_TOL <= ge and e + BLINK_TOL >= gs:
                used.add(j)
                det_matched[k] = True
                break
    for j, (gs, _) in enumerate(gt):
        out[gs // BLOCK, 3] += 1
        out[gs // BLOCK, 4] += j in used
    for k, (s, _) in enumerate(events):
        out[min(s // BLOCK, nb - 1), 5] += 1
        out[min(s // BLOCK, nb - 1), 6] += det_matched[k]
    return out


def summarise(c):
    tp, fp, fn, gt, gtm, det, detm = c.sum(axis=0) if c.ndim == 2 else c
    bp, br = detm / max(det, 1), gtm / max(gt, 1)
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": f1(tp, fp, fn),
            "blink_precision": bp, "blink_recall": br, "blink_f1": 2 * bp * br / max(bp + br, 1e-9)}


def bootstrap(counts_by_cfg, seed=477):
    keys = list(counts_by_cfg)
    nb = len(next(iter(counts_by_cfg.values())))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(N_BOOT, nb))
    stats = {}
    for k in keys:
        c = counts_by_cfg[k]
        sums = c[idx].sum(axis=1)  # (N_BOOT, 7)
        tp, fp, fn, gt, gtm, det, detm = sums.T
        bp, br = detm / np.maximum(det, 1), gtm / np.maximum(gt, 1)
        stats[k] = {"f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1),
                    "blink_f1": 2 * bp * br / np.maximum(bp + br, 1e-9)}
    return stats


# --------------------------------------------------------------------- main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cnn1, cnn2 = CNNEyeState(CNN_PATH), CNNEyeState(CNN2_PATH)

    def with_cnn2(v):
        p = np.full(len(v["valid"]), np.nan, np.float32)
        ok = v["valid"]
        if ok.any():
            p[ok] = 0.5 * (cnn2.predict_proba(v["left_crop"][ok]) + cnn2.predict_proba(v["right_crop"][ok]))
        v["p_cnn2"] = p
        rel = v["pose"][:, 1] - np.nanmedian(v["pose"][ok, 1])
        v["pitch_rel"] = rel
        return v

    dev = [with_cnn2(load_video(n, cnn1)) for n in DEV]
    test = [with_cnn2(load_video(n, cnn1)) for n in TEST]
    test_dark = [with_cnn2(load_video(n + "_lowlight", cnn1)) for n in TEST]
    held = [v for v in dev if v["name"] in HELD_OUT]

    # ---------------- tuning (DEV only)
    tuning = []
    b_d2, _ = best_threshold(held, lambda v: v["p_cnn2"], PROB_GRID)
    tuning.append({"config": "D2", "tuned_on": "eb8_03", "param": "cnn_threshold", "value": b_d2.threshold,
                   "dev_f1": b_d2.f1})
    b_g, _ = best_threshold(held, lambda v: np.where(c_pred(v), v["p_cnn2"], -1.0), PROB_GRID)
    tuning.append({"config": "G", "tuned_on": "eb8_03", "param": "cnn_threshold", "value": b_g.threshold,
                   "dev_f1": b_g.f1})
    best_f = None
    for w in (0.25, 0.5, 0.75):
        def fused(v, w=w):
            open_level = np.where(np.isnan(v["ear_base"]), FALLBACK / K_C, v["ear_base"])
            return w * np.clip(1 - v["ear"] / open_level, 0, 1) + (1 - w) * v["p_bs"]
        b, _ = best_threshold(dev, fused, np.arange(0.05, 0.951, 0.0125))
        if best_f is None or b.f1 > best_f[1].f1:
            best_f = (w, b)
    tuning.append({"config": "F", "tuned_on": "DEV (all 4)", "param": "weight / fused_threshold",
                   "value": f"{best_f[0]} / {best_f[1].threshold}", "dev_f1": best_f[1].f1})
    pd.DataFrame(tuning).round(4).to_csv(OUT / "tuning.csv", index=False)
    print(pd.DataFrame(tuning).round(4).to_string(index=False))

    ada = dict(k=K_C, window_frames=900, percentile=80, warmup_frames=30, fallback_threshold=FALLBACK,
               min_threshold=0.10, max_threshold=0.35, fusion="mean")
    new_cfgs = {
        "D2": make_cfg(f"D2_cnn_v2_p{b_d2.threshold:.3f}", "cnn", model_path="models/eye_state_cnn_v2.onnx",
                       threshold=float(b_d2.threshold)),
        "F": make_cfg(f"F_ear_adaptive_blendshape_w{best_f[0]}", "ear_adaptive_blendshape",
                      weight=best_f[0], fused_threshold=float(best_f[1].threshold), **ada),
        "G": make_cfg(f"G_ear_adaptive_cnn_check_p{b_g.threshold:.3f}", "ear_adaptive_cnn_check",
                      model_path="models/eye_state_cnn_v2.onnx", cnn_threshold=float(b_g.threshold), **ada),
    }
    save_cfg(new_cfgs["D2"], "midterm_D2_cnn_v2.yaml",
             "# Config D2 (midterm) - CNN v2 (MRL + CEW + DEV webcam crops); threshold tuned on eb8_03 only\n")
    save_cfg(new_cfgs["F"], "midterm_F_ear_blendshape_fusion.yaml",
             "# Config F (midterm) - adaptive EAR + eyeBlink blendshape score fusion (needs non-V1 field)\n"
             "# weight and threshold tuned on all DEV videos\n")
    save_cfg(new_cfgs["G"], "midterm_G_ear_adaptive_cnn_check.yaml",
             "# Config G (midterm) - adaptive EAR (Config C) confirmed by CNN v2; V1 inputs only\n"
             "# cnn_threshold tuned on eb8_03 only (held out of CNN v2 training)\n")

    week5 = {k: load_config(CFG_DIR / f) for k, f in (
        ("A", "baseline_w4_ear_fixed.yaml"), ("B", "w5_B_ear_fixed_tuned.yaml"), ("C", "w5_C_ear_adaptive.yaml"),
        ("D", "w5_D_cnn.yaml"), ("E", "w5_E_blendshape.yaml"))}
    configs = {**week5, **new_cfgs}
    order = ["A", "B", "C", "D", "D2", "E", "F", "G"]
    v1_ok = {"A": "yes", "B": "yes", "C": "yes", "D": "yes (+frame)", "D2": "yes (+frame)",
             "E": "no (needs blendshapes)", "F": "no (needs blendshapes)", "G": "yes (+frame)"}

    # ---------------- run every config on TEST (+ low light)
    rows, counts, counts_dark, per_video, hard = [], {}, {}, [], []
    conds = {
        "glasses (eb8_11)": (lambda v: str(v["glasses"]) == "YES", None),
        "no glasses": (lambda v: str(v["glasses"]) != "YES", None),
        "head pitch within 5 deg of driver's normal": (None, lambda v: np.abs(v["pitch_rel"]) < 5),
        "head pitch 5-10 deg from normal": (None, lambda v: (np.abs(v["pitch_rel"]) >= 5) & (np.abs(v["pitch_rel"]) < 10)),
        "head pitch >= 10 deg from normal": (None, lambda v: np.abs(v["pitch_rel"]) >= 10),
    }
    for key in order:
        cfg = configs[key]
        per = [run(cfg, v) for v in test]
        per_dark = [run(cfg, v) for v in test_dark]
        cnt = np.concatenate([block_counts(v, c, e) for v, (c, e, _) in zip(test, per)])
        cnt_dark = np.concatenate([block_counts(v, c, e) for v, (c, e, _) in zip(test_dark, per_dark)])
        counts[key], counts_dark[key] = cnt, cnt_dark
        ms = float(np.mean([t for _, _, t in per]))
        tot = cnt.sum(axis=0)
        rows.append({"config": key, "name": cfg["name"], "v1_compatible": v1_ok[key],
                     "closed_frames": int(tot[0] + tot[2]), "TP": int(tot[0]), "FP": int(tot[1]), "FN": int(tot[2]),
                     **summarise(cnt), "low_light_f1": summarise(cnt_dark)["f1"],
                     "low_light_blink_f1": summarise(cnt_dark)["blink_f1"], "sg2_ms_per_frame": ms})
        for v, (c, e, _) in zip(test, per):
            per_video.append({"config": key, "video": v["name"], **summarise(block_counts(v, c, e))})
        for cname, (vid_sel, mask_fn) in conds.items():
            parts = [block_counts(v, c, e, mask_fn(v) if mask_fn else None)
                     for v, (c, e, _) in zip(test, per) if vid_sel is None or vid_sel(v)]
            cc = np.concatenate(parts)
            hard.append({"config": key, "condition": cname, "closed_frames": int(cc[:, 0].sum() + cc[:, 2].sum()),
                         "f1": summarise(cc)["f1"], "FP": int(cc[:, 1].sum()), "FN": int(cc[:, 2].sum())})
        hard.append({"config": key, "condition": "low light (synthetic)", "closed_frames": int(tot[0] + tot[2]),
                     "f1": summarise(cnt_dark)["f1"], "FP": int(cnt_dark[:, 1].sum()), "FN": int(cnt_dark[:, 2].sum())})
        print(f"done {key}: F1 {rows[-1]['f1']:.3f} blink F1 {rows[-1]['blink_f1']:.3f}", flush=True)

    # ---------------- bootstrap CIs + paired differences vs C
    boot = bootstrap(counts)
    boot_dark = bootstrap(counts_dark)
    ci_rows, paired = [], []
    for key in order:
        for metric in ("f1", "blink_f1"):
            for cond, b in (("TEST", boot), ("TEST low-light", boot_dark)):
                lo, hi = np.percentile(b[key][metric], [2.5, 97.5])
                ci_rows.append({"config": key, "test_set": cond, "metric": metric, "ci95_low": lo, "ci95_high": hi})
                if key != "C":
                    d = b[key][metric] - b["C"][metric]
                    paired.append({"config": key, "vs": "C", "test_set": cond, "metric": metric,
                                   "mean_diff": d.mean(), "ci95_low": np.percentile(d, 2.5),
                                   "ci95_high": np.percentile(d, 97.5), "p_better_than_C": (d > 0).mean()})
    res = pd.DataFrame(rows)
    ci = pd.DataFrame(ci_rows)
    for metric in ("f1", "blink_f1"):
        sub = ci[(ci.test_set == "TEST") & (ci.metric == metric)].set_index("config")
        res[f"{metric}_ci95"] = res.config.map(lambda k: f"{sub.loc[k, 'ci95_low']:.3f}-{sub.loc[k, 'ci95_high']:.3f}")
    res.round(4).to_csv(OUT / "comparison.csv", index=False)
    ci.round(4).to_csv(OUT / "bootstrap_ci.csv", index=False)
    pd.DataFrame(paired).round(4).to_csv(OUT / "paired_vs_C.csv", index=False)
    pd.DataFrame(per_video).round(4).to_csv(OUT / "per_video.csv", index=False)
    hard_df = pd.DataFrame(hard)
    hard_df.round(4).to_csv(OUT / "difficult_cases.csv", index=False)
    print(res[["config", "v1_compatible", "precision", "recall", "f1", "f1_ci95", "blink_f1", "blink_f1_ci95",
               "low_light_f1", "sg2_ms_per_frame"]].round(3).to_string(index=False))
    print(pd.DataFrame(paired).query("test_set == 'TEST'").round(3).to_string(index=False))
    plots(res, ci, hard_df, order)


def plots(res, ci, hard_df, order):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric, title in ((axes[0], "f1", "Closed-frame F1"), (axes[1], "blink_f1", "Blink F1")):
        sub = ci[(ci.test_set == "TEST") & (ci.metric == metric)].set_index("config").loc[order]
        point = res.set_index("config").loc[order, metric]
        y = np.arange(len(order))[::-1]
        colours = ["C1" if k == "C" else ("C2" if k == "G" else "C0") for k in order]
        ax.hlines(y, sub.ci95_low, sub.ci95_high, color=colours, lw=3)
        ax.scatter(point, y, color=colours, zorder=3)
        ax.set_yticks(y, order)
        ax.set_xlabel(f"{title} on TEST (dot = value, bar = 95% CI)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_ylabel("configuration")
    fig.suptitle("SG2 midterm: primary vs alternatives with 95% bootstrap confidence intervals")
    fig.tight_layout()
    fig.savefig(FIG / "confidence_intervals.png", dpi=130)
    plt.close(fig)

    piv = hard_df.pivot(index="condition", columns="config", values="f1")[order]
    piv = piv.loc[list(dict.fromkeys(hard_df.condition))]
    ax = piv.plot.barh(figsize=(9, 5), width=0.85)
    ax.set_xlabel("F1 (closed frames), TEST")
    ax.set_title("SG2 midterm: difficult cases (pitch relative to each driver's normal)")
    ax.invert_yaxis()
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / "difficult_cases.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
