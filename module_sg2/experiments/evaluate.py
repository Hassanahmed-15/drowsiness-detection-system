"""Week 5 SG2 experiment: compare eye-state configurations on the SAME data.

Protocol
  * Data: Eyeblink8 (8 videos) + Talking Face (1 video), frame-level labels
    from the official .tag files (see common.parse_tag). Fixed split by video:
    DEV (threshold tuning) and TEST (reporting). Plus a synthetic low-light
    copy of the TEST videos.
  * Every configuration receives identical SG1-compatible landmarks/crops
    (experiments/cache), so differences come only from the SG2 method.
  * Thresholds are tuned on DEV by maximising F1 of the "closed" class and
    then frozen; all reported numbers are on TEST.
  * Frame metrics (closed = positive): precision, recall, F1, balanced
    accuracy, false-closed rate (open frame called closed) and false-open
    rate (closed frame called open). Blink metrics: event precision/recall/F1
    (detected closure overlaps an annotated blink, +-2 frames, one-to-one).
  * Final numbers come from running the real EyeStateModule (sg2_eye), not a
    re-implementation.

Run from module_sg2/ after prepare_videos.py and train_cnn.py:
    python experiments/evaluate.py
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
from common import DEV, MODULE_DIR, RESULTS_DIR, TEST, blink_intervals, load_cache  # noqa: E402
from sg2_eye.classifiers import CNNEyeState, EARAdaptive  # noqa: E402
from sg2_eye.geometry import eye_aspect_ratio  # noqa: E402
from sg2_eye.module import EyeStateModule  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
CFG_DIR = MODULE_DIR / "configs"
CNN_PATH = MODULE_DIR / "models" / "eye_state_cnn.onnx"
BLINK_TOL = 2
BLINK_CFG = {"min_closed_frames": 1, "max_blink_ms": 500, "max_gap_frames": 2}


# --------------------------------------------------------------------------- data
def load_video(name, cnn):
    c = load_cache(name)
    valid = c["valid"].astype(bool)
    n = len(valid)
    l_ear = np.full(n, np.nan, np.float32)
    r_ear = np.full(n, np.nan, np.float32)
    for i in np.where(valid)[0]:
        l_ear[i] = eye_aspect_ratio(c["left_pts"][i])
        r_ear[i] = eye_aspect_ratio(c["right_pts"][i])
    lw = np.linalg.norm(c["left_pts"][:, 0] - c["left_pts"][:, 3], axis=1)
    rw = np.linalg.norm(c["right_pts"][:, 0] - c["right_pts"][:, 3], axis=1)
    p_cnn = np.full(n, np.nan, np.float32)
    if valid.any():
        pl = cnn.predict_proba(c["left_crop"][valid])
        pr = cnn.predict_proba(c["right_crop"][valid])
        p_cnn[valid] = 0.5 * (pl + pr)
    # causal per-driver open-eye EAR level used by the adaptive method
    ada = EARAdaptive(k=1.0, min_threshold=0.0, max_threshold=10.0, fallback_threshold=np.nan)
    base = np.full(n, np.nan, np.float32)
    ear = 0.5 * (l_ear + r_ear)
    for i in np.where(valid)[0]:
        base[i] = ada.current_threshold()
        ada.history.append(float(ear[i]))
    return {"name": name, **c, "valid": valid, "l_ear": l_ear, "r_ear": r_ear, "ear": ear,
            "lw": lw, "rw": rw, "p_cnn": p_cnn, "ear_base": base,
            "p_bs": np.nanmean(c["blink_bs"], axis=1) if valid.any() else np.full(n, np.nan)}


# ------------------------------------------------------------------------ metrics
def frame_metrics(y, pred):
    m = y >= 0
    y, pred = y[m] == 1, pred[m]
    tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
    fn, tn = int(np.sum(~pred & y)), int(np.sum(~pred & ~y))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"closed_frames": tp + fn, "open_frames": tn + fp, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "balanced_acc": 0.5 * (rec + spec), "false_closed_rate": fp / max(fp + tn, 1),
            "false_open_rate": fn / max(fn + tp, 1)}


def match_events(gt, det, tol=BLINK_TOL):
    used, tp = set(), 0
    for s, e in det:
        for j, (gs, ge) in enumerate(gt):
            if j not in used and s - tol <= ge and e + tol >= gs:
                used.add(j)
                tp += 1
                break
    return tp


def blink_metrics(videos_events):
    tp = sum(v["tp"] for v in videos_events)
    n_det = sum(v["n_det"] for v in videos_events)
    n_gt = sum(v["n_gt"] for v in videos_events)
    p, r = tp / max(n_det, 1), tp / max(n_gt, 1)
    return {"gt_blinks": n_gt, "detected_events": n_det, "blink_TP": tp,
            "blink_precision": p, "blink_recall": r, "blink_f1": 2 * p * r / max(p + r, 1e-9)}


# ------------------------------------------------------------- vectorised sweeps
def scores_and_rule(v, method, fusion="mean"):
    """Return (score, closed_if) where closed = score < thr (lt) or > thr (gt)."""
    if method == "ear_fixed":
        if fusion == "width_weighted":
            w = v["lw"] + v["rw"]
            return (v["l_ear"] * v["lw"] + v["r_ear"] * v["rw"]) / np.where(w > 0, w, 1), "lt"
        return v["ear"], "lt"
    if method == "ear_adaptive":
        return v["ear"] / v["ear_base"], "lt"  # closed if EAR < k * open-level
    if method == "cnn":
        return v["p_cnn"], "gt"
    if method == "blendshape":
        return v["p_bs"], "gt"
    raise ValueError(method)


def sweep(videos, method, grid, fusion="mean"):
    rows = []
    y = np.concatenate([v["label"] for v in videos])
    s_all, rule = [], None
    for v in videos:
        s, rule = scores_and_rule(v, method, fusion)
        s_all.append(s)
    s = np.concatenate(s_all)
    ok = ~np.isnan(s)
    for thr in grid:
        if method == "ear_adaptive":  # same clipping as the module
            base = np.concatenate([v["ear_base"] for v in videos])
            pred = np.concatenate([v["ear"] for v in videos]) < np.clip(thr * base, 0.10, 0.35)
            pred &= ~np.isnan(base)
        else:
            pred = (s < thr) if rule == "lt" else (s > thr)
        pred &= ok
        rows.append({"method": method, "fusion": fusion, "threshold": round(float(thr), 4),
                     **frame_metrics(np.where(ok, y, -1), pred)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- module replays
def run_module(cfg, v):
    module = EyeStateModule(cfg)
    module.reset()
    ts = v["timestamp_ms"]
    closed = np.zeros(len(ts), bool)
    events, t_total = [], 0.0
    for i in range(len(ts)):
        ok = bool(v["valid"][i])
        block = {"frame_id": i, "timestamp_ms": float(ts[i]), "face_valid": ok,
                 "left_eye_pts": v["left_pts"][i].tolist() if ok else None,
                 "right_eye_pts": v["right_pts"][i].tolist() if ok else None,
                 "blendshapes": {"eyeBlinkLeft": float(v["blink_bs"][i, 0]),
                                 "eyeBlinkRight": float(v["blink_bs"][i, 1])} if ok else None}
        t0 = time.perf_counter()
        rec = module.process(block, crops=(v["left_crop"][i], v["right_crop"][i]))
        t_total += time.perf_counter() - t0
        closed[i] = rec["eye_state"] == "closed"
        if rec["blink_event"]:
            events.append((rec["blink_event"]["start_frame"], rec["blink_event"]["end_frame"]))
    gt = blink_intervals(v["blink_id"])
    return closed, {"tp": match_events(gt, events), "n_det": len(events), "n_gt": len(gt)}, \
        1000 * t_total / len(ts)


def evaluate_config(cfg, videos, frame_mask=None):
    ys, preds, ev, ms = [], [], [], []
    for v in videos:
        closed, e, t = run_module(cfg, v)
        y = v["label"].copy()
        if frame_mask is not None:
            y[~frame_mask(v)] = -1
        ys.append(np.where(v["valid"], y, -1))
        preds.append(closed)
        ev.append(e)
        ms.append(t)
    return {**frame_metrics(np.concatenate(ys), np.concatenate(preds)), **blink_metrics(ev),
            "sg2_ms_per_frame": float(np.mean(ms))}


def make_cfg(name, method, **params):
    clf = {"method": method, **params}
    if method == "cnn":
        clf["model_path"] = "models/eye_state_cnn.onnx"
    return {"name": name, "classifier": clf, "blink": dict(BLINK_CFG), "crop": {"size": 32, "scale": 1.6}}


def save_cfg(cfg, filename, header):
    disk = json.loads(json.dumps(cfg))
    (CFG_DIR / filename).write_text(header + yaml.safe_dump(disk, sort_keys=False))


def pick(df):
    return df.loc[df.f1.idxmax()]


# --------------------------------------------------------------------------- main
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cnn = CNNEyeState(CNN_PATH)
    dev = [load_video(n, cnn) for n in DEV]
    test = [load_video(n, cnn) for n in TEST]
    test_dark = [load_video(n + "_lowlight", cnn) for n in TEST]

    # ---------------- dataset summary + per-driver EAR (motivates adaptive threshold)
    rows = []
    for v in dev + test + test_dark:
        lab = v["label"]
        rows.append({"video": v["name"], "split": "DEV" if v["name"] in DEV else "TEST",
                     "glasses": str(v["glasses"]), "frames": len(lab),
                     "face_valid_rate": round(float(v["valid"].mean()), 4),
                     "open_frames": int((lab == 0).sum()), "closed_frames": int((lab == 1).sum()),
                     "gt_blinks": len(blink_intervals(v["blink_id"])),
                     "median_open_EAR": round(float(np.nanmedian(v["ear"][(lab == 0) & v["valid"]])), 3),
                     "median_closed_EAR": round(float(np.nanmedian(v["ear"][(lab == 1) & v["valid"]])), 3),
                     "median_face_brightness": round(float(np.nanmedian(v["brightness"])), 1),
                     "median_abs_yaw_deg": round(float(np.nanmedian(np.abs(v["pose"][:, 0]))), 1)})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- DEV sweeps (tuning)
    prob_grid = np.concatenate([np.arange(0.05, 0.95, 0.025), np.arange(0.95, 0.9951, 0.005)])
    grids = {"ear_fixed": np.arange(0.08, 0.305, 0.005), "ear_adaptive": np.arange(0.20, 0.951, 0.025),
             "cnn": prob_grid, "blendshape": prob_grid}
    sweeps = {m: sweep(dev, m, g) for m, g in grids.items()}
    sweeps["ear_fixed_width"] = sweep(dev, "ear_fixed", grids["ear_fixed"], "width_weighted")
    test_sweeps = {m: sweep(test, m, g) for m, g in grids.items()}
    all_sweeps = pd.concat([d.assign(split="DEV") for d in sweeps.values()] +
                           [d.assign(split="TEST") for d in test_sweeps.values()])
    all_sweeps.round(4).to_csv(RESULTS_DIR / "threshold_sweep.csv", index=False)
    best = {m: pick(d) for m, d in sweeps.items()}
    for m, b in best.items():
        print(f"DEV best {m}: thr={b.threshold} F1={b.f1:.3f}")

    # ---------------- configurations A-E (thresholds frozen from DEV)
    configs = {
        "A": make_cfg("A_baseline_ear_fixed_0.20", "ear_fixed", threshold=0.20, fusion="mean"),
        "B": make_cfg(f"B_ear_fixed_tuned_{best['ear_fixed'].threshold:.3f}", "ear_fixed",
                      threshold=float(best["ear_fixed"].threshold), fusion="mean"),
        "C": make_cfg(f"C_ear_adaptive_k{best['ear_adaptive'].threshold:.3f}", "ear_adaptive",
                      k=float(best["ear_adaptive"].threshold), window_frames=900, percentile=80,
                      warmup_frames=30, fallback_threshold=float(best["ear_fixed"].threshold),
                      min_threshold=0.10, max_threshold=0.35, fusion="mean"),
        "D": make_cfg(f"D_cnn_p{best['cnn'].threshold:.3f}", "cnn", threshold=float(best["cnn"].threshold)),
        "E": make_cfg(f"E_blendshape_p{best['blendshape'].threshold:.3f}", "blendshape",
                      threshold=float(best["blendshape"].threshold)),
    }
    notes = {"A": "Week 4 baseline: EAR, literature threshold 0.20",
             "B": "EAR, threshold tuned on DEV",
             "C": "EAR, per-driver adaptive threshold k x open-eye EAR (80th pct, last 30 s)",
             "D": "Learned CNN on 32x32 eye crop (MRL+CEW trained, ONNX)",
             "E": "MediaPipe eyeBlink blendshape (needs SG1 interface change)"}
    for key, fn in (("B", "w5_B_ear_fixed_tuned.yaml"), ("C", "w5_C_ear_adaptive.yaml"),
                    ("D", "w5_D_cnn.yaml"), ("E", "w5_E_blendshape.yaml")):
        save_cfg(configs[key], fn, f"# Config {key} - {notes[key]}\n# Generated by experiments/evaluate.py "
                                   f"(threshold tuned on DEV videos {DEV})\n")

    results, per_video, hard = [], [], []
    for key, cfg in configs.items():
        r = evaluate_config(cfg, test)
        rd = evaluate_config(cfg, test_dark)
        results.append({"config": key, "name": cfg["name"], "description": notes[key], "test_set": "TEST",
                        **r})
        results.append({"config": key, "name": cfg["name"], "description": notes[key],
                        "test_set": "TEST low-light (synthetic)", **rd})
        for v in test:
            per_video.append({"config": key, "video": v["name"], "glasses": str(v["glasses"]),
                              **evaluate_config(cfg, [v])})
        # difficult cases on TEST
        conds = {
            "glasses (eb8_11)": ([v for v in test if str(v["glasses"]) == "YES"], None),
            "no glasses": ([v for v in test if str(v["glasses"]) != "YES"], None),
            "|yaw| < 10 deg": (test, lambda v: np.abs(v["pose"][:, 0]) < 10),
            "|yaw| 10-20 deg": (test, lambda v: (np.abs(v["pose"][:, 0]) >= 10) & (np.abs(v["pose"][:, 0]) < 20)),
            "|yaw| >= 20 deg": (test, lambda v: np.abs(v["pose"][:, 0]) >= 20),
            "|pitch| >= 15 deg": (test, lambda v: np.abs(v["pose"][:, 1]) >= 15),
            "normal light": (test, None),
            "low light (synthetic)": (test_dark, None),
        }
        for cname, (vids, mask) in conds.items():
            hard.append({"config": key, "condition": cname, **evaluate_config(cfg, vids, mask)})
        print(f"done config {key}", flush=True)

    res = pd.DataFrame(results)
    cols = ["config", "name", "test_set", "closed_frames", "open_frames", "precision", "recall", "f1",
            "balanced_acc", "false_closed_rate", "false_open_rate", "FP", "FN", "gt_blinks", "detected_events",
            "blink_precision", "blink_recall", "blink_f1", "sg2_ms_per_frame", "description"]
    res[cols].round(4).to_csv(RESULTS_DIR / "week5_comparison.csv", index=False)
    pd.DataFrame(per_video).round(4).to_csv(RESULTS_DIR / "per_video_results.csv", index=False)
    hard_df = pd.DataFrame(hard)
    hard_df.round(4).to_csv(RESULTS_DIR / "difficult_cases.csv", index=False)
    print(res[cols[:-1]].round(3).to_string(index=False))

    # fusion study (pose-aware eye weighting) on TEST
    fus = []
    for fusion in ("mean", "width_weighted"):
        cfg = make_cfg(f"B_{fusion}", "ear_fixed", threshold=float(best["ear_fixed"].threshold), fusion=fusion)
        fus.append({"fusion": fusion, "threshold": float(best["ear_fixed"].threshold),
                    "all TEST": evaluate_config(cfg, test)["f1"],
                    "|yaw| >= 20 deg": evaluate_config(cfg, test, conds["|yaw| >= 20 deg"][1])["f1"]})
    pd.DataFrame(fus).round(4).to_csv(RESULTS_DIR / "fusion_study.csv", index=False)

    json.dump({k: v for k, v in configs.items()}, open(RESULTS_DIR / "week5_configs_used.json", "w"), indent=2)
    plots(sweeps, test_sweeps, best, test, configs, hard_df)
    failure_montage(test, configs)


# --------------------------------------------------------------------------- plots
def plots(sweeps, test_sweeps, best, test, configs, hard_df):
    # 1. EAR threshold study
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for d, split, ls in ((sweeps["ear_fixed"], "DEV", "-"), (test_sweeps["ear_fixed"], "TEST", "--")):
        ax1.plot(d.threshold, d.f1, ls, color="C0", label=f"F1 closed ({split})")
        ax2.plot(d.threshold, 100 * d.false_closed_rate, ls, color="C3", label=f"false-closed % ({split})")
        ax2.plot(d.threshold, 100 * d.false_open_rate, ls, color="C2", label=f"false-open % ({split})")
    for ax in (ax1, ax2):
        ax.axvline(0.20, color="grey", ls=":", label="A baseline 0.20")
        ax.axvline(best["ear_fixed"].threshold, color="k", ls=":", label=f"B tuned {best['ear_fixed'].threshold:.3f}")
        ax.set_xlabel("EAR threshold (closed if EAR < threshold)")
        ax.legend(fontsize=7)
    ax1.set_ylabel("F1 of closed frames")
    ax2.set_ylabel("error rate (%)")
    ax2.set_yscale("symlog", linthresh=1)
    fig.suptitle("SG2 Week 5 - fixed EAR threshold study (tuned on DEV, checked on TEST)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ear_threshold_sweep.png", dpi=130)
    plt.close(fig)

    # 2. precision-recall curves on TEST
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    names = {"ear_fixed": "EAR fixed (A/B)", "ear_adaptive": "EAR adaptive (C)", "cnn": "CNN (D)",
             "blendshape": "Blendshape (E)"}
    for i, (m, d) in enumerate(test_sweeps.items()):
        ax.plot(d.recall, d.precision, ".-", ms=3, color=f"C{i}", label=names[m])
    ax.set_xlabel("recall (closed frames found)")
    ax.set_ylabel("precision (closed calls correct)")
    ax.set_title("TEST precision-recall, threshold sweep")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "precision_recall_test.png", dpi=130)
    plt.close(fig)

    # 3. blink evidence: EAR and CNN signal over time with annotated blinks
    v = next(x for x in test if x["name"] == "eb8_04")
    s, e = 0, 600
    t = np.arange(s, e)
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    for gs, ge in blink_intervals(v["blink_id"]):
        if ge >= s and gs < e:
            for a in ax:
                a.axvspan(gs, ge, color="orange", alpha=0.25)
    closed = np.where(v["label"][s:e] == 1)[0] + s
    ax[0].plot(t, v["ear"][s:e], lw=1, label="EAR (mean of eyes)")
    ax[0].plot(t, np.clip(configs["C"]["classifier"]["k"] * v["ear_base"][s:e], 0.10, 0.35), "g--", lw=1,
               label="C adaptive threshold")
    ax[0].axhline(configs["B"]["classifier"]["threshold"], color="k", ls=":", lw=1, label="B fixed threshold")
    ax[0].scatter(closed, np.full(len(closed), 0.05), marker="|", color="r", label="GT fully closed")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_ylabel("EAR")
    ax[1].plot(t, v["p_cnn"][s:e], lw=1, color="C1", label="CNN P(closed)")
    ax[1].axhline(configs["D"]["classifier"]["threshold"], color="k", ls=":", lw=1)
    ax[1].set_ylabel("P(closed)")
    ax[1].set_xlabel("frame (eb8_04, 30 fps); orange = annotated blink")
    ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "blink_timeline_eb8_04.png", dpi=130)
    plt.close(fig)

    # 4. difficult cases
    piv = hard_df.pivot(index="condition", columns="config", values="f1")
    piv = piv.loc[[c for c in hard_df.condition.unique()]]
    ax = piv.plot.barh(figsize=(8, 4.5), width=0.8)
    ax.set_xlabel("F1 (closed frames), TEST")
    ax.set_title("SG2 difficult cases by configuration")
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "difficult_cases_f1.png", dpi=130)
    plt.close()


def failure_montage(test, configs):
    """Representative false-open / false-closed crops for configs B and D."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(4, 8, figsize=(9, 5))
    rows = []
    for key in ("B", "D"):
        thr = configs[key]["classifier"]["threshold"]
        for kind in ("false_open", "false_closed"):
            pool = []
            for v in test:
                s = v["ear"] if key == "B" else v["p_cnn"]
                closed = (s < thr) if key == "B" else (s > thr)
                want = (v["label"] == 1) & ~closed if kind == "false_open" else (v["label"] == 0) & closed
                pool += [(v, i) for i in np.where(want & v["valid"])[0]]
            pick = rng.choice(len(pool), size=min(8, len(pool)), replace=False) if pool else []
            rows.append((f"{key} {kind.replace('_', '-')}", [pool[i] for i in pick], len(pool)))
    for r, (title, items, total) in enumerate(rows):
        for c in range(8):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(items):
                v, i = items[c]
                ax.imshow(np.hstack([v["right_crop"][i], v["left_crop"][i]]), cmap="gray")
                ax.set_title(f"{v['name']}#{i}\nEAR {v['ear'][i]:.2f} p {v['p_cnn'][i]:.2f}", fontsize=5)
        axes[r, 0].text(-0.3, 0.5, f"{title}\n(n={total})", transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=7)
    fig.suptitle("SG2 TEST failure cases (eye crops: driver right | left)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "failure_cases.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
