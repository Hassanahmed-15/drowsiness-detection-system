"""Train the SG2 learned eye-state classifier (Config D) and export it to ONNX.

Training data (never the Week 5 test videos):
  * MRL Eye Dataset 2018 - 84,898 infrared eye crops, 37 people, with labels
    for glasses / reflections / lighting. Split BY PERSON:
    6 people held out as MRL-test (difficult-case breakdown), 4 for validation.
  * CEW eye patches (Closed Eyes in the Wild) - 4,846 visible-light crops,
    split by source face 85/15.

Run from module_sg2/:   python experiments/train_cnn.py
Outputs: models/eye_state_cnn.onnx, results/cnn_training.json,
         results/cnn_mrl_difficult_cases.csv
"""
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA_ROOT, MODULE_DIR, RESULTS_DIR  # noqa: E402

SEED = 477
SIZE = 32
MRL_TEST_SUBJECTS = ["s0001", "s0007", "s0013", "s0019", "s0025", "s0031"]
MRL_VAL_SUBJECTS = ["s0004", "s0010", "s0016", "s0028"]
EPOCHS, BATCH, LR = 12, 256, 2e-3


class EyeNet(nn.Module):
    """72,081 parameters, 32x32x1 grey crop -> one logit (closed)."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.features = nn.Sequential(block(1, 16), block(16, 16), nn.MaxPool2d(2),
                                      block(16, 32), block(32, 32), nn.MaxPool2d(2),
                                      block(32, 64), nn.MaxPool2d(2), block(64, 64),
                                      nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x))


def load_mrl():
    rows, imgs = [], []
    for p in sorted((DATA_ROOT / "mrl" / "mrlEyes_2018_01").glob("s*/*.png")):
        subj, _, _, glasses, state, refl, light, sensor = p.stem.split("_")
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        imgs.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
        rows.append({"source": "mrl", "group": subj, "closed": int(state == "0"),
                     "glasses": int(glasses), "reflections": int(refl), "lighting": int(light),
                     "sensor": sensor})
    return pd.DataFrame(rows), np.stack(imgs)


def load_cew():
    rows, imgs = [], []
    for p in sorted((DATA_ROOT / "cew" / "dataset_B_Eye_Images").glob("*/*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        imgs.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
        face = p.name.rsplit("_", 1)[0]  # left/right eye of one face stay together
        rows.append({"source": "cew", "group": face, "closed": int(p.parent.name.startswith("closed"))})
    return pd.DataFrame(rows), np.stack(imgs)


def standardise_t(x):
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True)
    return (x - mean) / (std + 1e-3 / 255.0)


def augment(x):
    """x: (B,1,H,W) in [0,1] on device."""
    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    ang = (torch.rand(b, device=x.device) - 0.5) * np.radians(24)
    scale = 0.85 + 0.35 * torch.rand(b, device=x.device)
    tx, ty = [(torch.rand(b, device=x.device) - 0.5) * 0.2 for _ in range(2)]
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    gamma = torch.exp((torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 1.4)  # 0.5 .. 2.0
    x = x.clamp(1e-4, 1) ** gamma
    blur = torch.rand(b, device=x.device) < 0.3
    x = torch.where(blur[:, None, None, None], F.avg_pool2d(x, 3, 1, 1, count_include_pad=False), x)
    x = x + torch.randn_like(x) * 0.03 * torch.rand(b, 1, 1, 1, device=x.device)
    return x


def predict(model, imgs, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 2048):
            x = torch.from_numpy(imgs[i:i + 2048]).float().div(255).unsqueeze(1).to(device)
            out.append(torch.sigmoid(model(standardise_t(x))).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p, thr=0.5):
    pred = p > thr
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"n": int(len(y)), "accuracy": round((tp + tn) / max(len(y), 1), 4),
            "precision_closed": round(prec, 4), "recall_closed": round(rec, 4),
            "f1_closed": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "false_closed_rate": round(fp / max(fp + tn, 1), 4), "false_open_rate": round(fn / max(fn + tp, 1), 4)}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    mrl_df, mrl_x = load_mrl()
    cew_df, cew_x = load_cew()
    print(f"loaded MRL {len(mrl_df)} + CEW {len(cew_df)} crops in {time.time() - t0:.0f}s", flush=True)

    split = np.where(mrl_df.group.isin(MRL_TEST_SUBJECTS), "test",
                     np.where(mrl_df.group.isin(MRL_VAL_SUBJECTS), "val", "train"))
    mrl_df["split"] = split
    faces = sorted(cew_df.group.unique())
    rng = np.random.default_rng(SEED)
    val_faces = set(rng.choice(faces, size=int(0.15 * len(faces)), replace=False))
    cew_df["split"] = np.where(cew_df.group.isin(val_faces), "val", "train")

    df = pd.concat([mrl_df, cew_df], ignore_index=True)
    x_all = np.concatenate([mrl_x, cew_x])
    y_all = df.closed.to_numpy()
    tr = np.where(df.split == "train")[0]
    va = np.where(df.split == "val")[0]
    te = np.where(df.split == "test")[0]
    # oversample CEW (visible light, in the wild) so it is not drowned by MRL infrared
    cew_tr = tr[df.source.to_numpy()[tr] == "cew"]
    tr_epoch = np.concatenate([tr, np.repeat(cew_tr, 4)])

    x_gpu = torch.from_numpy(x_all).to(device)
    y_gpu = torch.from_numpy(y_all).float().to(device)
    model = EyeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(tr_epoch) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    best, best_state, log = -1, None, []
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.from_numpy(np.random.permutation(tr_epoch)).to(device)
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            x = augment(x_gpu[idx].float().div(255).unsqueeze(1))
            loss = F.binary_cross_entropy_with_logits(model(standardise_t(x)).squeeze(1), y_gpu[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        p_val = predict(model, x_all[va], device)
        m_val = metrics(y_all[va], p_val)
        log.append({"epoch": epoch + 1, "train_loss": round(total / len(perm), 4), **{f"val_{k}": v for k, v in m_val.items()}})
        print(log[-1], flush=True)
        if m_val["f1_closed"] > best:
            best = m_val["f1_closed"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # held-out MRL people: overall + difficult-case breakdown
    p_te = predict(model, x_all[te], device)
    te_df = df.iloc[te].copy()
    te_df["p"] = p_te
    rows = [{"subset": "all MRL test people", **metrics(te_df.closed.to_numpy(), p_te)}]
    for col, names in (("glasses", {0: "no glasses", 1: "glasses"}),
                       ("reflections", {0: "no reflections", 1: "small reflections", 2: "big reflections"}),
                       ("lighting", {0: "bad lighting", 1: "good lighting"})):
        for v, label in names.items():
            sub = te_df[te_df[col] == v]
            if len(sub):
                rows.append({"subset": label, **metrics(sub.closed.to_numpy(), sub.p.to_numpy())})
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cnn_mrl_difficult_cases.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    model = model.cpu().eval()
    out = MODULE_DIR / "models" / "eye_state_cnn.onnx"
    torch.onnx.export(model, torch.zeros(2, 1, SIZE, SIZE), str(out), input_names=["eye"],
                      output_names=["closed_logit"], dynamic_axes={"eye": {0: "n"}, "closed_logit": {0: "n"}},
                      opset_version=17, dynamo=False)
    info = {"date": time.strftime("%Y-%m-%d"), "device": device, "seed": SEED, "epochs": EPOCHS,
            "params": int(sum(p.numel() for p in model.parameters())),
            "train_crops": int(len(tr)), "val_crops": int(len(va)), "mrl_test_crops": int(len(te)),
            "mrl_test_subjects": MRL_TEST_SUBJECTS, "mrl_val_subjects": MRL_VAL_SUBJECTS,
            "best_val_f1_closed": best, "log": log, "mrl_test": rows,
            "onnx": str(out.relative_to(MODULE_DIR)), "onnx_bytes": out.stat().st_size,
            "train_seconds": round(time.time() - t0)}
    (RESULTS_DIR / "cnn_training.json").write_text(json.dumps(info, indent=2))
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB), {info['params']} params, {info['train_seconds']}s")


if __name__ == "__main__":
    main()
