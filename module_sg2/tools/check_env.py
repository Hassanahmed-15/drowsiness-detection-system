"""Platform / development-stack readiness check (Jetson Orin Nano or development PC).

Identifies the board, OS, JetPack/L4T, storage and power mode, checks Python,
Git, OpenCV (GStreamer / CUDA / V4L support), CUDA, PyTorch, onnxruntime,
MediaPipe, lists cameras, and runs an SG2 smoke test on the committed real
sample. Nothing is installed or changed.

Run from module_sg2/:
    python tools/check_env.py                       # prints a Markdown report
    python tools/check_env.py --out ../embedded_sg6/reports/env_$(hostname).md
"""
import argparse
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
REPO_DIR = MODULE_DIR.parent


def sh(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"(failed: {exc})"


def read(path):
    try:
        return Path(path).read_text(errors="ignore").replace("\x00", "").strip()
    except OSError:
        return None


def ok(flag):
    return "✅" if flag else "❌"


def platform_section():
    model = read("/proc/device-tree/model")
    is_jetson = bool(model and "jetson" in model.lower()) or Path("/etc/nv_tegra_release").exists()
    rows = [("Hostname", platform.node()), ("OS", f"{platform.system()} {platform.release()}"),
            ("Machine", platform.machine())]
    if platform.system() == "Linux":
        rows.append(("Distribution", sh("lsb_release -ds 2>/dev/null || grep PRETTY_NAME /etc/os-release")))
    if is_jetson:
        rows += [("Board", model or "Not Verified"),
                 ("L4T release", read("/etc/nv_tegra_release") or "Not Verified"),
                 ("JetPack package", sh("dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null") or "not installed"),
                 ("Power mode", sh("nvpmodel -q 2>/dev/null | head -2 | tr '\\n' ' '") or "Not Verified (needs sudo?)")]
    elif platform.system() == "Darwin":
        rows.append(("Board", sh("sysctl -n machdep.cpu.brand_string") + " (development PC, NOT a Jetson)"))
    else:
        rows.append(("Board", "development PC, NOT a Jetson"))
    return is_jetson, rows


def storage_section():
    rows = []
    total, used, free = shutil.disk_usage("/")
    rows.append(("Root filesystem", f"{total / 1e9:.0f} GB total, {free / 1e9:.0f} GB free"))
    if platform.system() == "Linux":
        src = sh("findmnt -n -o SOURCE /")
        kind = "NVMe SSD" if "nvme" in src else ("microSD / eMMC" if "mmcblk" in src else "Not Verified")
        rows.append(("Root device", f"{src} ({kind})"))
        rows.append(("Block devices", sh("lsblk -d -o NAME,SIZE,MODEL 2>/dev/null | tr '\\n' ';'")))
        mem = read("/proc/meminfo") or ""
        m = dict(re.findall(r"^(\w+):\s+(\d+)", mem, re.M))
        if m:
            rows.append(("Memory", f"{int(m['MemTotal']) / 1e6:.1f} GB RAM, swap {int(m.get('SwapTotal', 0)) / 1e6:.1f} GB"))
    elif platform.system() == "Darwin":
        rows.append(("Memory", f"{int(sh('sysctl -n hw.memsize')) / 1e9:.0f} GB RAM"))
    return rows


def pkg(name, attr="__version__"):
    try:
        mod = importlib.import_module(name)
        return mod, getattr(mod, attr, "?")
    except Exception as exc:  # noqa: BLE001
        return None, f"MISSING ({exc.__class__.__name__})"


def stack_section():
    rows, flags = [], {}
    rows.append(("Python", f"{sys.version.split()[0]} ({sys.executable})"))
    rows.append(("Git", sh("git --version")))
    cv2, ver = pkg("cv2")
    info = cv2.getBuildInformation() if cv2 else ""

    def build(key):
        m = re.search(rf"^\s*{key}:\s*(.+)$", info, re.M)
        return m.group(1).strip() if m else "NO"
    if cv2:
        flags["gstreamer"] = build("GStreamer").upper().startswith("YES")
        flags["cuda_cv"] = "NVIDIA CUDA:" in info and "YES" in build("NVIDIA CUDA")
        rows.append(("OpenCV", f"{ver} | GStreamer: {build('GStreamer')} | V4L/V4L2: {build('v4l/v4l2')} | "
                               f"CUDA: {build('NVIDIA CUDA')} | GUI: {build('GUI')}"))
    else:
        rows.append(("OpenCV", ver))
    nvcc = sh("nvcc --version 2>/dev/null | tail -2 | tr '\\n' ' '") or read("/usr/local/cuda/version.json")
    rows.append(("CUDA toolkit", nvcc or "not found"))
    torch, tver = pkg("torch")
    if torch:
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else ("Apple MPS" if getattr(torch.backends, "mps", None)
                                                          and torch.backends.mps.is_available() else "CPU only")
        rows.append(("PyTorch", f"{tver} | CUDA available: {cuda} | device: {dev}"))
    else:
        rows.append(("PyTorch", f"{tver} (only needed for training, not for SG2 inference)"))
    ort, over = pkg("onnxruntime")
    rows.append(("onnxruntime", f"{over} | providers: {ort.get_available_providers()}" if ort else
                 f"{over} - primary config G falls back to backup C without it"))
    for name in ("numpy", "yaml", "mediapipe", "jsonschema"):
        rows.append((name, pkg(name)[1]))
    return rows, flags


def camera_section():
    rows = []
    if platform.system() == "Linux":
        devs = sorted(str(p) for p in Path("/dev").glob("video*"))
        rows.append(("/dev/video*", ", ".join(devs) or "none found"))
        rows.append(("v4l2-ctl --list-devices", sh("v4l2-ctl --list-devices 2>/dev/null | tr '\\n' ';'") or
                     "v4l-utils not installed (sudo apt install v4l-utils)"))
        rows.append(("CSI (nvargus-daemon)", sh("systemctl is-active nvargus-daemon 2>/dev/null") or "Not Verified"))
    else:
        rows.append(("Cameras", "use tools/camera_check.py --source 0 (macOS needs Camera permission for the terminal)"))
    return rows


def sg2_smoke():
    sys.path.insert(0, str(MODULE_DIR))
    try:
        from sg2_eye import EyeStateModule

        sample = REPO_DIR / "interfaces" / "sg2" / "examples" / "eb8_09_real_sg1.jsonl"
        blocks = [json.loads(line) for line in open(sample)]
        rows = []
        for cfg in ("primary", "backup"):
            m = EyeStateModule(MODULE_DIR / "configs" / f"{cfg}.yaml")
            t0 = time.perf_counter()
            for b in blocks:
                m.process(b)
            ms = 1000 * (time.perf_counter() - t0) / len(blocks)
            cnn = getattr(m.classifier, "cnn_available", None)
            rows.append((f"SG2 {cfg} on real sample", f"blinks {m.blink.blink_count}/8 annotated, {ms:.3f} ms/frame"
                         + ("" if cnn is None else f", CNN model loaded: {cnn}")))
        try:
            import numpy as np

            from sg2_eye.classifiers import CNNEyeState
            cnn = CNNEyeState(MODULE_DIR / "models" / "eye_state_cnn_v2.onnx")
            x = np.random.default_rng(0).integers(0, 255, (2, 32, 32), dtype=np.uint8)
            cnn.predict_proba(x)
            t0 = time.perf_counter()
            for _ in range(200):
                cnn.predict_proba(x)
            rows.append(("CNN v2 inference (2 eyes, CPU)", f"{1000 * (time.perf_counter() - t0) / 200:.3f} ms"))
        except Exception as exc:  # noqa: BLE001
            rows.append(("CNN v2 inference", f"not available: {exc.__class__.__name__} (run models/download_models.sh)"))
        return rows
    except Exception as exc:  # noqa: BLE001
        return [("SG2 smoke test", f"FAILED: {exc!r}")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="also write the Markdown report to this file")
    args = ap.parse_args()
    is_jetson, plat = platform_section()
    stack, flags = stack_section()
    lines = [f"# Environment report — {platform.node()}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `module_sg2/tools/check_env.py`. "
             f"{'Jetson detected.' if is_jetson else '**Not a Jetson** (development PC).'}", ""]
    for title, rows in (("Platform", plat), ("Storage / memory", storage_section()), ("Development stack", stack),
                        ("Cameras", camera_section()), ("SG2 smoke test", sg2_smoke())):
        lines += [f"## {title}", "", "| Item | Value |", "|---|---|"]
        lines += [f"| {k} | {str(v).replace('|', '/')} |" for k, v in rows]
        lines.append("")
    checks = [("Python 3.10+", sys.version_info >= (3, 10)), ("Git", "git version" in sh("git --version")),
              ("OpenCV importable", pkg("cv2")[0] is not None),
              ("OpenCV GStreamer (needed for CSI camera)", flags.get("gstreamer", False)),
              ("onnxruntime (primary config G)", pkg("onnxruntime")[0] is not None),
              ("MediaPipe (landmarks on device)", pkg("mediapipe")[0] is not None)]
    if is_jetson:
        torch = pkg("torch")[0]
        checks.append(("PyTorch with CUDA (only if a module needs it)", bool(torch and torch.cuda.is_available())))
    lines += ["## Readiness checklist", ""] + [f"- {ok(v)} {k}" for k, v in checks] + [""]
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report)
        print(f"\n[check_env] written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    os.environ.setdefault("GLOG_minloglevel", "2")
    main()
