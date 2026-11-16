"""Shared GT/event helpers (copied from dbse_blinkmodel_test.py)."""
def read_gt(p):
    b, started = {}, False
    for line in open(p):
        line = line.strip()
        if line == "#start":
            started = True
            continue
        if not started or not line or line.startswith("#"):
            continue
        fr, bid = map(int, line.split(":")[:2])
        if bid != -1:
            b.setdefault(bid, []).append(fr)
    return [(min(v), max(v)) for v in b.values()]


def events(flags):
    ev, s = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            ev.append((s, i - 1))
            s = None
    return ev


def match(gt, pred, tol=2):
    ov = lambda a, b: a[0] - tol <= b[1] and b[0] - tol <= a[1]  # noqa: E731
    found = sum(any(ov(g, p) for p in pred) for g in gt)
    fp = sum(not any(ov(p, g) for g in gt) for p in pred)
    return dict(gt=len(gt), pred=len(pred), gt_found=found, false_events=fp, missed=len(gt) - found,
                precision=round((len(pred) - fp) / len(pred), 3) if pred else 0.0,
                recall=round(found / len(gt), 3))


