"""
plot_metrics.py — render README figures from collected run data (PIL only,
no matplotlib dependency).

Inputs  : output/demo_ep01.csv  (from --collect)
          output/run_summary.json
Outputs : docs/grip_forces.png   — per-finger grip force during both cuts
          docs/serve_metrics.png — per-episode delivery error + grasp closure

Run:  python scripts/plot_metrics.py
"""
import csv, json, pathlib
from PIL import Image, ImageDraw, ImageFont

_ROOT = pathlib.Path(__file__).parent.parent
_OUT  = _ROOT / "output"
_DOCS = _ROOT / "docs"
_DOCS.mkdir(exist_ok=True)

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
def _font(size):
    for p in _FONTS:
        if pathlib.Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_MD, F_SM, F_XS = _font(22), _font(16), _font(13)

BG, GRID, AXIS = (13, 17, 27), (36, 42, 60), (150, 160, 190)
FINGER_COLORS = {
    "gh_index_N":  (80, 180, 255),
    "gh_middle_N": (100, 220, 140),
    "gh_ring_N":   (255, 180, 60),
    "gh_pinky_N":  (200, 100, 255),
    "gh_thumb_N":  (255, 90, 90),
}


def plot_grip_forces():
    rows = list(csv.DictReader(open(_OUT / "demo_ep01.csv")))
    ts   = [float(r["t"]) for r in rows]
    tmax = ts[-1]
    W, H = 1000, 560
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 20), "FF Master — blade contact force & per-finger grip",
           fill=(230, 235, 250), font=F_MD, anchor="mm")
    step = max(1, len(rows) // 1200)

    def panel(y0, y1, series, ylab, ystep):
        x0, x1 = 70, W - 30
        vmax = max(max(v for _, v in pts) for pts, _, _ in series) * 1.12 or 1.0
        def X(t): return x0 + (x1 - x0) * t / tmax
        def Y(v): return y1 - (y1 - y0) * v / vmax
        for gv in range(0, int(vmax) + 1, ystep):
            d.line([(x0, Y(gv)), (x1, Y(gv))], fill=GRID)
            d.text((x0 - 8, Y(gv)), f"{gv}", fill=AXIS, font=F_XS, anchor="rm")
        for tv in range(0, int(tmax) + 1, 2):
            d.line([(X(tv), y0), (X(tv), y1)], fill=GRID)
        d.text((24, (y0 + y1) // 2), ylab, fill=AXIS, font=F_SM, anchor="mm")
        for phase, col in (("SLICE", (255, 90, 50)), ("SLICE2", (255, 60, 200))):
            tcut = next((float(r["t"]) for r in rows if r["phase"] == phase), None)
            if tcut:
                d.line([(X(tcut), y0), (X(tcut), y1)], fill=col, width=2)
        lx = x0 + 10
        for pts, col, name in series:
            d.line([(X(t), Y(v)) for t, v in pts[::step]], fill=col, width=2)
            d.rectangle([lx, y0 + 6, lx + 13, y0 + 17], fill=col)
            d.text((lx + 18, y0 + 11), name, fill=(210, 215, 235),
                   font=F_XS, anchor="lm")
            lx += 88
        return X

    # top: blade contact force (the cutting story: two work-integral peaks)
    touch = [(float(r["t"]), float(r["touch_N"])) for r in rows]
    X = panel(56, 280, [(touch, (255, 120, 60), "blade N")], "N", 40)
    d.text((X(2.4), 60), "✂ cut 1", fill=(255, 90, 50), font=F_XS)
    d.text((X(9.1), 60), "✂ cut 2", fill=(255, 60, 200), font=F_XS)

    # bottom: per-finger handle grip forces (closed-loop servo hold + REGRASP dip)
    series = []
    for key, col in FINGER_COLORS.items():
        series.append(([(float(r["t"]), float(r[key])) for r in rows],
                       col, key.split("_")[1]))
    panel(316, 520, series, "N", 150)
    d.text((70, 530), "vertical lines: cut 1 / cut 2 · REGRASP grip relaxation visible ~4.5 s",
           fill=AXIS, font=F_XS)

    img.save(_DOCS / "grip_forces.png")
    print(f"  → {_DOCS/'grip_forces.png'}")


def plot_serve_metrics():
    d0 = json.load(open(_OUT / "run_summary.json"))
    errs = d0.get("serve_delivery_err_mm") or []
    clos = d0.get("grasp_closure_fingertips") or []
    n = len(errs)
    if not n:
        print("  (no serve data in run_summary.json — skipped)")
        return
    W, H = 1000, 360
    x0, y0, x1, y1 = 70, 46, W - 30, H - 60
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 20), "Futurist — delivery error & grasp closure per episode",
           fill=(230, 235, 250), font=F_MD, anchor="mm")
    bw = (x1 - x0) / n * 0.55
    emax = max(10.0, max(e or 0 for e in errs) * 1.3)
    for gv in range(0, int(emax) + 1, 5):
        yy = y1 - (y1 - y0) * gv / emax
        d.line([(x0, yy), (x1, yy)], fill=GRID)
        d.text((x0 - 8, yy), f"{gv}", fill=AXIS, font=F_XS, anchor="rm")
    d.text((22, (y0 + y1) // 2), "mm", fill=AXIS, font=F_SM, anchor="mm")
    for i, e in enumerate(errs):
        cx = x0 + (x1 - x0) * (i + 0.5) / n
        if e is not None:
            hh = (y1 - y0) * e / emax
            d.rectangle([cx - bw / 2, y1 - hh, cx + bw / 2, y1], fill=(80, 200, 255))
            d.text((cx, y1 - hh - 12), f"{e:.1f}", fill=(180, 225, 255),
                   font=F_XS, anchor="mm")
        cl = clos[i] if i < len(clos) and clos[i] else None
        lab = f"Ep{i+1}\n{cl[0]}/5 tips" if cl else f"Ep{i+1}"
        d.multiline_text((cx, y1 + 8), lab, fill=AXIS, font=F_XS,
                         anchor="ma", align="center")
    img.save(_DOCS / "serve_metrics.png")
    print(f"  → {_DOCS/'serve_metrics.png'}")


if __name__ == "__main__":
    plot_grip_forces()
    plot_serve_metrics()
