"""
stress_envelope.py — probe the success envelope over watermelon placement.

Runs one physics-only episode per (dx, dy) grid point and reports the
success/failure map. Answers "how far can the melon move before the
pipeline breaks?" — cutting AND serving must both succeed.

Run:  python scripts/stress_envelope.py            (~10 min)
      python scripts/stress_envelope.py --fast     (coarser grid)
"""
import argparse, pathlib, re, subprocess, sys

_ROOT   = pathlib.Path(__file__).parent.parent
_RECORD = _ROOT / "scripts" / "record_robot_video.py"


def probe(dx: float, dy: float) -> dict:
    r = subprocess.run(
        [sys.executable, str(_RECORD), "--quick", "--n-episodes", "1",
         "--wm-dx", f"{dx:.3f}", "--wm-dy", f"{dy:.3f}"],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=600)
    m = re.search(r"Ep 1: (OK|FAIL).*?(?:cut=([\d.]+)s)?.*?serve_err=(\S+)",
                  r.stdout)
    ok    = bool(m and m.group(1) == "OK")
    cut_t = float(m.group(2)) if (m and m.group(2)) else None
    serve = m.group(3) if m else "?"
    serve_ok = ok and serve.endswith("mm")
    return {"ok": ok, "cut_t": cut_t, "serve_ok": serve_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    dxs = [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04]
    dys = [0.0, 0.03] if not args.fast else [0.0]

    print(f"Probing {len(dxs) * len(dys)} placements "
          f"(dx ∈ [{dxs[0]},{dxs[-1]}] m, dy ∈ {dys}) …\n")
    rows = []
    for dy in dys:
        for dx in dxs:
            r = probe(dx, dy)
            tag = ("CUT+SERVE" if r["serve_ok"]
                   else "CUT only" if r["ok"] else "FAIL")
            ct = f"{r['cut_t']:.2f}s" if r["cut_t"] else "  —  "
            print(f"  dx={dx*100:+5.1f}cm dy={dy*100:+4.1f}cm  cut={ct}  {tag}",
                  flush=True)
            rows.append((dx, dy, tag, r["cut_t"]))

    print("\n── Success envelope ──")
    for dy in dys:
        oks = [dx for dx, d2, tag, _ in rows if d2 == dy and tag == "CUT+SERVE"]
        if oks:
            print(f"  dy={dy*100:+.1f}cm : full success for "
                  f"dx ∈ [{min(oks)*100:+.1f}, {max(oks)*100:+.1f}] cm")


if __name__ == "__main__":
    main()
