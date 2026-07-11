"""
selfcheck.py — one-command environment + physics verification.

    python scripts/selfcheck.py            # full check (~90 s)
    python scripts/selfcheck.py --fast     # skip the physics episode (~30 s)

Runs three stages and prints a PASS/FAIL summary:
  1. Environment  — Python / MuJoCo / NumPy versions, asset paths
  2. Unit tests   — the full pytest suite
  3. Physics      — one --quick episode end-to-end (cut + serve)
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).parent.parent


def _stage(name: str, fn) -> tuple[str, bool, str, float]:
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:                                   # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    return name, ok, detail, time.time() - t0


def check_environment() -> tuple[bool, str]:
    import numpy
    import mujoco
    py = ".".join(map(str, sys.version_info[:3]))
    missing = [p for p in ("assets/scene_robot.xml",
                           "assets/futurist_unlocked.urdf",
                           "models/bc_policy.npz")
               if not (_ROOT / p).exists()]
    if missing:
        return False, f"missing assets: {missing}"
    return True, f"Python {py} · MuJoCo {mujoco.__version__} · NumPy {numpy.__version__}"


def check_tests() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"],
                       cwd=_ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["no output"])[-1]
    return r.returncode == 0, tail


def check_physics() -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, "scripts/record_robot_video.py", "--quick", "--n-episodes", "1"],
        cwd=_ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    ok = r.returncode == 0 and "1/1 episodes successful" in out
    for line in out.splitlines():
        if "Avg cut time" in line:
            return ok, line.strip()
    return ok, "episode did not report a summary" if not ok else "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the physics episode (env + unit tests only)")
    args = ap.parse_args()

    stages = [("environment", check_environment), ("unit tests", check_tests)]
    if not args.fast:
        stages.append(("physics episode", check_physics))

    print("FF Master + Futurist — self check")
    print("─" * 60)
    results = [_stage(n, f) for n, f in stages]
    all_ok = True
    for name, ok, detail, dt in results:
        mark = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"  [{mark}]  {name:<16s} {detail}   ({dt:.1f}s)")
    print("─" * 60)
    print("ALL CHECKS PASSED" if all_ok else "SELF CHECK FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
