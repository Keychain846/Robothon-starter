"""
train_policy.py
---------------
Train the behaviour-cloning policy for FF Master's cutting arm on the
`--collect` demonstration CSVs, and save it as a small NumPy checkpoint that
`record_robot_video.py --policy` loads at runtime.

Pipeline (all NumPy, no torch / SB3 / gym):
    output/demo_ep*.csv  →  (observation, action) pairs  →  MLP  →  models/bc_policy.npz

Run:
    # first collect demos (fast, physics-only), then train:
    python scripts/record_robot_video.py --quick --collect --n-episodes 8
    python scripts/train_policy.py
"""
import sys, csv, pathlib, argparse

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np

from src.bc_policy import BCPolicy, featurize, PHASES

_ROOT   = pathlib.Path(__file__).parent.parent
_OUT    = _ROOT / "output"
_MODELS = _ROOT / "models"
_MODELS.mkdir(exist_ok=True)
_CKPT   = _MODELS / "bc_policy.npz"


def _rows_to_pairs(rows):
    X, Y = [], []
    for r in rows:
        ph = r["phase"]
        if ph not in PHASES:      # skip WAIT / any non-cut Master phase
            continue
        try:
            obs = featurize(
                float(r["bx"]), float(r["by"]), float(r["bz"]),
                float(r["wx"]), float(r["wy"]), float(r["wz"]),
                float(r["blade_dist"]), float(r["blade_speed_ms"]),
                float(r["touch_N"]), float(r["material_k_Npm"]), ph)
            act = [float(r["ctrl_sp"]), float(r["ctrl_sr"]),
                   float(r["ctrl_sy"]), float(r["ctrl_el"]), float(r["ctrl_wr"])]
        except (KeyError, ValueError):
            continue
        X.append(obs); Y.append(act)
    return np.asarray(X), np.asarray(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    csvs = sorted(_OUT.glob("demo_ep*.csv"))
    if not csvs:
        sys.exit("No demo_ep*.csv found — run "
                 "`record_robot_video.py --quick --collect --n-episodes 8` first.")
    rows = []
    for p in csvs:
        rows.extend(csv.DictReader(open(p)))
    X, Y = _rows_to_pairs(rows)
    print(f"Loaded {len(csvs)} demo file(s) → {len(X)} (obs,act) pairs "
          f"[obs_dim={X.shape[1]}, act_dim={Y.shape[1]}]")

    # hold out the last 10% (temporal) to report generalisation
    n = len(X); k = int(n * 0.9)
    perm = np.random.default_rng(0).permutation(n)
    tr, te = perm[:k], perm[k:]
    policy = BCPolicy.train(X[tr], Y[tr], hidden=args.hidden,
                            epochs=args.epochs, seed=0)

    pred = np.array([policy.predict(x) for x in X[te]])
    rmse = float(np.sqrt(np.mean((pred - Y[te]) ** 2)))
    per_joint = np.sqrt(np.mean((pred - Y[te]) ** 2, axis=0))
    print(f"\nHeld-out RMSE (rad): {rmse:.4f}  per-joint="
          f"[{', '.join(f'{v:.3f}' for v in per_joint)}]")

    policy.save(_CKPT)
    sz = _CKPT.stat().st_size / 1024.0
    print(f"Saved policy → {_CKPT}  ({sz:.1f} kB)")


if __name__ == "__main__":
    main()
