"""
bc_policy.py
------------
A tiny **behaviour-cloning** policy for FF Master's cutting arm, trained on the
`--collect` demonstration CSVs and run as an optional `--policy` mode.

Design goals
------------
* **No new runtime dependencies.** The network is a 2-hidden-layer MLP
  implemented in pure NumPy; weights ship as a small ``.npz`` and inference is a
  couple of matrix multiplies. (Training also uses only NumPy — a minimal Adam
  loop — so the whole thing runs without torch / SB3 / gym.)
* **Additive, not a replacement.** The policy predicts the 5 arm-joint control
  targets from the current observation; the autonomous FSM still owns phase
  transitions, the physics-grounded work-integral cut trigger, the second cut
  and the Futurist. So this is a *learned low-level controller under a scripted
  high-level plan* — and if the policy drifts, the cut still fires only when the
  blade has delivered enough cutting work.

Observation (featurised from a collected row or live sim state):
    [ bx-wx, by-wy, bz-wz,               # blade → watermelon offset (m)
      blade_dist, blade_speed_ms,        # scalar kinematics
      touch_N/200, material_k/20000 ]    # normalised force / stiffness
    ++ one-hot(phase)                     # which of the Master phases
Action:
    [ ctrl_sp, ctrl_sr, ctrl_sy, ctrl_el, ctrl_wr ]   # 5 arm joint targets
"""
from __future__ import annotations

import numpy as np

# Master arm-control phases seen in the cut pipeline (fixed order → stable
# one-hot encoding shared by trainer and runtime).
PHASES = [
    "APPROACH", "ALIGN", "CONTACT", "SLICE", "RETRACT", "DONE",
    "REGRASP", "REPOSITION2", "APPROACH2", "ALIGN2", "CONTACT2",
    "SLICE2", "RETRACT2", "DONE2",
]
_PHASE_IDX = {p: i for i, p in enumerate(PHASES)}

CONT_DIM = 7                       # continuous features
OBS_DIM  = CONT_DIM + len(PHASES)  # + one-hot phase
ACT_DIM  = 5


def featurize(bx, by, bz, wx, wy, wz, blade_dist, blade_speed_ms,
              touch_N, material_k, phase) -> np.ndarray:
    """Build the observation vector from raw scalars + phase name."""
    obs = np.zeros(OBS_DIM, dtype=np.float64)
    obs[0] = bx - wx
    obs[1] = by - wy
    obs[2] = bz - wz
    obs[3] = blade_dist
    obs[4] = blade_speed_ms
    obs[5] = touch_N / 200.0
    obs[6] = material_k / 20000.0
    idx = _PHASE_IDX.get(str(phase), None)
    if idx is not None:
        obs[CONT_DIM + idx] = 1.0
    return obs


class BCPolicy:
    """2-hidden-layer tanh MLP with input/output standardisation, in NumPy."""

    def __init__(self, W=None):
        self._W = W   # dict of parameters, or None until trained/loaded

    # ── inference ─────────────────────────────────────────────────────
    def predict(self, obs: np.ndarray) -> np.ndarray:
        W = self._W
        x = (obs - W["x_mu"]) / W["x_sd"]
        h1 = np.tanh(x @ W["W1"] + W["b1"])
        h2 = np.tanh(h1 @ W["W2"] + W["b2"])
        y = h2 @ W["W3"] + W["b3"]
        return y * W["y_sd"] + W["y_mu"]

    # ── persistence ───────────────────────────────────────────────────
    def save(self, path):
        np.savez(path, **self._W)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls({k: d[k] for k in d.files})

    # ── training (pure-NumPy Adam) ────────────────────────────────────
    @classmethod
    def train(cls, X: np.ndarray, Y: np.ndarray, hidden=64, epochs=400,
              lr=2e-3, batch=256, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        x_mu, x_sd = X.mean(0), X.std(0) + 1e-6
        y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-6
        Xn = (X - x_mu) / x_sd
        Yn = (Y - y_mu) / y_sd
        n, di = Xn.shape
        do = Yn.shape[1]

        def he(a, b):
            return rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
        P = {"W1": he(di, hidden), "b1": np.zeros(hidden),
             "W2": he(hidden, hidden), "b2": np.zeros(hidden),
             "W3": he(hidden, do), "b3": np.zeros(do)}
        M = {k: np.zeros_like(v) for k, v in P.items()}
        V = {k: np.zeros_like(v) for k, v in P.items()}
        b1a, b2a, eps = 0.9, 0.999, 1e-8
        t = 0
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb, yb = Xn[idx], Yn[idx]
                # forward
                z1 = xb @ P["W1"] + P["b1"]; a1 = np.tanh(z1)
                z2 = a1 @ P["W2"] + P["b2"]; a2 = np.tanh(z2)
                yp = a2 @ P["W3"] + P["b3"]
                # backward (MSE)
                m = len(idx)
                dy = (yp - yb) * (2.0 / m)
                g = {}
                g["W3"] = a2.T @ dy;         g["b3"] = dy.sum(0)
                da2 = dy @ P["W3"].T * (1 - a2 ** 2)
                g["W2"] = a1.T @ da2;        g["b2"] = da2.sum(0)
                da1 = da2 @ P["W2"].T * (1 - a1 ** 2)
                g["W1"] = xb.T @ da1;        g["b1"] = da1.sum(0)
                # Adam
                t += 1
                for k in P:
                    M[k] = b1a * M[k] + (1 - b1a) * g[k]
                    V[k] = b2a * V[k] + (1 - b2a) * g[k] ** 2
                    mh = M[k] / (1 - b1a ** t)
                    vh = V[k] / (1 - b2a ** t)
                    P[k] -= lr * mh / (np.sqrt(vh) + eps)
            if verbose and (ep + 1) % max(1, epochs // 8) == 0:
                a1 = np.tanh(Xn @ P["W1"] + P["b1"])
                a2 = np.tanh(a1 @ P["W2"] + P["b2"])
                mse = float(np.mean(((a2 @ P["W3"] + P["b3"]) - Yn) ** 2))
                print(f"  epoch {ep+1:4d}/{epochs}  train MSE(norm)={mse:.4f}")

        W = dict(P)
        W.update({"x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd})
        return cls(W)
