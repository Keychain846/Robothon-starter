"""
Unit tests for the behaviour-cloning policy (src/bc_policy.py) and its shipped
checkpoint (models/bc_policy.npz).
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pytest

from src.bc_policy import BCPolicy, featurize, PHASES, OBS_DIM, ACT_DIM

_ROOT = pathlib.Path(__file__).parent.parent
_CKPT = _ROOT / "models" / "bc_policy.npz"


def test_featurize_shape_and_onehot():
    obs = featurize(0.5, -0.1, 1.2, 0.44, -0.30, 0.64,
                    0.63, 0.0, 0.0, 2200.0, "CONTACT")
    assert obs.shape == (OBS_DIM,)
    # exactly one phase bit set, and it is CONTACT's
    onehot = obs[OBS_DIM - len(PHASES):]
    assert onehot.sum() == 1.0
    assert onehot[PHASES.index("CONTACT")] == 1.0


def test_featurize_unknown_phase_all_zero_onehot():
    obs = featurize(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "WAIT")
    assert obs[OBS_DIM - len(PHASES):].sum() == 0.0


def test_train_predict_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    # learnable linear-ish target: action = A @ obs
    X = rng.standard_normal((512, OBS_DIM))
    A = rng.standard_normal((OBS_DIM, ACT_DIM)) * 0.1
    Y = X @ A + 0.01 * rng.standard_normal((512, ACT_DIM))
    pol = BCPolicy.train(X, Y, hidden=32, epochs=60, verbose=False)
    y0 = pol.predict(X[0])
    assert y0.shape == (ACT_DIM,)
    assert np.all(np.isfinite(y0))
    # fit should be much better than predicting the mean
    pred = np.array([pol.predict(x) for x in X])
    assert np.mean((pred - Y) ** 2) < np.mean((Y - Y.mean(0)) ** 2)
    # save / load reproduces predictions exactly
    p = tmp_path / "p.npz"
    pol.save(p)
    pol2 = BCPolicy.load(str(p))
    assert np.allclose(pol2.predict(X[3]), pol.predict(X[3]))


@pytest.mark.skipif(not _CKPT.exists(),
                    reason="shipped checkpoint not present")
def test_shipped_checkpoint_predicts_finite_actions():
    pol = BCPolicy.load(str(_CKPT))
    for ph in ("APPROACH", "CONTACT", "SLICE2"):
        obs = featurize(0.53, -0.13, 1.0, 0.44, -0.30, 0.64,
                        0.4, 0.1, 20.0, 11000.0, ph)
        act = pol.predict(obs)
        assert act.shape == (ACT_DIM,)
        assert np.all(np.isfinite(act))
        # arm targets must stay in a sane joint range
        assert np.all(np.abs(act) < 4.0)
