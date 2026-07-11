"""Unit tests for src/video_effects.py — render-only cosmetic effects."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import mujoco
import numpy as np

from src.video_effects import JuiceSplash, blade_glow, paste_inset

_MINI_XML = """
<mujoco>
  <worldbody>
    <body name="wm">
      <geom name="j1" type="sphere" size="0.01" pos="0.05 0 0.02" rgba="1 0 0 0"/>
      <geom name="j2" type="sphere" size="0.01" pos="-0.05 0 0.02" rgba="1 0 0 0"/>
      <geom name="blade" type="box" size="0.01 0.01 0.01" rgba="0.7 0.7 0.7 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model():
    return mujoco.MjModel.from_xml_string(_MINI_XML)


def test_juice_burst_sets_alpha_and_velocity():
    m = _model()
    js = JuiceSplash(m, ["j1", "j2", "missing_geom"])
    assert len(js.ids) == 2                       # missing geoms skipped
    js.burst(m, force_n=150.0)
    assert js.countdown == 50
    assert all(m.geom_rgba[g, 3] == 1.0 for g in js.ids)
    assert np.linalg.norm(js.vel) > 0


def test_juice_step_moves_fades_and_resets():
    m = _model()
    js = JuiceSplash(m, ["j1", "j2"])
    p0 = m.geom_pos[js.ids[0]].copy()
    js.burst(m, force_n=300.0, frames=3)
    for _ in range(3):
        js.step(m, dt=0.03)
    # after the countdown expires the spray is parked back and invisible
    assert js.countdown == 0
    assert np.allclose(m.geom_pos[js.ids[0]], p0)
    assert m.geom_rgba[js.ids[0], 3] == 0.0


def test_juice_burst_force_scales_speed():
    m = _model()
    js = JuiceSplash(m, ["j1"])
    js.burst(m, force_n=90.0)      # clipped at 0.6x
    v_lo = np.linalg.norm(js.vel)
    js.burst(m, force_n=420.0)     # 2.8x
    v_hi = np.linalg.norm(js.vel)
    assert v_hi > v_lo * 3


def test_blade_glow_active_and_reset():
    m = _model()
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "blade")
    rgba0 = m.geom_rgba[gid].copy()
    blade_glow(m, gid, rgba0, blade_speed_ms=0.75, active=True)
    assert m.geom_rgba[gid, 0] > rgba0[0]         # shifted toward red-hot
    blade_glow(m, gid, rgba0, blade_speed_ms=0.75, active=False)
    assert np.allclose(m.geom_rgba[gid, :3], rgba0[:3])


def test_paste_inset_writes_pixels_and_border():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    inset = np.full((20, 30, 3), 200, dtype=np.uint8)
    out = paste_inset(frame, inset, x0=50, y0=40)
    assert (out[40:60, 50:80] == 200).all()
    assert (out[39, 49:81] == [100, 110, 160]).all()   # top border
