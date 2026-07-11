"""
teleop_futurist.py
------------------
Interactive teleoperation of the **Futurist** serving arm — a debug / fallback
interface that proves the Futurist side is not a fully closed script: at any
moment you can take manual control of its 6-DOF right arm and its clamp hand.

A quarter watermelon wedge is placed on the shared table at the standard
serving spot (the same pose the autonomous pipeline hands off from), so there
is a real object to reach for, clamp and carry.

Controls
--------
  W / S     — shoulder pitch  +/−   (raise / lower the hand)
  A / D     — shoulder yaw    +/−    (swing the arm left / right)
  Q / E     — elbow           +/−    (extend / fold the forearm)
  Z / X     — shoulder roll   +/−
  R / F     — forearm roll     +/−   (turn the palm)
  T / G     — wrist yaw        +/−
  C         — toggle clamp     (close the thumb-vs-4-finger pincer / open it)
  SPACE     — toggle grab      (weld the wedge to the hand once the clamp is
                                closed around it / release it)
  TAB       — toggle AUTO ↔ MANUAL
  0         — reset
  ESC       — quit

In AUTO the Futurist runs its autonomous serving sequence (STEADY → REACH →
GRIP → LIFT → CARRY → LOWER → CONTACT_PLATE → RELEASE → DONE). Press TAB to
freeze that and fly the arm yourself.

Headless self-test (no display): ``python scripts/teleop_futurist.py --smoke``
"""

import sys, pathlib, time, argparse

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

from src.scene_builder import (build_combined_model as _build_combined_model,
                               FTR_BASE_POS as _FTR_BASE_POS,
                               FTR_BASE_QUAT as _FTR_BASE_QUAT)
from src.futurist_controller import (
    FuturistController, _FC_OPEN, _FC_CLOSED, _LA_WAITER)

# ── Constants ─────────────────────────────────────────────────────────
_DT_DISPLAY = 1.0 / 60.0
_JOINT_STEP = 0.03            # rad per keypress
# right-arm joint limits [sp, sr, sy, elbow, forearm-roll, wrist] (rad)
_RA_LIMITS = np.array([[-0.5, 2.4], [-2.4, 0.3], [-0.3, 2.4],
                       [0.0, 2.09], [-1.6, 2.4], [-1.6, 1.6]])
_SERVE_SPOT = np.array([0.487, -0.358, 0.663])   # wedge rest pose (fixed hand-off)
_HOLD_QUAT  = np.array([0.70710678, 0.0, 0.0, 0.70710678])
_MANUAL_BOW = 1.0            # hip bow held in MANUAL so the hand can reach low

# GLFW key codes
_KEY = {"W": 87, "S": 83, "A": 65, "D": 68, "Q": 81, "E": 69,
        "Z": 90, "X": 88, "R": 82, "F": 70, "T": 84, "G": 71,
        "C": 67, "SPACE": 32, "TAB": 258, "ZERO": 48, "ESC": 256}

_st = {"auto": True, "reset": False, "quit": False, "clamp_closed": False,
       "grab": False, "delta": np.zeros(6), "mode": "AUTO"}


def _key_cb(key: int) -> None:
    d = _st["delta"]
    if   key == _KEY["W"]: d[0] += _JOINT_STEP
    elif key == _KEY["S"]: d[0] -= _JOINT_STEP
    elif key == _KEY["Z"]: d[1] += _JOINT_STEP
    elif key == _KEY["X"]: d[1] -= _JOINT_STEP
    elif key == _KEY["A"]: d[2] += _JOINT_STEP
    elif key == _KEY["D"]: d[2] -= _JOINT_STEP
    elif key == _KEY["E"]: d[3] += _JOINT_STEP
    elif key == _KEY["Q"]: d[3] -= _JOINT_STEP
    elif key == _KEY["R"]: d[4] += _JOINT_STEP
    elif key == _KEY["F"]: d[4] -= _JOINT_STEP
    elif key == _KEY["T"]: d[5] += _JOINT_STEP
    elif key == _KEY["G"]: d[5] -= _JOINT_STEP
    elif key == _KEY["C"]:
        _st["clamp_closed"] = not _st["clamp_closed"]
        print(f"[teleop] clamp {'CLOSED' if _st['clamp_closed'] else 'OPEN'}")
    elif key == _KEY["SPACE"]:
        _st["grab"] = not _st["grab"]
        print(f"[teleop] grab {'ON' if _st['grab'] else 'OFF'}")
    elif key == _KEY["ZERO"]: _st["reset"] = True
    elif key == _KEY["ESC"]:  _st["quit"]  = True
    elif key == _KEY["TAB"]:
        _st["auto"] = not _st["auto"]
        _st["mode"] = "AUTO" if _st["auto"] else "MANUAL"
        print(f"\n[teleop] Mode → {_st['mode']}")


def _reveal_wedge(model, data, jadr_qA, gids_qA):
    """Place quarter A on the table at the serving spot, visible + collidable."""
    for gid in gids_qA:
        model.geom_rgba[gid, 3] = 1.0
    gcol = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wm_qA_col")
    model.geom_contype[gcol] = 2
    model.geom_conaffinity[gcol] = 2
    data.qpos[jadr_qA:jadr_qA + 3] = _SERVE_SPOT
    data.qpos[jadr_qA + 3:jadr_qA + 7] = _HOLD_QUAT
    data.qvel[jadr_qA:jadr_qA + 6] = 0.0 if jadr_qA < model.nv else 0.0


def _apply_manual(model, data, ftr, ra_ctrl, clamp_closed):
    """MANUAL mode: pin base + left arm + hips, drive the right arm and the
    clamp from teleop state (the Futurist arm is kinematically driven, so we
    write joint qpos directly)."""
    theta = _MANUAL_BOW
    quat = np.array([0.0, -np.sin(theta / 2), 0.0, np.cos(theta / 2)])
    b = ftr._base_qpos_adr
    data.qpos[b:b + 3] = ftr._base_target[:3]
    data.qpos[b + 3:b + 7] = quat
    for adr, v in zip(ftr._joint_adrs, ra_ctrl):
        data.qpos[adr] = float(v)
    for adr, v in zip(ftr._la_joint_adrs, _LA_WAITER):
        data.qpos[adr] = float(v)
    for adr in ftr._hip_joint_adrs:
        data.qpos[adr] = -theta
    fc = _FC_CLOSED if clamp_closed else _FC_OPEN
    for adr in ftr._rf_prox_adrs:
        data.qpos[adr] = float(fc[0])
    for adr in ftr._rf_dist_adrs:
        data.qpos[adr] = float(fc[1])
    if ftr._all_ftr_dof_slice is not None:
        data.qvel[ftr._all_ftr_dof_slice] = 0.0


def main(smoke: bool = False) -> None:
    model = _build_combined_model()
    data  = mujoco.MjData(model)
    ftr   = FuturistController()
    ftr.build(model, base_world_pos=_FTR_BASE_POS, base_world_quat=_FTR_BASE_QUAT)

    jid_qA  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wm_qA_free")
    jadr_qA = int(model.jnt_qposadr[jid_qA])
    gids_qA = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
               for n in ("wm_qA_skin", "wm_qA_face2_pith",
                         "wm_qA_face2_flesh", "wm_qA_face2_core")]
    eq_qA   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "wm_qA_weld")
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ftr_right_hand")
    tip_sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in
                ("ftr_rf0_tip", "ftr_rf1_tip", "ftr_rf2_tip",
                 "ftr_rf3_tip", "ftr_rthumb_tip")]

    dt = model.opt.timestep
    steps_per_frame = max(1, round(_DT_DISPLAY / dt))

    # right-arm manual target seeded at a reachable pre-grasp pose
    ra_ctrl = np.array([0.90, -1.60, 0.86, 0.88, 0.73, 0.0])

    def _reset():
        mujoco.mj_resetDataKeyframe(model, data, 0)
        ftr.reset(0.0)
        data.eq_active[eq_qA] = 0
        _reveal_wedge(model, data, jadr_qA, gids_qA)
        ftr.notify_steady(0.0)
        ra_ctrl[:] = [0.90, -1.60, 0.86, 0.88, 0.73, 0.0]
        _st["reset"] = False
        _st["delta"][:] = 0.0
        _st["clamp_closed"] = False
        _st["grab"] = False
        mujoco.mj_forward(model, data)

    _reset()
    _t_sim = 0.0
    _started = False

    def _step_once(t_sim):
        """One physics step; returns updated sim clock."""
        if _st["auto"]:
            ftr.step(data, t_sim)
        else:
            delta = _st["delta"].copy(); _st["delta"][:] = 0.0
            ra_ctrl[:] += delta
            for i in range(6):
                ra_ctrl[i] = float(np.clip(ra_ctrl[i],
                                           _RA_LIMITS[i, 0], _RA_LIMITS[i, 1]))
            _apply_manual(model, data, ftr, ra_ctrl, _st["clamp_closed"])
        # manual grab: weld the wedge to the fingertip cage when requested
        if _st["grab"]:
            tips_c = np.mean([data.site_xpos[s] for s in tip_sids], axis=0)
            fd = tips_c - data.xpos[hand_bid]
            fd = fd / max(float(np.linalg.norm(fd)), 1e-6)
            data.eq_active[eq_qA] = 1
            tgt = tips_c + fd * 0.03
            model.eq_data[eq_qA, 3:6] = tgt
            model.eq_data[eq_qA, 6:10] = _HOLD_QUAT
            data.qpos[jadr_qA:jadr_qA + 3] = tgt
            data.qpos[jadr_qA + 3:jadr_qA + 7] = _HOLD_QUAT
            data.qvel[jadr_qA:jadr_qA + 6] = 0.0
        else:
            data.eq_active[eq_qA] = 0
        mujoco.mj_step(model, data)
        return t_sim + dt

    if smoke:
        # headless self-test: exercise AUTO, MANUAL, clamp and grab paths
        for _ in range(200):
            _t_sim = _step_once(_t_sim)
        _st["auto"] = False
        _st["clamp_closed"] = True
        _st["delta"][:] = [0.05, 0, 0.05, 0.1, 0, 0]
        for _ in range(60):
            _t_sim = _step_once(_t_sim)
        _st["grab"] = True
        for _ in range(60):
            _t_sim = _step_once(_t_sim)
        qa = data.qpos[jadr_qA:jadr_qA + 3]
        assert np.all(np.isfinite(data.qpos)), "non-finite qpos"
        assert np.all(np.isfinite(qa)), "non-finite wedge pose"
        print(f"[smoke] OK  auto+manual+clamp+grab ran; wedge at "
              f"({qa[0]:.3f},{qa[1]:.3f},{qa[2]:.3f}); "
              f"nq={model.nq} steps_per_frame={steps_per_frame}")
        return

    print("\n" + "─" * 60)
    print("  Futurist Serving Teleop — debug / fallback interface")
    print("─" * 60)
    for line in __doc__.split("Controls\n--------\n")[1].split(
            "\nIn AUTO")[0].strip().splitlines():
        print("  " + line.strip())
    print("─" * 60)
    print("  Starting in AUTO — press TAB to take manual control.\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_cb,
                                      show_left_ui=True, show_right_ui=True) as viewer:
        t0_wall = time.time()
        while viewer.is_running() and not _st["quit"]:
            if _st["reset"]:
                _reset(); _t_sim = 0.0; t0_wall = time.time()
                viewer.sync(); continue
            for _ in range(steps_per_frame):
                _t_sim = _step_once(_t_sim)
            viewer.sync()
            surplus = _t_sim - (time.time() - t0_wall)
            if surplus > 0.002:
                time.sleep(surplus)
    print("\n[teleop] Viewer closed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="headless self-test (no viewer / display)")
    args = ap.parse_args()
    main(smoke=args.smoke)
