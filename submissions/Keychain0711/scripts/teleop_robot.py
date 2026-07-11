"""
teleop_robot.py
--------------
Interactive teleoperation of the FF Master knife-cutting task.

Launches the MuJoCo passive viewer and lets the user drive the robot arm
in real time, then manually trigger the cut — or watch the autonomous
controller while switching to manual intervention at any moment.

Controls
--------
  W / S     — shoulder pitch +/−  (raise / lower blade)
  A / D     — wrist yaw   +/−    (swing blade left / right)
  Q / E     — elbow       +/−    (extend / retract arm)
  Z / X     — shoulder roll +/−  (tilt blade in/out)
  SPACE     — trigger cut NOW (first or second, in auto mode only)
  TAB       — toggle auto ↔ manual mode
  R         — reset episode
  ESC       — quit

In AUTO mode the autonomous 17-phase controller runs; pressing SPACE forces
an immediate notify_cut() call so you can experiment with cut timing.

In MANUAL mode the five right-arm joint targets are set directly from
accumulated key presses; the cut sequence is triggered by SPACE.
"""

import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

from src.robot_arm_controller import RobotArmController, ArmConfig
from src.cut_trigger_robot    import CutTriggerRobot
from src.feedback_controller  import FeedbackController

# ── Paths ─────────────────────────────────────────────────────────────
_ROOT  = pathlib.Path(__file__).parent.parent
_SCENE = _ROOT / "assets" / "scene_robot.xml"

# ── Constants ─────────────────────────────────────────────────────────
_DT_DISPLAY   = 1.0 / 60.0    # 60 Hz viewer sync
_JOINT_STEP   = 0.04           # radian increment per keypress
_CTRL_LIMITS  = np.array([[-1.6, 0.8],   # shoulder pitch [min, max]
                            [-0.8, 0.8],   # shoulder roll
                            [-0.8, 0.8],   # shoulder yaw
                            [-1.4, 0.1],   # elbow
                            [-0.6, 0.6]])  # wrist yaw

# GLFW key codes (uppercase ASCII for letters)
_KEY = {
    "W": 87, "S": 83, "A": 65, "D": 68,
    "Q": 81, "E": 69, "Z": 90, "X": 88,
    "R": 82, "SPACE": 32, "TAB": 258, "ESC": 256,
}

# ── Shared state (main thread reads; key_callback writes) ──────────────
# Python's GIL makes single-element reads/writes thread-safe for simple types.
_st = {
    "auto":  True,           # autonomous controller active
    "reset": False,          # reset flag
    "cut":   False,          # manual cut trigger flag
    "quit":  False,          # exit flag
    "delta": np.zeros(5),    # accumulated joint increments [sp, sr, sy, el, wr]
    "mode_label": "AUTO",
}

# Left-arm natural hang pose (ctrl[10:15])
_LEFT_NATURAL = np.array([-0.30,  0.30, 0.00, -0.80, 0.00], dtype=float)
# Guard pose during cut phases
_LEFT_GUARD   = np.array([-0.18,  0.28, 0.12, -0.42, 0.00], dtype=float)

# Biomimetic grip (closed = 0, open = values below)
_GRIP_OPEN  = np.array([0.9, 0.5, 0.3,  0.9, 0.5, 0.3,  0.9, 0.5, 0.3,
                         0.9, 0.4, 0.2, -0.9,-0.5,-0.3], dtype=float)
_GRIP_CLOSE = np.zeros(15, dtype=float)


def _key_cb(key: int) -> None:
    """Viewer-thread key callback — updates shared state."""
    d = _st["delta"]
    if   key == _KEY["W"]:     d[0] += _JOINT_STEP
    elif key == _KEY["S"]:     d[0] -= _JOINT_STEP
    elif key == _KEY["Z"]:     d[1] += _JOINT_STEP
    elif key == _KEY["X"]:     d[1] -= _JOINT_STEP
    elif key == _KEY["D"]:     d[4] += _JOINT_STEP
    elif key == _KEY["A"]:     d[4] -= _JOINT_STEP
    elif key == _KEY["E"]:     d[3] += _JOINT_STEP
    elif key == _KEY["Q"]:     d[3] -= _JOINT_STEP
    elif key == _KEY["R"]:     _st["reset"] = True
    elif key == _KEY["SPACE"]: _st["cut"]   = True
    elif key == _KEY["ESC"]:   _st["quit"]  = True
    elif key == _KEY["TAB"]:
        _st["auto"] = not _st["auto"]
        _st["mode_label"] = "AUTO" if _st["auto"] else "MANUAL"
        print(f"\n[teleop] Mode → {_st['mode_label']}")


def _ss(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def main() -> None:
    print(f"Loading scene: {_SCENE}")
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    data  = mujoco.MjData(model)

    # Sensor addresses
    sn = mujoco.mjtObj.mjOBJ_SENSOR
    def _sadr(name):
        return model.sensor_adr[mujoco.mj_name2id(model, sn, name)]
    adr_bpos   = _sadr("blade_pos")
    adr_wpos   = _sadr("wm_pos")
    adr_left   = _sadr("left_pos")
    adr_btouch = _sadr("blade_touch")
    adr_bvel   = _sadr("blade_vel")

    # Wrist joint qpos address
    jnt_rw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_yaw_joint")
    qa_rw  = model.jnt_qposadr[jnt_rw]

    dt         = model.opt.timestep
    # Number of physics steps per 60 Hz display frame
    _steps_per_frame = max(1, round(_DT_DISPLAY / dt))

    arm      = RobotArmController(ArmConfig())
    cut      = CutTriggerRobot(model, data)
    feedback = FeedbackController(model)   # 04-Control: closed-loop in ALIGN phases

    def _reset():
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:] = 0.0
        data.ctrl[10:15] = _LEFT_NATURAL
        data.ctrl[15:30] = _GRIP_CLOSE
        cut.reset(model, data)
        arm.reset()
        _manual_ctrl[:] = [-0.70, 0.0, 0.0, -0.65, 0.0]
        _st["reset"] = False
        _st["cut"]   = False
        _st["delta"][:] = 0.0
        mujoco.mj_forward(model, data)
        print("[teleop] Episode reset.")

    _manual_ctrl = np.array([-0.70, 0.0, 0.0, -0.65, 0.0], dtype=float)

    _reset()

    print("\n" + "─" * 58)
    print("  FF Master Teleop — Controls")
    print("─" * 58)
    print("  W / S    shoulder pitch  (raise / lower blade)")
    print("  A / D    wrist yaw       (left / right)")
    print("  Q / E    elbow           (extend / retract)")
    print("  Z / X    shoulder roll   (tilt blade in / out)")
    print("  SPACE    trigger cut     (forces notify_cut)")
    print("  TAB      toggle AUTO ↔ MANUAL")
    print("  R        reset episode")
    print("  ESC      quit")
    print("─" * 58)
    print("  Starting in AUTO mode — press TAB to take manual control.\n")

    _align_start_t   = None
    _retract_start_t = None
    _done_reached_t  = None
    _serve_triggered = False
    _prev_phase      = "APPROACH"
    _qA_serve_start  = None
    _serve_anim_t    = None
    _PLATE_POS       = np.array([0.40, -0.50, 0.615])
    _LEAN_MAX        = 0.22
    _LEAN_ADUR       = 1.0
    _LEAN_RDUR       = 1.5

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_cb,
                                      show_left_ui=True, show_right_ui=True) as viewer:
        _t0_wall = time.time()

        while viewer.is_running() and not _st["quit"]:
            # ── Reset ────────────────────────────────────────────────
            if _st["reset"]:
                _align_start_t  = None
                _retract_start_t = None
                _done_reached_t  = None
                _serve_triggered = False
                _prev_phase      = "APPROACH"
                _qA_serve_start  = None
                _serve_anim_t    = None
                _reset()
                _t0_wall = time.time()
                viewer.sync()
                continue

            # ── Physics steps ────────────────────────────────────────
            for _ in range(_steps_per_frame):
                t_sim = data.time
                sd    = data.sensordata

                blade_xyz = np.array(sd[adr_bpos:adr_bpos + 3])
                wm_xyz    = np.array(sd[adr_wpos:adr_wpos + 3])
                half_xyz  = np.array(sd[adr_left:adr_left + 3])
                touch_N        = float(sd[adr_btouch])
                blade_speed_ms = float(np.linalg.norm(sd[adr_bvel:adr_bvel + 3]))
                wrist_q   = float(data.qpos[qa_rw])

                _second = arm.phase in (
                    "REPOSITION2", "APPROACH2", "ALIGN2", "CONTACT2",
                    "SLICE2", "RETRACT2", "DONE2",
                    "KNIFE_DOWN", "SWEEP", "LIFT", "FINAL",
                )
                _target_xyz = half_xyz if _second else wm_xyz

                # ── Arm control ───────────────────────────────────────
                if _st["auto"]:
                    ctrl5 = arm.ctrl_target(t_sim, blade_xyz=blade_xyz,
                                            wm_xyz=_target_xyz,
                                            wrist_q=wrist_q, touch_N=touch_N)
                    data.ctrl[:5] = ctrl5
                    # 04-Control: Jacobian feedback in ALIGN phases (XY only)
                    if arm.phase in ("ALIGN", "ALIGN2"):
                        _fb_tgt = np.array([_target_xyz[0], _target_xyz[1], blade_xyz[2]])
                        data.ctrl[:5] = feedback.servo(model, data, _fb_tgt, data.ctrl[:5])
                else:
                    # Manual: apply accumulated increments
                    delta = _st["delta"].copy()
                    _st["delta"][:] = 0.0
                    _manual_ctrl += delta
                    for ji in range(5):
                        _manual_ctrl[ji] = float(np.clip(
                            _manual_ctrl[ji], _CTRL_LIMITS[ji, 0], _CTRL_LIMITS[ji, 1]))
                    data.ctrl[:5] = _manual_ctrl

                # ── Phase transitions ─────────────────────────────────
                if arm.phase != _prev_phase:
                    if arm.phase == "ALIGN":
                        _align_start_t = t_sim
                    elif arm.phase == "RETRACT":
                        _retract_start_t = t_sim
                    elif (arm.phase == "DONE"
                          and _done_reached_t is None
                          and cut.cut_fired):
                        _done_reached_t = t_sim
                        cut.prepare_second_cut(model)
                        arm.notify_reposition(t_sim)
                    _prev_phase = arm.phase

                # Serve animation trigger
                if arm.phase == "DONE2" and not _serve_triggered and cut.cut2_fired:
                    arm.notify_sweep(t_sim)
                    _serve_triggered = True

                # ── Manual cut trigger ────────────────────────────────
                if _st["cut"]:
                    _st["cut"] = False
                    if not cut.cut_fired:
                        arm.notify_cut(t_sim)
                        print(f"[teleop] Manual cut at t={t_sim:.3f}s  phase={arm.phase}")
                    elif not cut.cut2_fired and _done_reached_t is not None:
                        arm.notify_cut2(t_sim)
                        print(f"[teleop] Manual cut2 at t={t_sim:.3f}s")

                # ── Auto cut detection ────────────────────────────────
                if not cut.cut_fired:
                    if cut.step(model, data,
                                touch_N=touch_N, blade_speed_ms=blade_speed_ms):
                        arm.notify_cut(t_sim)
                        print(f"[teleop] AUTO CUT  t={t_sim:.3f}s  phase={arm.phase}  "
                              f"F={touch_N:.0f}N")
                elif not cut.cut2_fired and _done_reached_t is not None:
                    if cut.step2(model, data,
                                 touch_N=touch_N, blade_speed_ms=blade_speed_ms):
                        arm.notify_cut2(t_sim)
                        print(f"[teleop] AUTO CUT2 t={t_sim:.3f}s")

                # ── Waist lean ────────────────────────────────────────
                if arm.phase in ("ALIGN", "CONTACT", "SLICE",
                                  "ALIGN2", "CONTACT2", "SLICE2") and _align_start_t:
                    _lean = -_LEAN_MAX * _ss(
                        min((t_sim - _align_start_t) / _LEAN_ADUR, 1.0))
                elif arm.phase == "RETRACT" and _retract_start_t:
                    _lean = -_LEAN_MAX * (1.0 - _ss(
                        (t_sim - _retract_start_t) / _LEAN_RDUR))
                elif arm.phase in ("KNIFE_DOWN", "SWEEP"):
                    _lean = -0.14
                elif arm.phase in ("LIFT", "FINAL"):
                    _lean = -0.07
                else:
                    _lean = 0.0
                data.ctrl[5] = 0.0
                data.ctrl[6] = _lean
                data.ctrl[7] = 0.0

                # ── Head gaze ─────────────────────────────────────────
                if arm.phase in ("ALIGN", "CONTACT", "SLICE",
                                  "ALIGN2", "CONTACT2", "SLICE2"):
                    data.ctrl[9] = -0.28
                elif arm.phase in ("KNIFE_DOWN", "SWEEP"):
                    data.ctrl[9] = -0.24
                elif arm.phase in ("LIFT", "FINAL"):
                    data.ctrl[8] = 0.16
                    data.ctrl[9] = -0.12
                else:
                    data.ctrl[9] = -0.10

                # ── Left arm guard ────────────────────────────────────
                if arm.phase in ("ALIGN", "CONTACT", "SLICE",
                                  "ALIGN2", "CONTACT2", "SLICE2"):
                    _g = _ss(float(np.clip(
                        (t_sim - (_align_start_t or t_sim)) / 0.60, 0.0, 1.0)))
                    data.ctrl[10:15] = _LEFT_NATURAL * (1 - _g) + _LEFT_GUARD * _g
                elif arm.phase == "FINAL":
                    data.ctrl[10:15] = np.array([-0.50, 0.45, 0.00, -0.55, 0.00])
                else:
                    data.ctrl[10:15] = _LEFT_NATURAL

                # ── Fingers ───────────────────────────────────────────
                if arm.phase in ("SWEEP", "LIFT"):
                    data.ctrl[15:30] = _GRIP_OPEN * 0.55
                elif arm.phase == "FINAL":
                    data.ctrl[15:30] = _GRIP_OPEN * 0.35
                else:
                    data.ctrl[15:30] = _GRIP_CLOSE

                # ── Plate animation ───────────────────────────────────
                if arm.phase in ("SWEEP", "LIFT", "FINAL") and cut.cut2_fired:
                    if _serve_anim_t is None:
                        _serve_anim_t   = t_sim
                        _qA_serve_start = cut.get_qA_pos(data)
                    _alpha = _ss(float(np.clip(
                        (t_sim - _serve_anim_t) / 1.20, 0.0, 1.0)))
                    _np = _qA_serve_start + _alpha * (_PLATE_POS - _qA_serve_start)
                    _np = _np.copy()
                    _np[2] += 0.028 * float(np.sin(np.pi * _alpha))
                    cut.set_qA_weld_pos(model, data, _np)

                mujoco.mj_step(model, data)

            # ── Sync viewer ───────────────────────────────────────────
            viewer.sync()

            # ── Real-time pacing ──────────────────────────────────────
            _elapsed   = time.time() - _t0_wall
            _sim_time  = data.time
            _surplus   = _sim_time - _elapsed
            if _surplus > 0.002:
                time.sleep(_surplus)

    print("\n[teleop] Viewer closed.")


if __name__ == "__main__":
    main()
