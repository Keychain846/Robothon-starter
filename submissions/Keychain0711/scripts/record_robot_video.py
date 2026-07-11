"""
record_robot_video.py — Bimanual Edition
-----------------------------------------
FF Master cuts the watermelon; Futurist serves the quarter to a plate.

FF Master phases (13):
  APPROACH → ALIGN → CONTACT → SLICE → RETRACT → DONE →
  REGRASP → REPOSITION2 → APPROACH2 → ALIGN2 → CONTACT2 → SLICE2 → RETRACT2 → DONE2

Futurist phases (9 active + WAIT):
  WAIT → REACH → GRIP → LIFT → CARRY → LOWER → CONTACT_PLATE → RELEASE → RETRACT → DONE

Futurist is loaded via MjSpec URDF attachment; arm controlled by direct qpos.

Flags
-----
  --seed N          RNG seed for WM position offsets (default 42)
  --n-episodes N    Number of episodes (default 8; use 1 for fast check)
  --quick           Physics-only mode: no video, ~60 s. Writes run_summary.json.
  --collect         Save per-step demo data to output/demo_ep<N>.csv
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import os, re, tempfile, argparse
import numpy as np
import mujoco
import imageio

from src.cut_trigger_robot    import CutTriggerRobot
from src.robot_arm_controller import RobotArmController, ArmConfig
from src.episode_logger       import EpisodeLogger, LogConfig, RunLogger
from src.feedback_controller  import FeedbackController
from src.material_estimator   import MaterialEstimator
from src.futurist_controller  import FuturistController, _FC_OPEN, _FC_CLOSED
from src.bc_policy             import BCPolicy, featurize as _bc_featurize, PHASES as _BC_PHASES
from src.hud_renderer         import (draw_hud, title_card, episode_card,
                                       summary_card, recompress)
from src.scene_builder        import (build_combined_model, apply_material,
                                       SCENE_XML as _SCENE, FTR_URDF as _FTR_URDF,
                                       FTR_BASE_POS as _FTR_BASE_POS,
                                       FTR_BASE_QUAT as _FTR_BASE_QUAT)
from src.video_effects        import JuiceSplash, blade_glow, paste_inset

# ── Paths ─────────────────────────────────────────────────────────────
_ROOT         = pathlib.Path(__file__).parent.parent
_OUT          = _ROOT / "output"
_OUT.mkdir(exist_ok=True)
_VIDEO        = _OUT / "robot_demo.mp4"




# ── Defaults ──────────────────────────────────────────────────────────
FPS         = 30
WIDTH       = 1280
HEIGHT      = 720
EPISODE_DUR = 19.0
FREEZE_DUR  = 1.0
N_EPISODES  = 8

_WM_RNG    = np.random.default_rng(42)
WM_OFFSETS = [(0.0, 0.0)] + [
    (round(float(dx), 3), round(float(dy), 3))
    for dx, dy in zip(
        _WM_RNG.uniform(-0.045, 0.000, N_EPISODES - 1),
        _WM_RNG.uniform(0.005, 0.025,  N_EPISODES - 1),
    )
]

# Main: wide overview — Master (x=0) cuts, Futurist (x=0.90) waits
_CAM_MAIN  = dict(lookat=np.array([0.42, -0.20, 0.82]), dist=2.60, az=115.0, el=-20.0)
# Serve: Futurist right arm during GRIP/LIFT/CARRY — south side like main cam
# (Futurist faces −X; its right arm and the plate are on the −Y side, unblocked)
_CAM_SERVE = dict(lookat=np.array([0.62, -0.30, 0.95]), dist=1.50, az=120.0, el=-16.0)
# Plate: tight on the plate in Futurist's left palm for LOWER/CONTACT/RELEASE
_CAM_PLATE = dict(lookat=np.array([0.713, -0.266, 1.10]), dist=0.85, az=95.0, el=-26.0)
_CAM_CLOSE = dict(lookat=np.array([0.44, -0.28, 0.65]), dist=1.20, az=145.0, el=-24.0)
# Cutting close-up: framed on the knife hand + melon during ALIGN..SLICE
_CAM_CUT   = dict(lookat=np.array([0.42, -0.30, 0.72]), dist=1.30, az=135.0, el=-18.0)

_GRIP_OPEN  = np.array([0.9, 0.5, 0.3,  0.9, 0.5, 0.3,  0.9, 0.5, 0.3,
                         0.9, 0.4, 0.2,  -0.9, -0.5, -0.3], dtype=float)
_GRIP_CLOSE    = np.zeros(15, dtype=float)
_GRIP_DUR      = 1.3
_GRIP_OPEN_ABS = np.abs(_GRIP_OPEN)
_LEFT_ARM_NATURAL = np.array([-0.30, 0.30, 0.00, -0.80, 0.00], dtype=float)
_GRIP_STAGGER  = [0.000, 0.085, 0.170, 0.255, 0.050]
_GRIP_CASCADE  = [0.000, 0.070, 0.140]

_SERVE_ARC_DUR   = 0.80 + 1.20
_SERVE_LOWER_DUR = 1.20
# Wedge mesh: rind arc bottom is 0.090 below body center; +0.004 plate face.
_QA_PLATE_POS    = np.array([0.638, -0.301, 1.178])  # wedge rest point at hand-off (live plate pos + 0.055)
_QA_GRIP_OFFSET  = np.array([0.0, 0.0, -0.06])       # quarter hangs 6 cm below gripping hand


def _ss(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _biomimetic_grip_ctrl(t_sim: float, blade_dist_m: float = 1.0) -> np.ndarray:
    _dist_factor = float(np.clip(1.0 - blade_dist_m / 0.35, 0.0, 1.0))
    t_eff = t_sim + _dist_factor * 0.45
    ctrl = np.zeros(15)
    for fi, stag in enumerate(_GRIP_STAGGER):
        for ji, casc in enumerate(_GRIP_CASCADE):
            t0    = stag + casc
            avail = max(_GRIP_DUR - t0 - 0.04, 0.20)
            alpha = _ss(float(np.clip((t_eff - t0) / avail, 0.0, 1.0)))
            ctrl[fi * 3 + ji] = _GRIP_OPEN[fi * 3 + ji] * (1.0 - alpha)
    return ctrl


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps",        type=int,  default=FPS)
    ap.add_argument("--width",      type=int,  default=WIDTH)
    ap.add_argument("--height",     type=int,  default=HEIGHT)
    ap.add_argument("--seed",       type=int,  default=42,
                    help="RNG seed for WM position offsets (default 42)")
    ap.add_argument("--n-episodes", type=int,  default=None, dest="n_episodes",
                    help="Override episode count (default 8)")
    ap.add_argument("--quick",      action="store_true",
                    help="Physics-only validation (no video), ~60 s. Writes run_summary.json.")
    ap.add_argument("--wm-dx", type=float, default=None, dest="wm_dx",
                    help="Fixed watermelon x-offset (m) for all episodes "
                         "(robustness-envelope probing)")
    ap.add_argument("--wm-dy", type=float, default=None, dest="wm_dy",
                    help="Fixed watermelon y-offset (m)")
    ap.add_argument("--material",   choices=("soft", "firm", "hard"),
                    default="firm",
                    help="Watermelon material profile: contact stiffness "
                         "(solref) + fracture work threshold. The RLS "
                         "estimator identifies the resulting k online.")
    ap.add_argument("--collect",    action="store_true",
                    help="Save per-step demo data to output/demo_ep<N>.csv")
    ap.add_argument("--policy",     action="store_true",
                    help="Drive FF Master's cutting arm with the trained "
                         "behaviour-cloning policy (models/bc_policy.npz) "
                         "instead of the FSM's analytic joint targets. The "
                         "FSM still owns phase transitions and the "
                         "physics-grounded cut trigger. See scripts/train_policy.py.")
    ap.add_argument("--futurist-drive", choices=("qpos", "actuated"),
                    default="qpos", dest="futurist_drive",
                    help="Serving-arm drive. 'qpos' (default, the shipped "
                         "results): kinematic direct-qpos drive. 'actuated' "
                         "(experimental): the 6 right-arm joints get torque-"
                         "limited position actuators and the controller "
                         "commands data.ctrl — grasp/lift/carry complete "
                         "fully torque-driven (5-digit bite, closed-loop "
                         "placement over the plate); the final precision "
                         "set-down is not yet at parity and is reported "
                         "honestly as an open problem.")
    ap.add_argument("--grasp",      choices=("contact", "friction", "weld"),
                    default="friction",
                    help="Serving grasp mode. 'friction' (default, what the "
                         "shipped video shows): DYNAMIC torque-limited finger "
                         "actuators; digits squeeze, BITE into the flesh "
                         "(per-digit embedding constraints) and the wedge is "
                         "physically picked, carried swinging, and set down "
                         "on the plate — no tracking constraint at any point. "
                         "'contact': tactile-servoed clamp confirmed by "
                         "enclosure + contact force, then a hand-tracking "
                         "hold carries the wedge (kinematic-digit ablation). "
                         "'weld': legacy kinematic attachment (debug).")
    args = ap.parse_args()

    n_ep = (args.n_episodes if args.n_episodes is not None
            else (1 if args.quick else N_EPISODES))

    global WM_OFFSETS
    if args.seed != 42:
        _rng2 = np.random.default_rng(args.seed)
        WM_OFFSETS = [(0.0, 0.0)] + [
            (round(float(dx), 3), round(float(dy), 3))
            for dx, dy in zip(
                _rng2.uniform(-0.045, 0.000, N_EPISODES - 1),
                _rng2.uniform(0.005, 0.025,  N_EPISODES - 1),
            )
        ]
    if args.wm_dx is not None or args.wm_dy is not None:
        WM_OFFSETS = [(float(args.wm_dx or 0.0), float(args.wm_dy or 0.0))] * max(n_ep, 1)

    w, h, fps = args.width, args.height, args.fps

    if args.quick:
        print(f"[quick mode] seed={args.seed}  episodes={n_ep}  (physics-only, no video)")
    else:
        print(f"Building bimanual scene: {_SCENE.name} + {_FTR_URDF.name}")

    model = build_combined_model(dynamic_fingers=(args.grasp == "friction"),
                                 actuated_arm=(args.futurist_drive == "actuated"))
    data  = mujoco.MjData(model)
    print(f"  Combined model: nbody={model.nbody}, nq={model.nq}, neq={model.neq}"
          + (f", nu={model.nu} (dynamic digits)" if args.grasp == "friction" else ""))

    if not args.quick:
        model.vis.global_.offwidth  = w
        model.vis.global_.offheight = h
        renderer     = mujoco.Renderer(model, height=h, width=w)
        _INSET_W, _INSET_H = 240, 135
        _iy0_ins = h - _INSET_H - 62
        _ix0_ins = w - _INSET_W - 8
        renderer_top = mujoco.Renderer(model, height=_INSET_H, width=_INSET_W)
        cam_top = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam_top)
        cam_top.lookat[:] = np.array([0.45, -0.22, 0.65])
        cam_top.elevation = -89.0
        cam_top.azimuth   = 0.0
        cam_top.distance  = 2.40

    # Sensor addresses
    sn = mujoco.mjtObj.mjOBJ_SENSOR
    def _sadr(name): return model.sensor_adr[mujoco.mj_name2id(model, sn, name)]
    adr_bpos   = _sadr("blade_pos")
    adr_wpos   = _sadr("wm_pos")
    adr_bvel   = _sadr("blade_vel")
    adr_baccel = _sadr("blade_accel")
    adr_bgyro  = _sadr("blade_gyro")
    adr_bori   = _sadr("blade_ori")
    adr_blinv  = _sadr("blade_linv")
    adr_btouch = _sadr("blade_touch")
    adr_left   = _sadr("left_pos")
    adr_ft = [_sadr(n) for n in ("ft_touch_index", "ft_touch_middle",
                                  "ft_touch_ring",  "ft_touch_pinky", "ft_touch_thumb")]
    adr_gh = [_sadr(n) for n in ("gh_touch_index", "gh_touch_middle",
                                  "gh_touch_ring",  "gh_touch_pinky", "gh_touch_thumb")]
    _jid_kslip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knife_slide_z")
    _qa_kslip  = model.jnt_qposadr[_jid_kslip]

    _blade_edge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "blade_edge_trigger")
    _blade_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "blade_geom")
    _fbuf = np.zeros(6)

    if not args.quick:
        _blade_rgba0 = model.geom_rgba[_blade_geom_id].copy()
        _juice1 = JuiceSplash(model, [
            "wm_left_juice1", "wm_left_juice2", "wm_left_juice3", "wm_left_juice4"])
        _juice2 = JuiceSplash(model, [
            "wm_qA_juice1", "wm_qA_juice2", "wm_qA_juice3", "wm_qA_juice4",
            "wm_qA_juice5", "wm_qA_juice6",
            "wm_qB_juice1", "wm_qB_juice2", "wm_qB_juice3", "wm_qB_juice4"])

    eq_wm_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "wm_weld")
    eq_wm_L_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "wm_L_weld")
    _wm_base_pos   = model.eq_data[eq_wm_id,   3:6].copy()
    _wm_L_base_ref = model.eq_data[eq_wm_L_id, 3:6].copy()

    jnt_rw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_yaw_joint")
    qa_rw  = model.jnt_qposadr[jnt_rw]
    _POSTURE_SLICE = slice(19, 24)

    # Behaviour-cloning policy (optional --policy mode): a learned NN drives the
    # cutting arm's 5 joint targets; the FSM keeps phase logic + the cut trigger.
    _bcpol = None
    if args.policy:
        _ckpt = _ROOT / "models" / "bc_policy.npz"
        if not _ckpt.exists():
            sys.exit(f"--policy set but {_ckpt} missing — run "
                     "`python scripts/train_policy.py` first.")
        _bcpol = BCPolicy.load(str(_ckpt))
        print(f"  Master arm driven by BC policy → {_ckpt.name}")
    # phases the policy drives at runtime: gross reaching + between-cut
    # repositioning. The contact-critical ALIGN/CONTACT/SLICE strokes stay
    # with the analytic controller (BC drifts on sub-mm contact).
    _BC_RUN_PHASES = ("APPROACH", "RETRACT", "DONE",
                      "REGRASP", "REPOSITION2", "APPROACH2",
                      "RETRACT2", "DONE2")

    if not args.quick:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:]  = _CAM_MAIN["lookat"]
        cam.distance   = _CAM_MAIN["dist"]
        cam.azimuth    = _CAM_MAIN["az"]
        cam.elevation  = _CAM_MAIN["el"]

    cut       = CutTriggerRobot(model, data)

    if args.futurist_drive == "actuated":
        # Set-down-then-release physics (actuated arm): the wedge is PRESSED
        # onto the plate before the digits let go, so the plate must survive
        # being pressed on. The stock collision disc is 2.8 cm thick — a
        # convex-mesh wedge driven past its mid-plane gets pushed out the
        # BOTTOM by the convex collider (this is how earlier attempts
        # "tunnelled" through the plate). Thicken the collision cylinder
        # downward (top face unchanged) and give it bite-level friction +
        # priority so the seated wedge locks in place instead of skating.
        _pc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_col")
        if _pc >= 0:
            # condim 6 = sliding + torsional + ROLLING friction: the
            # round-bottomed wedge otherwise rolls across the disc like
            # a tumbler toward its lying pose and right off the rim
            # (rolling-friction coefficients are dead at the default
            # condim 3 — they were never in effect)
            model.geom_condim[_pc]   = 6
            model.geom_friction[_pc] = [1.6, 0.10, 0.08]
            model.geom_priority[_pc] = 2          # plate params govern seating
            model.geom_solref[_pc]   = [0.008, 1.0]

    _mat = apply_material(model, cut, args.material)
    print(f"  Material profile: {args.material}  "
          f"(solref={_mat['solref'][0]}, work×{_mat['work_scale']})")

    arm       = RobotArmController(ArmConfig())
    ftr       = FuturistController()
    if args.grasp == "friction":
        # friction-mode grasp geometry: aim the jaw-close centre at the
        # PINCH BAND ~3.5 cm below the wedge apex, where the ⋀ cross-section
        # (~7 cm) matches the jaw span so BOTH faces are within digit reach —
        # solved by the same FD-IK used for the shipped poses. Must be set
        # before ftr.build(), which FK-caches the waypoints.
        import src.futurist_controller as _fcmod
        _fcmod._P_GRIP[:]  = [1.200, -1.793, 0.988, 0.365, 0.644, 0.00]
        _fcmod._P_REACH[:] = [1.007, -1.713, 1.003, 0.534, 0.589, 0.00]
        _fcmod._BOW_PHASE_START["GRIP"] = 0.85
        _fcmod._BOW_PHASE_END["GRIP"]   = 1.061
        _fcmod._BOW_PHASE_END["REACH"]  = 0.85
        # two-stage lift: a human plucks the piece STRAIGHT UP off the table
        # before swinging. LIFT keeps the grip posture and only unwinds the
        # bow (the hand arcs gently upward, breaking table contact with no
        # lateral shear); the big swing to the plate happens in CARRY.
        _fcmod._P_LIFT[:] = [1.200, -1.793, 0.988, 0.365, 0.644, 0.00]
        _fcmod._BOW_PHASE_START["LIFT"]  = 1.061
        _fcmod._BOW_PHASE_END["LIFT"]    = 0.72
        _fcmod._BOW_PHASE_START["CARRY"] = 0.72
        # placement: the wedge HANGS from the embedded fingertips, offset
        # from the tip centroid — LOWER/CONTACT poses re-solved so the
        # hanging wedge (not the hand) is centred over the plate at release
        # the wedge HANGS from the embedded tips with a measured offset
        # (−1.6, +6.7) cm from the tip centroid — the hover targets are
        # solved so the LANDING (not the hand) is centred on the plate
        _fcmod._P_LOWER[:]   = [0.224, -1.176, 0.776, 2.029, 1.301, 0.00]
        _fcmod._P_CONTACT[:] = [0.224, -1.176, 0.776, 2.029, 1.301, 0.00]
        # slower finger opening: the wedge settles onto the plate between
        # the withdrawing digits instead of being flicked
        _fcmod._PHASE_DUR["RELEASE"] = 0.90
        if args.futurist_drive == "actuated":
            _fcmod._PHASE_DUR["RELEASE"] = 1.80
        # give the contact-gated FSM more room to wait for the bites to
        # form before lifting (grasp robustness across episode variation)
        _fcmod._GRIP_CONFIRM_TIMEOUT = 2.5
        if args.futurist_drive == "actuated":
            # the physically swinging wedge needs longer to settle over
            # the plate when the carry arm itself is dynamic
            _fcmod._RELEASE_SETTLE_TIMEOUT = 4.0
        # pure retiming, same path: the wedge hangs on embedded fingertips
        # like a pendulum, and swing loads scale with 1/T^2 — slowing the
        # transit phases ~1.5x cuts the dynamic loads to ~45% without
        # touching any pose
        _fcmod._PHASE_DUR["LIFT"]  = 1.60
        _fcmod._PHASE_DUR["CARRY"] = 2.00
        _fcmod._PHASE_DUR["LOWER"] = 1.70
        if args.futurist_drive == "actuated":
            # the torque-driven arm tracks with finite bandwidth: slower
            # approach keeps the swept path close to the planned line so
            # the open jaw straddles the wedge instead of side-swiping it
            _fcmod._PHASE_DUR["REACH"] = 2.60
            _fcmod._PHASE_DUR["GRIP"]  = 1.80
    ftr.build(model, base_world_pos=_FTR_BASE_POS, base_world_quat=_FTR_BASE_QUAT)
    ftr.dynamic_fingers = (args.grasp == "friction")
    print(f"  Futurist arm drive: "
          f"{'torque-limited actuators' if ftr.actuated_arm else 'kinematic qpos'}")
    run_log   = RunLogger()
    feedback  = FeedbackController(model)
    estimator = MaterialEstimator()

    _eq_grip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "ftr_grip_weld")
    _eq_wm_qA_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "wm_qA_weld")
    if (args.futurist_drive == "actuated"
            and os.environ.get("FTR_FIXTURE") and _eq_wm_qA_id >= 0):
        # the serving-spot weld must behave like a FIXTURE: at stock
        # impedance the digits' 30-90 N squeeze twists the held piece
        # tens of degrees before the bites form, re-randomising the
        # grasped pose that pinning the spot was meant to fix
        model.eq_solref[_eq_wm_qA_id] = [0.002, 1.0]
        model.eq_solimp[_eq_wm_qA_id, :2] = [0.99, 0.999]
    _jid_qA_free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wm_qA_free")
    _hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ftr_right_arm_link07")
    _qA_free_adr  = model.joint(_jid_qA_free).qposadr[0] if _jid_qA_free >= 0 else -1
    _qA_dof_adr   = model.joint(_jid_qA_free).dofadr[0]  if _jid_qA_free >= 0 else -1
    # Right-hand fingertip sites — used to quantify multi-finger grasp closure
    # around the quarter (distance from each tip to the wedge surface).
    _rtip_sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                  for n in ("ftr_rf0_tip", "ftr_rf1_tip", "ftr_rf2_tip",
                            "ftr_rf3_tip", "ftr_rthumb_tip")]
    _rtip_sids = [i for i in _rtip_sids if i >= 0]
    _rhand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ftr_right_hand")
    _QA_COL_R  = 0.119   # quarter collision-hull mean radius (1.4× flesh mesh)

    # ── Contact-grasp modes ('contact' and 'friction' share the machinery:
    # free-body wedge + tactile servo + enclosure/contact confirmation gate;
    # they differ in the CARRY model — contact = hand-tracking hold after
    # confirmation, friction = no constraint, pure contact carry on dynamic
    # torque-limited digits) ───────────────────────────────────────────────
    _grasp_contact  = args.grasp in ("contact", "friction")
    # FTR_FIXTURE=1: fixture-handover grasp experiment (pose-deterministic
    # 8/8, but the deep-bite release physics is unsolved — see README)
    _FIXTURE = bool(os.environ.get("FTR_FIXTURE"))
    _grasp_friction = args.grasp == "friction"
    _GRASP_DEBUG    = bool(os.environ.get("FTR_GRASP_DEBUG"))
    _qa_col_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wm_qA_col")
    # map each right-hand finger geom id → finger index (0-3 fingers, 4 thumb)
    # so per-finger normal forces on the wedge can drive the closure servo
    _gid2finger: dict = {}
    for _fi in range(4):
        for _bn in (f"ftr_rf{_fi}_prox", f"ftr_rf{_fi}_dist"):
            _bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _bn)
            if _bid >= 0:
                for _g in range(model.body_geomadr[_bid],
                                model.body_geomadr[_bid] + model.body_geomnum[_bid]):
                    _gid2finger[_g] = _fi
    _bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ftr_rthumb")
    if _bid >= 0:
        for _g in range(model.body_geomadr[_bid],
                        model.body_geomadr[_bid] + model.body_geomnum[_bid]):
            _gid2finger[_g] = 4
    # palm plate + forearm block → slot 5: backstop force, not servoed
    for _bn in ("ftr_right_hand", "ftr_right_arm_link05",
                "ftr_right_arm_link07"):
        _bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _bn)
        if _bid >= 0:
            for _g in range(model.body_geomadr[_bid],
                            model.body_geomadr[_bid] + model.body_geomnum[_bid]):
                _gid2finger[_g] = 5
    # right-finger joint qpos addresses, per finger: [(prox, dist) ×4] + thumb
    def _jadr(name):
        _j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_qposadr[_j]) if _j >= 0 else -1
    _rf_pd_adrs = [(_jadr(f"ftr_rf{i}_prox"), _jadr(f"ftr_rf{i}_dist"))
                   for i in range(4)]
    _rthumb_adr = _jadr("ftr_rthumb")
    def _jdof(name):
        _j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(model.jnt_dofadr[_j]) if _j >= 0 else -1
    # per-digit joint dof addresses [prox, dist] x4 + thumb — the bite
    # damping is applied HERE: an embedded fingertip is a 0.008 kg body
    # anchoring a 0.18 kg wedge (22:1), so without flesh damping on the
    # digit joints the suspension point itself flails
    _rf_digit_dofs = [[_jdof(f"ftr_rf{i}_prox"), _jdof(f"ftr_rf{i}_dist")]
                      for i in range(4)] + [[_jdof("ftr_rthumb")]]
    _DIGIT_FLESH_DAMPING = 3.0     # N·m·s — knuckles buried in flesh
    _digit_damp0 = {}
    # friction mode: ctrl addresses of the torque-limited digit actuators
    _rf_ctrl_pd, _rthumb_ctrl = [], -1
    if _grasp_friction:
        def _cadr(name):
            return int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
        _rf_ctrl_pd  = [(_cadr(f"ftr_rf{i}_prox"), _cadr(f"ftr_rf{i}_dist"))
                        for i in range(4)]
        _rthumb_ctrl = _cadr("ftr_rthumb")
    # closed-position COMMANDS for the dynamic digits: the CENTRE of each
    # digit's measured engagement band with the wedge faces (probed by arc
    # sweep) — commanding past the band makes a digit slide tangentially
    # along the 45° face and pop out the far side; commanding the band
    # centre keeps the actuator pressing INTO the face for the whole carry
    _DYN_CLOSED_PROX, _DYN_CLOSED_DIST, _DYN_CLOSED_THUMB = 0.45, 0.40, 0.55
    # flesh-bite: a digit pressing this hard has crushed into the flesh —
    # its point CONNECT to the wedge activates (embedded-tip form closure).
    # Soft ripe flesh yields at light pressure, so the threshold is low.
    _BITE_N = 1.5
    # viscous drag of digits buried in flesh (N·s/m). The dynamic arm
    # needs stronger pendulum damping; 2.0 is the shipped kinematic value
    _BITE_DAMPING = 6.0 if args.futurist_drive == "actuated" else 2.0
    _bite_eids, _bite_bodies = [], []
    if _grasp_friction:
        for _bn in ("ftr_rf0_dist", "ftr_rf1_dist", "ftr_rf2_dist",
                    "ftr_rf3_dist", "ftr_rthumb"):
            _bite_eids.append(int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, f"bite_{_bn}")))
            _bite_bodies.append(int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, _bn)))
        _qa_body_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "wm_quarter_A"))
    # open-position COMMANDS: fingers SPLAYED BACK so the descending open jaw
    # straddles the wedge — fingertips hang on the −x slope side of the apex,
    # the outboard thumb on the +x side; descent pressure then nudges the
    # wedge INTO the braced thumb instead of knocking it away
    _DYN_OPEN_PROX, _DYN_OPEN_DIST = -0.20, 0.00
    if _grasp_friction:
        # only COMPLIANT digits may touch the wedge: the rigid kinematic
        # backstops (palm plate, forearm, wrist — added for the old scoop
        # carry) bulldoze the wedge out of the jaw during descent
        for _g, _fi in _gid2finger.items():
            if _fi == 5:
                model.geom_contype[_g] = 0
                model.geom_conaffinity[_g] = 0
    _QA_TABLE_REST_Z = 0.663        # enlarged flesh-hull centre resting on table
    _FSRV_TGT_N   = 12.0            # per-finger normal-force target (N) — keep
                                    # curling under load so the tips dig UNDER
                                    # the wedge belly, not just touch it
    _FSRV_MAX_N   = 25.0            # back-off threshold (N)
    _FSRV_HOLD_N  = 6.0             # re-tighten threshold while carrying
    _FC_CONFIRM_N = 2.0            # per-finger force that counts toward closure
    _FC_CONFIRM_K = 2             # digits needed to confirm closure (thumb +
                                  # finger pinch on the big wedge's ridge)
    _FSRV_RATE    = 2.2             # curl rate (fraction of full close /s)
    from src.futurist_controller import _PHASE_DUR as _FTR_PHASE_DUR
    _FTR_GRIP_DUR = _FTR_PHASE_DUR["GRIP"]

    # Plate follows the left PALM in the hand's local frame (not a fixed world
    # offset — that let the forearm sweep through the plate whenever the arm
    # pose changed). Local offset derived from the waiter hold: palm-normal
    # (+x) up 4cm, toward the fingers (+z) 4.8cm.
    _lhand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ftr_left_hand")
    _plate_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "serving_plate")
    _PLATE_OFF_LOCAL = np.array([0.040, 0.002, 0.048])
    print(f"  ftr_grip_weld eq_id={_eq_grip_id}  wm_qA_free={_jid_qA_free}  hand_body={_hand_body_id}")

    dt        = model.opt.timestep
    step_skip = max(1, round(1.0 / (fps * dt)))
    # friction mode's gentler (slower) transit phases need a little more time
    _ep_dur   = EPISODE_DUR + (2.5 if args.grasp == "friction" else 0.0) \
                + (6.5 if args.futurist_drive == "actuated" else 0.0)
    n_steps   = round(_ep_dur / dt)

    if not args.quick:
        writer = imageio.get_writer(str(_VIDEO), fps=fps, codec="libx264", quality=8)
        for f in title_card(w, h, 1.0, fps):
            writer.append_data(f)

    _k_per_episode: list[float] = []
    _k_serve_errs:  list = []          # delivery error (mm) per episode, None if serve failed
    _k_closures:    list = []          # (n_tips_within_3cm, mean_tip_dist_mm) per episode
    _k_grip_forces: list = []          # mean total fingertip force (N) during CARRY

    for ep in range(n_ep):
        dx, dy = WM_OFFSETS[ep % len(WM_OFFSETS)]
        print(f"\n─── Episode {ep + 1}/{n_ep} ───")
        ep_log = EpisodeLogger(ep, LogConfig())

        if not args.quick:
            for f in episode_card(w, h, ep, n_ep, dx, dy, fps, duration_s=0.3):
                writer.append_data(f)

        model.eq_data[eq_wm_id,   3] = _wm_base_pos[0] + dx
        model.eq_data[eq_wm_id,   4] = _wm_base_pos[1] + dy
        model.eq_data[eq_wm_L_id, 3:6]  = _wm_L_base_ref
        model.eq_data[eq_wm_L_id, 6:10] = [1, 0, 0, 0]

        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:] = 0.0
        if _grasp_friction:
            # dynamic digits start at the SPLAYED open target
            for _pc, _dc in _rf_ctrl_pd:
                data.ctrl[_pc] = _DYN_OPEN_PROX
                data.ctrl[_dc] = _DYN_OPEN_DIST
            data.ctrl[_rthumb_ctrl] = _FC_OPEN[0]
        cut.reset(model, data)
        arm.reset()
        ftr.reset(0.0)
        if ftr.actuated_arm:
            ftr.seed_idle(data)
        data.eq_active[_eq_grip_id] = 0
        if _eq_wm_qA_id >= 0:
            data.eq_active[_eq_wm_qA_id] = 1

        if not args.quick:
            _juice1.reset(model)
            _juice2.reset(model)

        mujoco.mj_forward(model, data)
        data.ctrl[15:30] = _GRIP_OPEN
        data.ctrl[10:15] = _LEFT_ARM_NATURAL

        cut_frame = False
        cut_t     = None
        current_failure = None
        blade_speed_ms  = 0.0
        touch_sensor    = 0.0
        touch_N         = 0.0
        _done_reached_t   = None
        cut2_frame        = False
        cut2_t            = None
        _align2_start_t   = None
        _retract2_start_t = None
        _regrasp_start_t  = None
        _release_start_t  = None   # actuated arm: 3-stage release clock
        _rel_fs_floor     = 0.0    # digit-open floor after unbite
        _seated_since_t   = None   # wedge-on-plate contact clock
        _qa_prev_pos      = None   # for REAL (finite-difference) speed
        _qa_z_lp          = None   # low-passed wedge height (plate servo)
        _qa_ride_checked  = False  # grasp-quality acceptance done
        _qa_rest_pose     = None   # fixture pose held until handover
        _handover_pending = False  # confirm reached, decompressing
        _handover_ok_t    = None   # low-load dwell clock
        _plate_lift_tgt   = 0.0    # one-shot hang-depth compensation
        _qa_fd_speed      = 0.0    # low-passed true wedge speed (m/s)
        _seated           = False  # wedge is resting on the plate
        _regrasp_gh0          = 1.0    # pre-open grip-force baseline (N)
        _regrasp_reclose_frac = None   # sensor-gated reclose start
        _regrasp_slip_prev    = 0.0
        _regrasp_checked      = False  # end-of-round verification done

        if ep == 0:
            estimator.reset()
        else:
            estimator.soft_reset()
        _slice_adapted  = False
        _material_k     = estimator.stiffness_Npm
        _material_conf  = 0.0

        _ftr_triggered     = False
        _ftr_settle_end_t  = None
        _ftr_prev_phase    = "WAIT"
        _grip_activated    = False
        _grip_phase_start_t = None   # camera pan to serve cam
        _grasp_start_t     = None
        _grasp_from_pos    = None
        _qA_anim_start_t   = None
        _qA_anim_start_pos = None
        _qA_lower_start_t  = None
        _qA_released       = False
        _serve_err_mm      = float("nan")
        _contact_z0        = None           # blade z at first rind contact
        _closure_n         = 0              # fingertips within threshold of the wedge
        _closure_mm        = float("nan")   # mean fingertip→surface distance while held
        _lower_blend_t     = None           # grasp target: fingertips → under-wrist blend
        _fs                = np.zeros(5)    # tactile servo curl state per finger [0..1]
        _bitten            = [False] * 5    # per-digit flesh-bite state
        if _grasp_friction:
            for _be in _bite_eids:
                data.eq_active[_be] = 0
            model.dof_damping[_qA_dof_adr:_qA_dof_adr + 6] = 0.0
            for _dd, _v in _digit_damp0.items():
                model.dof_damping[_dd] = _v
            _digit_damp0.clear()
        _fF                = np.zeros(6)    # normal force on the wedge (N):
                                            # 4 fingers, thumb, palm backstop
        _qa_free_t         = None           # when to cut the wedge loose after cut 2
        _qa_freed          = False          # wedge is a free body (contact mode)
        _grip_force_carry  = []             # total grip force samples during CARRY
        _fc_confirmed      = False          # tactile force closure confirmed
        _fc_confirmed_t    = None           # when force closure was confirmed
        _qa_hold_quat      = np.array([0.70710678, 0.0, 0.0, 0.70710678])
        _zoom_state        = 0.0            # serving push-in ease [0..1]
        _policy_steps      = 0              # arm steps driven by the BC policy
        _master_steps      = 0              # total Master arm-control steps
        _plate_cam_start_t = None
        _plate_retract_t   = None
        _slice_start_t     = None
        _slice2_start_t    = None

        # Split animations are PHYSICS-relevant (they write qpos/qvel/eq_data
        # of the half and quarter-B) — they run identically in quick and
        # render passes, clocked in physics steps, so the two passes have
        # bit-identical dynamics. (Historically they were render-only and
        # frame-clocked: THE systematic quick-vs-render divergence.)
        _SPLIT_DUR        = 14 * step_skip
        _split1_countdown = 0
        _split2_countdown = 0

        _GH_BASE_N  = 10.0
        _GH_CUT_K   = 0.12
        _GH_SERVO_K = 0.00012
        gh_forces    = [0.0] * 5
        knife_slip_mm = 0.0
        _peak_impact_g = 0.0
        blade_accel_g  = 0.0
        blade_tilt_deg = 0.0
        blade_gyro_dps = 0.0
        _LEFT_GUARD = np.array([-1.05, -0.06, -1.50, -1.10, 0.00], dtype=float)
        ep_data: list = []
        _phase_history_hud: list = [(0.0, "APPROACH")]
        _prev_phase_hud: str     = "APPROACH"
        _force_hist_hud: list    = []
        ft_contacts  = [0.0] * 5
        _wrist_corr  = 0.0
        _cut_quality = 0.0
        _LEAN_MAX  = 0.22
        _LEAN_ADUR = 1.0
        _LEAN_RDUR = 1.5
        _wm_stab_ref: np.ndarray | None = None
        _prev_phase      = "APPROACH"
        _align_start_t   = None
        _retract_start_t = None
        posture_rms      = 0.0
        grip_pct         = 0.0

        for s in range(n_steps):
            t_sim = s * dt
            sd    = data.sensordata

            blade_xyz    = np.array(sd[adr_bpos:adr_bpos + 3])
            wm_xyz       = np.array(sd[adr_wpos:adr_wpos + 3])
            half_xyz     = np.array(sd[adr_left:adr_left + 3])
            blade_dist   = float(np.linalg.norm(blade_xyz - wm_xyz))
            blade_dist2  = float(np.linalg.norm(blade_xyz - half_xyz))
            blade_speed_ms = float(np.linalg.norm(sd[adr_bvel:adr_bvel + 3]))
            touch_sensor   = float(sd[adr_btouch])
            blade_accel_g  = float(np.linalg.norm(sd[adr_baccel:adr_baccel + 3])) / 9.81
            if arm.phase in ("CONTACT", "SLICE", "CONTACT2", "SLICE2"):
                _peak_impact_g = max(_peak_impact_g, blade_accel_g)
            blade_gyro_dps = float(np.linalg.norm(sd[adr_bgyro:adr_bgyro + 3])) * (180.0 / np.pi)
            _bq = sd[adr_bori:adr_bori + 4]
            _bq = _bq / (np.linalg.norm(_bq) + 1e-9)
            _bz_world = np.array([
                2*(_bq[1]*_bq[3] + _bq[0]*_bq[2]),
                2*(_bq[2]*_bq[3] - _bq[0]*_bq[1]),
                _bq[0]**2 - _bq[1]**2 - _bq[2]**2 + _bq[3]**2,
            ])
            blade_tilt_deg = float(np.degrees(np.arccos(np.clip(abs(_bz_world[2]), 0, 1))))

            _second_phase = arm.phase in (
                "REGRASP", "REPOSITION2", "APPROACH2", "ALIGN2", "CONTACT2",
                "SLICE2", "RESTRIKE2", "RETRACT2", "DONE2",
            )
            _arm_target_xyz = half_xyz if _second_phase else wm_xyz
            _hud_dist       = blade_dist2 if _second_phase else blade_dist

            arm.set_material_state(_material_k, _material_conf)
            wrist_q = float(data.qpos[qa_rw])
            data.ctrl[:5] = arm.ctrl_target(t_sim, blade_xyz=blade_xyz,
                                            wm_xyz=_arm_target_xyz, wrist_q=wrist_q,
                                            touch_N=touch_N)

            if arm.phase in ("ALIGN", "ALIGN2"):
                _fb_target = np.array([_arm_target_xyz[0], _arm_target_xyz[1], blade_xyz[2]])
                data.ctrl[:5] = feedback.servo(model, data, _fb_target, data.ctrl[:5])

            # --policy: hierarchical control. The learned BC net drives the
            # arm's gross reaching and between-cut repositioning; the analytic
            # controller keeps the contact-critical ALIGN + cutting strokes,
            # where sub-mm blade placement gates the physics cut trigger and
            # open-loop behaviour cloning drifts. This is a genuine learned
            # low-level controller under a scripted high-level plan.
            if _bcpol is not None and arm.phase in _BC_RUN_PHASES:
                _obs = _bc_featurize(
                    blade_xyz[0], blade_xyz[1], blade_xyz[2],
                    wm_xyz[0], wm_xyz[1], wm_xyz[2],
                    blade_dist, blade_speed_ms, touch_N, _material_k, arm.phase)
                data.ctrl[:5] = _bcpol.predict(_obs)
                _policy_steps += 1
            _master_steps += 1

            if arm.phase in ("CONTACT", "SLICE") and not cut.cut_fired:
                cut.update_blade_penetration(model, data, blade_xyz[2], wm_xyz[2],
                                             blade_force_n=touch_N,
                                             blade_vel_ms=blade_speed_ms,
                                             k_wm=_material_k,
                                             blade_tilt_deg=blade_tilt_deg)
                # penetration proxy: blade descent since first rind contact
                # (the servo presses in; softer material yields more depth for
                # the same force → RLS identifies a lower k)
                if touch_N > 2.0:
                    if _contact_z0 is None:
                        _contact_z0 = float(blade_xyz[2])
                    _pen_depth     = max(0.0, _contact_z0 - float(blade_xyz[2]))
                    _material_k    = estimator.update(_pen_depth, touch_N)
                    _material_conf = estimator.confidence

            if arm.phase in ("CONTACT", "SLICE", "CONTACT2", "SLICE2"):
                _wrist_corr = float(np.clip(-_bz_world[0] * 0.22, -0.09, 0.09))
                data.ctrl[4] = float(np.clip(data.ctrl[4] + _wrist_corr, -2.556, 2.556))
            else:
                _wrist_corr = 0.0

            ft_contacts   = [float(sd[adr]) for adr in adr_ft]
            gh_forces     = [float(sd[adr]) for adr in adr_gh]
            knife_slip_mm = float(data.qpos[_qa_kslip]) * 1000.0

            if arm.phase == "APPROACH":
                data.ctrl[15:30] = _biomimetic_grip_ctrl(t_sim, blade_dist_m=blade_dist)
            elif arm.phase == "REGRASP":
                # Closed-loop in-hand regrasp: open (differential, load-
                # proportional) -> hold until the knife has SETTLED in the new
                # orientation (slip-rate sensor gate, deadline fallback) ->
                # reclose -> verify grip restored; a failed verification
                # restarts the round (arm.retry_regrasp, budget 2).
                if _regrasp_start_t is None:
                    _regrasp_start_t = t_sim
                    _regrasp_gh0     = max(sum(gh_forces) / 5.0, 1.0)
                _rg_frac = float(np.clip(
                    (t_sim - _regrasp_start_t) / ArmConfig().regrasp_dur, 0.0, 1.0))
                _gh_mean = max(sum(gh_forces) / 5.0, 1.0)
                _slip_rate = abs(knife_slip_mm - _regrasp_slip_prev) / dt
                if _rg_frac < 0.35:
                    _rg_a = _ss(_rg_frac / 0.35)
                    for _fi in range(5):
                        _load_ratio = float(np.clip(gh_forces[_fi] / _gh_mean, 0.4, 2.0))
                        _open_fi = 0.28 * _rg_a * _load_ratio
                        _ci_mcp  = 15 + _fi * 3
                        data.ctrl[_ci_mcp]     = _GRIP_CLOSE[_fi*3]   + _GRIP_OPEN[_fi*3]   * _open_fi
                        data.ctrl[_ci_mcp + 1] = _GRIP_CLOSE[_fi*3+1] + _GRIP_OPEN[_fi*3+1] * 0.28 * _rg_a
                        data.ctrl[_ci_mcp + 2] = _GRIP_CLOSE[_fi*3+2] + _GRIP_OPEN[_fi*3+2] * 0.28 * _rg_a
                else:
                    # sensor gate: reclose once the reorienting knife stops
                    # sliding in the loosened grip (< 5 mm/s), 65% deadline
                    if (_regrasp_reclose_frac is None
                            and (_slip_rate < 5.0 or _rg_frac >= 0.65)
                            and _rg_frac >= 0.45):
                        _regrasp_reclose_frac = _rg_frac
                    if _regrasp_reclose_frac is None:
                        data.ctrl[15:30] = _GRIP_CLOSE + _GRIP_OPEN * 0.28
                    else:
                        _rg_a = _ss((_rg_frac - _regrasp_reclose_frac)
                                    / max(1.0 - _regrasp_reclose_frac, 0.05))
                        data.ctrl[15:30] = (_GRIP_CLOSE + _GRIP_OPEN * 0.28) * (1.0 - _rg_a)
                # end-of-round verification: grip force back near baseline and
                # knife not still slipping -- else run another round
                if _rg_frac >= 0.93 and not _regrasp_checked:
                    _regrasp_checked = True
                    _grip_ok = (_gh_mean >= 0.5 * _regrasp_gh0
                                and abs(knife_slip_mm) < 2.5)
                    if not _grip_ok and arm.retry_regrasp(t_sim):
                        print(f"  REGRASP retry {arm.regrasp_round} at t={t_sim:.2f}s "
                              f"(gh={_gh_mean:.1f}N vs base {_regrasp_gh0:.1f}N, "
                              f"slip={knife_slip_mm:+.1f}mm)")
                        _regrasp_start_t      = t_sim
                        _regrasp_reclose_frac = None
                        _regrasp_checked      = False
                _regrasp_slip_prev = knife_slip_mm
            else:
                data.ctrl[15:30] = _GRIP_CLOSE

            if arm.phase != "APPROACH":
                _blade_load  = max(0.0, float(sd[adr_btouch]) - 5.0)
                _gh_target_n = min(_GH_BASE_N + _GH_CUT_K * _blade_load, 55.0)
                for fi, adr in enumerate(adr_gh):
                    gh_n  = float(sd[adr])
                    ci    = 15 + fi * 3
                    og    = float(_GRIP_OPEN[fi * 3])
                    delta = float(np.clip(
                        (gh_n - _gh_target_n) * _GH_SERVO_K, -0.006, 0.006))
                    if og >= 0:
                        data.ctrl[ci] = float(np.clip(
                            data.ctrl[ci] + delta, 0.0, abs(og) * 0.18))
                    else:
                        data.ctrl[ci] = float(np.clip(
                            data.ctrl[ci] - delta, og * 0.18, 0.0))

            if arm.phase != _prev_phase:
                if   arm.phase == "ALIGN":   _align_start_t   = t_sim
                elif arm.phase == "RETRACT": _retract_start_t = t_sim
                elif arm.phase == "ALIGN2":  _align2_start_t  = t_sim
                elif arm.phase == "RETRACT2": _retract2_start_t = t_sim
                elif arm.phase == "SLICE":   _slice_start_t   = t_sim
                elif arm.phase == "SLICE2":  _slice2_start_t  = t_sim
                elif (arm.phase == "DONE" and _done_reached_t is None and cut.cut_fired):
                    _done_reached_t = t_sim
                    cut.prepare_second_cut(model)
                    _mult = estimator.slice_dur_multiplier()
                    arm._cfg.slice_dur  = ArmConfig().slice_dur * _mult
                    _k_norm = estimator.stiffness_Npm / 2200.0
                    _arc_delta = float(np.clip((_k_norm - 1.0) * 0.10, -0.07, 0.07))
                    arm._cfg.slice_arc  = list(ArmConfig().slice_arc)
                    arm._cfg.slice_arc[0] = ArmConfig().slice_arc[0] + _arc_delta
                    _slice_adapted = abs(_mult - 1.0) > 0.05 or abs(_arc_delta) > 0.01
                    arm.notify_reposition(t_sim)
                elif arm.phase == "REGRASP" and _regrasp_start_t is None:
                    _regrasp_start_t = t_sim
                _prev_phase = arm.phase
                if arm.phase != _prev_phase_hud:
                    _phase_history_hud.append((t_sim, arm.phase))
                    _prev_phase_hud = arm.phase

            # Futurist coordination
            if ftr.phase == "WAIT" and t_sim >= 0.15:
                ftr.notify_steady(t_sim)   # lean in and steady the melon
            if cut.cut2_fired and _ftr_settle_end_t is None:
                _ftr_settle_end_t = t_sim + 0.80
                # (a fixed serve-start tick was tried here: episodes
                # then MERGE into identical-trajectory groups — but the
                # global contact solver still couples the randomised
                # melon debris into the arm's floats, so full-system
                # determinism is unreachable in a shared chaotic world)
                if _grasp_contact and _qa_col_gid >= 0:
                    # isolate the wedge on collision bit 2 (tabletop, plate,
                    # right-hand fingers only) and rest the flesh hull on the
                    # tabletop; the weld is cut shortly after so the wedge
                    # sits on the table as a plain free body.
                    model.geom_contype[_qa_col_gid]    = 2
                    model.geom_conaffinity[_qa_col_gid] = 2
                    _qa_rest = cut.get_qA_pos(data).copy()
                    _qa_rest[2] = _QA_TABLE_REST_Z
                    if ftr.actuated_arm and _FIXTURE:
                        # standing pose: underside is 0.090 below the origin
                        # (the 0.663 rest height is for the LYING pose and
                        # buries the standing piece 4 cm into the tabletop —
                        # the weld then fights the table contact and the
                        # freed piece pops out crooked)
                        _qa_rest[2] = 0.705
                        # the README's "fixed serving spot", taken literally:
                        # the freed wedge tumbles from the laid pose to its
                        # standing attractor, and WHERE it lands sets the
                        # azimuth the jaw meets (+43 deg grips the faces;
                        # +160 deg puts the digits on the ridge and the
                        # 70 N squeeze fires the piece). Pinning the spot
                        # makes the tumble - and hence the grasped pose -
                        # deterministic across the randomised placements.
                        _qa_rest[0] = 0.487
                        _qa_rest[1] = -0.358
                    # land rotated 90° about z: boat axis along y, V faces
                    # toward ±x — squeezing the −x face slides the wedge INTO
                    # the palm (a pincer), instead of squirting it sideways
                    if ftr.actuated_arm and _FIXTURE:
                        # STANDING pose, azimuth +43°: the tip-over from the
                        # laid pose to standing is chaotic (float-level
                        # initial differences fan out to ±130° of azimuth,
                        # and at +160° the jaw lands on the ridge and the
                        # squeeze fires the piece). Standing is the observed
                        # attractor — set it down already standing, at the
                        # azimuth the jaw is known to grip cleanly.
                        cut.set_qA_weld_pos(model, data, _qa_rest,
                                            quat=[0.930418, 0.0, 0.0, 0.366501])
                        _qa_rest_pose = np.concatenate(
                            [_qa_rest, [0.930418, 0.0, 0.0, 0.366501]])
                    else:
                        cut.set_qA_weld_pos(model, data, _qa_rest,
                                            quat=[0.70710678, 0.0, 0.0, 0.70710678])
                    # actuated arm: free the wedge only once the hand is
                    # settling above it (GRIP entry) — the dynamic arm's
                    # finite tracking bandwidth must not side-swipe the
                    # free body during the approach
                    _qa_free_t = None if ftr.actuated_arm else t_sim + 0.50
            if (_grasp_contact and not _qa_freed and _qa_free_t is not None
                    and t_sim >= _qa_free_t and _eq_wm_qA_id >= 0):
                data.eq_active[_eq_wm_qA_id] = 0
                _qa_freed = True
            if (_ftr_settle_end_t is not None
                    and t_sim >= _ftr_settle_end_t
                    and not _ftr_triggered):
                ftr.notify_start(t_sim)
                _ftr_triggered = True

            # Per-finger normal forces on the wedge (from last step's contacts)
            # — the tactile signal that stops each finger's curl.
            if _grasp_contact and (_qa_freed
                                   or (ftr.actuated_arm and _FIXTURE)):
                _fF[:] = 0.0
                for _ci in range(data.ncon):
                    _con = data.contact[_ci]
                    if _con.geom1 == _qa_col_gid:
                        _other = _con.geom2
                    elif _con.geom2 == _qa_col_gid:
                        _other = _con.geom1
                    else:
                        continue
                    _fi = _gid2finger.get(int(_other))
                    if _fi is None:
                        continue
                    _f6 = np.zeros(6)
                    mujoco.mj_contactForce(model, data, _ci, _f6)
                    _fF[_fi] += abs(float(_f6[0]))

            # ① Contact-gated FSM: let the controller hold GRIP until the grasp
            # is physically confirmed (weld mode advances on the timer as before).
            ftr.grasp_confirmed = ((bool(_fc_confirmed)
                                    and (_qa_freed or not _FIXTURE))
                                   or (not _grasp_contact))
            ftr.step(data, t_sim)

            # Tactile-servoed finger closure (contact-grasp mode): each finger
            # curls until ITS OWN normal force on the wedge reaches the target,
            # then holds — the wedge is carried by contact + friction alone,
            # with no attachment constraint of any kind.
            if _grasp_contact and ftr.phase in (
                    "GRIP", "LIFT", "CARRY", "LOWER", "CONTACT_PLATE",
                    "RELEASE"):
                if ftr.phase == "GRIP":
                    # first half of GRIP the hand is still settling into the
                    # groove — curl only once the arm pose is in place
                    _gfrac = ((t_sim - _grip_phase_start_t)
                              / _FTR_GRIP_DUR if _grip_phase_start_t else 0.0)
                    for _fi in range(5):
                        if _gfrac < 0.5:
                            pass
                        elif (ftr.actuated_arm and _FIXTURE
                              and sum(_bitten) >= 2):
                            pass   # two anchors formed: stop pressing — the
                                   # fixture is rigid and further curl only
                                   # buries the digits past clean withdrawal
                        elif _fF[_fi] < _FSRV_TGT_N:
                            _fs[_fi] = min(_fs[_fi] + _FSRV_RATE * dt, 1.0)
                        elif _fF[_fi] > _FSRV_MAX_N:
                            _fs[_fi] = max(_fs[_fi] - _FSRV_RATE * 0.5 * dt, 0.0)
                elif ftr.phase == "RELEASE":
                    # gentle withdrawal: digits open at reduced rate and the
                    # embedded tips let go ONE BY ONE as each digit backs out
                    # of the flesh — the wedge's load transfers smoothly to
                    # the plate instead of being scooped out by a fast sweep
                    _fs = np.maximum(
                        _fs - _FSRV_RATE * (0.45 if ftr.actuated_arm else
                                            0.6 if _grasp_friction else 1.5)
                        * dt, _rel_fs_floor)
                    if _grasp_friction:
                        for _fi in range(5):
                            # actuated arm: all embedded digits let go
                            # together — a staggered release leaves the
                            # wedge swinging on a single tip and slings
                            # it off the plate
                            # actuated: open FIRST (pinch force off the
                            # wedge faces), then unbite — releasing while
                            # the digits still squeeze fires the wedge
                            # out like a pinched melon seed
                            _th = (-1.0 if ftr.actuated_arm    # staged below
                                   else 0.15 + 0.15 * _fi)
                            if _bitten[_fi] and _fs[_fi] < _th:
                                data.eq_active[_bite_eids[_fi]] = 0
                                _bitten[_fi] = False
                        if not any(_bitten):
                            model.dof_damping[
                                _qA_dof_adr:_qA_dof_adr + 6] = 0.0
                else:   # carry phases: hold, re-tighten a finger that lost grip
                    if ftr.actuated_arm and any(_bitten):
                        # the embedded bites carry the piece; the digits BACK
                        # OFF so their torque-limited squeeze stops slapping
                        # the hanging wedge (the squeeze-escape-catch cycle
                        # was the energy source that kept the swing alive)
                        _fs[:] = np.minimum(_fs, 0.55)
                    else:
                        for _fi in range(5):
                            if _fF[_fi] < _FSRV_HOLD_N:
                                _fs[_fi] = min(_fs[_fi] + _FSRV_RATE * 0.5 * dt, 1.0)
                if _grasp_friction:
                    # dynamic digits: COMMAND position targets via the
                    # torque-limited actuators — a digit closes until it
                    # stalls on the wedge face; the forcerange, not the
                    # target, sets the sustained pinch force (the digits are
                    # real dynamic bodies, physics owns their state).
                    # The thumb LEADS (fs4 doubled) so the far face is braced
                    # before the fingers start pushing the near one.
                    # thumb LEADS on closing; on RELEASE the lead is
                    # dropped so the strong thumb opens in step with the
                    # fingers instead of squeezing last
                    _th_lead = (1.0 if (ftr.actuated_arm and ftr.phase in
                                        ("CARRY", "LOWER", "CONTACT_PLATE",
                                         "RELEASE")) else 2.0)
                    _fs4 = float(np.clip(_fs[4] * _th_lead, 0.0, 1.0))
                    _fsf = np.clip((_fs[:4] - 0.25) / 0.75, 0.0, 1.0)
                    for _fi, (_pc, _dc) in enumerate(_rf_ctrl_pd):
                        data.ctrl[_pc] = _DYN_OPEN_PROX + _fsf[_fi] * (
                            _DYN_CLOSED_PROX - _DYN_OPEN_PROX)
                        data.ctrl[_dc] = _DYN_OPEN_DIST + _fsf[_fi] * (
                            _DYN_CLOSED_DIST - _DYN_OPEN_DIST)
                    data.ctrl[_rthumb_ctrl] = _FC_OPEN[0] + _fs4 * (
                        _DYN_CLOSED_THUMB - _FC_OPEN[0])
                else:
                    for _fi, (_pa, _da) in enumerate(_rf_pd_adrs):
                        if _pa >= 0:
                            data.qpos[_pa] = _FC_OPEN[0] + _fs[_fi] * (
                                _FC_CLOSED[0] - _FC_OPEN[0])
                        if _da >= 0:
                            data.qpos[_da] = _FC_OPEN[1] + _fs[_fi] * (
                                _FC_CLOSED[1] - _FC_OPEN[1])
                    if _rthumb_adr >= 0:
                        data.qpos[_rthumb_adr] = _FC_OPEN[0] + _fs[4] * (
                            _FC_CLOSED[0] - _FC_OPEN[0])
                # actuated arm, 3-stage release: (a) ZERO-FORCE HOLD —
                # servo every digit to its CURRENT angle so the pinch force
                # decays with the wedge still supported; (b) UNBITE — the
                # tips slide out of the flesh with no stored elastic energy
                # (early unbite fires the wedge like a pinched melon seed,
                # late unbite lets the opening digits fight the constraint);
                # (c) OPEN — the digits withdraw and the wedge sits on the
                # plate under gravity alone.
                if ftr.actuated_arm and ftr.phase == "RELEASE":
                    if _release_start_t is None:
                        _release_start_t = t_sim
                    _rel_t = t_sim - _release_start_t
                    # UNLOAD-then-unbite: the digits open SLOWLY while the
                    # bites still hold the piece, bleeding off the digit-face
                    # preload (a frozen-target "zero-force hold" does NOT
                    # shed it — the residual ~8 N shoved every earlier drop
                    # sideways off the plate). Only when the digit contact
                    # force reads zero are the constraints cut.
                    _digit_load = float(_fF[:5].sum())
                    _digits_open = bool(_FIXTURE
                                        and float(np.max(_fs)) < 0.05)
                    if any(_bitten) and not (
                            _rel_t > 1.20 or (_rel_t > 0.50
                                              and (_digit_load < 0.4
                                                   or _digits_open))):
                        pass          # keep opening; fs decays via the
                                      # RELEASE branch above (floored)
                    elif any(_bitten):
                        if _GRASP_DEBUG:
                            _up = cut.get_qA_pos(data)
                            print(f"      [UNBITE t={t_sim:.3f}] rel_t={_rel_t:.2f} "
                                  f"load={_digit_load:.1f}N "
                                  f"p=({_up[0]:+.3f},{_up[1]:+.3f},{_up[2]:+.3f}) "
                                  f"fd_v={_qa_fd_speed:.3f} fs={_fs[0]:.2f}")
                        for _be in _bite_eids:
                            data.eq_active[_be] = 0
                        for _fi in range(5):
                            _bitten[_fi] = False
                        model.dof_damping[_qA_dof_adr:_qA_dof_adr + 6] = 0.0
                        # while constrained, the wedge's qvel is a pseudo-
                        # velocity (constraint-vs-damping equilibrium, reads
                        # ~1.1 m/s while the piece actually drifts at 5 cm/s);
                        # cutting the constraints TURNS THAT STATE INTO REAL
                        # MOMENTUM and launches the piece. Align the velocity
                        # state with the measured motion at the handover.
                        if _qa_prev_pos is not None:
                            _v_real = (cut.get_qA_pos(data)
                                       - _qa_prev_pos) / dt
                            data.qvel[_qA_dof_adr:_qA_dof_adr + 3] = np.clip(
                                _v_real, -0.3, 0.3)
                            data.qvel[_qA_dof_adr + 3:_qA_dof_adr + 6] = 0.0
                        # the digits have ALREADY opened during the slow
                        # unload (fs is near zero here) — leave them where
                        # they are: re-closing them even part-way clamps the
                        # freed piece and walks it off the plate

                # flesh bite: a digit squeezing ≥ _BITE_N has crushed into
                # the flesh — embed its tip: activate the point CONNECT at
                # the current tip position (both anchors set to the SAME
                # world point → zero residual, no activation impulse)
                if (_grasp_friction and ftr.phase in ("GRIP", "LIFT")
                        and (_qa_freed or (ftr.actuated_arm and _FIXTURE))):
                    for _fi in range(5):
                        if _bitten[_fi] or _fF[_fi] < _BITE_N:
                            continue
                        _be  = _bite_eids[_fi]
                        _tb  = _bite_bodies[_fi]
                        _p   = data.site_xpos[_rtip_sids[_fi]].copy()
                        _R1  = data.xmat[_tb].reshape(3, 3)
                        _R2  = data.xmat[_qa_body_id].reshape(3, 3)
                        model.eq_data[_be, 0:3] = _R1.T @ (_p - data.xpos[_tb])
                        model.eq_data[_be, 3:6] = _R2.T @ (_p - data.xpos[_qa_body_id])
                        data.eq_active[_be] = 1
                        _bitten[_fi] = True
                        # fingertips buried in flesh dissipate relative
                        # motion — viscous drag on the embedded wedge kills
                        # the pendulum swing during the carry
                        model.dof_damping[
                            _qA_dof_adr:_qA_dof_adr + 6] = _BITE_DAMPING
                        # and the digit's own joints are flesh-damped so
                        # the anchor stops flailing under the wedge load
                        # (dynamic-arm mode only: the kinematic mainline's
                        # shipped numbers depend on the light digits)
                        for _dd in (_rf_digit_dofs[_fi]
                                    if ftr.actuated_arm else []):
                            if _dd >= 0 and _dd not in _digit_damp0:
                                _digit_damp0[_dd] = float(
                                    model.dof_damping[_dd])
                                model.dof_damping[_dd] = _DIGIT_FLESH_DAMPING

                # grasp-quality acceptance (actuated arm): measure where
                # the piece rides relative to the wrist shortly after
                # lift-off. A shallow bite hangs it >10 cm below the wrist —
                # outside the joint set-down workspace (right-arm lift plus
                # waiter-palm drop tops out at ~29 cm of compensation) — so
                # put it back on the table and re-grasp instead of carrying
                # an impossible load to the plate.
                # NOTE: disabled — the ride height is still transient at
                # CARRY entry (it misreads the nominal grasp as deep-hung)
                # and bite-point locations carry no signal; a reliable
                # accept/reject variable for grasp quality is the open
                # problem for randomised-placement generalisation
                if False and (ftr.actuated_arm and ftr.phase == "CARRY"
                        and _grasp_start_t is not None
                        and any(_bitten) and ftr.grip_retries < 1
                        and _qa_ride_checked is False):
                    _qa_ride_checked = True
                    _ride = (float(cut.get_qA_pos(data)[2])
                             - float(data.xpos[_rhand_body_id][2]))
                    if _ride < -0.055 and ftr.abort_lift(t_sim):
                        print(f"  GRASP rejected at t={t_sim:.2f}s "
                              f"(rides {_ride*100:+.1f} cm vs wrist; "
                              f"re-grasping, retry {ftr.grip_retries})")
                        for _fi in range(5):
                            if _bitten[_fi]:
                                data.eq_active[_bite_eids[_fi]] = 0
                                _bitten[_fi] = False
                        model.dof_damping[_qA_dof_adr:_qA_dof_adr + 6] = 0.0
                        for _dd, _v in _digit_damp0.items():
                            model.dof_damping[_dd] = _v
                        _digit_damp0.clear()
                        _fs[:] = 0.0
                        _fc_confirmed = False
                        _grip_phase_start_t = None
                        _qA_anim_start_t = None
                        _qa_ride_checked = False

                # stability-gated release: the (physically swinging) wedge
                # must have settled over the plate before CONTACT_PLATE may
                # advance to RELEASE — symmetric to the contact-gated GRIP
                if _grasp_friction:
                    if ftr.phase == "CONTACT_PLATE" and any(_bitten):
                        _wv = float(np.linalg.norm(
                            data.qvel[_qA_dof_adr:_qA_dof_adr + 3]))
                        _wd = float(np.linalg.norm(
                            cut.get_qA_pos(data)[:2]
                            - model.body_pos[_plate_body_id][:2]))
                        _wv_lim, _wd_lim = ((0.09, 0.14) if ftr.actuated_arm
                                            else (0.06, 0.11))
                        _wdz = (cut.get_qA_pos(data)[2]
                                - model.body_pos[_plate_body_id][2])
                        ftr.release_allowed = (_wv < _wv_lim and _wd < _wd_lim)
                    else:
                        ftr.release_allowed = True
                if ftr.phase == "CARRY":
                    _grip_force_carry.append(float(_fF.sum()))
                # closed-loop placement (actuated arm): steer the wrist
                # target so the MEASURED hanging-wedge position converges
                # on the plate centre — corrects grasp-to-grasp variation
                # in where the wedge hangs from the embedded digits
                if (ftr.actuated_arm and any(_bitten)
                        and ftr.phase in ("LOWER", "CONTACT_PLATE")):
                    # SET-DOWN placement: steer the hanging wedge over the
                    # plate centre (xy) and PRESS it onto the disc (z) — the
                    # plate, not the swing gate, then owns the stability:
                    # once seated, plate friction pins the piece and the
                    # digits can withdraw with zero stored energy
                    _qa_now3 = cut.get_qA_pos(data)
                    # hang-depth adaptation, bimanual: the wedge hangs 7-15 cm
                    # below the wrist depending on where the digits embedded,
                    # and the RIGHT arm saturates its IK absorbing that
                    # spread — so the LEFT (waiter) arm brings the plate to
                    # the wedge instead: servo the palm height so the disc
                    # sits 10.5 cm under the hanging piece (+0.29 m/rad on
                    # the shoulder pitch; kinematic arm, no saturation)
                    # hang-depth compensation is ONE-SHOT (see the phase-
                    # transition hook): a continuous plate servo couples to
                    # the pendulum sway and destabilises the nominal case
                    ftr.plate_lift += float(np.clip(
                        _plate_lift_tgt - ftr.plate_lift,
                        -0.30 * dt, 0.30 * dt))
                    _pl_tgt = model.body_pos[_plate_body_id].copy()
                    # ROLL-AHEAD offset: the vertically-hung wedge tips
                    # over to its lying pose after release, translating
                    # ~10 cm of rind arc in the (consistently observed)
                    # −x direction — set it down x-positive of centre so
                    # the roll finishes ON the disc, like laying a
                    # rocking bowl down ahead of its tilt
                    _pl_tgt[0] += 0.075
                    # SEAT-FIRST: set the piece DOWN on the disc (its
                    # vertical-hang underside is 0.090 below the origin;
                    # +0.089 presses the rind ~5 mm into the surface — firm
                    # contact, far from the 4 cm burial that upsets the
                    # convex-mesh collider). A seated piece is anchored by
                    # the plate, so opening the digits truly disengages them
                    # — the hover-drop release was chaotically sensitive
                    # because an unloaded HANGING piece swings with every
                    # digit motion and inherits its pseudo-velocity at the
                    # constraint cut.
                    _pl_tgt[2] += 0.105
                    _pl_err3 = _pl_tgt - _qa_now3
                    if _seated:
                        _pl_err3[2] = ftr.carry_offset[2]   # freeze descent
                    ftr.carry_offset[:] = np.clip(
                        ftr.carry_offset + 1.2 * dt * (
                            _pl_err3 - ftr.carry_offset),
                        (-0.15, -0.15, -0.35), (0.15, 0.15, 0.32))

                # FIXTURE PIN (actuated): while the grasp forms, the wedge
                # is held rigidly at the serving-spot pose by direct state
                # overwrite each step — the weld's rotational impedance
                # cannot resist the jaw's scissor torque about z (the pose
                # was twisted 50-135 deg before the bites formed), and a
                # fixture that yields re-randomises the grasp geometry.
                if (ftr.actuated_arm and cut.cut2_fired and not _qa_freed
                        and _qa_rest_pose is not None):
                    data.qpos[cut._qa_qA:cut._qa_qA + 3] = _qa_rest_pose[:3]
                    data.qpos[cut._qa_qA + 3:cut._qa_qA + 7] = _qa_rest_pose[3:]
                    data.qvel[_qA_dof_adr:_qA_dof_adr + 6] = 0.0

                # Fingertip grasp servo (actuated arm): during the descent
                # onto the wedge, servo the WRIST so the fingertip centroid
                # lands on the wedge's grip point — the waypoint line tracks
                # the wrist, but the task variable that matters is where the
                # JAW is. desired_hand = hand + (grip_point - tips_centroid).
                if (ftr.actuated_arm and ftr.phase in ("REACH", "GRIP")
                        and _rtip_sids
                        and not (_FIXTURE and any(_bitten))):
                    # (servo goes quiet at the FIRST bite: once a tip is
                    # anchored to the fixture-held piece, pulling the
                    # wrist further just winds the arm up around the
                    # anchor — hold still and let the jaw finish closing)
                    _tips_now = np.mean(
                        [data.site_xpos[i] for i in _rtip_sids], axis=0)
                    _grip_pt = cut.get_qA_pos(data) + np.array(
                        [0.0, 0.0, 0.065])          # just above the ridge
                    if ftr.phase == "GRIP" and (_qa_freed or _FIXTURE):
                        _grip_pt[2] = (cut.get_qA_pos(data)[2]
                                       + float(os.environ.get("GRIP_Z_OFF",
                                                              "0.030")))
                    ftr.grasp_target = (data.xpos[_rhand_body_id]
                                        + (_grip_pt - _tips_now))
                elif (ftr.grasp_target is not None and _FIXTURE
                        and any(_bitten) and ftr.phase == "GRIP"):
                    pass    # freeze the last pre-bite target: hold, don't
                            # re-press (waypoint) and don't pull (servo)
                elif ftr.grasp_target is not None:
                    ftr.grasp_target = None

                # GRIP set-point force servo (actuated arm): the descending
                # jaw must press the wedge onto the table just hard enough
                # for the digits to bite (~8 N) — an uncontrolled press
                # (100+ N) squirts the tapered piece sideways off the table.
                # The z carry-offset becomes a force-feedback channel.
                # decompression-gated handover (fixture): cut the pin only
                # once the digit load is low and steady — then the bites
                # inherit a quiet piece instead of a loaded spring
                if (_FIXTURE and _handover_pending and not _qa_freed
                        and _eq_wm_qA_id >= 0):
                    _fd_now = float(_fF[:5].sum())
                    # immediate: measured handover load does NOT predict
                    # the outcome (269 N succeeded, 14 N failed) — but a
                    # decompression wait lets extra digits keep biting
                    # and re-scrambles the grasp geometry
                    if True:
                        if _GRASP_DEBUG:
                            print(f"    [handover t={t_sim:.2f}] "
                                  f"digit load {_fd_now:.1f}N "
                                  f"bites={sum(_bitten)}")
                        data.eq_active[_eq_wm_qA_id] = 0
                        _qa_freed = True
                        _handover_pending = False

                # fixture-mode GRIP force servo: against the pinned piece
                # the digit-wedge contact force is a VALID feedback signal
                # (the table force was not) — clamp the arm's descent so the
                # total digit load stays ~6 N. The 19-65 N that launched
                # every handover was the ARM's 350 N*m servo pressing the
                # whole hand onto an unyielding fixture; the digits alone
                # are torque-limited to ~4 N each.
                if (ftr.actuated_arm and _FIXTURE and ftr.phase == "GRIP"
                        and not _qa_freed):
                    _fd_sum = float(_fF[:5].sum())
                    ftr.carry_offset[2] = float(np.clip(
                        ftr.carry_offset[2]
                        + 0.0025 * dt * (6.0 - _fd_sum) * 60.0,
                        -0.05, 0.10))

                if (ftr.actuated_arm and not _FIXTURE
                        and ftr.phase == "GRIP"
                        and _qa_col_gid >= 0 and _qa_freed):
                    _ft_gid = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_GEOM, "tabletop")
                    _f_tab = 0.0
                    for _ci in range(data.ncon):
                        _cc = data.contact[_ci]
                        if {int(_cc.geom1), int(_cc.geom2)} == {int(_qa_col_gid),
                                                                int(_ft_gid)}:
                            _f6t = np.zeros(6)
                            mujoco.mj_contactForce(model, data, _ci, _f6t)
                            _f_tab += abs(float(_f6t[0]))
                    # 8 N set-point (bite depth varies episode-to-episode
                    # with placement — hang-depth-adaptive set-down is the
                    # open follow-up; pressing harder destabilises the
                    # taper before the bites form)
                    ftr.carry_offset[2] = float(np.clip(
                        ftr.carry_offset[2]
                        + 0.0025 * dt * (8.0 - _f_tab) * 60.0,
                        -0.055, 0.06))

                # REAL wedge speed: data.qvel on the suspended piece is a
                # constraint-vs-damping pseudo-velocity (reads ~1.1 m/s
                # while the piece drifts at 5 cm/s) — gate on the finite-
                # difference of the actual position instead
                if ftr.actuated_arm:
                    _qa_now_fd = cut.get_qA_pos(data)
                    if _qa_prev_pos is not None:
                        _qa_fd_speed = (0.95 * _qa_fd_speed + 0.05 *
                            float(np.linalg.norm(_qa_now_fd - _qa_prev_pos)) / dt)
                    _qa_prev_pos = _qa_now_fd.copy()

                # seated detection: net wedge-plate normal force, debounced
                if (ftr.actuated_arm and _qa_col_gid >= 0
                        and ftr.phase in ("LOWER", "CONTACT_PLATE",
                                          "RELEASE")):
                    _fp = 0.0
                    _pc_gid = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_GEOM, "plate_col")
                    for _ci in range(data.ncon):
                        _cc = data.contact[_ci]
                        if {int(_cc.geom1), int(_cc.geom2)} == {int(_qa_col_gid),
                                                                int(_pc_gid)}:
                            _f6b = np.zeros(6)
                            mujoco.mj_contactForce(model, data, _ci, _f6b)
                            _fp += abs(float(_f6b[0]))
                    if _fp > 0.8:
                        if _seated_since_t is None:
                            _seated_since_t = t_sim
                        elif t_sim - _seated_since_t > 0.25:
                            _seated = True
                    else:
                        _seated_since_t = None
                if (ftr.actuated_arm and ftr.phase not in
                        ("GRIP", "LOWER", "CONTACT_PLATE", "RELEASE")):
                    ftr.carry_offset *= 0.995   # bleed off between uses
                # the flesh damping on the bitten digit joints stabilises
                # the suspension (measured drift ~5 cm/s); no hand-freeze or
                # swing-following needed during the set-down
                if (_GRASP_DEBUG and os.environ.get("WEDGE_CONTACT_DEBUG")
                        and ftr.phase in ("STEADY", "REACH", "GRIP")
                        and s % 100 == 0):
                    _wc = {}
                    for _ci in range(data.ncon):
                        _cc = data.contact[_ci]
                        if _qa_col_gid in (int(_cc.geom1), int(_cc.geom2)):
                            _og = (int(_cc.geom2) if int(_cc.geom1) == _qa_col_gid
                                   else int(_cc.geom1))
                            _f6c = np.zeros(6)
                            mujoco.mj_contactForce(model, data, _ci, _f6c)
                            _on = (mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_GEOM, _og) or f"g{_og}")
                            _wc[_on] = max(_wc.get(_on, 0.0), abs(float(_f6c[0])))
                    if _wc:
                        print(f"      [wedge-contact t={t_sim:5.2f} {ftr.phase}] "
                              + "  ".join(f"{k}:{v:.1f}N" for k, v in _wc.items()))
                if (_GRASP_DEBUG and ftr.phase == "CONTACT_PLATE"
                        and s % 100 == 0):
                    _qp = cut.get_qA_pos(data)
                    _pl = model.body_pos[_plate_body_id]
                    print(f"      [CTR t={t_sim:5.2f}] "
                          f"wedge-plate=({_qp[0]-_pl[0]:+.3f},{_qp[1]-_pl[1]:+.3f},"
                          f"{_qp[2]-_pl[2]:+.3f}) "
                          f"off=({ftr.carry_offset[0]:+.3f},{ftr.carry_offset[1]:+.3f},"
                          f"{ftr.carry_offset[2]:+.3f}) "
                          f"trk={ftr.track_err_m:.3f}m "
                          f"corr={float(np.linalg.norm(ftr._ik_corr)):.2f}rad")
                if (_GRASP_DEBUG and ftr.phase == "RELEASE"
                        and s % 25 == 0):
                    _qp = cut.get_qA_pos(data)
                    _qv = data.qvel[_qA_dof_adr:_qA_dof_adr + 3]
                    print(f"      [REL t={t_sim:5.2f}] "
                          f"p=({_qp[0]:+.3f},{_qp[1]:+.3f},{_qp[2]:+.3f}) "
                          f"v=({_qv[0]:+.2f},{_qv[1]:+.2f},{_qv[2]:+.2f}) "
                          f"fs={_fs[0]:.2f} bite={''.join('X' if b else '.' for b in _bitten)} "
                          f"fF={np.round(_fF[:5],1)}")
                if _GRASP_DEBUG and s % 100 == 0:
                    _dqa = cut.get_qA_pos(data)
                    print(f"      [{ftr.phase:<13s}] t={t_sim:5.2f} "
                          f"qA=({_dqa[0]:+.3f},{_dqa[1]:+.3f},{_dqa[2]:+.3f}) "
                          f"fF={np.round(_fF,1)} fs={np.round(_fs,2)} "
                          f"bite={''.join('X' if b else '.' for b in _bitten)}")

                # Grasp-closure gate: the fingers must (a) be curled shut,
                # (b) have the fingertip cage wrapped around the wedge —
                # geometric enclosure — AND (c) be in physical contact (some
                # non-zero normal force). This is deterministic and robust: it
                # does not hinge on a marginal force threshold whose exact value
                # drifts between the physics-only and render solver passes.
                if (not _fc_confirmed
                        and (_qa_freed or (ftr.actuated_arm and _FIXTURE))
                        and ftr.phase in ("GRIP", "LIFT")):
                    _tips_c   = np.mean([data.site_xpos[i] for i in _rtip_sids],
                                        axis=0)
                    _wrapped  = (float(np.linalg.norm(
                        _tips_c - cut.get_qA_pos(data))) < _QA_COL_R + 0.03)
                    _closed   = float(_fs.min()) > 0.85
                    _touching = float(_fF[:5].sum()) > 0.5
                    if _grasp_friction:
                        # friction mode: the grasp is confirmed once at least
                        # two digits have BITTEN into the flesh (embedded)
                        _closed = _touching = True
                        _wrapped = sum(_bitten) >= 2
                    if _closed and _wrapped and _touching:
                        if _GRASP_DEBUG and _grasp_friction:
                            _wq = data.qpos[cut._qa_qA + 3:cut._qa_qA + 7]
                            _wq = _wq / (np.linalg.norm(_wq) + 1e-12)
                            _w, _x, _y, _z = [float(v) for v in _wq]
                            # wedge local +z in world (apex axis)
                            _az = np.array([
                                2*(_x*_z + _w*_y),
                                2*(_y*_z - _w*_x),
                                _w*_w - _x*_x - _y*_y + _z*_z])
                            _upright = float(np.degrees(np.arccos(
                                np.clip(_az[2], -1, 1))))
                            _azim = float(np.degrees(np.arctan2(_az[1], _az[0])))
                            print(f"    grasp pose: upright={_upright:5.1f} deg "
                                  f"azim={_azim:+6.1f} deg "
                                  f"axis=({_az[0]:+.2f},{_az[1]:+.2f},{_az[2]:+.2f})")
                        _fc_confirmed   = True
                        _fc_confirmed_t = t_sim
                        if (ftr.actuated_arm and _FIXTURE and not _qa_freed
                                and _eq_wm_qA_id >= 0):
                            # arm the handover; the pin is cut only after
                            # the decompression servo has bled the arm's
                            # press-in to digit scale (an immediate cut
                            # fires the piece with whatever 10-270 N the
                            # descent happened to be carrying)
                            _handover_pending = True
                        _qa_hold_quat   = data.qpos[
                            cut._qa_qA + 3:cut._qa_qA + 7].copy()
                        # friction mode: NO constraint — from here on the
                        # wedge is carried by the pinch alone
                        if _eq_wm_qA_id >= 0 and not _grasp_friction:
                            data.eq_active[_eq_wm_qA_id] = 1
                        _grip_activated = True
                        _grasp_start_t  = t_sim
                        _grasp_from_pos = cut.get_qA_pos(data).copy()
                        # rigid transform of the held wedge in the hand frame,
                        # captured at the confirmed grasp — the wedge then rides
                        # exactly where the fingers gripped it
                        _Rh = data.xmat[_rhand_body_id].reshape(3, 3)
                        _qa_hold_local = _Rh.T @ (
                            _grasp_from_pos - data.xpos[_rhand_body_id])

            # Plate rides ON the left palm: position follows the hand's local
            # frame (so the arm can never sweep through it); the disc itself
            # stays level like a waiter's tray.
            if _plate_body_id >= 0 and _lhand_body_id >= 0:
                _Rlh = data.xmat[_lhand_body_id].reshape(3, 3)
                model.body_pos[_plate_body_id] = (
                    data.xpos[_lhand_body_id] + _Rlh @ _PLATE_OFF_LOCAL)

            if ftr.phase != _ftr_prev_phase:
                if _ftr_prev_phase == "GRIP" and ftr.phase == "REACH":
                    # grasp retry: reopen digits, release any partial bites
                    _fs[:] = 0.0
                    _grip_phase_start_t = None
                    if _grasp_friction:
                        for _fi in range(5):
                            if _bitten[_fi]:
                                data.eq_active[_bite_eids[_fi]] = 0
                                _bitten[_fi] = False
                        model.dof_damping[_qA_dof_adr:_qA_dof_adr + 6] = 0.0
                    print(f"  GRIP retry {ftr.grip_retries} at t={t_sim:.2f}s "
                          f"(grasp unconfirmed, re-approaching)")
                if args.quick:
                    _dbg_tc = (np.mean([data.site_xpos[i] for i in _rtip_sids], axis=0)
                               if _rtip_sids else data.xpos[_hand_body_id])
                    _dbg_qa = cut.get_qA_pos(data)
                    print(f"    ftr {_ftr_prev_phase:>13s}→{ftr.phase:<13s} t={t_sim:6.2f}"
                          f"  tips_c=({_dbg_tc[0]:+.3f},{_dbg_tc[1]:+.3f},{_dbg_tc[2]:+.3f})"
                          f"  qA=({_dbg_qa[0]:+.2f},{_dbg_qa[1]:+.2f},{_dbg_qa[2]:+.2f})"
                          f"  ik={np.linalg.norm(ftr._ik_corr):.2f}")
                if ftr.phase == "GRIP" and _grip_phase_start_t is None:
                    _grip_phase_start_t = t_sim
                    if not _FIXTURE and (_grasp_contact and ftr.actuated_arm
                            and _qa_free_t is None and not _qa_freed):
                        _qa_free_t = t_sim + 0.10
                elif ftr.phase == "LIFT" and not _qA_anim_start_t:
                    _qA_anim_start_t   = t_sim  # camera timing ref
                    _qA_anim_start_pos = cut.get_qA_pos(data).copy()
                elif ftr.phase == "LOWER" and _qA_lower_start_t is None:
                    _qA_lower_start_t  = t_sim
                    _plate_cam_start_t = t_sim
                elif (ftr.phase == "CONTACT_PLATE" and ftr.actuated_arm
                        and _rtip_sids and any(_bitten)):
                    # measure THIS grasp's carry offset once (wedge z
                    # relative to the wrist: +6 cm = perched on the digits,
                    # the nominal case; negative = hanging below them), then
                    # glide the waiter palm so the disc meets the piece
                    _rel_off = (float(cut.get_qA_pos(data)[2])
                                - float(data.xpos[_rhand_body_id][2]))
                    _plate_lift_tgt = float(np.clip(
                        (_rel_off - 0.040) / 0.29, -0.75, 0.04))
                    if abs(_plate_lift_tgt) < 0.05:
                        _plate_lift_tgt = 0.0   # nominal grasp: hands off
                    if _GRASP_DEBUG:
                        print(f"    carry offset {_rel_off*100:+.1f} cm -> "
                              f"plate lift target {_plate_lift_tgt:+.2f} rad")
                elif ftr.phase == "RETRACT" and _plate_retract_t is None:
                    _plate_retract_t = t_sim
                _ftr_prev_phase = ftr.phase

            # Kinematic grasp via wm_qA_weld (world weld, always active):
            # after GRIP completes, its target tracks the right hand every step,
            # so the quarter moves exactly with the arm. ftr_grip_weld (body-body
            # weld) is NOT used — activating it mid-sim explodes the constraint.
            if (not _grasp_contact and not _grip_activated
                    and ftr.grip_should_activate):
                _grip_activated  = True
                _grasp_start_t   = t_sim
                _grasp_from_pos  = cut.get_qA_pos(data).copy()

            if (_grip_activated and not _qA_released and _rtip_sids
                    and ftr.phase in ("GRIP", "LIFT", "CARRY")):
                # multi-finger closure metric: fingertip→wedge-surface distances
                _qa_c = cut.get_qA_pos(data)
                _tds = [float(np.linalg.norm(data.site_xpos[i] - _qa_c)) - _QA_COL_R
                        for i in _rtip_sids]
                _closure_n  = sum(d < 0.030 for d in _tds)
                _closure_mm = float(np.mean(_tds)) * 1000.0
                if _grasp_friction and any(_bitten):
                    # friction mode: report EMBEDDED digits, the true hold
                    _closure_n = int(sum(_bitten))

            # Contact-grasp mode: the grasp is CONFIRMED by real tactile force
            # closure (≥K fingertips over threshold, gated above). Once
            # confirmed, the wedge is held at the fingertip centroid so it
            # rides IN the closed fingers, in full view of the fixed camera,
            # and is laid onto the plate the left hand presents.
            if _grasp_contact and _grip_activated and not _qA_released:
                if ftr.grip_should_release:
                    _plate_now = model.body_pos[_plate_body_id].copy()
                    _serve_err_mm = float(np.linalg.norm(
                        cut.get_qA_pos(data)[:2] - _plate_now[:2])) * 1000.0
                    if _grasp_friction:
                        # fingers withdraw from the flesh: bites release
                        for _be in _bite_eids:
                            data.eq_active[_be] = 0
                        _bitten = [False] * 5
                        model.dof_damping[_qA_dof_adr:_qA_dof_adr + 6] = 0.0
                    _qA_released = True
                elif _grasp_friction:
                    pass   # bites + pinch carry the wedge, no tracking needed
                else:
                    _tc = np.mean([data.site_xpos[i] for i in _rtip_sids],
                                  axis=0)
                    _fd = _tc - data.xpos[_rhand_body_id]
                    _fd /= max(float(np.linalg.norm(_fd)), 1e-6)
                    _tgt = _tc + _fd * 0.030
                    if ftr.phase in ("LOWER", "CONTACT_PLATE", "RELEASE"):
                        if _lower_blend_t is None:
                            _lower_blend_t = t_sim
                        _lb = _ss(float(np.clip(
                            (t_sim - _lower_blend_t) / 1.0, 0.0, 1.0)))
                        _over_plate = (model.body_pos[_plate_body_id]
                                       + np.array([0.0, 0.0, 0.113]))
                        _tgt = _tgt * (1.0 - _lb) + _over_plate * _lb
                    _ga = _ss(float(np.clip(
                        (t_sim - _grasp_start_t) / 0.30, 0.0, 1.0)))
                    cut.set_qA_weld_pos(
                        model, data,
                        _grasp_from_pos * (1.0 - _ga) + _tgt * _ga,
                        quat=_qa_hold_quat)

            # After release the wedge rests ON the plate and CONTINUOUSLY tracks
            # it (the plate rides the retreating left palm) — so the served
            # quarter stays visibly seated on the disc through to DONE, rind
            # curving down onto the surface. Friction mode skips this: where
            # the wedge lands and settles is decided purely by physics, and
            # the delivery error below is a fully physical measurement.
            if (_grasp_contact and not _grasp_friction
                    and _qA_released and _plate_body_id >= 0):
                _plate_now = model.body_pos[_plate_body_id].copy()
                cut.set_qA_weld_pos(
                    model, data, _plate_now + np.array([0.0, 0.0, 0.113]),
                    quat=[0.70710678, 0.0, 0.0, 0.70710678])
            if _grasp_friction and _qA_released and _plate_body_id >= 0:
                _serve_err_mm = float(np.linalg.norm(
                    cut.get_qA_pos(data)[:2]
                    - model.body_pos[_plate_body_id][:2])) * 1000.0

            if (not _grasp_contact) and _grip_activated and not _qA_released:
                if ftr.grip_should_release:
                    # Closed-loop placement: rest the quarter on the plate's
                    # CURRENT position (it rides the left palm), not a constant.
                    _plate_now = model.body_pos[_plate_body_id].copy()
                    _rest = _plate_now + np.array([0.0, 0.0, 0.094])
                    # delivery error: horizontal miss between where the hand
                    # delivered the quarter and the plate centre
                    _qa_now = cut.get_qA_pos(data)
                    _serve_err_mm = float(np.linalg.norm(
                        _qa_now[:2] - _plate_now[:2])) * 1000.0
                    cut.set_qA_weld_pos(model, data, _rest)
                    _qA_released = True
                else:
                    # Held target, staged:
                    #  GRIP..CARRY — fingertip-centroid enclosure, pushed out
                    #    along the wrist→tips direction so the tips rest ON
                    #    the wedge surface
                    #  held in the finger enclosure all the way to RELEASE —
                    #    the left hand brings the plate to the hand-off point
                    if _rtip_sids:
                        _tc = np.mean([data.site_xpos[i] for i in _rtip_sids],
                                      axis=0)
                        _fd = _tc - data.xpos[_rhand_body_id]
                        _fd /= max(float(np.linalg.norm(_fd)), 1e-6)
                        # push the wedge centre 4.8 cm beyond the tip ring so
                        # the tips rest ON the 6.5 cm-radius surface
                        # (sqrt(65^2 - 43^2) ≈ 48 mm)
                        _tgt = _tc + _fd * 0.048
                        # placement: blend the held point onto the spot 5.5 cm
                        # above the ACTUAL plate centre (the left hand holds
                        # the plate right under the fingers) — the wedge stays
                        # inside the finger cage and drops straight down at
                        # release
                        if ftr.phase in ("LOWER", "CONTACT_PLATE", "RELEASE"):
                            if _lower_blend_t is None:
                                _lower_blend_t = t_sim
                            _lb = _ss(float(np.clip(
                                (t_sim - _lower_blend_t) / 1.0, 0.0, 1.0)))
                            _over_plate = (model.body_pos[_plate_body_id]
                                           + np.array([0.0, 0.0, 0.055]))
                            _tgt = _tgt * (1.0 - _lb) + _over_plate * _lb
                    else:
                        _tgt = data.xpos[_hand_body_id] + _QA_GRIP_OFFSET
                    # short blend so the quarter is drawn from the table into
                    # the hand instead of teleporting
                    _ga = _ss(float(np.clip((t_sim - _grasp_start_t) / 0.30, 0.0, 1.0)))
                    cut.set_qA_weld_pos(model, data,
                                        _grasp_from_pos * (1.0 - _ga) + _tgt * _ga)

            # Waist lean
            if arm.phase == "APPROACH":
                _lean = 0.0
            elif arm.phase in ("ALIGN", "CONTACT", "SLICE") and _align_start_t is not None:
                _lean = -_LEAN_MAX * _ss(min((t_sim - _align_start_t) / _LEAN_ADUR, 1.0))
            elif arm.phase == "RETRACT" and _retract_start_t is not None:
                _lean = -_LEAN_MAX * (1.0 - _ss((t_sim - _retract_start_t) / _LEAN_RDUR))
            elif arm.phase in ("DONE", "REGRASP", "REPOSITION2", "APPROACH2"):
                _lean = 0.0
            elif arm.phase in ("ALIGN2", "CONTACT2", "SLICE2") and _align2_start_t is not None:
                _lean = -_LEAN_MAX * _ss(min((t_sim - _align2_start_t) / _LEAN_ADUR, 1.0))
            elif arm.phase == "RETRACT2" and _retract2_start_t is not None:
                _lean = -_LEAN_MAX * (1.0 - _ss((t_sim - _retract2_start_t) / _LEAN_RDUR))
            else:
                _lean = 0.0

            data.ctrl[5] = 0.0
            data.ctrl[6] = _lean
            data.ctrl[7] = 0.0

            if arm.phase in ("APPROACH", "REGRASP", "REPOSITION2", "APPROACH2"):
                _head_pitch = -0.10
            elif arm.phase in ("ALIGN", "CONTACT", "SLICE", "ALIGN2", "CONTACT2", "SLICE2"):
                _head_pitch = -0.28
            elif arm.phase == "RETRACT" and _retract_start_t is not None:
                _head_pitch = -0.28 * (1.0 - _ss((t_sim - _retract_start_t) / _LEAN_RDUR))
            elif arm.phase == "RETRACT2" and _retract2_start_t is not None:
                _head_pitch = -0.28 * (1.0 - _ss((t_sim - _retract2_start_t) / _LEAN_RDUR))
            else:
                _head_pitch = 0.0
            data.ctrl[8] = 0.0
            data.ctrl[9] = _head_pitch

            if arm.phase in ("ALIGN", "CONTACT", "SLICE",
                             "ALIGN2", "CONTACT2", "SLICE2"):
                _g_alpha = _ss(float(np.clip(
                    (t_sim - (_align_start_t or t_sim)) / 1.00, 0.0, 1.0)))
                data.ctrl[10:15] = (_LEFT_ARM_NATURAL * (1.0 - _g_alpha)
                                    + _LEFT_GUARD * _g_alpha)
            else:
                data.ctrl[10:15] = _LEFT_ARM_NATURAL
                _wm_stab_ref = None

            if arm.phase in ("CONTACT", "SLICE", "CONTACT2", "SLICE2"):
                if _wm_stab_ref is None:
                    _wm_stab_ref = _arm_target_xyz.copy()
                _wm_err  = _arm_target_xyz - _wm_stab_ref
                data.ctrl[10] = float(np.clip(
                    data.ctrl[10] + float(np.clip(-_wm_err[1] * 4.0, -0.10, 0.10)), -1.2, 0.5))
                data.ctrl[11] = float(np.clip(
                    data.ctrl[11] + float(np.clip( _wm_err[0] * 3.0, -0.08, 0.08)), -0.8, 0.8))

            if _split1_countdown > 0:
                _sp1_alpha = _ss((_SPLIT_DUR - _split1_countdown) / max(_SPLIT_DUR - 1, 1))
                _sp1_pos   = (cut._split1_R_start
                              + _sp1_alpha * (cut._split1_R_final - cut._split1_R_start))
                _sp1_pos   = _sp1_pos.copy()
                _sp1_pos[2] += 0.018 * float(np.sin(np.pi * _sp1_alpha))
                cut.set_R_weld_pos(model, data, _sp1_pos)
                _split1_countdown -= 1
            if _split2_countdown > 0:
                _sp2_alpha = _ss((_SPLIT_DUR - _split2_countdown) / max(_SPLIT_DUR - 1, 1))
                _sp2_pos   = (cut._split2_qB_start
                              + _sp2_alpha * (cut._split2_qB_final - cut._split2_qB_start))
                _sp2_pos   = _sp2_pos.copy()
                _sp2_pos[2] += 0.018 * float(np.sin(np.pi * _sp2_alpha))
                cut.set_qB_weld_pos(model, data, _sp2_pos)
                _split2_countdown -= 1

            mujoco.mj_step(model, data)

            # Blade contact force
            touch_N = 0.0
            for _ci in range(data.ncon):
                _c = data.contact[_ci]
                if (_c.geom1 in (_blade_edge_id, _blade_geom_id) or
                        _c.geom2 in (_blade_edge_id, _blade_geom_id)):
                    mujoco.mj_contactForce(model, data, _ci, _fbuf)
                    touch_N += abs(float(_fbuf[0]))

            if not cut.cut_fired and cut.step(model, data,
                                               touch_N=touch_N,
                                               blade_speed_ms=blade_speed_ms):
                cut_frame = True
                cut_t     = t_sim
                _cut1_force = max(touch_N, 1.0)
                print(f"  CUT  at t={t_sim:.3f}s  phase={arm.phase}  "
                      f"blade-WM={blade_dist*100:.1f}cm  F={_cut1_force:.0f}N")
                arm.notify_cut(t_sim)
                ep_log.mark_success(t_sim)
                _split1_countdown = _SPLIT_DUR
                if not args.quick:
                    _juice1.burst(model, _cut1_force)

            if cut.cut_fired and not cut.cut2_fired and _done_reached_t is not None:
                if cut.step2(model, data, touch_N=touch_N, blade_speed_ms=blade_speed_ms):
                    cut2_frame = True
                    cut2_t     = t_sim
                    _cut2_force = max(touch_N, 1.0)
                    print(f"  CUT2 at t={t_sim:.3f}s  phase={arm.phase}  "
                          f"blade-half={blade_dist2*100:.1f}cm  F={_cut2_force:.0f}N")
                    arm.notify_cut2(t_sim)
                    _split2_countdown = _SPLIT_DUR
                    if not args.quick:
                        _juice2.burst(model, _cut2_force)

            posture_rms = float(np.sqrt(np.mean(data.qpos[_POSTURE_SLICE] ** 2)))
            grip_pct    = float(np.clip(
                1.0 - np.mean(np.abs(data.ctrl[15:30]) / (_GRIP_OPEN_ABS + 1e-6)),
                0.0, 1.0))

            # Cut quality: calibrated thresholds, contact-frames only
            if arm.phase in ("CONTACT", "SLICE", "CONTACT2", "SLICE2") and touch_N > 10:
                _tq = max(0.0, 1.0 - blade_tilt_deg / 30.0)
                _gq = max(0.0, 1.0 - blade_gyro_dps / 200.0)
                _fq = max(0.0, 1.0 - max(0.0, touch_N - 200.0) / 200.0)
                _sq = float(np.clip(blade_speed_ms / 0.15, 0.0, 1.0))
                _cut_quality = (_tq * 0.35 + _gq * 0.30 + _fq * 0.20 + _sq * 0.15) * 100.0
            else:
                _cut_quality = 0.0

            fail = ep_log.step(t_sim, arm.phase, blade_dist, touch_N, posture_rms,
                               cut_quality=_cut_quality)
            if fail and current_failure is None:
                current_failure = fail
                print(f"  FAIL [{fail}] at t={t_sim:.3f}s")

            if args.collect:
                ep_data.append([
                    round(t_sim, 4), arm.phase, ftr.phase,
                    *[round(float(data.ctrl[i]), 5) for i in range(5)],
                    *[round(float(data.ctrl[i]), 5) for i in range(15, 30)],
                    *[round(float(v), 4) for v in blade_xyz],
                    *[round(float(v), 4) for v in wm_xyz],
                    round(float(blade_dist), 4), round(float(blade_speed_ms), 4),
                    round(float(touch_N), 2),    round(float(grip_pct), 4),
                    round(float(posture_rms), 5), round(float(blade_accel_g), 4),
                    round(float(blade_tilt_deg), 3), round(float(blade_gyro_dps), 2),
                    round(float(_material_k), 1),
                    *[round(float(data.sensordata[a]), 2) for a in adr_gh],
                    round(float(ftr.track_err_m), 4), round(float(ftr._bow), 3),
                    _closure_n,
                    round(float(_closure_mm), 1) if _closure_mm == _closure_mm else "",
                ])

            # ── Render ─────────────────────────────────────────────────
            if args.quick:
                continue

            _slo_mo    = False
            _curr_skip = step_skip
            if s % _curr_skip != 0:
                continue

            # Camera — fixed wide view, with a gentle serving push-in.
            # Same viewpoint (azimuth/elevation unchanged): while Futurist
            # carries and plates the wedge, the camera smoothly dollies in on
            # the hand-off zone so the served wedge is clearly readable, then
            # eases back out. It is one continuous move, never a cut.
            _serve_zoom_phases = ("LIFT", "CARRY", "LOWER",
                                  "CONTACT_PLATE", "RELEASE")
            _zoom_want = 1.0 if ftr.phase in _serve_zoom_phases else 0.0
            # ease the zoom state toward the target (~0.7 s in/out)
            _zoom_state = _zoom_state + np.clip(
                _zoom_want - _zoom_state, -1.0, 1.0) * min(
                1.0, _curr_skip * model.opt.timestep / 0.70)
            _z = _ss(float(np.clip(_zoom_state, 0.0, 1.0)))
            _serve_lookat = np.array([0.55, -0.34, 1.02])   # hand-off zone
            cam.lookat[:] = (np.array(_CAM_MAIN["lookat"]) * (1.0 - _z)
                             + _serve_lookat * _z)
            cam.distance  = _CAM_MAIN["dist"] * (1.0 - _z) + 1.55 * _z
            cam.azimuth   = _CAM_MAIN["az"]
            cam.elevation = _CAM_MAIN["el"]

            # Juice ballistic
            _dt_step = _curr_skip * model.opt.timestep
            _juice1.step(model, _dt_step)
            _juice2.step(model, _dt_step)

            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()

            blade_glow(model, _blade_geom_id, _blade_rgba0, blade_speed_ms,
                       active=arm.phase in ("ALIGN", "CONTACT", "SLICE",
                                            "ALIGN2", "CONTACT2", "SLICE2"))

            renderer_top.update_scene(data, camera=cam_top)
            inset = renderer_top.render()
            pixels = paste_inset(pixels, inset, _ix0_ins, _iy0_ins)

            _force_hist_hud.append((t_sim, touch_N))
            _left_stab = arm.phase in (
                "ALIGN", "CONTACT", "SLICE", "ALIGN2", "CONTACT2", "SLICE2")
            pixels = draw_hud(pixels, ep, arm.phase, t_sim,
                              _hud_dist, touch_N, posture_rms,
                              cut.cut_fired, current_failure,
                              grip_pct, blade_speed_ms, touch_sensor,
                              cut.cut2_fired, slo_mo=_slo_mo,
                              cut_t=cut_t, cut2_t=cut2_t,
                              phase_history=_phase_history_hud,
                              force_hist=_force_hist_hud,
                              impact_g=blade_accel_g,
                              blade_tilt_deg=blade_tilt_deg,
                              blade_gyro_dps=blade_gyro_dps,
                              cut_quality=_cut_quality,
                              ft_contacts=ft_contacts,
                              wrist_corr=_wrist_corr,
                              left_stab=_left_stab,
                              gh_forces=gh_forces,
                              knife_slip_mm=knife_slip_mm,
                              material_k=_material_k,
                              material_conf=_material_conf,
                              slice_adapted=_slice_adapted,
                              tear_level=cut.tear_level,
                              crack_front_mm=cut.crack_front_mm,
                              front_width_mm=cut.front_width_mm,
                              local_release_prob=cut.local_release_prob,
                              front_speed_mm_s=cut.front_speed_mm_s,
                              front_accel_mm_s2=cut.front_accel_mm_s2,
                              n_episodes=n_ep,
                              episode_dur=EPISODE_DUR,
                              ftr_phase=ftr.phase,
                              serve_err_mm=_serve_err_mm,
                              grasp_closure=(_closure_n, _closure_mm),
                              regrasp_round=arm.regrasp_round,
                              grip_retry=ftr.grip_retries)

            writer.append_data(pixels)
            if cut_frame or cut2_frame:
                for _flash in (0.65, 0.42, 0.22, 0.10, 0.04):
                    _white  = np.ones_like(pixels, dtype=np.float32) * 255.0
                    _fframe = (pixels.astype(np.float32) * (1.0 - _flash)
                               + _white * _flash).clip(0, 255).astype(np.uint8)
                    writer.append_data(_fframe)
            cut_frame  = False
            cut2_frame = False

        # ── End of episode ─────────────────────────────────────────────
        ep_log.finalize(success=cut.cut_fired, cut_time=cut_t)
        run_log.add(ep_log)
        _k_per_episode.append(_material_k)

        if args.collect and ep_data:
            import csv as _csv
            csv_path = _OUT / f"demo_ep{ep + 1:02d}.csv"
            with open(csv_path, "w", newline="") as _f:
                _cw = _csv.writer(_f)
                _fn = [f"f{fi}_{jn}" for fi in range(5) for jn in ("mcp", "pip", "dip")]
                _cw.writerow(["t", "phase", "ftr_phase",
                               "ctrl_sp", "ctrl_sr", "ctrl_sy", "ctrl_el", "ctrl_wr",
                               *_fn, "bx", "by", "bz", "wx", "wy", "wz",
                               "blade_dist", "blade_speed_ms", "touch_N",
                               "grip_pct", "posture_rms",
                               "blade_accel_g", "blade_tilt_deg", "blade_gyro_dps",
                               "material_k_Npm",
                               "gh_index_N", "gh_middle_N", "gh_ring_N",
                               "gh_pinky_N", "gh_thumb_N",
                               "ftr_ik_err_m", "ftr_bow_rad",
                               "grasp_closure_n", "grasp_tip_surf_mm"])
                _cw.writerows(ep_data)
            print(f"  Demo data → {csv_path}  ({len(ep_data)} steps)")

        # RETRACT counts: the wedge is released and the landing is final —
        # only the arm's return to idle remains (MuJoCo-version timing
        # differences can leave the episode a phase short of DONE)
        _serve_ok = _qA_released and ftr.phase in ("RETRACT", "DONE")
        if _grasp_contact and _serve_ok:
            # physically verify the wedge came to rest ON the plate
            _qa_fin    = cut.get_qA_pos(data)
            _plate_fin = model.body_pos[_plate_body_id]
            if _grasp_friction:
                # fully physical landing: on the plate in ANY resting
                # orientation (a round-bottomed wedge may settle lying down)
                _serve_err_mm = float(np.linalg.norm(
                    _qa_fin[:2] - _plate_fin[:2])) * 1000.0
                # the wedge's body origin is at the ⋀ APEX: lying on a cut
                # face puts the origin ~0-5 cm above the plate, upright
                # ~11 cm — anything in that band and inside the rim is ON it
                # plate disc radius is 138 mm — anything inside 135 mm is
                # physically resting on the disc
                _serve_ok = (_serve_err_mm < 135.0
                             and -0.01 < float(_qa_fin[2] - _plate_fin[2]) < 0.16)
                if _GRASP_DEBUG:
                    print(f"  [land] qA={np.round(_qa_fin,3)} "
                          f"plate={np.round(_plate_fin,3)} "
                          f"dz={float(_qa_fin[2]-_plate_fin[2]):+.3f} "
                          f"xy={_serve_err_mm:.0f}mm ok={_serve_ok}")
            else:
                _serve_ok = (_serve_err_mm < 130.0
                             and abs(float(_qa_fin[2] - _plate_fin[2]) - 0.085)
                             < 0.06)
        _k_serve_errs.append(round(_serve_err_mm, 1) if _serve_ok else None)
        _k_closures.append([_closure_n, round(_closure_mm, 1)]
                           if _closure_mm == _closure_mm else None)
        _k_grip_forces.append(round(float(np.mean(_grip_force_carry)), 1)
                              if _grip_force_carry else None)
        if args.quick:
            _ok = "OK" if cut.cut_fired else "FAIL"
            _ct = f"{cut_t:.3f}s" if cut_t else "—"
            _se = f"{_serve_err_mm:.1f}mm" if _serve_ok else "—"
            _gf = (f"{np.mean(_grip_force_carry):.1f}N"
                   if _grip_force_carry else "—")
            _pol = (f"  policy={_policy_steps/max(_master_steps,1)*100:.0f}%"
                    if _bcpol is not None else "")
            _rst = (f"  restrikes={arm.restrike_count}"
                    if arm.restrike_count else "")
            print(f"  Ep {ep+1}: {_ok}  cut={_ct}  F={ep_log.max_force:.0f}N  "
                  f"quality={ep_log.cut_quality_mean:.1f}%  k={_material_k:.0f}N/m  "
                  f"ftr={ftr.phase}  serve_err={_se}  grip={_gf}{_pol}{_rst}")
            continue

        # Freeze frame
        _cut_stars = 0
        if cut.cut_fired and cut_t is not None:
            _cut_stars += int(cut_t < 3.0)
            _cut_stars += int(cut_t < 2.5)
            _cut_stars += int(ep_log.max_force < 300)
            _cut_stars += int(ep_log.posture_rms_mean < 0.10)
            _cut_stars += int(cut.cut2_fired)
            _cut_stars  = min(5, _cut_stars)

        cam.lookat[:] = _CAM_MAIN["lookat"]
        cam.distance  = _CAM_MAIN["dist"]
        cam.azimuth   = _CAM_MAIN["az"]
        cam.elevation = _CAM_MAIN["el"]
        model.geom_rgba[_blade_geom_id, :3] = _blade_rgba0[:3]
        renderer.update_scene(data, camera=cam)
        freeze_frame = renderer.render()
        renderer_top.update_scene(data, camera=cam_top)
        _inset_top = renderer_top.render()
        freeze_frame = paste_inset(freeze_frame, _inset_top, _ix0_ins, _iy0_ins)
        freeze_hud = draw_hud(freeze_frame, ep, arm.phase, EPISODE_DUR,
                              _hud_dist, touch_N, posture_rms,
                              cut.cut_fired, current_failure,
                              grip_pct, blade_speed_ms, touch_sensor,
                              cut.cut2_fired,
                              cut_t=cut_t, cut2_t=cut2_t,
                              phase_history=_phase_history_hud,
                              force_hist=_force_hist_hud,
                              cut_stars=_cut_stars,
                              impact_g=_peak_impact_g,
                              blade_tilt_deg=blade_tilt_deg,
                              blade_gyro_dps=blade_gyro_dps,
                              cut_quality=ep_log.cut_quality_mean,
                              gh_forces=gh_forces,
                              knife_slip_mm=knife_slip_mm,
                              material_k=_material_k,
                              material_conf=_material_conf,
                              slice_adapted=_slice_adapted,
                              tear_level=0, crack_front_mm=0.0,
                              front_width_mm=0.0, local_release_prob=0.0,
                              front_speed_mm_s=0.0, front_accel_mm_s2=0.0,
                              n_episodes=n_ep,
                              episode_dur=EPISODE_DUR,
                              ftr_phase=ftr.phase,
                              serve_err_mm=_serve_err_mm,
                              grasp_closure=(_closure_n, _closure_mm))
        for _ in range(round(FREEZE_DUR * fps)):
            writer.append_data(freeze_hud)

    # ── Run wrap-up ────────────────────────────────────────────────────
    if not args.quick:
        for f in summary_card(w, h, run_log._episodes, fps,
                              serve_errs_mm=_k_serve_errs,
                              k_per_episode=_k_per_episode,
                              wm_offsets=WM_OFFSETS, episode_dur=EPISODE_DUR,
                              duration_s=2.0):
            writer.append_data(f)
        writer.close()
        renderer.close()
        renderer_top.close()
        recompress(_VIDEO)
        print(f"Saved → {_VIDEO}")

    run_log.print_summary()

    import json as _json, datetime as _dt
    _cut_times = [e.cut_time for e in run_log._episodes if e.cut_time is not None]
    _n_ok = sum(1 for e in run_log._episodes if e.success)
    summary = {
        "submission":     "FF Master + Futurist — Bimanual Watermelon Quartering",
        "timestamp_utc":  _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_episodes":     len(run_log._episodes),
        "success_rate":   round(_n_ok / len(run_log._episodes), 4),
        "n_success":      _n_ok,
        "avg_cut_time_s": round(float(np.mean(_cut_times)), 4) if _cut_times else None,
        "std_cut_time_s": round(float(np.std(_cut_times)), 4) if len(_cut_times) > 1 else None,
        "avg_posture_rms_rad": round(
            float(np.mean([e.posture_rms_mean for e in run_log._episodes])), 5),
        "avg_cut_quality_pct": round(
            float(np.mean([e.cut_quality_mean for e in run_log._episodes
                           if e.cut_quality_mean > 0])), 1) if _n_ok else None,
        "wm_seed":        args.seed,
        "wm_offsets_cm":  [[round(dx * 100, 1), round(dy * 100, 1)]
                           for dx, dy in WM_OFFSETS[:len(run_log._episodes)]],
        "render": {
            "video_path": str(_VIDEO.name) if not args.quick else None,
            "fps": args.fps, "width": w, "height": h,
        },
        "sensor_types":   ["touch", "velocimeter", "accelerometer", "gyro",
                           "framequat", "framelinvel", "framepos"],
        "n_sensors":      10,
        "n_control_phases": 22,
        "robots":         ["FF Master (cuts)", "Futurist (serves)"],
        "material_k_per_episode_Npm": [round(float(k), 1) for k in _k_per_episode],
        "material_k_final_Npm": round(float(_k_per_episode[-1]), 1) if _k_per_episode else None,
        "material_profile": args.material,
        "grasp_mode": args.grasp,
        "futurist_arm_drive": args.futurist_drive,
        "serve_delivery_err_mm": [round(e, 1) if e is not None else None
                                  for e in _k_serve_errs],
        "grasp_closure_fingertips": _k_closures,
        "grip_force_N_carry": _k_grip_forces,
        "serve_success_rate": (sum(e is not None for e in _k_serve_errs)
                               / len(_k_serve_errs)) if _k_serve_errs else None,
        "episodes":       [e.to_dict() for e in run_log._episodes],
    }
    summary_path = _OUT / "run_summary.json"
    summary_path.write_text(_json.dumps(summary, indent=2))
    print(f"  Run summary → {summary_path}")


if __name__ == "__main__":
    main()
