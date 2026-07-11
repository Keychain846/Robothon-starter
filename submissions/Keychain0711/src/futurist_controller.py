from __future__ import annotations
"""
futurist_controller.py
----------------------
Direct-qpos arm controller for the Futurist robot (serving role).

Controls: right arm (reach/grip/lift/carry/lower/contact/release),
          left arm (natural hanging → serving complement pose),
          hip pitch (forward bow during lower/contact phases).

Joint control: direct qpos assignment (Futurist has no MuJoCo actuators).

Phases (right arm drives the sequence):
  WAIT          — arm at idle resting pose; waiting for coordination trigger
  REACH         — arm extends toward quarter pickup position
  GRIP          — arm holds grip pose; weld activated externally on entry
  LIFT          — arm raises with quarter attached
  CARRY         — arm carries quarter over serving plate
  LOWER         — arm descends toward plate surface; hips bow forward
  CONTACT_PLATE — arm at plate level; quarter visibly touches plate (brief pause)
  RELEASE       — arm holds contact; weld released, quarter settles on plate
  RETRACT       — arm pulls back after release (clears plate area)
  DONE          — arm returns to idle
"""

import numpy as np
import mujoco

# ── Right arm joints ──────────────────────────────────────────────────────────
_RA_JOINT_NAMES = [
    "ftr_idx20_right_arm_joint1",  # shoulder pitch  (+ = forward)
    "ftr_idx21_right_arm_joint2",  # shoulder roll   (− = arm down)
    "ftr_idx22_right_arm_joint3",  # shoulder yaw
    "ftr_idx23_right_arm_joint4",  # elbow  right:  [0, +2.094] — positive = bent
    "ftr_idx24_right_arm_joint5",  # forearm roll: +1.6 → palm faces world-down
    "ftr_idx26_right_arm_joint7",  # wrist yaw
]

# ── Left arm joints ───────────────────────────────────────────────────────────
_LA_JOINT_NAMES = [
    "ftr_idx13_left_arm_joint1",   # shoulder pitch  [−2.967, 2.967]
    "ftr_idx14_left_arm_joint2",   # shoulder roll   [−0.524, 1.658]  + = arm out
    "ftr_idx15_left_arm_joint3",   # shoulder yaw    [−2.967, 2.967]
    "ftr_idx16_left_arm_joint4",   # elbow left:     [−2.094, 0]      − = bent
    "ftr_idx17_left_arm_joint5",   # forearm roll: −1.6 → palm faces world-up
    "ftr_idx19_left_arm_joint7",   # wrist yaw       [−0.524, 0.524]
]

# ── Hip pitch joints (forward lean = positive) ────────────────────────────────
# left  idx03: [−1.919, 0.785]   right  idx09: [−1.919, 0.785]
_HIP_JOINT_NAMES = [
    "ftr_idx03_left_hip_pitch",
    "ftr_idx09_right_hip_pitch",
]

# ── Right arm keyframe poses [sp, sr, sy, el, ro, wr] (radians) ───────────────
# Futurist base x=0.90 facing −X, shoulder at (0.938,−0.30,1.322).
# j5 (forearm roll) = +1.6 turns the palm world-down for grasping.
# FK-verified (j1=sp, j2=sr, j3=sy, j4=el, j5=ro, j7=wr):
#   GRIP:    j2=−1.66,j3=0.40,j4=1.20,j5=1.6 → wrist≈(0.716,−0.243,0.99)
#   CONTACT: j2=−1.66,j3=0.50,j4=1.92,j5=1.6 → wrist≈(0.726,−0.249,1.176)
#            = 7cm above plate held in left palm at (0.713,−0.266,1.10)
# STEADY: while FF Master cuts, Futurist leans in (49° hip bow) and rests its
# open right hand on the melon's +X+Y shoulder — clear of both cut planes
# (cut1 x=0.44, cut2 y=−0.30) — genuinely steadying the workpiece.
# FK: wrist (0.559,−0.175,0.800), fingertips on the rind.
_P_STEADY   = np.array([ 0.60, -1.66,  0.10,  0.96,  1.60,  0.00])
_BOW_STEADY = 0.85
_STEADY_ENTER_DUR = 1.60   # ease-in from IDLE after the episode starts

# GRIP is reached together with a 54° hip bow (see _BOW tables): the pelvis
# pitches forward about the hip while the hip-pitch joints counter-rotate the
# same angle, so the legs stay vertical and only the torso leans — a natural
# bend. The fingertip centroid lands 2 cm above the wedge resting at
# (0.52,−0.35,0.61) — the closing fingers visibly cage the piece before it
# attaches.
_P_IDLE    = np.array([ 0.00, -1.45,  0.00,  0.15,  0.00,  0.00])  # arm hanging down
_P_REACH   = np.array([ 0.904, -1.603, 0.862, 0.881, 0.734, 0.00]) # hover: jaw open 9 cm above the wedge
                                                                   # ridge, jaw centre over it — GRIP is
                                                                   # a straight descent onto the ridge
_P_GRIP    = np.array([ 1.084, -1.731, 0.810, 0.579, 0.765, 0.00]) # CLAMP on the RIDGE: jaw-close centre
                                                                   # (hand-local [0.038,0,0.11]) sits on
                                                                   # the enlarged wedge's top ridge
                                                                   # (0.487,−0.357,0.666). The wedge is
                                                                   # bigger than the hand, so the clamp
                                                                   # grips its thin upper ridge — 4 fingers
                                                                   # on one flat face, thumb on the other
_P_LIFT    = np.array([ 0.90, -1.51,  1.00,  1.65,  2.40,  0.00])  # body straightens, wrist→(0.66,−0.32,
                                                                   # 1.28) — the weld-held wedge rides UP
                                                                   # HIGH, clear of both robots' bodies so
                                                                   # the fixed camera sees it the whole way
_P_CARRY   = np.array([ 1.05, -1.56,  0.70,  1.92,  1.60,  0.00])  # peak held high over the plate
                                                                   # (0.709,−0.274,1.401) — well above
                                                                   # the plate, no dip below it
_P_LOWER   = np.array([ 0.30, -1.18,  0.80,  2.00,  1.20,  0.00])  # the whole HAND descends onto the
                                                                   # plate so the wedge is set down by the
                                                                   # fingers (not dropped from above)
_P_CONTACT = np.array([ 0.30, -1.18,  0.80,  2.00,  1.20,  0.00])  # fingers at the plate; opening them
                                                                   # lays the wedge on the disc
_P_DONE    = np.array([ 0.00, -1.45,  0.00,  0.15,  0.00,  0.00])  # arm returns to hanging

# ── Left arm keyframe poses [sp, sr, sy, el, ro, wr] ──────────────────────────
# Waiter hold: left arm extended, forearm rolled −1.6 so the palm faces up
# (palm·ẑ = 1.00). Wrist at (0.673,−0.334,1.082) — this brings the plate to
# the right arm's hand-off point so the wedge is RELEASED from the fingers
# directly above the plate (both arms meet; no slide-off).
_LA_WAITER  = np.array([ 0.20,  1.60,  0.65, -1.40, -1.60,  0.00])
_LA_IDLE    = _LA_WAITER
_LA_REACH   = _LA_WAITER
_LA_GRIP    = _LA_WAITER
_LA_LIFT    = _LA_WAITER
_LA_CARRY   = _LA_WAITER
_LA_LOWER   = _LA_WAITER
_LA_CONTACT = _LA_WAITER
_LA_DONE    = _LA_WAITER

# ── Hip pitch keyframes [left_pitch, right_pitch] ─────────────────────────────
_HP_NEUTRAL   = np.array([0.00, 0.00])
_HP_SLIGHT    = np.array([0.08, 0.08])
_HP_MID       = np.array([0.16, 0.16])
_HP_BOW       = np.array([0.26, 0.26])
_HP_DEEP_BOW  = np.array([0.32, 0.32])

# ── Finger curl per phase (rad, [prox, dist]) ────────────────────────────────
# Right hand: opens on approach, closes around the quarter at GRIP, holds
# through CARRY..CONTACT, releases the wrap at RELEASE.
_FC_OPEN   = np.array([0.10, 0.15])
_FC_CLOSED = np.array([0.85, 1.05])
_FC_RIGHT_START = {
    "REACH": _FC_OPEN,   "GRIP": _FC_OPEN,    "LIFT": _FC_CLOSED,
    "CARRY": _FC_CLOSED, "LOWER": _FC_CLOSED, "CONTACT_PLATE": _FC_CLOSED,
    "RELEASE": _FC_CLOSED, "RETRACT": _FC_OPEN, "DONE": _FC_OPEN,
}
_FC_RIGHT_END = {
    "REACH": _FC_OPEN,   "GRIP": _FC_CLOSED,  "LIFT": _FC_CLOSED,
    "CARRY": _FC_CLOSED, "LOWER": _FC_CLOSED, "CONTACT_PLATE": _FC_CLOSED,
    "RELEASE": _FC_OPEN, "RETRACT": _FC_OPEN, "DONE": _FC_OPEN,
}
# Left hand: constant gentle spread supporting the plate.
_FC_LEFT = np.array([0.25, 0.20])

# ── Torso bow (hip bend: pelvis pitches, legs counter-rotate to stay vertical) ─
# The floating base (pelvis) is re-pinned each step at pitch θ about its own
# origin, while both hip-pitch joints are offset by −θ so the legs remain
# upright — the robot bends at the hip like a person picking something up
# (arm-only reach bottoms out ~30cm short of the quarter).
_BOW_GRIP  = 1.012  # rad ≈ 58° — jaw centre on the enlarged wedge ridge
_BOW_HOVER = 0.850  # rad ≈ 49° — REACH hover, jaw open 9 cm above the ridge
_BOW_PHASE_START = {
    "REACH": _BOW_STEADY, "GRIP": _BOW_HOVER, "LIFT": _BOW_GRIP,
    "CARRY": 0.0,       "LOWER": 0.0,       "CONTACT_PLATE": 0.0,
    "RELEASE": 0.0,     "RETRACT": 0.0,     "DONE": 0.0,
}
_BOW_PHASE_END = {
    "REACH": _BOW_HOVER, "GRIP": _BOW_GRIP, "LIFT": 0.0,
    "CARRY": 0.0,       "LOWER": 0.0,       "CONTACT_PLATE": 0.0,
    "RELEASE": 0.0,     "RETRACT": 0.0,     "DONE": 0.0,
}

_PHASE_SEQUENCE = [
    "WAIT", "REACH", "GRIP", "LIFT", "CARRY",
    "LOWER", "CONTACT_PLATE", "RELEASE", "RETRACT", "DONE",
]
_PHASE_DUR = {
    "WAIT":          999.0,
    "REACH":         1.20,
    "GRIP":          1.20,   # first half: hand settles at the scoop point;
                             # second half: tactile servo closes the scoop
    "LIFT":          1.05,   # slow departure — the gripped wedge rides the
                             # fingers up, not flung off them
    "CARRY":         1.30,   # supinate high up — the wedge rolls from the
                             # side-grip onto the flat palm-up hook
    "LOWER":         1.20,   # slow descent — plate contact must be clearly visible
    "CONTACT_PLATE": 0.90,   # pause on plate before release
    "RELEASE":       0.55,
    "RETRACT":       1.10,
    "DONE":          999.0,
}
# ① max extra time GRIP will wait for grasp confirmation before advancing
# anyway (fallback so a missed grasp cannot stall the sequence forever).
_GRIP_CONFIRM_TIMEOUT = 1.0
# max extra time CONTACT_PLATE will wait for the carried piece to settle
# before releasing anyway.
_RELEASE_SETTLE_TIMEOUT = 2.0

# Right arm phase interpolation tables
_PHASE_START_POSE = {
    "REACH":         _P_STEADY,   # flows straight from steadying the melon
    "GRIP":          _P_REACH,
    "LIFT":          _P_GRIP,
    "CARRY":         _P_LIFT,
    "LOWER":         _P_CARRY,
    "CONTACT_PLATE": _P_LOWER,
    "RELEASE":       _P_CONTACT,
    "RETRACT":       _P_CONTACT,
    "DONE":          _P_DONE,
}
_PHASE_END_POSE = {
    "REACH":         _P_REACH,
    "GRIP":          _P_GRIP,
    "LIFT":          _P_LIFT,
    "CARRY":         _P_CARRY,
    "LOWER":         _P_LOWER,
    "CONTACT_PLATE": _P_CONTACT,
    "RELEASE":       _P_CONTACT,
    "RETRACT":       _P_DONE,
    "DONE":          _P_DONE,
}

# Left arm phase interpolation tables.
# The plate hand stays TUCKED while the torso is bowed (REACH..LIFT) so the
# tray never swings over the work area; it re-extends to the waiter hold as
# the body straightens during CARRY.
_LA_PHASE_START = {
    "REACH":         _LA_WAITER,
    "GRIP":          _LA_WAITER,
    "LIFT":          _LA_WAITER,
    "CARRY":         _LA_WAITER,
    "LOWER":         _LA_WAITER,
    "CONTACT_PLATE": _LA_WAITER,
    "RELEASE":       _LA_WAITER,
    "RETRACT":       _LA_WAITER,
    "DONE":          _LA_WAITER,
}
_LA_PHASE_END = {
    "REACH":         _LA_WAITER,
    "GRIP":          _LA_WAITER,
    "LIFT":          _LA_WAITER,
    "CARRY":         _LA_WAITER,
    "LOWER":         _LA_WAITER,
    "CONTACT_PLATE": _LA_WAITER,
    "RELEASE":       _LA_WAITER,
    "RETRACT":       _LA_WAITER,
    "DONE":          _LA_WAITER,
}

# Hip pitch phase interpolation tables.
# All neutral: the hip joints are reserved for the bow counter-rotation
# (see _BOW tables) that keeps the legs vertical while the pelvis pitches.
_HP_PHASE_START = {ph: _HP_NEUTRAL for ph in (
    "REACH", "GRIP", "LIFT", "CARRY", "LOWER",
    "CONTACT_PLATE", "RELEASE", "RETRACT", "DONE")}
_HP_PHASE_END = dict(_HP_PHASE_START)


def _ss(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class FuturistController:
    """Phase-based direct-qpos controller for Futurist: right arm + left arm + hip bow."""

    def __init__(self):
        self._phase: str = "WAIT"
        self._phase_start_t: float = 0.0
        self._joint_adrs: list[int] = []          # right arm qpos addresses
        self._joint_vel_adrs: list[int] = []
        self._la_joint_adrs: list[int] = []       # left arm qpos addresses
        self._hip_joint_adrs: list[int] = []      # hip pitch qpos addresses
        self._all_ftr_qpos_slice: slice | None = None
        self._all_ftr_dof_slice:  slice | None = None
        self._last_pose: np.ndarray = _P_IDLE.copy()
        self._last_la_pose: np.ndarray = _LA_IDLE.copy()
        self._last_hp: np.ndarray = _HP_NEUTRAL.copy()
        self._bow: float = 0.0
        self._last_fc_right: np.ndarray = _FC_OPEN.copy()
        self._rf_prox_adrs: list[int] = []
        self._rf_dist_adrs: list[int] = []
        self._lf_prox_adrs: list[int] = []
        self._lf_dist_adrs: list[int] = []
        self._model: mujoco.MjModel | None = None
        self._hand_bid: int = -1
        self._arm_dof_adrs: list[int] = []
        self._ik_corr: np.ndarray = np.zeros(len(_RA_JOINT_NAMES))
        self._waypoints: dict = {}
        self.track_err_m: float = 0.0
        self._base_qpos_adr: int = -1
        self._base_vel_adr: int = -1
        # x=0.90, y=-0.30, z=0.98, quat=[0,0,0,1] = 180° around Z → facing −X
        # Moved from x=1.23 to x=0.90 so arm can reach serving stand at x=0.75
        self._base_target: np.ndarray = np.array([0.90, -0.30, 0.98, 0.0, 0.0, 0.0, 1.0])
        self.grip_should_activate: bool = False
        self.grip_should_release: bool = False
        # Contact-gated FSM: GRIP does not advance to LIFT until the grasp is
        # physically confirmed. The record pipeline sets this from its
        # geometric-enclosure + tactile gate; it defaults True so bare-scene
        # callers (tests, teleop AUTO) still advance on the phase timer.
        self.grasp_confirmed: bool = True
        # GRIP retry: if the confirm window times out unconfirmed, the FSM
        # backs off to REACH and re-approaches instead of lifting a missed
        # grasp. Budgeted (2 retries) so a hopeless grasp cannot loop forever.
        self.grip_retries: int = 0
        # Stability-gated release: CONTACT_PLATE does not advance to RELEASE
        # until the carried piece has settled (set by the record pipeline in
        # friction mode; default True so other callers advance on the timer).
        self.release_allowed: bool = True
        # Dynamic right-hand digits (friction-carry architecture): when True,
        # the right fingers are torque-limited ACTUATED joints — this
        # controller must NOT overwrite their qpos nor zero their qvel; the
        # record pipeline drives them via data.ctrl instead. The kinematic
        # arm dofs then also carry their true finite-difference velocity
        # (mocap-style) instead of zero, so contact friction correctly
        # transmits the hand's motion to whatever the digits are holding.
        self.dynamic_fingers: bool = False
        # Actuated serving arm: when the model was built with
        # actuated_arm=True (torque-limited position actuators on the 6
        # right-arm joints), this controller commands data.ctrl targets
        # and physics owns the arm state — auto-detected in build().
        self.actuated_arm: bool = False
        self._ra_act_ids: list[int] = []
        # task-space carry correction (m), added to the wrist waypoint
        # target: the record pipeline closes an outer loop on the CARRIED
        # PIECE's measured position (visual-servo-style placement) so the
        # hanging wedge — not just the hand — is centred over the plate
        self.carry_offset: np.ndarray = np.zeros(3)
        # when not None, the IK loop tracks THIS wrist target instead of
        # the waypoint line — used for fingertip-on-wedge grasp servoing
        self.grasp_target: np.ndarray | None = None
        # left-shoulder-pitch offset (rad): the waiter arm LOWERS or
        # RAISES the palm-held plate to meet the hanging wedge — the
        # record pipeline servos this on the measured hang depth
        # (bimanual set-down: kinematic left arm has no IK saturation)
        self.plate_lift: float = 0.0
        self._prev_qpos: np.ndarray | None = None
        self._vel_buf: np.ndarray | None = None

    def build(self, model: mujoco.MjModel,
              base_world_pos: np.ndarray | None = None,
              base_world_quat: np.ndarray | None = None):
        """Look up joint addresses in the compiled model."""
        self._joint_adrs = []
        self._joint_vel_adrs = []
        _arm_lo, _arm_hi = [], []
        for jname in _RA_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"FuturistController: joint '{jname}' not found in model")
            self._joint_adrs.append(int(model.jnt_qposadr[jid]))
            self._joint_vel_adrs.append(int(model.jnt_dofadr[jid]))
            _arm_lo.append(float(model.jnt_range[jid][0]))
            _arm_hi.append(float(model.jnt_range[jid][1]))
        self._arm_lo = np.array(_arm_lo)
        self._arm_hi = np.array(_arm_hi)

        # torque-limited arm servos (present when the model was built
        # with actuated_arm=True) — drive ctrl instead of qpos
        self._ra_act_ids = [
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n))
            for n in _RA_JOINT_NAMES]
        self.actuated_arm = all(a >= 0 for a in self._ra_act_ids)

        self._la_joint_adrs = []
        for jname in _LA_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"FuturistController: left arm joint '{jname}' not found")
            self._la_joint_adrs.append(int(model.jnt_qposadr[jid]))

        self._hip_joint_adrs = []
        for jname in _HIP_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"FuturistController: hip joint '{jname}' not found")
            self._hip_joint_adrs.append(int(model.jnt_qposadr[jid]))

        # Articulated finger joints (added by _add_fingers in the model builder;
        # absent in bare-scene tests). prox list includes the thumb.
        self._rf_prox_adrs, self._rf_dist_adrs = [], []
        self._lf_prox_adrs, self._lf_dist_adrs = [], []
        self._rf_all_qadrs, self._rf_all_dadrs = [], []   # every right digit
        for jid in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not jname or not jname.startswith("ftr_"):
                continue
            adr = int(model.jnt_qposadr[jid])
            stem = jname[4:]
            if stem.startswith("rf") or stem == "rthumb":
                self._rf_all_qadrs.append(adr)
                self._rf_all_dadrs.append(int(model.jnt_dofadr[jid]))
            if stem.startswith("rf") and stem.endswith("_prox") or stem == "rthumb":
                self._rf_prox_adrs.append(adr)
            elif stem.startswith("rf") and stem.endswith("_dist"):
                self._rf_dist_adrs.append(adr)
            elif stem.startswith("lf") and stem.endswith("_prox") or stem == "lthumb":
                self._lf_prox_adrs.append(adr)
            elif stem.startswith("lf") and stem.endswith("_dist"):
                self._lf_dist_adrs.append(adr)

        jid_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ftr_floating")
        if jid_base < 0:
            raise ValueError("FuturistController: freejoint 'ftr_floating' not found")
        self._base_qpos_adr = int(model.jnt_qposadr[jid_base])
        self._base_vel_adr  = int(model.jnt_dofadr[jid_base])

        _ftr_qpos_addrs: list[int] = []
        _ftr_dof_addrs:  list[int] = []
        for jid in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname and jname.startswith("ftr_"):
                jtype = model.jnt_type[jid]
                nq = 7 if jtype == 0 else 1
                nd = 6 if jtype == 0 else 1
                qa = int(model.jnt_qposadr[jid])
                da = int(model.jnt_dofadr[jid])
                _ftr_qpos_addrs.extend(range(qa, qa + nq))
                _ftr_dof_addrs.extend(range(da, da + nd))

        if _ftr_qpos_addrs:
            self._all_ftr_qpos_slice = slice(min(_ftr_qpos_addrs), max(_ftr_qpos_addrs) + 1)
        if _ftr_dof_addrs:
            self._all_ftr_dof_slice  = slice(min(_ftr_dof_addrs),  max(_ftr_dof_addrs)  + 1)

        if base_world_pos is not None:
            self._base_target[:3] = base_world_pos
        if base_world_quat is not None:
            self._base_target[3:] = base_world_quat

        # ── Task-space tracking setup (damped-least-squares IK) ──────────
        # FK-derive the wrist waypoint of every keyframe (with its bow angle)
        # on a scratch MjData; at runtime step() closes the loop on the
        # actual wrist position toward the interpolated waypoint line.
        self._model = model
        self._hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                           "ftr_right_hand")
        self._arm_dof_adrs = [int(model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in _RA_JOINT_NAMES]
        self._ik_corr = np.zeros(len(_RA_JOINT_NAMES))

        scratch = mujoco.MjData(model)
        self._waypoints: dict[int, np.ndarray] = {}

        def _fk_wrist(pose: np.ndarray, bow: float) -> np.ndarray:
            scratch.qpos[:] = 0.0
            bq = self._base_target.copy()
            if abs(bow) > 1e-9:
                bq[3:] = [0.0, -np.sin(bow / 2.0), 0.0, np.cos(bow / 2.0)]
            scratch.qpos[self._base_qpos_adr:self._base_qpos_adr + 7] = bq
            for adr, val in zip(self._joint_adrs, pose):
                scratch.qpos[adr] = float(val)
            for adr in self._hip_joint_adrs:
                scratch.qpos[adr] = -bow
            mujoco.mj_kinematics(model, scratch)
            return scratch.xpos[self._hand_bid].copy()

        for ph in _PHASE_SEQUENCE:
            key_s = (id(_PHASE_START_POSE.get(ph, _P_IDLE)),
                     round(_BOW_PHASE_START.get(ph, 0.0), 4))
            key_e = (id(_PHASE_END_POSE.get(ph, _P_IDLE)),
                     round(_BOW_PHASE_END.get(ph, 0.0), 4))
            for key, pose, bow in (
                    (key_s, _PHASE_START_POSE.get(ph, _P_IDLE),
                     _BOW_PHASE_START.get(ph, 0.0)),
                    (key_e, _PHASE_END_POSE.get(ph, _P_IDLE),
                     _BOW_PHASE_END.get(ph, 0.0))):
                if key not in self._waypoints:
                    self._waypoints[key] = _fk_wrist(np.asarray(pose), bow)

    def _phase_waypoints(self) -> tuple[np.ndarray, np.ndarray] | None:
        ph = self._phase
        if ph in ("WAIT", "DONE") or not self._waypoints:
            return None
        ks = (id(_PHASE_START_POSE.get(ph, _P_IDLE)),
              round(_BOW_PHASE_START.get(ph, 0.0), 4))
        ke = (id(_PHASE_END_POSE.get(ph, _P_IDLE)),
              round(_BOW_PHASE_END.get(ph, 0.0), 4))
        if ks in self._waypoints and ke in self._waypoints:
            return self._waypoints[ks], self._waypoints[ke]
        return None

    @property
    def phase(self) -> str:
        return self._phase

    def reset(self, t: float = 0.0):
        self._phase = "WAIT"
        self._phase_start_t = t
        self._bow = 0.0
        self._ik_corr = np.zeros(len(_RA_JOINT_NAMES))
        self.track_err_m = 0.0
        self.grip_should_activate = False
        self.grip_should_release = False
        self.grasp_confirmed = True
        self.release_allowed = True
        self.grip_retries = 0
        self.carry_offset = np.zeros(3)
        self.grasp_target = None
        self.plate_lift = 0.0
        self._prev_qpos = None

    def seed_idle(self, data: mujoco.MjData):
        """Initialise the arm at the idle pose (call after episode reset).
        With an actuated arm this seeds both qpos and the ctrl targets so
        the episode does not open with a wake-up transient."""
        for adr, val in zip(self._joint_adrs, _P_IDLE):
            data.qpos[adr] = float(val)
        if self.actuated_arm:
            for aid, val in zip(self._ra_act_ids, _P_IDLE):
                data.ctrl[aid] = float(val)

    def notify_steady(self, t: float):
        """Lean in and steady the melon while Master cuts (episode start)."""
        if self._phase == "WAIT":
            self._phase = "STEADY"
            self._phase_start_t = t

    def abort_lift(self, t: float) -> bool:
        """Grasp-quality rejection: the piece was picked up but hangs
        outside the set-down workspace (bite too shallow). Put it back
        and re-approach — same retry budget as the confirm gate."""
        if self._phase in ("LIFT", "CARRY") and self.grip_retries < 2:
            self.grip_retries += 1
            self._phase = "REACH"
            self._phase_start_t = t
            self.grip_should_activate = False
            return True
        return False

    def notify_start(self, t: float):
        """Trigger the serving sequence (called when Master finishes cut2)."""
        if self._phase in ("WAIT", "STEADY"):
            self._phase = "REACH"
            self._phase_start_t = t
            self.grip_should_activate = False
            self.grip_should_release = False

    def step(self, data: mujoco.MjData, t: float):
        """Advance phase state machine and write right arm, left arm, and hip qpos."""
        self.grip_should_activate = False
        self.grip_should_release = False

        if self._phase == "WAIT":
            self._bow = 0.0
            self._set_all(data, _P_IDLE, _LA_IDLE, _HP_NEUTRAL)
            self._pin_base(data)
            return
        if self._phase == "DONE":
            self._bow = 0.0
            self._set_all(data, _P_DONE, _LA_DONE, _HP_NEUTRAL)
            self._pin_base(data)
            return
        if self._phase == "STEADY":
            # lean in and rest the open hand on the melon while Master cuts
            a = _ss(min((t - self._phase_start_t) / _STEADY_ENTER_DUR, 1.0))
            self._bow = _BOW_STEADY * a
            self._last_fc_right = _FC_OPEN * (1.0 - a) + np.array([0.35, 0.30]) * a
            ra = _P_IDLE * (1.0 - a) + _P_STEADY * a
            self._set_all(data, ra, _LA_WAITER, _HP_NEUTRAL)
            self._pin_base(data)
            return

        elapsed = t - self._phase_start_t
        dur = _PHASE_DUR.get(self._phase, 1.0)
        alpha = _ss(float(np.clip(elapsed / max(dur, 1e-6), 0.0, 1.0)))

        ra = (_PHASE_START_POSE.get(self._phase, _P_IDLE) * (1.0 - alpha)
              + _PHASE_END_POSE.get(self._phase, _P_IDLE) * alpha)
        la = (_LA_PHASE_START.get(self._phase, _LA_IDLE) * (1.0 - alpha)
              + _LA_PHASE_END.get(self._phase, _LA_IDLE) * alpha)
        if abs(self.plate_lift) > 1e-9:
            la = la.copy()
            la[0] += float(np.clip(self.plate_lift, -0.45, 0.45))
        hp = (_HP_PHASE_START.get(self._phase, _HP_NEUTRAL) * (1.0 - alpha)
              + _HP_PHASE_END.get(self._phase, _HP_NEUTRAL) * alpha)
        self._bow = (_BOW_PHASE_START.get(self._phase, 0.0) * (1.0 - alpha)
                     + _BOW_PHASE_END.get(self._phase, 0.0) * alpha)
        self._last_fc_right = (
            _FC_RIGHT_START.get(self._phase, _FC_OPEN) * (1.0 - alpha)
            + _FC_RIGHT_END.get(self._phase, _FC_OPEN) * alpha)

        # Damped-least-squares task-space correction: close the loop on the
        # measured wrist position toward the interpolated waypoint line.
        wp = self._phase_waypoints()
        if wp is not None and self._hand_bid >= 0:
            x_tgt = wp[0] * (1.0 - alpha) + wp[1] * alpha + self.carry_offset
            if self.grasp_target is not None:
                x_tgt = self.grasp_target
            err = x_tgt - data.xpos[self._hand_bid]
            en = float(np.linalg.norm(err))
            if 1e-4 < en < (0.45 if self.actuated_arm else 0.30):
                jacp = np.zeros((3, self._model.nv))
                mujoco.mj_jacBody(self._model, data, jacp, None, self._hand_bid)
                J = jacp[:, self._arm_dof_adrs]
                dq = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(3), err)
                _lim  = 0.75 if self.actuated_arm else 0.25
                self._ik_corr = np.clip(self._ik_corr * 0.98 + 0.35 * dq,
                                        -_lim, _lim)
            else:
                self._ik_corr *= 0.90
            ra = np.clip(ra + self._ik_corr, self._arm_lo, self._arm_hi)
            self.track_err_m = en
        else:
            self._ik_corr *= 0.90

        self._set_all(data, ra, la, hp)
        self._pin_base(data)

        if elapsed >= dur:
            # ① Contact-gated FSM: hold GRIP (keep the fingers closing on the
            # wedge at the end pose) until the grasp is physically confirmed,
            # so the arm only lifts once it has actually grasped — with a
            # timeout so a missed grasp can never stall the sequence forever.
            if self._phase == "GRIP" and not self.grasp_confirmed:
                if elapsed < dur + _GRIP_CONFIRM_TIMEOUT:
                    return
                # confirm window expired with no physical grasp: back off to
                # REACH and re-approach (the record pipeline reopens the
                # digits on this transition). After the retry budget is
                # spent, fall through and advance on the timer as before.
                if self.grip_retries < 2:
                    self.grip_retries += 1
                    self._phase = "REACH"
                    self._phase_start_t = t
                    return
            # Symmetric stability gate on the OTHER end of the carry: hold
            # CONTACT_PLATE until the (physically swinging) wedge has settled
            # over the plate — a waiter steadies the piece before letting go.
            if (self._phase == "CONTACT_PLATE" and not self.release_allowed
                    and elapsed < dur + _RELEASE_SETTLE_TIMEOUT):
                return

            if self._phase == "GRIP":
                self.grip_should_activate = True
            elif self._phase == "RELEASE":
                self.grip_should_release = True

            idx = _PHASE_SEQUENCE.index(self._phase)
            if idx + 1 < len(_PHASE_SEQUENCE):
                self._phase = _PHASE_SEQUENCE[idx + 1]
                self._phase_start_t = t

    def _set_all(self, data: mujoco.MjData,
                 ra_pose: np.ndarray,
                 la_pose: np.ndarray,
                 hp: np.ndarray):
        self._last_pose    = np.asarray(ra_pose, dtype=float)
        self._last_la_pose = np.asarray(la_pose, dtype=float)
        # hip counter-rotation keeps legs vertical while the pelvis bows
        self._last_hp      = np.asarray(hp, dtype=float) - float(getattr(self, "_bow", 0.0))
        if self.actuated_arm:
            for aid, val in zip(self._ra_act_ids, self._last_pose):
                data.ctrl[aid] = float(val)
        else:
            for adr, val in zip(self._joint_adrs, self._last_pose):
                data.qpos[adr] = float(val)
        for adr, val in zip(self._la_joint_adrs, self._last_la_pose):
            data.qpos[adr] = float(val)
        for adr, val in zip(self._hip_joint_adrs, self._last_hp):
            data.qpos[adr] = float(val)
        self._apply_fingers(data)

    def _apply_fingers(self, data: mujoco.MjData):
        if not self.dynamic_fingers:
            # kinematic digits: write the curl table directly
            fc = getattr(self, "_last_fc_right", _FC_OPEN)
            for adr in self._rf_prox_adrs:
                data.qpos[adr] = float(fc[0])
            for adr in self._rf_dist_adrs:
                data.qpos[adr] = float(fc[1])
        for adr in self._lf_prox_adrs:
            data.qpos[adr] = float(_FC_LEFT[0])
        for adr in self._lf_dist_adrs:
            data.qpos[adr] = float(_FC_LEFT[1])

    def _base_with_bow(self) -> np.ndarray:
        """Base qpos[7] pitched by self._bow about the pelvis origin (hip bend).

        The hip-pitch joints receive a −θ offset in _set_all, so the legs
        counter-rotate and stay vertical while the torso leans forward.
        """
        theta = float(getattr(self, "_bow", 0.0))
        if abs(theta) < 1e-6:
            return self._base_target
        # quat = Ry(−θ) ⊗ Z180  →  [w,x,y,z] = [0, −sin(θ/2), 0, cos(θ/2)]
        quat = np.array([0.0, -np.sin(theta / 2.0), 0.0, np.cos(theta / 2.0)])
        return np.concatenate([self._base_target[:3], quat])

    def _pin_base(self, data: mujoco.MjData):
        """Pin Futurist base and freeze ALL Futurist joints, then re-apply controlled poses.

        With ``dynamic_fingers`` the right-hand digit joints are real dynamic
        DOFs driven by their own actuators — their qpos/qvel state is saved
        across the wholesale freeze and restored, so physics owns them.
        """
        if self._base_qpos_adr < 0:
            return
        _dyn_qadrs = list(self._rf_all_qadrs) if (self.dynamic_fingers
                                                  and self._rf_all_qadrs) else []
        _dyn_dadrs = list(self._rf_all_dadrs) if (self.dynamic_fingers
                                                  and self._rf_all_dadrs) else []
        if self.actuated_arm:
            _dyn_qadrs += self._joint_adrs
            _dyn_dadrs += self._joint_vel_adrs
        _dyn = bool(_dyn_qadrs)
        if _dyn:
            _dq = [float(data.qpos[a]) for a in _dyn_qadrs]
            _dv = [float(data.qvel[a]) for a in _dyn_dadrs]
        base = self._base_with_bow()
        data.qpos[self._base_qpos_adr:self._base_qpos_adr + 7] = base
        if self._all_ftr_qpos_slice is not None:
            data.qpos[self._all_ftr_qpos_slice] = 0.0
            data.qpos[self._base_qpos_adr:self._base_qpos_adr + 7] = base
            if not self.actuated_arm:
                for adr, val in zip(self._joint_adrs, self._last_pose):
                    data.qpos[adr] = float(val)
            for adr, val in zip(self._la_joint_adrs, self._last_la_pose):
                data.qpos[adr] = float(val)
            for adr, val in zip(self._hip_joint_adrs, self._last_hp):
                data.qpos[adr] = float(val)
            self._apply_fingers(data)
        if self._all_ftr_dof_slice is not None:
            data.qvel[self._all_ftr_dof_slice] = 0.0
            if _dyn and self._model is not None:
                # mocap-style kinematic driving: the commanded dofs carry
                # their true finite-difference velocity, so contact friction
                # transmits the hand's motion to the pinched wedge instead
                # of the solver believing the hand is stationary
                if self._vel_buf is None:
                    self._vel_buf = np.zeros(self._model.nv)
                if self._prev_qpos is not None:
                    mujoco.mj_differentiatePos(
                        self._model, self._vel_buf,
                        self._model.opt.timestep,
                        self._prev_qpos, data.qpos)
                    sl = self._all_ftr_dof_slice
                    data.qvel[sl] = np.clip(self._vel_buf[sl], -4.0, 4.0)
                self._prev_qpos = data.qpos.copy()
        elif self._base_vel_adr >= 0:
            data.qvel[self._base_vel_adr:self._base_vel_adr + 6] = 0.0
        if _dyn:
            for a, v in zip(_dyn_qadrs, _dq):
                data.qpos[a] = v
            for a, v in zip(_dyn_dadrs, _dv):
                data.qvel[a] = v
