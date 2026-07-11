"""
scene_builder.py
----------------
Runtime assembly of the two-robot scene: FF Master MJCF + Futurist URDF are
merged with ``MjSpec.attach()``, the Futurist hands get procedurally-built
articulated finger sets, and the serving-grasp equalities are declared.

Extracted from record_robot_video.py so the scene construction is unit-
testable on its own and the recording script stays control/render-focused.
"""
from __future__ import annotations

import os
import pathlib
import re
import tempfile

import mujoco
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).parent.parent
SCENE_XML   = ROOT / "assets" / "scene_robot.xml"
FTR_URDF    = (ROOT / "assets" / "futurist_unlocked.urdf").resolve()
FTR_MESHDIR = (ROOT / ".." / ".." / "assets" / "Futurist").resolve()

# ── Futurist world position — x=0.90 so arm (len≈0.70m) reaches stand at x=0.75 ──
FTR_BASE_POS  = np.array([0.90, -0.30, 0.98])
FTR_BASE_QUAT = np.array([0.0,  0.0,  0.0,  1.0])   # 180° around Z → faces −X

_FINGER_RGBA = [0.75, 0.75, 0.78, 1.0]

# ── Material profiles: contact stiffness + fracture work ─────────────
# solref timeconst sets how stiff the rind contact is (smaller = harder);
# the work threshold scales the fracture energy. The RLS estimator then
# identifies the resulting k online from (depth, force) pairs.
MATERIALS = {
    "soft": dict(solref=(0.050, 1.0), work_scale=0.65),
    "firm": dict(solref=(0.020, 1.0), work_scale=1.00),
    "hard": dict(solref=(0.009, 1.0), work_scale=1.60),
}


def apply_material(model: mujoco.MjModel, cut, name: str) -> dict:
    """Apply a MATERIALS profile to the melon geoms + fracture threshold."""
    mat = MATERIALS[name]
    for gn in ("wm_whole", "wm_left_col", "wm_right_col"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gn)
        if gid >= 0:
            model.geom_solref[gid, 0] = mat["solref"][0]
            model.geom_solref[gid, 1] = mat["solref"][1]
    cut.work_threshold *= mat["work_scale"]
    return mat


def add_fingers(spec: mujoco.MjSpec, body_name: str, side: str,
                collide: bool = False, actuated: bool = False):
    """Articulated 9-DOF finger set on a hand hull (4×2-joint fingers + thumb).

    Hand local frame: palm surface on +x side, fingers extend along +z.
    Each finger: proximal + distal body, hinge about local +y (curl toward
    palm). Joints are named ``{side}f{i}_prox/dist`` / ``{side}thumb``.

    ``collide=True`` puts the finger/palm geoms on collision bit 2 so they
    contact the quarter wedge (and only it — bit 2 is isolated from the
    melon, blade and Master).

    ``actuated=True`` gives every digit a torque-limited position actuator
    (named after its joint) and joint damping, making the digits DYNAMIC
    bodies: commanding the closed pose makes each digit close until it
    contacts the wedge, where the force limit — not the position target —
    sets the sustained pinch force. Grip-force regulation by construction,
    the classic underactuated-gripper trick. Kinematic mode (default) is
    byte-identical to the shipped model.
    """
    _ct, _ca = (2, 2) if collide else (0, 0)
    hand = spec.body(body_name)
    r    = 0.008
    seg1, seg2 = 0.042, 0.034

    def _servo(joint, kp, kv, frange):
        a = spec.add_actuator()
        a.name = joint.name
        a.target = joint.name
        a.trntype  = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[0] = 0.0; a.biasprm[1] = -kp; a.biasprm[2] = -kv
        a.forcerange = [-frange, frange]
        a.ctrlrange  = list(joint.range)
        try:
            joint.damping = 0.02        # MuJoCo ≤3.3: scalar field
        except TypeError:
            joint.damping[:] = 0.02     # MuJoCo ≥3.4: per-DOF array

    def _bite(g):
        """Digits BITE into the flesh (the wedge need not stay intact):
        priority=1 makes the digit's SOFT solref govern its wedge contacts,
        so the actuator force sinks the digit 1-2 cm into the hull — an
        embedded fingertip is form-closed against upward slip (the flesh
        above the tip resists), modelled here as deep soft penetration plus
        interlock-level friction. contype=0/conaffinity=2 pairs the digits
        with the wedge/plate only (no table snagging)."""
        g.priority = 1
        g.solref = [0.04, 1.0]
        g.friction = [3.0, 0.02, 0.003]
        g.contype = 0
        g.conaffinity = 2

    for i, y in enumerate((-0.040, -0.013, 0.014, 0.041)):
        prox = hand.add_body()
        prox.name = f"{side}f{i}_prox"
        prox.pos  = [0.015, y, 0.078]
        j = prox.add_joint()
        j.name = f"{side}f{i}_prox"; j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = [0.0, 1.0, 0.0];    j.range = [-0.2, 1.6]
        if actuated:
            _servo(j, kp=8.0, kv=0.08, frange=0.30)
        g = prox.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size[0] = r
        g.fromto = [0, 0, 0, 0, 0, seg1]
        g.rgba = _FINGER_RGBA; g.mass = 0.01; g.contype = _ct; g.conaffinity = _ca
        if actuated:
            _bite(g)

        dist = prox.add_body()
        dist.name = f"{side}f{i}_dist"
        dist.pos  = [0.0, 0.0, seg1]
        j = dist.add_joint()
        j.name = f"{side}f{i}_dist"; j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = [0.0, 1.0, 0.0];    j.range = [-0.2, 1.8]
        if actuated:
            _servo(j, kp=5.0, kv=0.05, frange=0.15)
        g = dist.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size[0] = r * 0.9
        g.fromto = [0, 0, 0, 0, 0, seg2]
        g.rgba = _FINGER_RGBA; g.mass = 0.008; g.contype = _ct; g.conaffinity = _ca
        if actuated:
            _bite(g)
        s = dist.add_site()
        s.name = f"{side}f{i}_tip"; s.pos = [0, 0, seg2]; s.size = [0.004, 0.004, 0.004]

    # palm plate: thin pad the fingers grow out of (the original hand hull is
    # stripped on the right hand — without this the hand looks hollow)
    g = hand.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    # extended toward the fingertips (+z): the palm face is the backstop the
    # scooped wedge leans against — it must reach down level with the wedge
    # flank, not just cover the knuckles
    g.size = [0.010, 0.052, 0.052]
    g.pos  = [0.006, -0.008, 0.058]
    g.rgba = _FINGER_RGBA; g.mass = 0.02; g.contype = _ct; g.conaffinity = _ca

    # TRUE opposable thumb: mounted OUTBOARD (+x) of the finger row with a
    # REVERSED hinge (−y axis) so it curls toward −x, against the fingers'
    # +x curl. The open jaw then STRADDLES the wedge — open thumb outside one
    # flat face, open fingers near the other — and closing loads both faces:
    # a genuine opposed pinch.
    thumb = hand.add_body()
    thumb.name = f"{side}thumb"
    thumb.pos  = [0.127, -0.008, 0.045]
    j = thumb.add_joint()
    j.name = f"{side}thumb"; j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis = [0.0, -1.0, 0.0]; j.range = [-0.2, 1.6]
    if actuated:
        # one thumb balances four fingers (like a human hand): ~4× the
        # torque budget so the pinch is force-balanced and the wedge is
        # squeezed in place instead of walked sideways
        _servo(j, kp=24.0, kv=0.20, frange=1.20)
    g = thumb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.size[0] = r * 1.1
    g.fromto = [0, 0, 0, 0.010, -0.006, 0.068]
    g.rgba = _FINGER_RGBA; g.mass = 0.012; g.contype = _ct; g.conaffinity = _ca
    if actuated:
        _bite(g)
    if actuated:
        # T-bar pad across the tip (line contact along the wedge's long
        # axis), friction mode only: a single-hinge thumb tip is a point a
        # shifting wedge escapes in 1-2 cm; the bar keeps the far face
        # loaded under drift and blocks sliding out along the ridge
        g = thumb.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size[0] = r * 1.1
        g.fromto = [0.010, -0.066, 0.068, 0.010, 0.054, 0.068]
        g.rgba = _FINGER_RGBA; g.mass = 0.010; g.contype = _ct; g.conaffinity = _ca
        _bite(g)
    s = thumb.add_site()
    s.name = f"{side}thumb_tip"; s.pos = [0.010, -0.006, 0.068]; s.size = [0.004, 0.004, 0.004]


# Right-arm servo table: (URDF joint name, kp, kv, torque limit N·m).
# Shoulder/elbow carry the bowed-torso reach + wedge load; distal joints are
# lighter. Torque limits keep contact forces physical (the arm can stall).
_ARM_SERVOS = [
    ("idx20_right_arm_joint1", 900.0, 90.0, 350.0),   # shoulder pitch
    ("idx21_right_arm_joint2", 900.0, 90.0, 350.0),   # shoulder roll
    ("idx22_right_arm_joint3", 700.0, 70.0, 250.0),   # shoulder yaw
    ("idx23_right_arm_joint4", 700.0, 70.0, 250.0),   # elbow
    ("idx24_right_arm_joint5", 350.0, 35.0, 120.0),   # forearm roll
    ("idx26_right_arm_joint7", 250.0, 25.0,  80.0),   # wrist yaw
]


def _add_arm_actuators(spec: mujoco.MjSpec):
    """Give the Futurist right arm torque-limited position actuators.

    Turns the serving arm's 6 joints into real dynamic DOFs (same servo
    pattern as the digits): the controller then commands targets via
    ``data.ctrl`` and physics owns the joint state — no kinematic overwrite.
    """
    by_name = {j.name: j for j in spec.joints}
    for jname, kp, kv, frange in _ARM_SERVOS:
        j = by_name[jname]
        a = spec.add_actuator()
        a.name = jname
        a.target = jname
        a.trntype  = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[0] = 0.0; a.biasprm[1] = -kp; a.biasprm[2] = -kv
        a.forcerange = [-frange, frange]
        rng = list(j.range)
        a.ctrlrange = rng if rng[0] < rng[1] else [-3.14, 3.14]
        try:
            j.damping = 1.0
        except TypeError:
            j.damping[:] = 1.0


def build_combined_model(dynamic_fingers: bool = False,
                         actuated_arm: bool = False) -> mujoco.MjModel:
    """Combine scene_robot.xml + Futurist URDF via MjSpec; add grip weld.

    ``dynamic_fingers=True`` builds the right-hand digits as torque-limited
    ACTUATED dynamic joints (see add_fingers) for the friction-carry grasp;
    the default keeps them kinematic, byte-identical to the shipped model.

    ``actuated_arm=True`` additionally gives the serving arm's 6 joints
    torque-limited position actuators (see _add_arm_actuators), so the arm
    is torque-driven end-to-end instead of kinematically overwritten.
    """
    urdf_text = FTR_URDF.read_text()
    meshdir   = str(FTR_MESHDIR)
    compiler_block = (
        "  <mujoco>\n"
        f'    <compiler meshdir="{meshdir}" discardvisual="false"/>\n'
        "  </mujoco>\n"
    )
    if "<mujoco>" not in urdf_text:
        urdf_text = re.sub(r"(<robot[^>]*>\n)", r"\1" + compiler_block, urdf_text, count=1)

    # Strip the right-hand wedge hull (a big cone-like mitten mesh): the
    # articulated fingers are the hand now, and the hull visually collided
    # with Master's blade while Futurist steadies the melon during cut 2.
    urdf_text = re.sub(r'<link name="right_hand">.*?</link>',
                       '<link name="right_hand"/>', urdf_text, flags=re.S)

    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(urdf_text)
        tmp_urdf = f.name
    try:
        ftr_spec = mujoco.MjSpec.from_file(tmp_urdf)
        base = ftr_spec.worldbody.find_child("base_link")
        base.add_freejoint(name="floating")

        add_fingers(ftr_spec, "left_hand",  side="l")   # palm-up: holds plate
        add_fingers(ftr_spec, "right_hand", side="r",   # closes around quarter
                    collide=True,                       # contact-grasp capable
                    actuated=dynamic_fingers)           # friction-carry mode
        if actuated_arm:
            _add_arm_actuators(ftr_spec)
            # the palm plate and the forearm/wrist backstop blocks share
            # collision bit 2 and overlap by construction — harmless while
            # the arm was kinematic, but a ~500 N internal contact that
            # stalls the torque-limited joints once the arm is dynamic
            for b1, b2 in (("right_hand", "right_arm_link05"),
                           ("right_hand", "right_arm_link07"),
                           ("right_arm_link05", "right_arm_link07")):
                ex = ftr_spec.add_exclude()
                ex.bodyname1 = b1
                ex.bodyname2 = b2

        # The forearm block (link05, 19 cm) and wrist (link07) are the
        # power-grasp backstop: the wedge is squeezed between the curling
        # fingers and this flat face — a human holds a melon wedge against
        # the palm/forearm too, not by fingertips. Bit 2 pairs them with the
        # wedge only (must be set at COMPILE time to enter the broadphase).
        for _ln in ("right_arm_link05", "right_arm_link07"):
            for _g in ftr_spec.body(_ln).geoms:
                _g.contype = 2
                _g.conaffinity = 2

        scene_spec = mujoco.MjSpec.from_file(str(SCENE_XML))
        attach_frame = scene_spec.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
        scene_spec.attach(ftr_spec, prefix="ftr_", suffix="", frame=attach_frame)

        if actuated_arm:
            # the plate's collision cylinder is thickened at runtime for the
            # set-down release; its contacts must serve ONLY the wedge — the
            # dynamic arm sweeping past (bowed GRIP posture) or the digits
            # pressing during set-down must not be shoved by it
            _plate_excl = ["ftr_right_arm_link05", "ftr_right_arm_link07",
                           "ftr_right_hand", "ftr_rthumb"] + [
                f"ftr_rf{i}_{seg}" for i in range(4)
                for seg in ("prox", "dist")]
            for _bn in _plate_excl:
                ex = scene_spec.add_exclude()
                ex.bodyname1 = "serving_plate"
                ex.bodyname2 = _bn
        gw = scene_spec.add_equality()
        gw.type    = mujoco.mjtEq.mjEQ_WELD
        gw.name    = "ftr_grip_weld"
        gw.name1   = "ftr_right_arm_link07"
        gw.name2   = "wm_quarter_A"
        gw.objtype = mujoco.mjtObj.mjOBJ_BODY
        gw.active  = False

        if dynamic_fingers:
            # Flesh-bite constraints (friction mode): when a digit's squeeze
            # exceeds the bite threshold the flesh yields and the fingertip
            # EMBEDS (the wedge need not stay intact). Each embedding is a
            # point CONNECT between that digit's tip body and the wedge —
            # transmits force in any direction (form closure of a buried
            # tip) but no torque, so the wedge can still pivot about the
            # grip points. Activated per-digit at the current coincident
            # point (zero residual at activation → no impulse).
            for _bn in ("ftr_rf0_dist", "ftr_rf1_dist", "ftr_rf2_dist",
                        "ftr_rf3_dist", "ftr_rthumb"):
                bq = scene_spec.add_equality()
                bq.type    = mujoco.mjtEq.mjEQ_CONNECT
                bq.name    = f"bite_{_bn}"
                bq.name1   = _bn
                bq.name2   = "wm_quarter_A"
                bq.objtype = mujoco.mjtObj.mjOBJ_BODY
                bq.active  = False

        return scene_spec.compile()
    finally:
        os.unlink(tmp_urdf)
