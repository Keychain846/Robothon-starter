"""
FeedbackController — Jacobian-transpose position feedback for the FF Master right arm.

Converts a Cartesian position error at blade_tip into joint-angle corrections
(Δq) for the 5 right-arm joints using MuJoCo's analytic site Jacobian.
The correction is added on top of the nominal time-interpolated trajectory
produced by RobotArmController, making the ALIGN / ALIGN2 phases genuinely
closed-loop without replacing the existing planner.

Math
----
    J_arm  ∈ ℝ^{3×5}  — translational Jacobian (blade_tip site, arm DOFs only)
    error  ∈ ℝ^3       — target_xyz − blade_xyz  (Cartesian)
    Δq     ∈ ℝ^5       — Jacobian-transpose step: Δq = J_arm^T @ error × Kp
    ctrl   ∈ ℝ^5       — clip(nominal + Δq, joint_limits)

Usage
-----
    fb = FeedbackController(model)

    # inside the sim loop, after arm.ctrl_target() sets data.ctrl[:5]:
    if arm.phase in ("ALIGN", "ALIGN2"):
        target_xy = np.array([wm_xyz[0], wm_xyz[1], blade_xyz[2]])
        data.ctrl[:5] = fb.servo(model, data, target_xy, data.ctrl[:5])
"""

import numpy as np
import mujoco

# Right-arm joint names (must match scene_robot.xml)
_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
)

# Joint limits [min, max] in radians, from scene_robot.xml
_ARM_LIMITS = np.array([
    [-3.08,   2.04 ],   # shoulder pitch
    [-2.993,  0.061],   # shoulder roll
    [-2.556,  2.556],   # shoulder yaw
    [-2.3556, 0.0  ],   # elbow
    [-2.556,  2.556],   # wrist yaw
])


class FeedbackController:
    """
    Jacobian-transpose position feedback for the FF Master right arm.

    Parameters
    ----------
    model    : MuJoCo model (used to cache joint DOF addresses)
    Kp       : proportional gain  [rad / m]  — tuned for dt = 0.002 s
    max_dq   : per-joint correction clip  [rad / step]
    """

    def __init__(self, model: mujoco.MjModel, *,
                 Kp: float = 1.6, max_dq: float = 0.07) -> None:
        sn = mujoco.mjtObj.mjOBJ_SITE
        jn = mujoco.mjtObj.mjOBJ_JOINT

        self._site_id = mujoco.mj_name2id(model, sn, "blade_tip")
        self._jnt_ids = [mujoco.mj_name2id(model, jn, n) for n in _ARM_JOINT_NAMES]
        # Velocity DOF addresses → Jacobian column indices
        self._vadr = [model.jnt_dofadr[jid] for jid in self._jnt_ids]
        self._Kp   = float(Kp)
        self._max_dq = float(max_dq)

        # Pre-allocated Jacobian buffers; reused every step to avoid GC pressure
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def compute_correction(self, model: mujoco.MjModel, data: mujoco.MjData,
                           target_xyz: np.ndarray) -> np.ndarray:
        """
        Compute Δq (shape 5,) for the right arm joints.

        Uses the Jacobian-transpose method::

            Δq = J_arm^T @ (target_xyz − blade_xyz) × Kp

        where J_arm is computed by mj_jacSite for the current robot state.
        Output is clipped to ±max_dq per joint.
        """
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self._site_id)
        J_arm = self._jacp[:, self._vadr]                        # (3, 5)
        error = np.asarray(target_xyz, float) - data.site_xpos[self._site_id]
        dq    = J_arm.T @ error * self._Kp                      # (5,)
        return np.clip(dq, -self._max_dq, self._max_dq)

    def servo(self, model: mujoco.MjModel, data: mujoco.MjData,
              target_xyz: np.ndarray,
              nominal_ctrl: np.ndarray) -> np.ndarray:
        """
        Return ctrl[0:5] = clip(nominal_ctrl + Δq, joint_limits).

        Safe to call every step: corrections are bounded and joint-limited.
        """
        dq   = self.compute_correction(model, data, target_xyz)
        ctrl = np.asarray(nominal_ctrl, float) + dq
        return np.clip(ctrl, _ARM_LIMITS[:, 0], _ARM_LIMITS[:, 1])
