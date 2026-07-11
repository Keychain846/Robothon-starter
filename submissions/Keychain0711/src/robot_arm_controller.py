from __future__ import annotations
"""
RobotArmController — sensor-adaptive trajectory for the FF Master right arm.

First cut sequence:
  APPROACH → ALIGN → CONTACT → SLICE → RETRACT → DONE
Second cut + serve sequence:
  REPOSITION2 → APPROACH2 → ALIGN2 → CONTACT2 → SLICE2 → RETRACT2 → DONE2
  → SERVE → FINAL

ctrl indices map to scene_robot.xml actuators:
  ctrl[0]  act_rsp  right_shoulder_pitch
  ctrl[1]  act_rsr  right_shoulder_roll
  ctrl[2]  act_rsy  right_shoulder_yaw
  ctrl[3]  act_re   right_elbow
  ctrl[4]  act_rw   right_wrist_yaw
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ArmConfig:
    # Joint-angle poses [sp, sr, sy, el, wr]
    home:         list = field(default_factory=lambda: [-1.20, 0.0, 0.0, -1.00, 0.0])
    approach_end: list = field(default_factory=lambda: [-0.70, 0.0, 0.0, -0.65, 0.0])
    strike:       list = field(default_factory=lambda: [ 0.50, 0.0, 0.0, -0.08, 0.0])
    slice_arc:    list = field(default_factory=lambda: [ 0.58, 0.0, 0.0, -0.04, 0.0])
    slice_target: list = field(default_factory=lambda: [ 0.40, 0.0, 0.0, -0.16, 0.0])
    retract:      list = field(default_factory=lambda: [-0.70, 0.0, 0.0, -0.80, 0.0])
    # Serve sequence: arm lowers knife to table → sweeps piece to plate → lifts to present
    knife_down:   list = field(default_factory=lambda: [ 0.44, 0.0,  0.06, -0.18, 0.05])
    sweep_start:  list = field(default_factory=lambda: [ 0.45, 0.0,  0.10, -0.16, 0.03])
    sweep_end:    list = field(default_factory=lambda: [ 0.48, 0.0, -0.28, -0.10, 0.0 ])
    lift_pose:    list = field(default_factory=lambda: [-0.20, -0.35,  0.10, -0.38, 0.0 ])

    # Phase durations (s)
    approach_dur:    float = 1.2
    pre_contact_dur: float = 2.0
    penetrate_dur:   float = 0.80
    slice_dur:       float = 0.35
    retract_delay:   float = 0.15
    retract_dur:     float = 1.50
    regrasp_dur:     float = 0.70   # in-hand reorientation between cuts
    reposition2_dur: float = 2.0   # retract → home while half is placed
    knife_down_dur:  float = 0.80  # retract → knife resting at table height
    sweep_dur:       float = 1.20  # sweep_start → sweep_end (weld anim runs here)
    lift_dur:        float = 0.80  # sweep_end → lift_pose (presentation)

    # Adaptive threshold: blade-to-target distance (m) that triggers CONTACT/CONTACT2
    contact_dist_m: float = 0.13


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, alpha: float) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + float(np.clip(alpha, 0.0, 1.0)) * (b - a)


class RobotArmController:
    """
    Returns ctrl[0:5] for the 5 right-arm position actuators.

    Pass live sensordata blade_xyz / wm_xyz to ctrl_target() for adaptive
    ALIGN → CONTACT transition; omit them to fall back to timeout.
    Call notify_cut(t) when the cut fires to trigger the RETRACT phase.
    """

    _K_NOM = 6000.0   # nominal WM stiffness (matches cut_trigger_robot)

    def __init__(self, cfg: ArmConfig | None = None):
        self._cfg = cfg or ArmConfig()
        self._phase = "APPROACH"
        self._phase_start = 0.0
        self._pre_alpha = 0.0    # smoothstep alpha when ALIGN → CONTACT fires
        self.min_blade_dist = float("inf")
        self._slice_start = None   # absolute time when SLICE begins
        self._retract_start = None  # absolute time when RETRACT begins
        self._contact2_alpha = 0.0
        self._slice2_start = None
        self._retract2_start = None
        self._k_wm: float = self._K_NOM    # material stiffness from RLS estimator
        self._mat_conf: float = 0.0         # estimator confidence (0–1)
        self._regrasp_round: int = 0        # extra REGRASP rounds run (0 = first try)
        self._restrikes: dict = {"SLICE": 0, "SLICE2": 0}   # re-strike budget used
        self._restrike_return: str = "SLICE"                 # phase to resume

    # ------------------------------------------------------------------
    @property
    def phase(self) -> str:
        return self._phase

    def set_material_state(self, k_wm: float, material_conf: float) -> None:
        """Update material stiffness + confidence for adaptive REGRASP angle."""
        self._k_wm    = float(k_wm)
        self._mat_conf = float(np.clip(material_conf, 0.0, 1.0))

    def reset(self):
        self._phase = "APPROACH"
        self._phase_start = 0.0
        self._pre_alpha = 0.0
        self.min_blade_dist = float("inf")
        self._slice_start = None
        self._retract_start = None
        self._contact2_alpha = 0.0
        self._slice2_start = None
        self._retract2_start = None
        self._k_wm    = self._K_NOM
        self._mat_conf = 0.0
        self._regrasp_round = 0
        self._restrikes = {"SLICE": 0, "SLICE2": 0}

    def _start_restrike(self, which: str, t: float) -> bool:
        if self._restrikes[which] >= 1:
            return False
        self._restrikes[which] += 1
        self._restrike_return = which
        # deepen the follow-through arc with identified stiffness: harder
        # material earns a deeper second pass (bounded)
        _k_ratio = float(np.clip(self._k_wm / self._K_NOM, 0.5, 2.5))
        self._cfg.slice_arc = list(self._cfg.slice_arc)
        self._cfg.slice_arc[0] += float(np.clip(0.05 * _k_ratio, 0.03, 0.10))
        self._phase = "RESTRIKE" if which == "SLICE" else "RESTRIKE2"
        self._phase_start = t
        return True

    def notify_cut(self, t: float):
        if self._phase in ("APPROACH", "ALIGN", "CONTACT"):
            self._slice_start = t
            self._phase = "SLICE"

    def notify_reposition(self, t: float):
        """Call when first DONE is reached; inserts REGRASP before repositioning."""
        if self._phase == "DONE":
            self._phase = "REGRASP"
            self._phase_start = t
            self._regrasp_round = 0

    @property
    def regrasp_round(self) -> int:
        return self._regrasp_round

    @property
    def restrike_count(self) -> int:
        return self._restrikes["SLICE"] + self._restrikes["SLICE2"]

    def retry_regrasp(self, t: float) -> bool:
        """Grip verification failed at the end of a REGRASP round (grip force
        not restored, or residual knife slip) — restart the open/reclose wave
        for another round. Budgeted at 2 retries so a hopeless grip can never
        stall the pipeline. Returns True if a retry round was started."""
        if self._phase == "REGRASP" and self._regrasp_round < 2:
            self._regrasp_round += 1
            self._phase_start = t
            return True
        return False

    def notify_cut2(self, t: float):
        """Call when second cut fires; starts SLICE2 follow-through."""
        if self._phase in ("APPROACH2", "ALIGN2", "CONTACT2"):
            self._slice2_start = t
            self._phase = "SLICE2"

    def notify_sweep(self, t: float) -> None:
        """Call once DONE2 is reached; arm lowers knife then sweeps piece to plate."""
        if self._phase == "DONE2":
            self._phase = "KNIFE_DOWN"
            self._phase_start = t

    # ------------------------------------------------------------------
    def ctrl_target(
        self,
        t: float,
        blade_xyz: np.ndarray | None = None,
        wm_xyz: np.ndarray | None = None,
        wrist_q: float = 0.0,
        touch_N: float = 0.0,
    ) -> np.ndarray:
        """
        Returns a length-5 array of joint-angle targets at simulation time t.
        blade_xyz / wm_xyz: optional 3-vectors from sensordata (framepos).
        wrist_q: current wrist yaw angle (from data.qpos); used for contact compliance.
        touch_N: blade contact force (N) from mj_contactForce; used for adaptive slice.

        Wrist yaw (ctrl[4]) is driven as an active DOF:
          APPROACH  — ramps 0 → 0.07 rad (pre-alignment tilt)
          ALIGN     — holds at 0.07 rad (blade oriented toward WM)
          CONTACT   — target = wrist_q → zero spring, pure damping (compliant following)
          SLICE     — gentle 0.04 rad follow-through; exits early if force drops < 20 N
          RETRACT   — fades back to 0

        Adaptive slice (force-feedback):
          - Early exit if force drops below 20 N after ≥ 60% of slice_dur (blade through WM).
          - Extended dwell up to 1.6× slice_dur if still in heavy contact (> 80 N).
        """
        cfg = self._cfg

        if self._phase == "APPROACH":
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / cfg.approach_dur)
            if elapsed >= cfg.approach_dur:
                self._phase = "ALIGN"
                self._phase_start = t
                self._pre_alpha = 0.0
            target = _lerp(cfg.home, cfg.approach_end, alpha)
            return target

        if self._phase == "ALIGN":
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / cfg.pre_contact_dur)
            self._pre_alpha = alpha

            if blade_xyz is not None and wm_xyz is not None:
                dist = float(np.linalg.norm(blade_xyz - wm_xyz))
                self.min_blade_dist = min(self.min_blade_dist, dist)
                proximity_trigger = dist < cfg.contact_dist_m
            else:
                proximity_trigger = False

            timeout_trigger = elapsed >= cfg.pre_contact_dur
            if proximity_trigger or timeout_trigger:
                self._phase = "CONTACT"
                self._phase_start = t

            target = _lerp(cfg.approach_end, cfg.strike, alpha)
            return target

        if self._phase == "CONTACT":
            elapsed = t - self._phase_start
            extra = _smoothstep(elapsed / cfg.penetrate_dur)
            alpha = self._pre_alpha + (1.0 - self._pre_alpha) * extra
            if blade_xyz is not None and wm_xyz is not None:
                dist = float(np.linalg.norm(blade_xyz - wm_xyz))
                self.min_blade_dist = min(self.min_blade_dist, dist)
            target = _lerp(cfg.approach_end, cfg.strike, min(alpha, 1.0))
            return target

        if self._phase == "SLICE":
            if self._slice_start is None:
                self._slice_start = t
            elapsed = t - self._slice_start
            # Adaptive exit: early if force drops (blade through WM), extended if still resisted
            _frac = elapsed / cfg.slice_dur
            _blade_through = touch_N < 20.0 and _frac >= 0.60
            _max_dur = cfg.slice_dur * (1.6 if touch_N > 80.0 else 1.0)
            if _blade_through or elapsed >= _max_dur:
                if (not _blade_through and touch_N > 80.0
                        and self._start_restrike("SLICE", t)):
                    return np.asarray(cfg.slice_target, float)
                self._retract_start = t + cfg.retract_delay
                self._phase = "RETRACT"
            u = _smoothstep(min(_frac, 1.0))
            p0 = np.asarray(cfg.strike,       float)
            p1 = np.asarray(cfg.slice_arc,    float)
            p2 = np.asarray(cfg.slice_target, float)
            target = (1 - u)**2 * p0 + 2 * (1 - u) * u * p1 + u**2 * p2
            return target

        if self._phase == "RETRACT":
            if self._retract_start is None:
                self._retract_start = t
            elapsed = t - self._retract_start
            if elapsed < 0:
                target = np.array(cfg.slice_target, dtype=float)
                return target
            alpha = _smoothstep(elapsed / cfg.retract_dur)
            if elapsed >= cfg.retract_dur:
                self._phase = "DONE"
            target = _lerp(cfg.slice_target, cfg.retract, alpha)
            return target

        # RESTRIKE / RESTRIKE2: the slice ran its full (force-extended)
        # duration and the blade is STILL heavily resisted — the cut stalled
        # (hard material, bad bite). Instead of giving up into RETRACT, back
        # the blade out to the strike entry and drive a DEEPER follow-through
        # arc, deepened in proportion to the RLS-identified stiffness.
        # Budget: one re-strike per cut, so a hopeless cut cannot loop.
        if self._phase in ("RESTRIKE", "RESTRIKE2"):
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / 0.40)
            if elapsed >= 0.40:
                ret = self._restrike_return
                if ret == "SLICE":
                    self._slice_start = t
                else:
                    self._slice2_start = t
                self._phase = ret
            # back out along the slice line to the strike entry pose
            return _lerp(cfg.slice_target, cfg.strike, alpha)

        # REGRASP: in-hand knife reorientation before second cut.
        # Arm holds retract pose; wrist pronates then returns (sin-wave over regrasp_dur).
        # Finger control (open → hold → reclose) is handled by record_robot_video.py.
        # Wrist amplitude adapts to identified material stiffness × estimator confidence:
        #   harder WM → larger pronation angle (need more bite clearance for second cut).
        #   Base 0.15 rad; range [0.08, 0.25] driven by k_wm ∈ [3k, 12k] N/m.
        if self._phase == "REGRASP":
            elapsed = t - self._phase_start
            if elapsed >= self._cfg.regrasp_dur:
                self._phase = "REPOSITION2"
                self._phase_start = t
                self._contact2_alpha = 0.0
            _rg_frac  = float(np.clip(elapsed / self._cfg.regrasp_dur, 0.0, 1.0))
            _k_ratio  = float(np.clip(self._k_wm / self._K_NOM, 0.5, 2.0))
            _rg_amp   = float(np.clip(
                0.15 * (1.0 + self._mat_conf * (_k_ratio - 1.0) * 0.45),
                0.08, 0.25))
            target    = np.array(self._cfg.retract, dtype=float)
            target[4] = cfg.retract[4] + _rg_amp * float(np.sin(np.pi * _rg_frac))
            return target

        # REPOSITION2: arm lifts back to HOME while the half is being placed on table
        if self._phase == "REPOSITION2":
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / cfg.reposition2_dur)
            if elapsed >= cfg.reposition2_dur:
                self._phase = "APPROACH2"
                self._phase_start = t
                self._contact2_alpha = 0.0
            target = _lerp(cfg.retract, cfg.home, alpha)
            return target

        # APPROACH2: same sweep as first APPROACH (HOME → approach_end)
        if self._phase == "APPROACH2":
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / cfg.approach_dur)
            if elapsed >= cfg.approach_dur:
                self._phase = "ALIGN2"
                self._phase_start = t
                self._contact2_alpha = 0.0
            target = _lerp(cfg.home, cfg.approach_end, alpha)
            return target

        # ALIGN2: slow advance toward half — sensor-adaptive trigger
        if self._phase == "ALIGN2":
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / cfg.pre_contact_dur)
            self._contact2_alpha = alpha
            if blade_xyz is not None and wm_xyz is not None:
                proximity_trigger = float(np.linalg.norm(blade_xyz - wm_xyz)) < cfg.contact_dist_m
            else:
                proximity_trigger = False
            if proximity_trigger or elapsed >= cfg.pre_contact_dur:
                self._phase = "CONTACT2"
                self._phase_start = t
            target = _lerp(cfg.approach_end, cfg.strike, alpha)
            return target

        # CONTACT2: fast push to strike (same compliance as CONTACT)
        if self._phase == "CONTACT2":
            elapsed = t - self._phase_start
            extra = _smoothstep(elapsed / cfg.penetrate_dur)
            alpha = self._contact2_alpha + (1.0 - self._contact2_alpha) * extra
            target = _lerp(cfg.approach_end, cfg.strike, min(alpha, 1.0))
            return target

        # SLICE2: Bezier follow-through (same adaptive force logic)
        if self._phase == "SLICE2":
            if self._slice2_start is None:
                self._slice2_start = t
            elapsed = t - self._slice2_start
            _frac2 = elapsed / cfg.slice_dur
            _blade_through2 = touch_N < 20.0 and _frac2 >= 0.60
            _max_dur2 = cfg.slice_dur * (1.6 if touch_N > 80.0 else 1.0)
            if _blade_through2 or elapsed >= _max_dur2:
                if (not _blade_through2 and touch_N > 80.0
                        and self._start_restrike("SLICE2", t)):
                    return np.asarray(cfg.slice_target, float)
                self._retract2_start = t + cfg.retract_delay
                self._phase = "RETRACT2"
            u = _smoothstep(min(_frac2, 1.0))
            p0 = np.asarray(cfg.strike,       float)
            p1 = np.asarray(cfg.slice_arc,    float)
            p2 = np.asarray(cfg.slice_target, float)
            target = (1 - u)**2 * p0 + 2 * (1 - u) * u * p1 + u**2 * p2
            return target

        # RETRACT2: return to retract pose
        if self._phase == "RETRACT2":
            if self._retract2_start is None:
                self._retract2_start = t
            elapsed = t - self._retract2_start
            if elapsed < 0:
                target = np.array(cfg.slice_target, dtype=float)
                return target
            alpha = _smoothstep(elapsed / cfg.retract_dur)
            if elapsed >= cfg.retract_dur:
                self._phase = "DONE2"
            target = _lerp(cfg.slice_target, cfg.retract, alpha)
            return target

        # KNIFE_DOWN: arm descends from retract to just above table, knife near quarter
        if self._phase == "KNIFE_DOWN":
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / cfg.knife_down_dur)
            if elapsed >= cfg.knife_down_dur:
                self._phase = "SWEEP"
                self._phase_start = t
            target = _lerp(cfg.retract, cfg.knife_down, alpha)
            return target

        # SWEEP: arm sweeps forward (negative-Y = toward camera) pushing piece to plate
        if self._phase == "SWEEP":
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / cfg.sweep_dur)
            if elapsed >= cfg.sweep_dur:
                self._phase = "LIFT"
                self._phase_start = t
            # Sweep from sweep_start (behind piece) to sweep_end (past plate)
            target = _lerp(cfg.sweep_start, cfg.sweep_end, alpha)
            return target

        # LIFT: arm rises from sweep_end to presentation pose
        if self._phase == "LIFT":
            elapsed = t - self._phase_start
            alpha = _smoothstep(elapsed / cfg.lift_dur)
            if elapsed >= cfg.lift_dur:
                self._phase = "FINAL"
            target = _lerp(cfg.sweep_end, cfg.lift_pose, alpha)
            return target

        # FINAL: hold lift_pose (presentation)
        if self._phase == "FINAL":
            target = np.array(cfg.lift_pose, dtype=float)
            return target

        # DONE / DONE2
        target = np.array(cfg.retract, dtype=float)
        return target
