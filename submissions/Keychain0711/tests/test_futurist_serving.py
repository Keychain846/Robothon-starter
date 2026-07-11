"""
Unit tests for the Futurist serving stack: combined-model assembly,
articulated fingers, phase machine, hip-bow kinematics, and DLS-IK
waypoint tracking.

Run with:  pytest tests/test_futurist_serving.py -v
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pytest
import mujoco

from scripts.record_robot_video import _QA_PLATE_POS
from src.scene_builder import (
    build_combined_model as _build_combined_model,
    FTR_BASE_POS as _FTR_BASE_POS, FTR_BASE_QUAT as _FTR_BASE_QUAT)
from src.futurist_controller import (
    FuturistController, _PHASE_SEQUENCE, _PHASE_DUR,
    _P_GRIP, _P_CONTACT, _BOW_GRIP)


@pytest.fixture(scope="module")
def combined():
    model = _build_combined_model()
    data  = mujoco.MjData(model)
    ftr   = FuturistController()
    ftr.build(model, base_world_pos=_FTR_BASE_POS,
              base_world_quat=_FTR_BASE_QUAT)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ftr.reset(0.0)
    mujoco.mj_forward(model, data)
    return model, data, ftr


class TestModelAssembly:
    def test_futurist_attached(self, combined):
        model, _, _ = combined
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "ftr_right_hand") >= 0

    def test_articulated_fingers_present(self, combined):
        model, _, ftr = combined
        # 4 fingers + thumb per hand → 5 prox (incl thumb) + 4 dist
        assert len(ftr._rf_prox_adrs) == 5
        assert len(ftr._rf_dist_adrs) == 4
        assert len(ftr._lf_prox_adrs) == 5
        assert len(ftr._lf_dist_adrs) == 4

    def test_fingertip_sites(self, combined):
        model, _, _ = combined
        for s in ("ftr_rf0_tip", "ftr_rf3_tip", "ftr_rthumb_tip",
                  "ftr_lf0_tip", "ftr_lthumb_tip"):
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, s) >= 0, f"missing site {s}"

    def test_grip_weld_exists_inactive(self, combined):
        model, data, _ = combined
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                               "ftr_grip_weld")
        assert eq >= 0
        assert data.eq_active[eq] == 0


class TestPhaseMachine:
    def test_full_sequence_reaches_done(self, combined):
        model, data, ftr = combined
        ftr.reset(0.0)
        ftr.notify_start(0.0)
        t, dt = 0.0, 0.02
        for _ in range(2000):
            t += dt
            ftr.step(data, t)
            mujoco.mj_forward(model, data)
            if ftr.phase == "DONE":
                break
        assert ftr.phase == "DONE"

    def test_grip_and_release_flags_fire_once(self, combined):
        model, data, ftr = combined
        ftr.reset(0.0)
        ftr.notify_start(0.0)
        t, dt = 0.0, 0.02
        grips, releases = 0, 0
        for _ in range(2000):
            t += dt
            ga, gr = ftr.grip_should_activate, ftr.grip_should_release
            ftr.step(data, t)
            mujoco.mj_forward(model, data)
            grips    += int(ftr.grip_should_activate and not ga)
            releases += int(ftr.grip_should_release and not gr)
            if ftr.phase == "DONE":
                break
        assert grips == 1 and releases == 1

    def test_durations_positive(self):
        for ph in _PHASE_SEQUENCE:
            assert _PHASE_DUR[ph] > 0


class TestContactGatedGrasp:
    """① GRIP must not advance to LIFT until the grasp is confirmed."""

    @staticmethod
    def _run_to_grip(ftr, model, data):
        ftr.reset(0.0)
        ftr.notify_start(0.0)
        t, dt = 0.0, 0.02
        while ftr.phase != "GRIP" and t < 30.0:
            t += dt
            ftr.grasp_confirmed = True   # let earlier phases advance normally
            ftr.step(data, t)
            mujoco.mj_forward(model, data)
        assert ftr.phase == "GRIP"
        return t, dt

    def test_grip_holds_while_unconfirmed_then_advances(self, combined):
        model, data, ftr = combined
        t, dt = self._run_to_grip(ftr, model, data)
        # step well past GRIP's nominal duration with the grasp UNconfirmed
        for _ in range(int(_PHASE_DUR["GRIP"] / dt) + 5):
            t += dt
            ftr.grasp_confirmed = False
            ftr.step(data, t)
            mujoco.mj_forward(model, data)
        assert ftr.phase == "GRIP", "GRIP advanced before the grasp was confirmed"
        # now confirm — it should release the hold and move to LIFT
        for _ in range(10):
            t += dt
            ftr.grasp_confirmed = True
            ftr.step(data, t)
            mujoco.mj_forward(model, data)
        assert ftr.phase == "LIFT"

    def test_grip_times_out_if_never_confirmed(self, combined):
        model, data, ftr = combined
        t, dt = self._run_to_grip(ftr, model, data)
        grip_start = t
        # never confirm; the timeout fallback must still advance the sequence
        while ftr.phase == "GRIP" and (t - grip_start) < 5.0:
            t += dt
            ftr.grasp_confirmed = False
            ftr.step(data, t)
            mujoco.mj_forward(model, data)
        assert ftr.phase != "GRIP", "GRIP stalled forever without confirmation"

    def test_default_confirmed_advances_on_timer(self, combined):
        # bare callers (tests, teleop AUTO) leave grasp_confirmed True by default
        _, _, ftr = combined
        ftr.reset(0.0)
        assert ftr.grasp_confirmed is True

    def test_grip_timeout_backs_off_to_reach_then_exhausts_budget(self, combined):
        model, data, ftr = combined
        t, dt = self._run_to_grip(ftr, model, data)
        # 1st confirm-window expiry: FSM must RETRY (back off to REACH),
        # not lift a missed grasp
        while ftr.phase == "GRIP" and t < 60.0:
            t += dt
            ftr.grasp_confirmed = False
            ftr.step(data, t)
            mujoco.mj_forward(model, data)
        assert ftr.phase == "REACH"
        assert ftr.grip_retries == 1
        # keep failing: after the retry budget (2) the FSM advances on the
        # timer so a hopeless grasp can never stall the episode
        while ftr.phase in ("REACH", "GRIP") and t < 120.0:
            t += dt
            ftr.grasp_confirmed = False
            ftr.step(data, t)
            mujoco.mj_forward(model, data)
        assert ftr.grip_retries == 2
        assert ftr.phase == "LIFT"

    def test_grip_retry_counter_resets(self, combined):
        _, _, ftr = combined
        ftr.grip_retries = 2
        ftr.reset(0.0)
        assert ftr.grip_retries == 0


def test_teleop_futurist_smoke():
    """② the Futurist teleop interface runs AUTO + MANUAL + clamp + grab."""
    from scripts.teleop_futurist import main
    main(smoke=True)   # raises on any failure


class TestDynamicFingers:
    """Force-controlled digit architecture for `--grasp friction`."""

    def test_default_build_has_no_finger_actuators(self, combined):
        model, _, _ = combined
        assert model.nu == 30   # FF Master's actuators only

    def test_dynamic_build_adds_torque_limited_servos(self):
        m = _build_combined_model(dynamic_fingers=True)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(m.nu)]
        digits = [n for n in names if n and n.startswith("ftr_r")]
        assert len(digits) == 9          # 4×(prox+dist) + thumb
        assert m.nu == 39
        i = names.index("ftr_rthumb")
        j = names.index("ftr_rf0_prox")
        # the thumb carries ~4× the per-finger torque budget (balanced jaw)
        assert m.actuator_forcerange[i, 1] == pytest.approx(
            4.0 * m.actuator_forcerange[j, 1])
        assert 0.0 < m.actuator_forcerange[j, 1] < 1.0

    def test_controller_exempts_dynamic_digits(self):
        m = _build_combined_model(dynamic_fingers=True)
        d = mujoco.MjData(m)
        ftr = FuturistController()
        ftr.build(m, base_world_pos=_FTR_BASE_POS,
                  base_world_quat=_FTR_BASE_QUAT)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        ftr.reset(0.0)
        ftr.notify_start(0.0)
        adr = ftr._rf_all_qadrs[0]
        dadr = ftr._rf_all_dadrs[0]
        # kinematic mode: the controller overwrites the digit joint
        ftr.dynamic_fingers = False
        d.qpos[adr] = 0.77
        ftr.step(d, 0.02)
        assert d.qpos[adr] != 0.77
        # dynamic mode: digit qpos AND qvel survive the freeze/re-pin
        ftr.dynamic_fingers = True
        d.qpos[adr] = 0.77
        d.qvel[dadr] = 0.33
        ftr.step(d, 0.04)
        assert d.qpos[adr] == 0.77
        assert d.qvel[dadr] == 0.33

    def test_kinematic_arm_carries_fd_velocity(self):
        """mocap-style driving: commanded dofs carry finite-diff velocity."""
        m = _build_combined_model(dynamic_fingers=True)
        d = mujoco.MjData(m)
        ftr = FuturistController()
        ftr.build(m, base_world_pos=_FTR_BASE_POS,
                  base_world_quat=_FTR_BASE_QUAT)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        ftr.reset(0.0)
        ftr.dynamic_fingers = True
        ftr.notify_start(0.0)
        # step through a moving phase: arm dofs must show non-zero velocity
        for i in range(40):
            ftr.step(d, i * m.opt.timestep)
            mujoco.mj_forward(m, d)
        arm_dofs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in ("ftr_idx20_right_arm_joint1",)]
        v = float(abs(d.qvel[m.jnt_dofadr[arm_dofs[0]]]))
        assert v > 1e-4, "kinematic arm dof shows zero velocity in dynamic mode"


class TestKinematics:
    @staticmethod
    def _settle(ftr, model, data, phase, n=80):
        """Step a phase toward its end pose without letting it advance."""
        ftr.reset(0.0)
        ftr._phase = phase
        ftr._phase_start_t = 0.0
        t_max = _PHASE_DUR[phase] * 0.99
        for i in range(n):
            ftr.step(data, min(i * 0.02, t_max))
            mujoco.mj_forward(model, data)

    def test_grip_waypoint_reaches_low(self, combined):
        """With the hip bow, the wrist must dip below z=0.85 to reach the
        quarter zone — impossible with the arm alone (limit ≈ 0.91)."""
        model, data, ftr = combined
        self._settle(ftr, model, data, "GRIP")
        wrist = data.xpos[ftr._hand_bid]
        assert wrist[2] < 0.85, f"bowed GRIP wrist too high: {wrist}"

    def test_contact_fingertips_over_plate(self, combined):
        model, data, ftr = combined
        self._settle(ftr, model, data, "CONTACT_PLATE")
        # overhand placement: the wrist trails behind while the fingertip
        # centroid reaches over the wedge rest point on the plate
        tips = np.mean([data.site_xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)]
            for s in ("ftr_rf0_tip", "ftr_rf1_tip", "ftr_rf2_tip",
                      "ftr_rf3_tip", "ftr_rthumb_tip")], axis=0)
        assert np.linalg.norm(tips[:2] - _QA_PLATE_POS[:2]) < 0.06
        # tips grip the wedge flank: above the plate surface, at or below
        # the wedge-centre rest height
        assert _QA_PLATE_POS[2] - 0.06 < tips[2] < _QA_PLATE_POS[2] + 0.02

    def test_ik_waypoints_precomputed(self, combined):
        _, _, ftr = combined
        assert len(ftr._waypoints) >= 6

    def test_ik_tracking_converges(self, combined):
        """Task-space error at the end of a held phase must be small."""
        model, data, ftr = combined
        self._settle(ftr, model, data, "LOWER", n=120)
        assert ftr.track_err_m < 0.03, \
            f"IK residual too large: {ftr.track_err_m*1000:.0f} mm"

    def test_legs_stay_vertical_during_bow(self, combined):
        """Hip counter-rotation: at full bow the hip-pitch qpos equals −bow."""
        model, data, ftr = combined
        self._settle(ftr, model, data, "GRIP")
        hip = data.qpos[ftr._hip_joint_adrs[0]]
        assert abs(hip + _BOW_GRIP) < 0.05


class TestActuatedArm:
    """Experimental torque-driven serving arm (--futurist-drive actuated)."""

    @pytest.fixture(scope="class")
    def actuated(self):
        model = _build_combined_model(dynamic_fingers=True, actuated_arm=True)
        data = mujoco.MjData(model)
        ftr = FuturistController()
        ftr.build(model, base_world_pos=_FTR_BASE_POS,
                  base_world_quat=_FTR_BASE_QUAT)
        ftr.dynamic_fingers = True
        return model, data, ftr

    def test_arm_servos_present_and_torque_limited(self, actuated):
        model, _, ftr = actuated
        assert ftr.actuated_arm is True
        for aid in ftr._ra_act_ids:
            assert aid >= 0
            assert 0 < model.actuator_forcerange[aid, 1] <= 400.0

    def test_kinematic_build_not_actuated(self, combined):
        _, _, ftr = combined
        assert ftr.actuated_arm is False

    def test_controller_commands_ctrl_not_qpos(self, actuated):
        model, data, ftr = actuated
        mujoco.mj_resetData(model, data)
        ftr.reset(0.0)
        ftr.seed_idle(data)
        ftr.notify_steady(0.2)
        ftr.step(data, 0.5)
        # ctrl targets written for every arm servo
        assert any(abs(data.ctrl[a]) > 1e-6 for a in ftr._ra_act_ids)
        # the arm tracks under physics: joints move toward targets
        for _ in range(400):
            ftr.step(data, 0.5)
            mujoco.mj_step(model, data)
        errs = [abs(data.qpos[j] - data.ctrl[a])
                for j, a in zip(ftr._joint_adrs, ftr._ra_act_ids)]
        assert max(errs) < 0.6   # coarse tracking, not kinematic teleport

    def test_self_collision_excluded(self, actuated):
        model, _, _ = actuated
        assert model.nexclude >= 3
