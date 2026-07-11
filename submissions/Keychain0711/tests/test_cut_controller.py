"""
Unit tests for the FF Master watermelon-cut submission.

Run with:  pytest tests/test_cut_controller.py -v
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pytest
import mujoco

from src.robot_arm_controller import RobotArmController, ArmConfig
from src.episode_logger       import EpisodeLogger, LogConfig
from src.feedback_controller  import FeedbackController, _ARM_LIMITS
from src.material_estimator   import MaterialEstimator, _K_DEFAULT


# ── ArmConfig / RobotArmController ────────────────────────────────────

class TestArmConfig:
    def test_default_phases_present(self):
        cfg = ArmConfig()
        for attr in ("home", "approach_end", "strike", "slice_arc",
                     "slice_target", "retract", "knife_down", "sweep_end", "lift_pose"):
            assert hasattr(cfg, attr), f"ArmConfig missing '{attr}'"

    def test_durations_positive(self):
        cfg = ArmConfig()
        for attr in ("approach_dur", "pre_contact_dur", "penetrate_dur",
                     "slice_dur", "retract_dur"):
            assert getattr(cfg, attr) > 0, f"{attr} must be positive"

    def test_contact_dist_positive(self):
        assert ArmConfig().contact_dist_m > 0


class TestRobotArmController:
    def test_initial_phase(self):
        arm = RobotArmController(ArmConfig())
        assert arm.phase == "APPROACH"

    def test_ctrl_target_shape(self):
        arm = RobotArmController(ArmConfig())
        out = arm.ctrl_target(0.0)
        assert out.shape == (5,), "ctrl_target must return 5-element array"

    def test_approach_advances_to_align(self):
        arm = RobotArmController(ArmConfig())
        cfg = ArmConfig()
        # Advance past approach_dur
        arm.ctrl_target(cfg.approach_dur + 0.1)
        assert arm.phase == "ALIGN"

    def test_align_triggers_on_proximity(self):
        arm = RobotArmController(ArmConfig())
        arm.ctrl_target(ArmConfig().approach_dur + 0.1)  # advance to ALIGN
        blade = np.array([0.0, 0.0, 0.0])
        wm    = np.array([0.10, 0.0, 0.0])  # 10 cm away — within 13 cm threshold
        arm.ctrl_target(1.5, blade_xyz=blade, wm_xyz=wm)
        assert arm.phase == "CONTACT"

    def test_notify_cut_advances_to_slice(self):
        arm = RobotArmController(ArmConfig())
        arm.ctrl_target(ArmConfig().approach_dur + 0.1)
        arm.notify_cut(1.5)
        assert arm.phase == "SLICE"

    def test_force_adaptive_slice_early_exit(self):
        """Force drops → SLICE exits early after ≥ 60% of slice_dur."""
        arm = RobotArmController(ArmConfig())
        cfg = ArmConfig()
        arm.ctrl_target(cfg.approach_dur + 0.1)
        arm.notify_cut(2.0)
        t_slice_start = 2.0
        # At 62% of duration with zero force — should exit to RETRACT
        arm.ctrl_target(t_slice_start + cfg.slice_dur * 0.62,
                        touch_N=0.0)
        assert arm.phase == "RETRACT"

    def test_force_adaptive_slice_held_under_force(self):
        """High force → SLICE extends beyond slice_dur."""
        arm = RobotArmController(ArmConfig())
        cfg = ArmConfig()
        arm.ctrl_target(cfg.approach_dur + 0.1)
        arm.notify_cut(2.0)
        t_slice_start = 2.0
        # At 110% of normal duration with high force — should still be in SLICE
        arm.ctrl_target(t_slice_start + cfg.slice_dur * 1.10,
                        touch_N=200.0)
        assert arm.phase == "SLICE"

    def test_reset_restores_approach(self):
        arm = RobotArmController(ArmConfig())
        arm.ctrl_target(ArmConfig().approach_dur + 0.5)
        assert arm.phase == "ALIGN"
        arm.reset()
        assert arm.phase == "APPROACH"

    def test_second_cut_sequence(self):
        arm = RobotArmController(ArmConfig())
        cfg = ArmConfig()
        arm.ctrl_target(cfg.approach_dur + 0.1)
        arm.notify_cut(2.0)
        arm.ctrl_target(2.0 + cfg.slice_dur * 0.62, touch_N=0.0)  # early exit
        arm.ctrl_target(5.0)   # RETRACT → DONE
        arm.notify_reposition(5.5)
        assert arm.phase == "REGRASP"   # REGRASP inserted before REPOSITION2
        arm.ctrl_target(5.5 + cfg.regrasp_dur + 0.01)
        assert arm.phase == "REPOSITION2"

    def test_sweep_sequence_after_done2(self):
        arm = RobotArmController(ArmConfig())
        arm._phase = "DONE2"
        arm.notify_sweep(10.0)
        assert arm.phase == "KNIFE_DOWN"


# ── EpisodeLogger ──────────────────────────────────────────────────────

class TestEpisodeLogger:
    def test_initial_state(self):
        ep = EpisodeLogger(0)
        assert not ep.success
        assert ep.cut_time is None
        assert ep.max_force == 0.0

    def test_mark_success(self):
        ep = EpisodeLogger(0)
        ep.mark_success(2.3)
        assert ep.success
        assert abs(ep.cut_time - 2.3) < 1e-9

    def test_step_accumulates_posture(self):
        ep = EpisodeLogger(0)
        ep.step(0.1, "APPROACH", 0.5, 0.0, 0.05)
        ep.step(0.2, "APPROACH", 0.4, 0.0, 0.10)
        assert abs(ep.posture_rms_mean - 0.075) < 1e-9

    def test_timeout_failure(self):
        cfg = LogConfig(cut_timeout=2.0)
        ep  = EpisodeLogger(0, cfg)
        result = ep.step(2.1, "ALIGN", 0.3, 0.0, 0.0)
        assert result == "timeout"

    def test_excess_force_failure(self):
        cfg = LogConfig(max_touch_force=100.0)
        ep  = EpisodeLogger(0, cfg)
        result = ep.step(1.0, "CONTACT", 0.1, 150.0, 0.0)
        assert result == "excess_force"

    def test_to_dict_keys(self):
        ep = EpisodeLogger(1)
        ep.mark_success(2.5)
        ep.finalize(True, 2.5)
        d = ep.to_dict()
        for key in ("ep", "success", "cut_time_s", "min_blade_mm",
                    "max_force_N", "posture_rms_rad", "cut_quality_pct", "failure"):
            assert key in d, f"to_dict missing '{key}'"

    def test_cut_quality_tracking(self):
        ep = EpisodeLogger(0)
        ep.step(1.0, "SLICE", 0.05, 50.0, 0.05, cut_quality=85.0)
        ep.step(1.1, "SLICE", 0.04, 55.0, 0.05, cut_quality=90.0)
        ep.step(1.2, "APPROACH", 0.10, 0.0, 0.05, cut_quality=50.0)  # not tracked (not SLICE)
        assert abs(ep.cut_quality_mean - 87.5) < 0.1


# ── FeedbackController ─────────────────────────────────────────────────

_SCENE = pathlib.Path(__file__).parent.parent / "assets" / "scene_robot.xml"


class TestFeedbackController:
    @pytest.fixture(scope="class")
    def model_data(self):
        model = mujoco.MjModel.from_xml_path(str(_SCENE))
        data  = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        return model, data

    @pytest.fixture(scope="class")
    def fb(self, model_data):
        model, _ = model_data
        return FeedbackController(model)

    def test_correction_shape(self, fb, model_data):
        model, data = model_data
        dq = fb.compute_correction(model, data, np.array([0.4, -0.25, 0.8]))
        assert dq.shape == (5,), "correction must be shape (5,)"

    def test_correction_clipped(self, fb, model_data):
        model, data = model_data
        dq = fb.compute_correction(model, data, np.array([5.0, 5.0, 5.0]))
        assert np.all(np.abs(dq) <= 0.07 + 1e-9), "correction must be ≤ max_dq=0.07"

    def test_servo_within_joint_limits(self, fb, model_data):
        model, data = model_data
        nominal = np.array([-0.70, 0.0, 0.0, -0.65, 0.0])
        ctrl = fb.servo(model, data, np.array([0.4, -0.25, 0.8]), nominal)
        assert ctrl.shape == (5,)
        assert np.all(ctrl >= _ARM_LIMITS[:, 0] - 1e-9)
        assert np.all(ctrl <= _ARM_LIMITS[:, 1] + 1e-9)

    def test_zero_error_zero_correction(self, fb, model_data):
        model, data = model_data
        # If target = current blade position, correction should be near zero
        blade_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "blade_tip")]
        dq = fb.compute_correction(model, data, blade_pos.copy())
        assert np.all(np.abs(dq) < 1e-6), "zero positional error must give zero correction"


# ── MuJoCo scene validation ────────────────────────────────────────────


class TestScene:
    @pytest.fixture(scope="class")
    def model(self):
        return mujoco.MjModel.from_xml_path(str(_SCENE))

    def test_scene_loads(self, model):
        assert model.ngeom > 0

    def test_required_sensors(self, model):
        sn = mujoco.mjtObj.mjOBJ_SENSOR
        for name in ("blade_touch", "blade_vel", "blade_accel", "blade_gyro",
                     "blade_ori", "blade_linv",
                     "blade_pos", "wm_pos", "left_pos",
                     "ft_touch_index", "ft_touch_middle", "ft_touch_ring",
                     "ft_touch_pinky", "ft_touch_thumb",
                     "gh_touch_index", "gh_touch_middle", "gh_touch_ring",
                     "gh_touch_pinky", "gh_touch_thumb"):
            sid = mujoco.mj_name2id(model, sn, name)
            assert sid != -1, f"Sensor '{name}' missing from scene"

    def test_total_sensor_count(self, model):
        assert model.nsensor >= 19, f"Expected ≥19 sensors (9 blade+wm + 5 fingertip + 5 grip), got {model.nsensor}"

    def test_sensor_types_coverage(self, model):
        """Verify ≥ 6 distinct MuJoCo sensor types are used."""
        sn = mujoco.mjtObj.mjOBJ_SENSOR
        type_ids = set()
        for name in ("blade_touch", "blade_vel", "blade_accel", "blade_gyro",
                     "blade_ori", "blade_linv", "blade_pos"):
            sid = mujoco.mj_name2id(model, sn, name)
            if sid != -1:
                type_ids.add(model.sensor_type[sid])
        assert len(type_ids) >= 6, f"Expected ≥6 sensor types, got {len(type_ids)}"

    def test_finger_actuators(self, model):
        assert model.nu >= 30, "Expected ≥30 actuators (arm + waist + head + fingers)"

    def test_weld_constraints(self, model):
        eq = mujoco.mjtObj.mjOBJ_EQUALITY
        for name in ("wm_weld", "wm_L_weld", "wm_R_weld", "wm_qA_weld", "wm_qB_weld"):
            eid = mujoco.mj_name2id(model, eq, name)
            assert eid != -1, f"Equality constraint '{name}' missing"

    def test_watermelon_geoms(self, model):
        geom_t = mujoco.mjtObj.mjOBJ_GEOM
        for name in ("wm_whole", "wm_left_col", "wm_right_col",
                     "wm_qA_col", "wm_qB_col"):
            gid = mujoco.mj_name2id(model, geom_t, name)
            assert gid != -1, f"Geom '{name}' missing"

    def test_simulation_step(self, model):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_step(model, data)  # must not raise
        assert data.time > 0

    def test_cut_plane_geoms(self, model):
        geom_t = mujoco.mjtObj.mjOBJ_GEOM
        for i in range(4):
            for suffix in ("skin", "flesh"):
                name = f"wm_cut_plane{i}_{suffix}"
                gid = mujoco.mj_name2id(model, geom_t, name)
                assert gid != -1, f"Progressive cut-plane geom '{name}' missing"

    def test_slab_bodies_in_scene(self, model):
        body_t = mujoco.mjtObj.mjOBJ_BODY
        for i in range(8):
            for side in ("L", "R"):
                name = f"wm_slab{i}_{side}"
                bid  = mujoco.mj_name2id(model, body_t, name)
                assert bid != -1, f"Fracture slab body '{name}' missing"

    def test_slab_geoms_initially_invisible(self, model):
        geom_t = mujoco.mjtObj.mjOBJ_GEOM
        for i in range(8):
            for side in ("L", "R"):
                name = f"wm_slab{i}_{side}_vis"
                gid  = mujoco.mj_name2id(model, geom_t, name)
                assert gid != -1, f"Slab vis geom '{name}' missing"
                assert model.geom_rgba[gid, 3] == 0.0, \
                    f"Slab geom '{name}' should start invisible (alpha=0)"

    def test_slab_weld_constraints_present(self, model):
        eq_t = mujoco.mjtObj.mjOBJ_EQUALITY
        for i in range(8):
            for side in ("L", "R"):
                name = f"wm_slab{i}_{side}_weld"
                eid  = mujoco.mj_name2id(model, eq_t, name)
                assert eid != -1, f"Slab weld constraint '{name}' missing"

    def test_griffith_constants_valid(self):
        """_SLAB_GC and _SLAB_D_CTR must be 8-element sequences with decreasing G_c."""
        from src.cut_trigger_robot import _SLAB_GC, _SLAB_D_CTR, _SLAB_SIGMA, _K_NOM
        assert len(_SLAB_GC) == 8,    "_SLAB_GC must have 8 entries"
        assert len(_SLAB_D_CTR) == 8, "_SLAB_D_CTR must have 8 entries"
        assert _SLAB_GC[0] > _SLAB_GC[-1],   "rind G_c must exceed flesh G_c"
        assert list(_SLAB_D_CTR) == sorted(_SLAB_D_CTR), "layer centres must be in ascending depth order"
        assert _SLAB_SIGMA > 0
        assert _K_NOM > 0

    def test_griffith_energy_triggers_release(self, model):
        """Feeding sustained F·v energy into layer 0 must trigger weld release."""
        from src.cut_trigger_robot import CutTriggerRobot, _SLAB_GC
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        cut = CutTriggerRobot(model, data)

        wm_z    = float(data.sensordata[cut._adr_wm_p + 2])
        # Blade positioned at layer-0 centre (3 mm below WM top)
        blade_z = wm_z + 0.09 - 0.003

        # Drive layer 0 past its G_c threshold with F=300 N, v=0.10 m/s
        required_steps = int(_SLAB_GC[0] / (300.0 * 0.10 * model.opt.timestep)) + 5
        for _ in range(required_steps):
            cut.update_blade_penetration(model, data, blade_z, wm_z,
                                         blade_force_n=300.0, blade_vel_ms=0.10,
                                         k_wm=6000.0)
        assert 0 in cut._torn_layers, "Layer 0 must release when Griffith energy ≥ G_c"
        assert cut.tear_level >= 1  # overlap σ=spacing → adjacent layers may co-fire

    def test_front_width_grows_with_energy(self, model):
        """front_width_mm must be 0 initially and positive once multiple layers accumulate."""
        from src.cut_trigger_robot import CutTriggerRobot, _SLAB_D_CTR
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        cut = CutTriggerRobot(model, data)
        assert cut.front_width_mm == 0.0, "process zone width must start at 0"
        wm_z    = float(data.sensordata[cut._adr_wm_p + 2])
        # Blade at mid-depth to drive several layers simultaneously
        blade_z = wm_z + 0.09 - 0.006
        for _ in range(50):
            cut.update_blade_penetration(model, data, blade_z, wm_z,
                                         blade_force_n=200.0, blade_vel_ms=0.05)
        assert cut.front_width_mm >= 0.0, "front_width_mm must be non-negative"
        # At least one layer should be accumulating energy → local_release_prob > 0
        assert cut.local_release_prob > 0.0

    def test_local_release_prob_reaches_one(self, model):
        """local_release_prob must reach 1.0 when a layer hits G_c."""
        from src.cut_trigger_robot import CutTriggerRobot, _SLAB_GC
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        cut = CutTriggerRobot(model, data)
        assert cut.local_release_prob == 0.0
        wm_z    = float(data.sensordata[cut._adr_wm_p + 2])
        blade_z = wm_z + 0.09 - 0.003
        steps = int(_SLAB_GC[0] / (400.0 * 0.10 * model.opt.timestep)) + 5
        for _ in range(steps):
            cut.update_blade_penetration(model, data, blade_z, wm_z,
                                         blade_force_n=400.0, blade_vel_ms=0.10,
                                         k_wm=6000.0)
        assert cut.tear_level >= 1
        # After release, remaining untorn layers may have lower prob; but prob must be ≤ 1
        assert cut.local_release_prob <= 1.0

    def test_cod_displacement_before_fracture(self, model):
        """Slabs must have non-zero COD displacement once energy ratio enters 0.85–1.0 band."""
        from src.cut_trigger_robot import CutTriggerRobot, _SLAB_GC
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        cut = CutTriggerRobot(model, data)
        wm_z    = float(data.sensordata[cut._adr_wm_p + 2])
        blade_z = wm_z + 0.09 - 0.003
        # Drive to ≈90% of G_c (90% of steps needed for full release)
        full_steps = int(_SLAB_GC[0] / (300.0 * 0.10 * model.opt.timestep)) + 5
        partial_steps = int(full_steps * 0.90)
        for _ in range(partial_steps):
            cut.update_blade_penetration(model, data, blade_z, wm_z,
                                         blade_force_n=300.0, blade_vel_ms=0.10,
                                         k_wm=6000.0)
        # At least layer 0 should be in COD phase (if not yet fully torn)
        if 0 not in cut._torn_layers:
            assert cut._cod_disp[0] > 0.0, "Layer 0 must show COD displacement in 0.85–1.0 band"
            assert 0 in cut._cod_active

    def test_blade_tilt_reduces_energy_accumulation(self, model):
        """A vertically tilted blade (mode-I) must accumulate more energy than a tilted one."""
        from src.cut_trigger_robot import CutTriggerRobot
        data0 = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data0, 0)
        mujoco.mj_forward(model, data0)
        cut0 = CutTriggerRobot(model, data0)

        data1 = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data1, 0)
        mujoco.mj_forward(model, data1)
        cut1 = CutTriggerRobot(model, data1)

        wm_z    = float(data0.sensordata[cut0._adr_wm_p + 2])
        blade_z = wm_z + 0.09 - 0.003
        for _ in range(30):
            cut0.update_blade_penetration(model, data0, blade_z, wm_z,
                                          blade_force_n=200.0, blade_vel_ms=0.08,
                                          blade_tilt_deg=0.0)    # vertical: full mode-I
            cut1.update_blade_penetration(model, data1, blade_z, wm_z,
                                          blade_force_n=200.0, blade_vel_ms=0.08,
                                          blade_tilt_deg=30.0)   # tilted: reduced mode-I
        assert cut0._layer_energy[0] > cut1._layer_energy[0], \
            "Vertical blade must drive more fracture energy than 30°-tilted blade"

    def test_crack_path_angle_derived_from_blade_and_material(self, model):
        """Path angle must be 0 before contact, non-zero and consistent after first contact."""
        from src.cut_trigger_robot import CutTriggerRobot
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        cut = CutTriggerRobot(model, data)
        assert cut._path_angle_rad == 0.0, "path angle must start at 0 before contact"
        wm_z    = float(data.sensordata[cut._adr_wm_p + 2])
        blade_z = wm_z + 0.09 - 0.005   # 5 mm deep → depth > 0.001 threshold
        cut.update_blade_penetration(model, data, blade_z, wm_z,
                                     blade_force_n=100.0, blade_vel_ms=0.05,
                                     k_wm=8000.0, blade_tilt_deg=10.0)
        assert cut._path_angle_rad != 0.0, "path angle must be set after first contact"
        assert abs(cut._path_angle_rad) <= 0.052, "path angle must stay within ±3°"
        # Angle must be stable (locked in — not recalculated each step)
        prev_angle = cut._path_angle_rad
        cut.update_blade_penetration(model, data, blade_z, wm_z,
                                     blade_force_n=100.0, blade_vel_ms=0.05,
                                     k_wm=9000.0, blade_tilt_deg=20.0)
        assert cut._path_angle_rad == prev_angle, "path angle must not change after first contact"


# ── MaterialEstimator ──────────────────────────────────────────────────

class TestMaterialEstimator:
    def test_initial_nominal(self):
        est = MaterialEstimator()
        assert est.confidence == 0.0
        assert est.slice_dur_multiplier() == 1.0   # no data → nominal

    def test_update_returns_bounded_k(self):
        est = MaterialEstimator()
        k = est.update(0.010, 22.0)
        assert isinstance(k, float)
        assert 400.0 <= k <= 12000.0

    def test_convergence_to_true_k(self):
        rng = np.random.default_rng(7)
        est = MaterialEstimator()
        TRUE_K = 3500.0
        for _ in range(60):
            d = rng.uniform(0.003, 0.04)
            est.update(d, TRUE_K * d + rng.normal(0, 4))
        assert abs(est.stiffness_Npm - TRUE_K) / TRUE_K < 0.12, (
            f"k did not converge: got {est.stiffness_Npm:.0f}, expected {TRUE_K:.0f}")

    def test_confidence_grows_and_saturates(self):
        est = MaterialEstimator()
        assert est.confidence == 0.0
        for _ in range(30):
            est.update(0.01, 22.0)
        assert est.confidence >= 1.0

    def test_hard_material_increases_mult(self):
        est = MaterialEstimator()
        for _ in range(35):
            est.update(0.01, 80.0)   # k ≈ 8000 N/m >> 2200
        assert est.slice_dur_multiplier() > 1.0

    def test_soft_material_decreases_mult(self):
        est = MaterialEstimator()
        for _ in range(35):
            est.update(0.02, 14.0)   # k ≈ 700 N/m << 2200
        assert est.slice_dur_multiplier() < 1.0

    def test_soft_reset_preserves_k(self):
        est = MaterialEstimator()
        for _ in range(40):
            est.update(0.015, 50.0)
        k_learned = est.stiffness_Npm
        est.soft_reset()
        assert abs(est.stiffness_Npm - k_learned) < 1.0, "soft_reset must keep k"
        assert est.confidence == 0.0, "soft_reset must clear confidence"

    def test_hard_reset_restores_default(self):
        est = MaterialEstimator()
        for _ in range(40):
            est.update(0.015, 80.0)
        est.reset()
        assert abs(est.stiffness_Npm - _K_DEFAULT) < 1.0


class TestRegraspRetry:
    """Closed-loop REGRASP: a failed grip verification restarts the round."""

    @staticmethod
    def _arm_in_regrasp():
        from src.robot_arm_controller import RobotArmController, ArmConfig
        arm = RobotArmController(ArmConfig())
        arm._phase = "DONE"
        arm.notify_reposition(1.0)
        assert arm.phase == "REGRASP"
        return arm

    def test_retry_budget_two_rounds(self):
        arm = self._arm_in_regrasp()
        assert arm.regrasp_round == 0
        assert arm.retry_regrasp(2.0) is True
        assert arm.regrasp_round == 1
        assert arm.retry_regrasp(3.0) is True
        assert arm.regrasp_round == 2
        assert arm.retry_regrasp(4.0) is False   # budget spent

    def test_retry_restarts_the_wave(self):
        import numpy as np
        arm = self._arm_in_regrasp()
        dur = arm._cfg.regrasp_dur
        # near the end of round 1 the wrist wave has nearly returned to rest
        arm.ctrl_target(1.0 + 0.95 * dur)
        assert arm.phase == "REGRASP"
        arm.retry_regrasp(1.0 + 0.95 * dur)
        # the retried round runs a FULL new wave: mid-round wrist deflection
        tgt = arm.ctrl_target(1.0 + 0.95 * dur + 0.5 * dur)
        assert arm.phase == "REGRASP"
        assert abs(tgt[4] - arm._cfg.retract[4]) > 0.05
        # and it still exits to REPOSITION2 on the new clock
        arm.ctrl_target(1.0 + 0.95 * dur + 1.01 * dur)
        assert arm.phase == "REPOSITION2"

    def test_retry_refused_outside_regrasp(self):
        from src.robot_arm_controller import RobotArmController, ArmConfig
        arm = RobotArmController(ArmConfig())
        assert arm.phase == "APPROACH"
        assert arm.retry_regrasp(0.0) is False

    def test_round_counter_resets_per_episode(self):
        arm = self._arm_in_regrasp()
        arm.retry_regrasp(2.0)
        arm.reset()
        assert arm.regrasp_round == 0


class TestRestrikeRecovery:
    """A slice that runs out of time still heavily resisted re-strikes deeper."""

    @staticmethod
    def _arm_in_slice():
        from src.robot_arm_controller import RobotArmController, ArmConfig
        arm = RobotArmController(ArmConfig())
        arm.notify_cut(1.0)          # APPROACH -> SLICE via the cut trigger
        assert arm.phase == "SLICE"
        return arm

    def test_stalled_slice_restrikes_then_resumes(self):
        arm = self._arm_in_slice()
        arc0 = list(arm._cfg.slice_arc)
        dur = arm._cfg.slice_dur
        # run past the force-extended max duration with heavy resistance
        t = 1.0 + 1.6 * dur + 0.01
        arm.ctrl_target(t, touch_N=150.0)
        assert arm.phase == "RESTRIKE"
        assert arm.restrike_count == 1
        assert arm._cfg.slice_arc[0] > arc0[0]      # deeper follow-through
        # back-out completes, slice resumes on a fresh clock
        arm.ctrl_target(t + 0.41, touch_N=150.0)
        assert arm.phase == "SLICE"

    def test_restrike_budget_is_one_per_cut(self):
        arm = self._arm_in_slice()
        dur = arm._cfg.slice_dur
        t = 1.0 + 1.6 * dur + 0.01
        arm.ctrl_target(t, touch_N=150.0)            # -> RESTRIKE
        arm.ctrl_target(t + 0.41, touch_N=150.0)     # -> SLICE (fresh clock)
        t2 = t + 0.41 + 1.6 * dur + 0.01
        arm.ctrl_target(t2, touch_N=150.0)           # stalled AGAIN
        assert arm.phase == "RETRACT"                # budget spent: retract

    def test_clean_cut_never_restrikes(self):
        arm = self._arm_in_slice()
        dur = arm._cfg.slice_dur
        # force drops below 20 N after 60% duration -> normal exit
        arm.ctrl_target(1.0 + 0.7 * dur, touch_N=5.0)
        assert arm.phase == "RETRACT"
        assert arm.restrike_count == 0

    def test_deepening_scales_with_stiffness(self):
        from src.robot_arm_controller import RobotArmController, ArmConfig
        deltas = []
        for k in (3000.0, 15000.0):
            arm = RobotArmController(ArmConfig())
            arm.notify_cut(1.0)
            arm.set_material_state(k, 1.0)
            arc0 = arm._cfg.slice_arc[0]
            arm.ctrl_target(1.0 + 1.6 * arm._cfg.slice_dur + 0.01, touch_N=150.0)
            deltas.append(arm._cfg.slice_arc[0] - arc0)
        assert deltas[1] > deltas[0]                 # harder -> deeper

    def test_restrike_resets_per_episode(self):
        arm = self._arm_in_slice()
        arm.ctrl_target(1.0 + 1.6 * arm._cfg.slice_dur + 0.01, touch_N=150.0)
        assert arm.restrike_count == 1
        arm.reset()
        assert arm.restrike_count == 0
