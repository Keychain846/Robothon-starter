"""Unit tests for src/scene_builder.py — two-robot scene assembly."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import mujoco
import numpy as np
import pytest

from src.scene_builder import (build_combined_model, apply_material,
                               MATERIALS, FTR_BASE_POS)


@pytest.fixture(scope="module")
def kin_model():
    return build_combined_model(dynamic_fingers=False)


@pytest.fixture(scope="module")
def dyn_model():
    return build_combined_model(dynamic_fingers=True)


def test_combined_model_has_both_robots(kin_model):
    assert mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_BODY, "ftr_base_link") >= 0
    assert mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_GEOM, "blade_geom") >= 0
    assert mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_BODY, "wm_quarter_A") >= 0


def test_dynamic_fingers_add_digit_actuators(kin_model, dyn_model):
    # 4 fingers x 2 joints + thumb = 9 torque-limited actuators on the right hand
    assert dyn_model.nu == kin_model.nu + 9
    for name in ("ftr_rf0_prox", "ftr_rf3_dist", "ftr_rthumb"):
        aid = mujoco.mj_name2id(dyn_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, name
        assert dyn_model.actuator_forcerange[aid, 1] > 0  # torque-limited


def test_bite_equalities_only_in_friction_mode(kin_model, dyn_model):
    for bn in ("ftr_rf0_dist", "ftr_rthumb"):
        assert mujoco.mj_name2id(
            dyn_model, mujoco.mjtObj.mjOBJ_EQUALITY, f"bite_{bn}") >= 0
        assert mujoco.mj_name2id(
            kin_model, mujoco.mjtObj.mjOBJ_EQUALITY, f"bite_{bn}") == -1


def test_right_digits_on_collision_bit2(dyn_model):
    gid_prox = None
    bid = mujoco.mj_name2id(dyn_model, mujoco.mjtObj.mjOBJ_BODY, "ftr_rf0_prox")
    gid_prox = dyn_model.body_geomadr[bid]
    assert dyn_model.geom_conaffinity[gid_prox] == 2
    # left-hand digits stay collision-free (plate holder)
    bid_l = mujoco.mj_name2id(dyn_model, mujoco.mjtObj.mjOBJ_BODY, "ftr_lf0_prox")
    gid_l = dyn_model.body_geomadr[bid_l]
    assert dyn_model.geom_contype[gid_l] == 0


def test_grip_weld_declared_inactive(kin_model):
    eq = mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_EQUALITY, "ftr_grip_weld")
    assert eq >= 0
    assert kin_model.eq_active0[eq] == 0


def test_material_profiles_order_stiffness(kin_model):
    class _Cut:  # minimal stand-in for CutTriggerRobot
        work_threshold = 2.0
    gid = mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_GEOM, "wm_whole")
    solrefs, works = [], []
    for name in ("soft", "firm", "hard"):
        cut = _Cut()
        apply_material(kin_model, cut, name)
        solrefs.append(float(kin_model.geom_solref[gid, 0]))
        works.append(cut.work_threshold)
    assert solrefs[0] > solrefs[1] > solrefs[2]     # soft: softest contact
    assert works[0] < works[1] < works[2]           # hard: most fracture work
    assert set(MATERIALS) == {"soft", "firm", "hard"}


def test_futurist_base_position_reachable(kin_model):
    # base placed so the arm (~0.70 m) reaches the shared prep table
    assert 0.7 < FTR_BASE_POS[0] < 1.1
