"""
CutTriggerRobot — contact detection + weld-based placement for the FF Master robot.

After each cut the pieces are positioned by UPDATING the equality-weld reference
rather than releasing them to free-body physics.  This guarantees they stay on
the table regardless of any collision or friction subtleties.

First cut  : wm_whole → wm_half_L (Y−0.11 m) + wm_half_R (Y+0.11 m), both welded.
Reposition : wm_L_weld reference animated back to WM centre (handled by the caller).
Second cut : wm_half_L → wm_quarter_A (Y−0.065 m) + wm_quarter_B (Y−0.200 m),
             both welded at quarter resting height (table_z + sphere_r).
"""

import mujoco
import numpy as np

_TABLE_Z   = 0.550   # table top surface z (from scene_robot.xml)
_HALF_R    = 0.090   # half-watermelon collision sphere radius
_QUARTER_R = 0.113   # quarter rest offset — 1.4× enlarged wedge rind radius
_HALF_REST = _TABLE_Z + _HALF_R      # 0.640 m — half sphere centre resting on table
_QTR_REST  = _TABLE_Z + _QUARTER_R  # 0.663 m — enlarged wedge centre resting on table

# Quarter body = Rx(-40°) * Rz(+20°):
#   Rz(+20°) aligns face2 (body −Y) with the horizontal camera direction.
#   Rx(−40°) then tilts the top of the piece AWAY from camera so face2 ends up
#   pointing 37° above horizontal — the piece "lies on its curved rind" with the
#   large flesh face tilted upward toward the camera, exactly like the reference photo.
#
# After compound rotation:
#   face2 normal → world (0.342, −0.720, 0.604) → 66% toward camera  (flesh face visible)
#   skin bottom  → world (0, −0.643, −0.766)     → 74% toward camera  (curved rind at bottom)
#   skin top     → world (0, +0.643, +0.766)     → faces away          (hidden behind piece)
#
# Hamilton product  q = Rx(−40°) * Rz(+20°):
_QTR_QUAT       = np.array([1.0, 0.0, 0.0, 0.0])  # wedge mesh is self-oriented (rind down, V up)
# Rz(180°): body −X (right-half flesh face) → world +X (toward camera)
_RIGHT_HALF_QUAT = np.array([0.0, 0.0, 0.0, 1.0])

# Work-integral cut trigger: accumulated F×v×dt must reach threshold before cut fires.
# At typical contact (F≈260 N, v≈0.08 m/s, dt=0.002 s): ~48 steps ≈ 0.10 s to fire.
_CUT_WORK_THRESHOLD = 2.0   # N·m (joules)

# 8-layer Griffith fracture model
# E_i = ∫ F·v · exp(-(depth-d_i)²/2σ²) · dt  (Gaussian depth weighting, σ = layer spacing)
# → adjacent layers share 60% energy → crack front propagates continuously, not jump-by-jump.
# Impulse: v_Y_i = V_Y · √(k_wm/K_NOM) · depth_scale(d_i)   (elastic energy ∝ k·d²)
#          v_Z_i = V_Z · √(k_wm/K_NOM)   (uniform upward kick)
# Cohesive zone: slab alpha 0→0.28 as E_i/G_c_i goes 0.5→1.0 (process zone pre-fracture)
_SLAB_GC    = (0.42, 0.40, 0.37, 0.35, 0.32, 0.30, 0.27, 0.25)   # J — G_c per layer pair
_SLAB_D_CTR = (0.0015, 0.003, 0.0045, 0.006, 0.0075, 0.009, 0.0105, 0.012)  # m — layer depths
_SLAB_SIGMA  = 0.0015   # m — Gaussian σ = layer spacing → 60.7% overlap with neighbors
_SLAB_Z_OFF  = (0.083, 0.070, 0.057, 0.045, 0.033, 0.020, 0.010, 0.000)   # WM-frame Z
_K_NOM       = 6000.0   # N/m — nominal stiffness; impulse scales as √(k_wm/_K_NOM)
_D_REF       = 0.012    # m — deepest layer (normalises depth-scale for impulse)
_SLAB_V_Y    = 0.38   # m/s  lateral base impulse (scaled by depth + stiffness)
_SLAB_V_Z    = 0.90   # m/s  upward base impulse


def _id(model, obj_type, name):
    return mujoco.mj_name2id(model, obj_type, name)


class CutTriggerRobot:
    """
    Monitors blade_edge_trigger ↔ wm_whole contact.
    On first hit  : hides whole WM, welds both halves to their table positions.
    On second hit : hides left half, welds both quarters to their table positions.
    All pieces are pinned by weld constraints — they cannot fall through the table.
    """

    def __init__(self, model, data):
        # per-instance work threshold — scaled by the material profile
        self.work_threshold = _CUT_WORK_THRESHOLD
        geom = mujoco.mjtObj.mjOBJ_GEOM
        body = mujoco.mjtObj.mjOBJ_BODY
        eq   = mujoco.mjtObj.mjOBJ_EQUALITY

        self._blade_edge_id = _id(model, geom, "blade_edge_trigger")
        self._blade_geom_id = _id(model, geom, "blade_geom")

        self._wm_id      = _id(model, geom, "wm_whole")
        self._wm_vis_id  = _id(model, geom, "wm_vis")
        self._wm_stem_id = _id(model, geom, "wm_stem")

        self._L_col  = _id(model, geom, "wm_left_col")
        self._L_skin = [_id(model, geom, n) for n in (
            "wm_left",
            "wm_left_green", "wm_left_pith", "wm_left_face", "wm_left_core",
            "wm_left_seed1", "wm_left_seed2", "wm_left_seed3", "wm_left_seed4",
            "wm_left_seed5", "wm_left_seed6", "wm_left_seed7",
        )]

        self._R_col  = _id(model, geom, "wm_right_col")
        self._R_skin = [_id(model, geom, n) for n in (
            "wm_right",
            "wm_right_green", "wm_right_pith", "wm_right_face", "wm_right_core",
            "wm_right_seed1", "wm_right_seed2", "wm_right_seed3", "wm_right_seed4",
            "wm_right_seed5", "wm_right_seed6", "wm_right_seed7",
        )]

        body_L = _id(model, body, "wm_half_L")
        body_R = _id(model, body, "wm_half_R")
        body_W = _id(model, body, "watermelon")
        jnt_L  = model.body_jntadr[body_L]
        jnt_R  = model.body_jntadr[body_R]
        jnt_W  = model.body_jntadr[body_W]
        self._qa_L = model.jnt_qposadr[jnt_L]
        self._qa_R = model.jnt_qposadr[jnt_R]
        self._qa_W = model.jnt_qposadr[jnt_W]
        self._qv_L = model.jnt_dofadr[jnt_L]
        self._qv_R = model.jnt_dofadr[jnt_R]

        self._eq_wm  = _id(model, eq, "wm_weld")
        self._eq_L   = _id(model, eq, "wm_L_weld")
        self._eq_R   = _id(model, eq, "wm_R_weld")
        self._eq_qA  = _id(model, eq, "wm_qA_weld")
        self._eq_qB  = _id(model, eq, "wm_qB_weld")

        # Parking/initial weld references — restored on reset()
        self._eq_L_base  = model.eq_data[self._eq_L,  3:6].copy()
        self._eq_R_base  = model.eq_data[self._eq_R,  3:6].copy()
        self._eq_qA_base = model.eq_data[self._eq_qA, 3:6].copy()
        self._eq_qB_base = model.eq_data[self._eq_qB, 3:6].copy()

        body_qA = _id(model, body, "wm_quarter_A")
        body_qB = _id(model, body, "wm_quarter_B")
        jnt_qA  = model.body_jntadr[body_qA]
        jnt_qB  = model.body_jntadr[body_qB]
        self._qa_qA = model.jnt_qposadr[jnt_qA]
        self._qa_qB = model.jnt_qposadr[jnt_qB]
        self._qv_qA = model.jnt_dofadr[jnt_qA]
        self._qv_qB = model.jnt_dofadr[jnt_qB]

        self._qA_col  = _id(model, geom, "wm_qA_col")
        self._qA_skin = [_id(model, geom, n) for n in (
            # face1 layers excluded: at body +X → after rotation appears 6cm to camera's
            # right as a separate floating disc, not overlapping with face2.
            # face2_green excluded: full 360° ring overrides the skin D-shape effect.
            # The skin ellipsoid (Z=0.050, only covers lower arc) provides green rind
            # at the outer/bottom edge only → D-shape appearance.
            "wm_qA_skin",
            "wm_qA_face2_pith", "wm_qA_face2_flesh", "wm_qA_face2_core",
            "wm_qA_seed1", "wm_qA_seed2", "wm_qA_seed3",
        )]
        self._qB_col  = _id(model, geom, "wm_qB_col")
        self._qB_skin = [_id(model, geom, n) for n in (
            "wm_qB_skin",
            "wm_qB_face2_pith", "wm_qB_face2_flesh", "wm_qB_face2_core",
            "wm_qB_seed1", "wm_qB_seed2", "wm_qB_seed3",
        )]

        sn = mujoco.mjtObj.mjOBJ_SENSOR
        self._adr_wm_p = model.sensor_adr[_id(model, sn, "wm_pos")]

        # Progressive cut-plane cross-sections (blade penetration visualisation)
        self._cut_plane_ids = [
            (_id(model, geom, f"wm_cut_plane{i}_skin"),
             _id(model, geom, f"wm_cut_plane{i}_flesh"))
            for i in range(4)
        ]

        # Layer-wise weld-release fracture slab bodies (4 pairs, 8 bodies total)
        def _slab_entry(i):
            L_jnt = model.body_jntadr[_id(model, body, f"wm_slab{i}_L")]
            R_jnt = model.body_jntadr[_id(model, body, f"wm_slab{i}_R")]
            return {
                "L_qa":  model.jnt_qposadr[L_jnt],
                "R_qa":  model.jnt_qposadr[R_jnt],
                "L_qv":  model.jnt_dofadr[L_jnt],
                "R_qv":  model.jnt_dofadr[R_jnt],
                "L_eid": _id(model, eq,   f"wm_slab{i}_L_weld"),
                "R_eid": _id(model, eq,   f"wm_slab{i}_R_weld"),
                "L_gid": _id(model, geom, f"wm_slab{i}_L_vis"),
                "R_gid": _id(model, geom, f"wm_slab{i}_R_vis"),
                "z_off": float(_SLAB_Z_OFF[i]),
            }
        self._slabs = [_slab_entry(i) for i in range(8)]
        self._torn_layers: set = set()
        self._layer_energy: list = [0.0] * 8   # per-layer Griffith energy accumulator (J)
        self._k_wm = _K_NOM                     # last known material stiffness for impulse scaling
        self._crack_front_cache: float = 0.0    # crack_front_mm from previous step
        self._crack_front_prev_mm: float = 0.0  # for finite-difference speed
        self._crack_front_speed: float = 0.0    # mm/s  (EMA-smoothed, ≥0)
        self._crack_front_accel: float = 0.0    # mm/s² (EMA-smoothed, signed)
        self._cod_disp: list = [0.0] * 8        # per-layer crack opening displacement (m)
        self._cod_active: set = set()            # layers that have entered COD phase
        self._last_tilt_deg: float = 0.0        # blade tilt at last update (for burst release)
        self._path_angle_rad: float = 0.0       # crack propagation plane angle (rad, ±3°)
                                                 # set once at first contact from blade tilt + k_wm

        self._cut_fired  = False
        self._cut2_fired = False
        self._cut_work1  = 0.0   # accumulated F×v×dt toward first cut
        self._cut_work2  = 0.0   # accumulated F×v×dt toward second cut

    @property
    def cut_fired(self):  return self._cut_fired

    @property
    def cut2_fired(self): return self._cut2_fired

    @property
    def tear_level(self) -> int:
        """Number of slab layers currently released (0–8)."""
        return len(self._torn_layers)

    @property
    def crack_front_mm(self) -> float:
        """Continuous crack front depth (mm): interpolates between torn layers via energy ratio."""
        n = len(self._slabs)
        if not self._torn_layers:
            # No layer torn yet — interpolate toward layer 0 based on energy progress
            r0 = min(self._layer_energy[0] / (_SLAB_GC[0] or 1.0), 1.0)
            return float(1000.0 * _SLAB_D_CTR[0] * r0)
        i_last = max(self._torn_layers)
        next_i = i_last + 1
        if next_i >= n:
            return float(1000.0 * _SLAB_D_CTR[i_last])
        r_next = min(self._layer_energy[next_i] / (_SLAB_GC[next_i] or 1.0), 1.0)
        d_last = _SLAB_D_CTR[i_last]
        d_next = _SLAB_D_CTR[next_i]
        return float(1000.0 * (d_last + r_next * (d_next - d_last)))

    @property
    def front_width_mm(self) -> float:
        """Active process zone width (mm): depth span of layers with E_i/G_c_i > 0.1."""
        k_f = float(np.sqrt(np.clip(self._k_wm / _K_NOM, 0.5, 2.0)))
        active = [
            _SLAB_D_CTR[i] for i in range(8)
            if i not in self._torn_layers
            and self._layer_energy[i] / (_SLAB_GC[i] * k_f) > 0.1
        ]
        if len(active) < 2:
            return 0.0
        return float(1000.0 * (max(active) - min(active)))

    @property
    def local_release_prob(self) -> float:
        """Max E_i/G_c_eff across untorn layers (0–1): fracture readiness of hottest plane."""
        untorn = [i for i in range(8) if i not in self._torn_layers]
        if not untorn:
            return 1.0
        k_f = float(np.sqrt(np.clip(self._k_wm / _K_NOM, 0.5, 2.0)))
        return float(min(
            max(self._layer_energy[i] / (_SLAB_GC[i] * k_f) for i in untorn),
            1.0
        ))

    @property
    def front_speed_mm_s(self) -> float:
        """EMA-smoothed crack front propagation speed (mm/s, always ≥ 0)."""
        return float(max(0.0, self._crack_front_speed))

    @property
    def front_accel_mm_s2(self) -> float:
        """EMA-smoothed crack front acceleration (mm/s²); positive = accelerating."""
        return float(self._crack_front_accel)

    @property
    def cut_work_pct(self) -> float:
        """Fraction of work threshold reached for first cut (0–1)."""
        return float(np.clip(self._cut_work1 / _CUT_WORK_THRESHOLD, 0.0, 1.0))

    @property
    def cut_work2_pct(self) -> float:
        """Fraction of work threshold reached for second cut (0–1)."""
        return float(np.clip(self._cut_work2 / _CUT_WORK_THRESHOLD, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _weld_body(self, model, data, eq_id, qa, qv, pos, quat=None):
        """Pin a freejoint body to world position `pos` (and optional `quat`) via weld."""
        p = np.asarray(pos, float)
        q = np.asarray(quat, float) if quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
        model.eq_data[eq_id, 3:6]  = p
        model.eq_data[eq_id, 6:10] = q
        data.qpos[qa:qa + 3]  = p
        data.qpos[qa + 3:qa + 7] = q
        data.qvel[qv:qv + 6]  = 0.0
        data.eq_active[eq_id] = 1   # keep weld ACTIVE — body cannot fall

    def _release_slab_layer(self, model, data, slab: dict,
                            wm_x: float, wm_y: float, layer_z: float,
                            v_scale: float = 1.0, layer_d: float = _D_REF,
                            blade_tilt_deg: float = 0.0,
                            cod_offset: float = 0.0) -> None:
        """Release one slab pair from COD-displaced position with angle-biased impulse.

        cod_offset  — crack opening displacement already applied (m); bodies start here,
                      impulse is reduced proportionally (remaining elastic energy only).
        blade_tilt  — tilted blade biases lateral ejection direction (sin θ × 0.25 factor).
        """
        depth_scale = float(np.clip(0.55 + 0.45 * (layer_d / _D_REF), 0.55, 1.0))
        v_y = _SLAB_V_Y * v_scale * depth_scale
        v_z = _SLAB_V_Z * v_scale
        # Bodies already partially open → only remaining elastic energy drives impulse
        _cod_max = 0.007
        cod_done = float(np.clip(cod_offset / _cod_max, 0.0, 1.0))
        v_y *= float(np.clip(1.0 - 0.60 * cod_done, 0.4, 1.0))
        v_z *= float(np.clip(1.0 - 0.30 * cod_done, 0.7, 1.0))
        # Blade angle biases lateral ejection (mode-II component)
        tilt_rad   = float(np.radians(np.clip(blade_tilt_deg, -30.0, 30.0)))
        angle_bias = float(np.sin(tilt_rad) * 0.25)
        # Crack path angle rotates the split plane: X component ∝ sin(path_angle)
        # Makes ejection direction depend on blade approach + material — path not predefined
        _pa  = self._path_angle_rad
        v_px = float(np.sin(_pa)) * v_y   # X-axis component of split velocity
        v_py = float(np.cos(_pa)) * v_y   # Y-axis component (dominant)
        for qa, qv, gid, y_off in (
            (slab["L_qa"], slab["L_qv"], slab["L_gid"], -cod_offset),
            (slab["R_qa"], slab["R_qv"], slab["R_gid"], +cod_offset),
        ):
            data.qpos[qa:qa + 3]     = [wm_x, wm_y + y_off, layer_z]
            data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[qv:qv + 6]     = 0.0
            model.geom_rgba[gid, 3]  = 0.88
        data.qvel[slab["L_qv"] + 0] = -v_px
        data.qvel[slab["L_qv"] + 1] = -(v_py + angle_bias * v_z)
        data.qvel[slab["R_qv"] + 0] = +v_px
        data.qvel[slab["R_qv"] + 1] = +(v_py - angle_bias * v_z)
        data.qvel[slab["L_qv"] + 2] = +v_z
        data.qvel[slab["R_qv"] + 2] = +v_z
        data.eq_active[slab["L_eid"]] = 0
        data.eq_active[slab["R_eid"]] = 0

    def update_blade_penetration(self, model, data, blade_z: float, wm_z: float,
                                 blade_force_n: float = 0.0,
                                 blade_vel_ms:  float = 0.0,
                                 k_wm: float = _K_NOM,
                                 blade_tilt_deg: float = 0.0) -> None:
        """
        Physics step: skin fade + cross-section reveal + 8-layer Griffith fracture.

        Energy (per plane):
          E_i += F·v · exp(-(depth-d_i)²/2σ²) · cos²(tilt) · dt
          cos²(tilt) = mode-I driving factor (tilted blade → more mode-II → less tensile)
          σ = layer spacing → 60.7% Gaussian overlap → continuous crack front

        Dynamic fracture toughness:
          G_c_eff_i = G_c_i · √(k_wm/K_NOM)    (stiffer WM needs more energy per area)

        Three-phase slab response (E_i / G_c_eff_i):
          0.5 → 0.85  cohesive zone: slab alpha 0→0.28 (process zone pre-fracture)
          0.85 → 1.0  COD phase: weld reference drifts laterally 0→7 mm (crack mouth opens)
          ≥ 1.0       fracture: weld released; impulse v_Y ∝ depth·√k, angle-biased by tilt
        """
        if self._cut_fired:
            return
        self._k_wm = k_wm
        self._last_tilt_deg = blade_tilt_deg
        depth = float(np.clip(wm_z + 0.09 - blade_z, 0.0, 0.18))
        # 1. Skin fade
        model.geom_rgba[self._wm_vis_id, 3] = float(np.clip(
            1.0 - depth / 0.045, 0.05, 1.0))
        # 2. Anatomical cross-section reveal
        for ld, (skin_id, flesh_id) in zip(
                (0.005, 0.025, 0.050, 0.075), self._cut_plane_ids):
            frac = float(np.clip((depth - ld) / 0.020, 0.0, 1.0))
            model.geom_rgba[skin_id,  3] = frac
            model.geom_rgba[flesh_id, 3] = frac
        # 3. Griffith 8-layer + COD + cohesive zone
        wm_x        = float(data.sensordata[self._adr_wm_p])
        wm_y        = float(data.sensordata[self._adr_wm_p + 1])
        dt          = float(model.opt.timestep)
        v_scale     = float(np.sqrt(np.clip(k_wm / _K_NOM, 0.25, 4.0)))
        k_factor    = float(np.sqrt(np.clip(k_wm / _K_NOM, 0.5, 2.0)))
        tilt_rad    = float(np.radians(np.abs(blade_tilt_deg)))
        mode_factor = float(np.clip(np.cos(tilt_rad) ** 2, 0.5, 1.0))
        # Lock in crack propagation plane at first contact — determined by blade approach
        # angle and material stiffness, not by fixed geometry. Max ±3° (±0.052 rad).
        if depth > 0.001 and self._path_angle_rad == 0.0:
            _tilt_c = float(np.sin(np.radians(blade_tilt_deg))) * 0.12
            _mat_c  = float(np.clip((k_wm - _K_NOM) / _K_NOM, -0.5, 0.5)) * 0.08
            self._path_angle_rad = float(np.clip(_tilt_c + _mat_c, -0.052, 0.052))
        for i, slab in enumerate(self._slabs):
            if i not in self._torn_layers:
                dz = depth - _SLAB_D_CTR[i]
                w  = float(np.exp(-0.5 * (dz / _SLAB_SIGMA) ** 2))
                self._layer_energy[i] += max(
                    0.0, blade_force_n * blade_vel_ms * w * mode_factor * dt)
                G_c_eff = _SLAB_GC[i] * k_factor
                e_ratio = self._layer_energy[i] / G_c_eff
                if e_ratio >= 1.0:
                    self._release_slab_layer(model, data, slab, wm_x, wm_y,
                                             wm_z + slab["z_off"],
                                             v_scale=v_scale,
                                             layer_d=_SLAB_D_CTR[i],
                                             blade_tilt_deg=blade_tilt_deg,
                                             cod_offset=self._cod_disp[i])
                    self._torn_layers.add(i)
                    self._cod_active.discard(i)
                elif e_ratio > 0.85:
                    # COD phase: animate weld ref laterally — crack mouth physically opens
                    if i not in self._cod_active:
                        layer_z = wm_z + slab["z_off"]
                        model.eq_data[slab["L_eid"], 3:6] = [wm_x, wm_y, layer_z]
                        model.eq_data[slab["R_eid"], 3:6] = [wm_x, wm_y, layer_z]
                        data.qpos[slab["L_qa"]:slab["L_qa"] + 3] = [wm_x, wm_y, layer_z]
                        data.qpos[slab["R_qa"]:slab["R_qa"] + 3] = [wm_x, wm_y, layer_z]
                        self._cod_active.add(i)
                    cod = float((e_ratio - 0.85) / 0.15) * 0.007
                    model.eq_data[slab["L_eid"], 4] = wm_y - cod
                    model.eq_data[slab["R_eid"], 4] = wm_y + cod
                    data.qpos[slab["L_qa"] + 1] = wm_y - cod
                    data.qpos[slab["R_qa"] + 1] = wm_y + cod
                    self._cod_disp[i] = cod
                    alpha = 0.28 + float((e_ratio - 0.85) / 0.15) * 0.55
                    model.geom_rgba[slab["L_gid"], 3] = alpha
                    model.geom_rgba[slab["R_gid"], 3] = alpha
                elif e_ratio > 0.5:
                    czm = float(np.clip((e_ratio - 0.5) * 2.0, 0.0, 1.0)) * 0.28
                    model.geom_rgba[slab["L_gid"], 3] = czm
                    model.geom_rgba[slab["R_gid"], 3] = czm
        # Crack front speed + acceleration (EMA-smoothed finite difference)
        _new_front = self.crack_front_mm
        _dt        = float(model.opt.timestep)
        _raw_spd   = max(0.0, (_new_front - self._crack_front_prev_mm) / max(_dt, 1e-6))
        _ema       = 0.12   # τ ≈ 8 steps ≈ 16 ms at dt=0.002 s
        _raw_acc   = (_raw_spd - self._crack_front_speed) / max(_dt, 1e-6)
        self._crack_front_accel = _ema * _raw_acc + (1.0 - _ema) * self._crack_front_accel
        self._crack_front_speed = _ema * _raw_spd + (1.0 - _ema) * self._crack_front_speed
        self._crack_front_prev_mm = _new_front
        self._crack_front_cache   = _new_front

    def _in_contact(self, model, data):
        for ci in range(data.ncon):
            c = data.contact[ci]
            if (c.geom1 == self._blade_edge_id and c.geom2 == self._wm_id) or \
               (c.geom2 == self._blade_edge_id and c.geom1 == self._wm_id):
                return True
        return False

    def step(self, model, data,
             touch_N: float = 0.0, blade_speed_ms: float = 0.0) -> bool:
        """Accumulate cutting work (F×v×dt); fire when threshold reached."""
        if self._cut_fired:
            return False
        if not self._in_contact(model, data):
            return False
        self._cut_work1 += touch_N * blade_speed_ms * float(model.opt.timestep)
        if self._cut_work1 < self.work_threshold:
            return False
        self._fire_cut(model, data)
        return True

    def _fire_cut(self, model, data):
        sd   = data.sensordata
        wm_x = float(sd[self._adr_wm_p])
        wm_y = float(sd[self._adr_wm_p + 1])
        wm_z = float(sd[self._adr_wm_p + 2])

        # Burst-release remaining slabs; honour COD offset and last known blade tilt
        v_scale = float(np.sqrt(np.clip(self._k_wm / _K_NOM, 0.25, 4.0)))
        for i, slab in enumerate(self._slabs):
            if i not in self._torn_layers:
                self._release_slab_layer(model, data, slab, wm_x, wm_y,
                                         wm_z + slab["z_off"],
                                         v_scale=v_scale, layer_d=_SLAB_D_CTR[i],
                                         blade_tilt_deg=self._last_tilt_deg,
                                         cod_offset=self._cod_disp[i])
                self._torn_layers.add(i)

        # Hide whole watermelon
        model.geom_rgba[self._wm_vis_id,  3] = 0.0
        model.geom_rgba[self._wm_stem_id, 3] = 0.0
        model.geom_contype[self._wm_id]    = 0
        model.geom_conaffinity[self._wm_id] = 0

        # Disable blade (no more contact with whole WM)
        model.geom_contype[self._blade_geom_id]    = 0
        model.geom_conaffinity[self._blade_geom_id] = 0
        model.geom_contype[self._blade_edge_id]    = 0
        model.geom_conaffinity[self._blade_edge_id] = 0

        # Release whole-WM weld only; half welds stay active at new positions
        data.eq_active[self._eq_wm] = 0

        # Left half stays at the WM cutting position (robot will cut it in-place).
        # Right half starts at the same position as left (separation animated externally).
        self._split1_L_final = np.array([wm_x, wm_y, _HALF_REST])
        self._split1_R_start = np.array([wm_x, wm_y, _HALF_REST])
        self._split1_R_final = np.array([wm_x, wm_y + 0.185, _HALF_REST])
        self._weld_body(model, data, self._eq_L, self._qa_L, self._qv_L,
                        self._split1_L_final)
        # Place R at same centre as L first; record_robot_video.py will animate it
        self._weld_body(model, data, self._eq_R, self._qa_R, self._qv_R,
                        self._split1_R_start, quat=_RIGHT_HALF_QUAT)

        for gid in self._L_skin: model.geom_rgba[gid, 3] = 1.0
        for gid in self._R_skin: model.geom_rgba[gid, 3] = 1.0

        # Remove progressive cut planes — halves now show the final cut faces
        for skin_id, flesh_id in self._cut_plane_ids:
            model.geom_rgba[skin_id,  3] = 0.0
            model.geom_rgba[flesh_id, 3] = 0.0
        # Hide slab bodies — they're flying but halves provide the definitive visual
        for slab in self._slabs:
            model.geom_rgba[slab["L_gid"], 3] = 0.0
            model.geom_rgba[slab["R_gid"], 3] = 0.0
        self._torn_layers.clear()

        mujoco.mj_forward(model, data)
        self._cut_fired = True

    def prepare_second_cut(self, model):
        """Re-enable blade so second cut can trigger."""
        model.geom_contype[self._blade_geom_id]    = 1
        model.geom_conaffinity[self._blade_geom_id] = 1
        model.geom_contype[self._blade_edge_id]    = 1
        model.geom_conaffinity[self._blade_edge_id] = 1

    def step2(self, model, data,
              touch_N: float = 0.0, blade_speed_ms: float = 0.0) -> bool:
        """Accumulate cutting work for second cut; fire when threshold reached."""
        if self._cut2_fired:
            return False
        in_contact2 = any(
            (c.geom1 == self._blade_edge_id and c.geom2 == self._L_col) or
            (c.geom2 == self._blade_edge_id and c.geom1 == self._L_col)
            for c in (data.contact[ci] for ci in range(data.ncon))
        )
        if not in_contact2:
            return False
        self._cut_work2 += touch_N * blade_speed_ms * float(model.opt.timestep)
        if self._cut_work2 < self.work_threshold:
            return False
        self._fire_cut2(model, data)
        return True

    def _fire_cut2(self, model, data):
        hx = float(data.qpos[self._qa_L])
        hy = float(data.qpos[self._qa_L + 1])

        # Hide left half (weld disabled after hiding)
        for gid in self._L_skin: model.geom_rgba[gid, 3] = 0.0
        model.geom_contype[self._L_col]    = 0
        model.geom_conaffinity[self._L_col] = 0
        data.eq_active[self._eq_L] = 0

        # Disable blade
        model.geom_contype[self._blade_geom_id]    = 0
        model.geom_conaffinity[self._blade_geom_id] = 0
        model.geom_contype[self._blade_edge_id]    = 0
        model.geom_conaffinity[self._blade_edge_id] = 0

        # Store animation start/end for the quarter-B separation (animated externally)
        # Quarter A is presented at a FIXED serving spot (0.487,−0.358) regardless
        # of where the melon was cut: the melon is quartered wherever it sits, then
        # the serving quarter is set down at the standard hand-off position the
        # Futurist's clamp is calibrated for. This makes the grasp position-robust
        # (a ±few-cm melon offset no longer moves the target out of the jaw).
        self._split2_qA_final = np.array([0.487, -0.358, _QTR_REST])
        self._split2_qB_start = np.array([hx, hy, _QTR_REST])
        self._split2_qB_final = np.array([hx - 0.09, hy - 0.04, _QTR_REST])
        # Quarters spread side-by-side in X (camera left↔right), both pulled toward
        # camera (−Y) relative to the half.  qA lands on the +X side — the Futurist
        # side — so the serving arm can reach it at full extension; qB on −X.
        # 0.19 m centre gap (> 2×0.065 radius → no sphere overlap), both on the table.
        self._weld_body(model, data, self._eq_qA, self._qa_qA, self._qv_qA,
                        self._split2_qA_final, quat=_QTR_QUAT)
        # qB starts at the centre (same as qA); animated to final by record_robot_video.py
        self._weld_body(model, data, self._eq_qB, self._qa_qB, self._qv_qB,
                        self._split2_qB_start, quat=_QTR_QUAT)

        for gid in self._qA_skin: model.geom_rgba[gid, 3] = 1.0
        model.geom_contype[self._qA_col]    = 1
        model.geom_conaffinity[self._qA_col] = 1

        for gid in self._qB_skin: model.geom_rgba[gid, 3] = 1.0
        model.geom_contype[self._qB_col]    = 1
        model.geom_conaffinity[self._qB_col] = 1

        mujoco.mj_forward(model, data)
        self._cut2_fired = True

    def set_R_weld_pos(self, model, data, pos) -> None:
        """Animate right-half weld during cut1 split (call each render frame)."""
        p = np.asarray(pos, float)
        model.eq_data[self._eq_R, 3:6] = p
        data.qpos[self._qa_R:self._qa_R + 3] = p
        data.qvel[self._qv_R:self._qv_R + 6] = 0.0

    def set_qB_weld_pos(self, model, data, pos) -> None:
        """Animate quarter-B weld during cut2 split (call each render frame)."""
        p = np.asarray(pos, float)
        model.eq_data[self._eq_qB, 3:6]  = p
        data.qpos[self._qa_qB:self._qa_qB + 3]     = p
        data.qpos[self._qa_qB + 3:self._qa_qB + 7] = _QTR_QUAT
        data.qvel[self._qv_qB:self._qv_qB + 6]     = 0.0

    def get_qA_pos(self, data) -> np.ndarray:
        """Current wm_quarter_A world position (from qpos)."""
        return data.qpos[self._qa_qA:self._qa_qA + 3].copy()

    def set_qA_weld_pos(self, model, data, pos, quat=None) -> None:
        """Teleport wm_quarter_A weld reference to pos (for smooth plate animation)."""
        p = np.asarray(pos, float)
        q = np.asarray(quat if quat is not None else _QTR_QUAT, float)
        model.eq_data[self._eq_qA, 3:6]  = p
        model.eq_data[self._eq_qA, 6:10] = q
        data.qpos[self._qa_qA:self._qa_qA + 3]     = p
        data.qpos[self._qa_qA + 3:self._qa_qA + 7] = q
        data.qvel[self._qv_qA:self._qv_qA + 6]     = 0.0

    def reset(self, model, data):
        """Restore all model/data mutations. Call before mj_resetDataKeyframe."""
        model.geom_rgba[self._wm_vis_id,  3] = 1.0
        model.geom_rgba[self._wm_stem_id, 3] = 1.0
        for skin_id, flesh_id in self._cut_plane_ids:
            model.geom_rgba[skin_id,  3] = 0.0
            model.geom_rgba[flesh_id, 3] = 0.0
        # Slab bodies: hide geoms; data.qpos/eq_active reset by mj_resetDataKeyframe
        for slab in self._slabs:
            model.geom_rgba[slab["L_gid"], 3] = 0.0
            model.geom_rgba[slab["R_gid"], 3] = 0.0
        self._torn_layers.clear()
        self._layer_energy = [0.0] * 8
        self._k_wm = _K_NOM
        self._crack_front_cache = 0.0
        self._cod_disp = [0.0] * 8
        self._cod_active.clear()
        self._last_tilt_deg = 0.0
        self._path_angle_rad = 0.0
        self._crack_front_prev_mm = 0.0
        self._crack_front_speed   = 0.0
        self._crack_front_accel   = 0.0
        model.geom_contype[self._wm_id]    = 1
        model.geom_conaffinity[self._wm_id] = 1

        model.geom_contype[self._blade_geom_id]    = 1
        model.geom_conaffinity[self._blade_geom_id] = 1
        model.geom_contype[self._blade_edge_id]    = 1
        model.geom_conaffinity[self._blade_edge_id] = 1

        for gid in self._L_skin: model.geom_rgba[gid, 3] = 0.0
        for gid in self._R_skin: model.geom_rgba[gid, 3] = 0.0

        model.geom_contype[self._L_col]    = 1
        model.geom_conaffinity[self._L_col] = 1
        model.geom_contype[self._R_col]    = 1
        model.geom_conaffinity[self._R_col] = 1

        for gid in self._qA_skin: model.geom_rgba[gid, 3] = 0.0
        model.geom_contype[self._qA_col]    = 0
        model.geom_conaffinity[self._qA_col] = 0
        for gid in self._qB_skin: model.geom_rgba[gid, 3] = 0.0
        model.geom_contype[self._qB_col]    = 0
        model.geom_conaffinity[self._qB_col] = 0

        # Restore weld eq_data to original parking positions
        for eq_id, base in (
            (self._eq_L,  self._eq_L_base),
            (self._eq_R,  self._eq_R_base),
            (self._eq_qA, self._eq_qA_base),
            (self._eq_qB, self._eq_qB_base),
        ):
            model.eq_data[eq_id, 3:6]  = base
            model.eq_data[eq_id, 6:10] = [1.0, 0.0, 0.0, 0.0]

        data.eq_active[self._eq_wm] = 1
        data.eq_active[self._eq_L]  = 1
        data.eq_active[self._eq_R]  = 1
        data.eq_active[self._eq_qA] = 1
        data.eq_active[self._eq_qB] = 1

        self._cut_fired  = False
        self._cut2_fired = False
        self._cut_work1  = 0.0
        self._cut_work2  = 0.0
