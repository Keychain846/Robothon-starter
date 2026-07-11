"""
video_effects.py
----------------
Render-only cosmetic effects for the demo video: juice-splash bursts on each
cut, blade speed glow, and the top-view inset compositor. None of these touch
qpos/qvel — they only mutate geom rgba/pos of decorative geoms and pixels.

Extracted from record_robot_video.py so the recording script stays focused on
control + physics; each effect is unit-testable without a renderer.
"""
from __future__ import annotations

import mujoco
import numpy as np

_GRAVITY = 9.81


class JuiceSplash:
    """Ballistic spray of decorative juice spheres, one burst per cut.

    The spheres are geoms parked (alpha 0) at their scene positions; a burst
    gives each an outward velocity ∝ cut force, then `step()` integrates a
    simple ballistic fall and fades them out over ~50 render frames.
    """

    def __init__(self, model: mujoco.MjModel, geom_names: list[str],
                 base_speed: float = 0.32):
        gt = mujoco.mjtObj.mjOBJ_GEOM
        self.ids = [gid for gid in (mujoco.mj_name2id(model, gt, n)
                                    for n in geom_names) if gid != -1]
        self.pos0 = (np.vstack([model.geom_pos[g].copy() for g in self.ids])
                     if self.ids else np.zeros((0, 3)))
        self.dirs = (np.array([p / (np.linalg.norm(p) + 1e-9) for p in self.pos0])
                     if len(self.pos0) else np.zeros((0, 3)))
        self.vel  = np.zeros_like(self.dirs)
        self.base_speed = base_speed
        self.countdown  = 0

    def reset(self, model: mujoco.MjModel):
        for i, gid in enumerate(self.ids):
            model.geom_rgba[gid, 3] = 0.0
            model.geom_pos[gid]     = self.pos0[i].copy()
        self.vel[:] = 0.0
        self.countdown = 0

    def burst(self, model: mujoco.MjModel, force_n: float, frames: int = 50):
        """Fire the spray with speed scaled by the cut's contact force."""
        fscale = float(np.clip(force_n / 150.0, 0.6, 2.8))
        self.vel[:] = self.dirs * (self.base_speed * fscale)
        for gid in self.ids:
            model.geom_rgba[gid, 3] = 1.0
        self.countdown = frames

    def step(self, model: mujoco.MjModel, dt: float):
        if self.countdown <= 0:
            return
        self.vel[:, 2] -= _GRAVITY * dt
        for i, gid in enumerate(self.ids):
            model.geom_pos[gid] += self.vel[i] * dt
        alpha = min(1.0, self.countdown / 12.0)
        for gid in self.ids:
            model.geom_rgba[gid, 3] = alpha
        self.countdown -= 1
        if self.countdown == 0:
            self.reset(model)


def blade_glow(model: mujoco.MjModel, blade_gid: int, rgba0: np.ndarray,
               blade_speed_ms: float, active: bool):
    """Lerp the blade colour white → orange-red with live blade speed."""
    if not active:
        model.geom_rgba[blade_gid, :3] = rgba0[:3]
        return
    glow = float(np.clip(blade_speed_ms / 0.75, 0.0, 1.0))
    target = (0.95, 0.25, 0.00)
    for c in range(3):
        model.geom_rgba[blade_gid, c] = float(
            rgba0[c] + glow * (target[c] - rgba0[c]))


def paste_inset(pixels: np.ndarray, inset: np.ndarray, x0: int, y0: int,
                border=(100, 110, 160)):
    """Composite the top-view inset into the frame with a 1 px border."""
    h, w = inset.shape[:2]
    pixels[y0:y0 + h, x0:x0 + w] = inset
    bc = list(border)
    pixels[y0 - 1:y0,         x0 - 1:x0 + w + 1] = bc
    pixels[y0 + h:y0 + h + 1, x0 - 1:x0 + w + 1] = bc
    pixels[y0 - 1:y0 + h + 1, x0 - 1:x0]         = bc
    pixels[y0 - 1:y0 + h + 1, x0 + w:x0 + w + 1] = bc
    return pixels
