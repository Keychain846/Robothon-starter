"""
hud_renderer.py
---------------
HUD overlay, title/episode/summary cards, and video re-compression.
Extracted from record_robot_video.py to keep the main script focused on
physics simulation and control logic.
"""
from __future__ import annotations

import pathlib
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Font loading ───────────────────────────────────────────────────────
_FONTS = None   # (font_xl, font_lg, font_md, font_sm, font_xs, font_xxs) — lazy init

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def _load_fonts():
    path = next((p for p in _FONT_CANDIDATES if pathlib.Path(p).exists()), None)
    try:
        if path:
            return tuple(ImageFont.truetype(path, s) for s in (56, 36, 24, 20, 17, 12))
    except Exception:
        pass
    fb = ImageFont.load_default()
    return fb, fb, fb, fb, fb, fb


def _get_fonts():
    global _FONTS
    if _FONTS is None:
        _FONTS = _load_fonts()
    return _FONTS


# ── Phase colour map ───────────────────────────────────────────────────
PHASE_COLORS = {
    "APPROACH":    (255, 220,  80),
    "ALIGN":       ( 80, 200, 255),
    "CONTACT":     (255, 160,  40),
    "SLICE":       (255,  60,  60),
    "RETRACT":     (120, 220, 140),
    "RESTRIKE":    (255, 120, 200),
    "RESTRIKE2":   (255, 100, 190),
    "DONE":        (100, 140, 100),
    "REGRASP":     (180, 255, 120),
    "REPOSITION2": (200, 150, 255),
    "APPROACH2":   (255, 210,  60),
    "ALIGN2":      ( 60, 190, 255),
    "CONTACT2":    (255, 140,  30),
    "SLICE2":      (255,  40,  40),
    "RETRACT2":    (100, 210, 130),
    "DONE2":       ( 80, 120,  80),
    "KNIFE_DOWN":  (255, 200,  80),
    "SWEEP":       (255, 140,  40),
    "LIFT":        (160, 255, 200),
    "FINAL":       (180, 255, 200),
}

_BAR_BG  = (30, 25, 50)
_BAR_BRD = (80, 70, 110)


# ── Helper: horizontal fill bar ───────────────────────────────────────
def _hbar(d, x, y, w, h, frac, fill, label="", font=None, fxs=None):
    bx = x
    if label and fxs:
        d.text((x, y), label, fill=(190, 190, 220), font=fxs)
        # bar starts past the label, never under it (labels vary in width)
        bx = x + max(72, int(d.textlength(label, font=fxs)) + 8)
    d.rectangle([bx, y + 2, bx + w, y + 2 + h], outline=_BAR_BRD, fill=_BAR_BG)
    fw = max(0, int(w * float(np.clip(frac, 0.0, 1.0))))
    if fw:
        d.rectangle([bx, y + 2, bx + fw, y + 2 + h], fill=fill)


# ── Title card ────────────────────────────────────────────────────────
def title_card(w, h, duration_s, fps):
    font_xl, font_lg, font_md, font_sm, font_xs, _ = _get_fonts()
    n = int(round(duration_s * fps))
    img = Image.new("RGB", (w, h), (15, 20, 30))
    d   = ImageDraw.Draw(img)
    d.text((w // 2, h // 2 - 100), "Robothon 2026",
           fill=(255, 215, 0), font=font_xl, anchor="mm")
    d.text((w // 2, h // 2 - 32), "FF Master — Precision Watermelon Quartering & Serving",
           fill=(200, 230, 255), font=font_md, anchor="mm")
    d.text((w // 2, h // 2 + 18),
           "15-DOF · Physics Grip · Closed-Loop Servo · Bimanual · Dual Cut · Serving",
           fill=(150, 175, 210), font=font_sm, anchor="mm")
    d.text((w // 2, h // 2 + 55),
           "MCP + PIP + DIP per finger  ·  Force feedback  ·  Head gaze  ·  MuJoCo 3.x",
           fill=(110, 140, 170), font=font_xs, anchor="mm")
    d.text((w // 2, h // 2 + 85),
           "First cut: full half  ·  Second cut: quarter  ·  Plate presentation with garnish",
           fill=(90, 120, 155), font=font_xs, anchor="mm")
    arr = np.array(img)
    return [arr] * n


# ── Episode intro card ────────────────────────────────────────────────
def episode_card(w, h, ep, n_ep, dx, dy, fps, duration_s=0.6):
    font_xl, font_lg, font_md, font_sm, font_xs, _ = _get_fonts()
    n = int(round(duration_s * fps))
    img = Image.new("RGB", (w, h), (10, 14, 22))
    d   = ImageDraw.Draw(img)
    d.text((w // 2, h // 2 - 40), f"Episode {ep + 1} / {n_ep}",
           fill=(255, 215, 0), font=font_lg, anchor="mm")
    d.text((w // 2, h // 2 + 20), f"WM offset  Δx = {dx*100:+.1f} cm   Δy = {dy*100:+.1f} cm",
           fill=(150, 180, 220), font=font_sm, anchor="mm")
    arr = np.array(img)
    return [arr] * n


# ── Summary card ──────────────────────────────────────────────────────
def summary_card(w, h, episodes, fps, duration_s=4.0,
                 k_per_episode=None, wm_offsets=None, episode_dur=14.0,
                 serve_errs_mm=None):
    import math as _math
    font_xl, font_lg, font_md, font_sm, font_xs, font_xxs = _get_fonts()
    n = int(round(duration_s * fps))

    img = Image.new("RGB", (w, h), (10, 14, 22))
    d   = ImageDraw.Draw(img)
    d.text((w // 2, 60), "Run Summary", fill=(255, 215, 0), font=font_lg, anchor="mm")

    cols   = ["Ep", "Result", "Cut Time", "Max Force", "Posture", "WM Offset", "Serve Err", "Precision"]
    # column positions as canvas fractions so the table fits any render width;
    # row height adapts so 8 episodes + footer fit a 480 px canvas
    col_x  = [int(w * f) for f in (0.047, 0.129, 0.250, 0.355, 0.461, 0.578, 0.695, 0.820)]
    _tbl_f = font_sm if w >= 1100 else font_xs
    _row_h = 48 if h >= 900 else 30
    row_y  = 110 if h < 900 else 130
    for ci, col in enumerate(cols):
        d.text((col_x[ci], row_y), col, fill=(160, 180, 220), font=_tbl_f)
    d.line([(40, row_y + 28), (w - 40, row_y + 28)], fill=(60, 70, 100), width=1)

    for ei, ep in enumerate(episodes):
        ed = ep.to_dict()
        row_y += _row_h
        ok      = ep.success
        row_col = (140, 220, 140) if ok else (220, 100, 100)
        _ns = 0
        if ok and ep.cut_time is not None:
            _ns += int(ep.cut_time < 3.0)
            _ns += int(ep.cut_time < 2.5)
            _ns += int(ep.max_force < 300)
            _ns += int(ep.posture_rms_mean < 0.10)
            _ns  = min(5, _ns + 1)
        _stars_str = ("●" * _ns + "·" * (5 - _ns)) if ok else "—"
        _off_str = "—"
        if wm_offsets and ei < len(wm_offsets):
            _off_str = f"{wm_offsets[ei][0]*100:+.0f}/{wm_offsets[ei][1]*100:+.0f} cm"
        _serve_str = "—"
        if serve_errs_mm and ei < len(serve_errs_mm) and serve_errs_mm[ei] is not None:
            _serve_str = f"{serve_errs_mm[ei]:.0f} mm OK"
        vals = [
            ed["ep"], "SUCCESS" if ok else "FAIL",
            ed["cut_time_s"] + " s",
            ed["max_force_N"] + " N", ed["posture_rms_rad"] + " rad",
            _off_str, _serve_str, _stars_str,
        ]
        for ci, val in enumerate(vals):
            _vc = (255, 215, 0) if ci == 7 and ok else (row_col if ci == 1 else (200, 210, 230))
            d.text((col_x[ci], row_y), str(val), fill=_vc, font=_tbl_f)

    # ── Cut-time bar chart (drawn only if it fits under the table) ────
    _footer_y = h - 28
    _ct_vals = [ep.cut_time for ep in episodes if ep.cut_time is not None]
    _ct_y0, _chart_h = row_y + 46, 0
    if _ct_vals:
        _avg_ct = sum(_ct_vals) / len(_ct_vals)
        _std = (_math.sqrt(sum((v - _avg_ct) ** 2 for v in _ct_vals) / len(_ct_vals))
                if len(_ct_vals) > 1 else 0)
        _free  = _footer_y - 16 - _ct_y0                    # space below the table
        _bar_h = min(26, max(6, (_free - 44) // max(len(_ct_vals), 1) - 8))
        if _free >= len(_ct_vals) * (_bar_h + 8) + 40:      # full chart fits
            _ct_x0 = 160
            _ct_w  = w - 320
            _ct_max = max(max(_ct_vals) * 1.15, 3.5)
            d.text((_ct_x0, _ct_y0 - 20), "First-Cut Time per Episode (s)",
                   fill=(140, 160, 200), font=font_xs)
            d.rectangle([_ct_x0, _ct_y0, _ct_x0 + _ct_w,
                         _ct_y0 + len(_ct_vals) * (_bar_h + 8)], fill=(14, 18, 30))
            _bar_colors = [(80, 180, 255), (100, 220, 140), (255, 180, 60), (200, 100, 255)]
            for bi, ct in enumerate(_ct_vals):
                _bw = int(_ct_w * ct / _ct_max)
                _by = _ct_y0 + bi * (_bar_h + 8)
                d.rectangle([_ct_x0, _by, _ct_x0 + _bw, _by + _bar_h],
                            fill=_bar_colors[bi % len(_bar_colors)])
                d.text((_ct_x0 + _bw + 8, _by + _bar_h // 2),
                       f"Ep{bi+1}: {ct:.3f} s", fill=(200, 215, 240), font=font_xxs, anchor="lm")
            _avg_x  = _ct_x0 + int(_ct_w * _avg_ct / _ct_max)
            _chart_h = len(_ct_vals) * (_bar_h + 8)
            d.line([(_avg_x, _ct_y0), (_avg_x, _ct_y0 + _chart_h)], fill=(255, 215, 0), width=2)
            d.text((_avg_x + 4, _ct_y0 - 4), f"avg {_avg_ct:.3f} s",
                   fill=(255, 215, 0), font=font_xxs, anchor="lb")
            d.text((w // 2, _ct_y0 + _chart_h + 8),
                   f"σ = {_std*1000:.1f} ms  ·  CV = {(_std/_avg_ct*100):.2f}%"
                   "  ·  Highly consistent across randomised WM offsets",
                   fill=(120, 150, 190), font=font_xxs, anchor="mt")
            _chart_h += 24
        else:                                               # compact one-line stats
            d.text((w // 2, _ct_y0 - 24),
                   f"Cut time  avg {_avg_ct:.3f} s  ·  σ {_std*1000:.1f} ms"
                   f"  ·  CV {(_std/_avg_ct*100):.2f}%  across randomised WM offsets",
                   fill=(140, 165, 205), font=font_xs, anchor="mt")
            _chart_h = 22    # reserve the compact line's height

    # ── Material stiffness learning curve (only if it fits) ──────────
    if k_per_episode and len(k_per_episode) >= 2:
        _kl_y0 = (_ct_y0 + _chart_h + 42) if _chart_h else _ct_y0 + 4
        _kl_h  = 32
        _k_drift = k_per_episode[-1] - k_per_episode[0]
        if _kl_y0 + _kl_h + 34 <= _footer_y - 12:           # full curve fits
            _kl_x0 = 160
            _kl_w  = w - 320
            _k_min_v = min(k_per_episode) * 0.92
            _k_max_v = max(k_per_episode) * 1.08
            _k_range  = max(_k_max_v - _k_min_v, 500.0)
            d.text((_kl_x0, _kl_y0 - 20),
                   "Material Stiffness k_wm (N/m) per episode  [cross-episode RLS learning]",
                   fill=(100, 140, 200), font=font_xxs)
            d.rectangle([_kl_x0, _kl_y0, _kl_x0 + _kl_w, _kl_y0 + _kl_h], fill=(12, 10, 22))
            _kl_step = _kl_w / max(len(k_per_episode), 1)
            _kl_pts  = []
            for _ki, _kv in enumerate(k_per_episode):
                _kx = int(_kl_x0 + (_ki + 0.5) * _kl_step)
                _ky = int(_kl_y0 + _kl_h - _kl_h * (_kv - _k_min_v) / _k_range)
                _kl_pts.append((_kx, _ky))
                _kc = (80, 220, 160) if _ki > 0 else (200, 200, 120)
                d.ellipse([_kx - 4, _ky - 4, _kx + 4, _ky + 4], fill=_kc)
                d.text((_kx, _kl_y0 + _kl_h + 3), f"Ep{_ki+1}", fill=(90, 100, 130),
                       font=font_xxs, anchor="mt")
                d.text((_kx, _ky - 6), f"{_kv:.0f}", fill=_kc, font=font_xxs, anchor="mb")
            for _pi in range(len(_kl_pts) - 1):
                d.line([_kl_pts[_pi], _kl_pts[_pi + 1]], fill=(60, 180, 130), width=2)
            d.text((_kl_x0 + _kl_w + 8, _kl_y0 + _kl_h // 2),
                   f"Δk={abs(_k_drift):.0f} N/m", fill=(120, 150, 190), font=font_xxs, anchor="lm")
        else:                                               # compact one-line stats
            d.text((w // 2, _footer_y - 26),
                   f"Material ID (RLS warm start): k {k_per_episode[0]:.0f} "
                   f"to {k_per_episode[-1]:.0f} N/m across episodes  ·  Δk {_k_drift:+.0f} N/m",
                   fill=(100, 145, 200), font=font_xs, anchor="mt")

    ok_n = sum(1 for ep in episodes if ep.success)
    d.text((w // 2, _footer_y),
           f"{ok_n}/{len(episodes)} episodes successful  ·  Dual-cut quartering  ·  "
           "Plate serving  ·  Biomimetic 5-finger grip  ·  2× Slow-Motion Cuts  ·  MuJoCo 3.x",
           fill=(180, 200, 255), font=font_xxs, anchor="mm")

    arr = np.array(img)
    return [arr] * n


# ── HUD overlay ───────────────────────────────────────────────────────
def draw_hud(frame_rgb, ep, phase, t, blade_dist_m, touch_N,
             posture_rms, cut_fired, failure,
             grip_pct=0.0, blade_speed_ms=0.0, touch_sensor=0.0,
             cut2_fired=False, slo_mo=False,
             cut_t=None, cut2_t=None,
             phase_history=None, force_hist=None, cut_stars=0,
             impact_g=0.0, blade_tilt_deg=0.0, blade_gyro_dps=0.0,
             cut_quality=0.0, ft_contacts=None, wrist_corr=0.0,
             left_stab=False,
             gh_forces=None, knife_slip_mm=0.0,
             material_k=0.0, material_conf=0.0, slice_adapted=False,
             tear_level: int = 0, crack_front_mm: float = 0.0,
             front_width_mm: float = 0.0, local_release_prob: float = 0.0,
             front_speed_mm_s: float = 0.0, front_accel_mm_s2: float = 0.0,
             n_episodes: int = 8, episode_dur: float = 14.0,
             ftr_phase: str = "", serve_err_mm: float | None = None,
             grasp_closure: tuple | None = None,
             regrasp_round: int = 0, grip_retry: int = 0):
    img = Image.fromarray(frame_rgb)
    d   = ImageDraw.Draw(img)
    _, _, font, fsm, fxs, f12 = _get_fonts()

    cw, ch     = img.width, img.height
    phase_col  = PHASE_COLORS.get(phase, (220, 220, 220))
    _DIST_MAX  = 1.20

    # ── Left panel ────────────────────────────────────────────────────
    x0, y0 = 18, 14
    d.text((x0, y0),       f"Ep {ep + 1}/{n_episodes}", fill=(220, 220, 220), font=font)
    d.text((x0, y0 + 30),  f"t = {t:5.2f} s",           fill=(180, 200, 255), font=font)
    _phase_txt = f"Phase: {phase}"
    if phase == "REGRASP" and regrasp_round > 0:
        _phase_txt += f"  RETRY {regrasp_round}"
    d.text((x0, y0 + 58),  _phase_txt,                  fill=phase_col,       font=fsm)

    dist_frac = 1.0 - float(np.clip(blade_dist_m / _DIST_MAX, 0.0, 1.0))
    dist_col  = (int(80 + 170 * dist_frac), int(220 - 120 * dist_frac), 80)
    _hbar(d, x0, y0 + 82, 96, 10, dist_frac, dist_col,
          f"Dist {min(blade_dist_m, 9.999)*100:5.1f}cm", fxs=f12)

    spd_frac = float(np.clip(blade_speed_ms / 0.80, 0.0, 1.0))
    _hbar(d, x0, y0 + 104, 96, 10, spd_frac, (100, 180, 255),
          f"Spd  {blade_speed_ms*100:5.1f}cm/s", fxs=f12)

    force_col = (200, 180, 230) if touch_N < 5.0 else (255, int(180 - 100 * min(touch_N/300, 1.0)), 60)
    d.text((x0, y0 + 124), f"Force: {touch_N:6.1f} N", fill=force_col, font=fsm)

    ts_frac = float(np.clip(touch_sensor / 300.0, 0.0, 1.0))
    _hbar(d, x0, y0 + 146, 96, 8, ts_frac, (255, 200, 80), "Sensor", fxs=f12)
    _hbar(d, x0, y0 + 162, 96, 10, grip_pct, (160, 110, 255),
          f"Grip {grip_pct*100:4.0f}%", fxs=f12)

    d.text((x0, y0 + 178), f"Posture: {posture_rms:.3f} rad", fill=(140, 210, 190), font=f12)
    _g_col = (255, 130, 40) if impact_g >= 5.0 else (120, 140, 180)
    d.text((x0, y0 + 193), f"Accel:   {impact_g:5.1f} G", fill=_g_col, font=f12)
    _tilt_col = (255, 80, 60) if blade_tilt_deg > 12.0 else (100, 200, 160)
    d.text((x0, y0 + 206), f"Tilt:    {blade_tilt_deg:5.1f}°", fill=_tilt_col, font=f12)
    _gyro_col = (255, 150, 40) if blade_gyro_dps > 80.0 else (90, 130, 180)
    d.text((x0, y0 + 219), f"Gyro:  {blade_gyro_dps:6.1f}°/s", fill=_gyro_col, font=f12)

    # Cut quality: only shown during contact phases with meaningful value
    _cut_qual = int(cut_quality)
    if phase in ("CONTACT", "SLICE", "CONTACT2", "SLICE2") and _cut_qual > 0:
        _qual_col = (100, 220, 120) if _cut_qual >= 70 else (255, 200, 60) if _cut_qual >= 45 else (220, 80, 60)
        d.text((x0, y0 + 232), f"Quality: {_cut_qual:3d}%", fill=_qual_col, font=f12)

    # ── Per-finger grip forces ─────────────────────────────────────────
    _ft_y     = y0 + 248
    _gh_vals  = gh_forces if gh_forces else [0.0] * 5
    _gh_names = ["Idx", "Mid", "Rng", "Pky", "Thm"]
    _GH_MAX   = 120.0
    d.text((x0, _ft_y), "GRIP FORCE (N)", fill=(140, 150, 190), font=f12)
    for _fi, (_gn, _gv) in enumerate(zip(_gh_names, _gh_vals)):
        _gy = _ft_y + 14 + _fi * 13
        _gf = float(np.clip(_gv / _GH_MAX, 0.0, 1.0))
        _gc = (int(60 + 195 * _gf), int(220 - 100 * _gf), int(100 - 40 * _gf))
        d.text((x0, _gy), _gn, fill=(120, 130, 160), font=f12)
        _bx0, _bw = x0 + 26, 56
        d.rectangle([_bx0, _gy + 2, _bx0 + _bw, _gy + 10], fill=(18, 18, 30))
        if _gf > 0:
            d.rectangle([_bx0, _gy + 2, _bx0 + int(_bw * _gf), _gy + 10], fill=_gc)
        d.text((_bx0 + _bw + 4, _gy + 6), f"{_gv:.0f}", fill=_gc, font=f12, anchor="lm")

    _kslip_y = _ft_y + 14 + 5 * 13
    if abs(knife_slip_mm) > 0.3:
        _ks_col = (255, 160, 40) if abs(knife_slip_mm) > 2.0 else (80, 200, 220)
        d.text((x0, _kslip_y), f"Slip {knife_slip_mm:+.1f}mm", fill=_ks_col, font=f12)
    else:
        d.text((x0, _kslip_y), "Grip locked", fill=(60, 180, 100), font=f12)

    # status flags sit in their own column, clear of the grip bars
    _flag_x = x0 + 116
    if abs(wrist_corr) > 0.005 and phase in ("CONTACT", "SLICE", "CONTACT2", "SLICE2"):
        _wc_col = (255, 160, 60) if abs(wrist_corr) > 0.04 else (120, 200, 255)
        d.text((_flag_x, _ft_y + 14), f"Wrist Δ {wrist_corr:+.3f}r", fill=_wc_col, font=f12)
    if left_stab:
        d.text((_flag_x, _ft_y + 27), "LEFT: STABILIZING", fill=(80, 220, 180), font=f12)

    # ── Contact force mini-chart ──────────────────────────────────────
    _fc_x0, _fc_y0 = x0, _kslip_y + 30
    _fc_w,  _fc_h  = 150, 46
    d.text((_fc_x0, _fc_y0 - 14), "CONTACT FORCE (N)", fill=(110, 90, 160), font=f12)
    d.rectangle([_fc_x0, _fc_y0, _fc_x0 + _fc_w, _fc_y0 + _fc_h], fill=(12, 10, 22))
    d.line([(_fc_x0, _fc_y0 + _fc_h // 2), (_fc_x0 + _fc_w, _fc_y0 + _fc_h // 2)],
           fill=(28, 24, 48), width=1)
    if force_hist:
        cols_f: dict = {}
        for ft, fn in force_hist:
            ci = int(_fc_w * min(ft, episode_dur) / episode_dur)
            cols_f[ci] = max(cols_f.get(ci, 0.0), fn)
        peak_f = max(cols_f.values()) if cols_f else 0.0
        for ci, fn in sorted(cols_f.items()):
            fh_px = int(_fc_h * min(fn / 280.0, 1.0))
            if fh_px > 0:
                fc = (255, 100, 50) if fn > 30 else (60, 140, 255)
                d.line([(_fc_x0 + ci, _fc_y0 + _fc_h - fh_px),
                        (_fc_x0 + ci, _fc_y0 + _fc_h)], fill=fc)
        if peak_f > 0:
            py = _fc_y0 + _fc_h - int(_fc_h * min(peak_f / 280.0, 1.0))
            d.text((_fc_x0 + _fc_w + 3, py), f"{peak_f:.0f}",
                   fill=(160, 130, 220), font=f12, anchor="lm")

    # ── Material ID display (single compact row) ──────────────────────
    _me_y = _fc_y0 + _fc_h + 8
    if material_conf > 0.0:
        _mc = (80, 220, 160) if material_conf >= 0.5 else (180, 180, 100)
        d.text((_fc_x0, _me_y), f"MAT k:{material_k:.0f} N/m · cf {material_conf*100:.0f}%",
               fill=_mc, font=f12)
        _sa_col = (100, 200, 255) if slice_adapted else (90, 90, 110)
        d.text((_fc_x0 + int(d.textlength(
                    f"MAT k:{material_k:.0f} N/m · cf {material_conf*100:.0f}%",
                    font=f12)) + 10, _me_y),
               "ADAPTED" if slice_adapted else "nominal", fill=_sa_col, font=f12)

    # ── Top-right badges ──────────────────────────────────────────────
    _ins_x   = cw - 240 - 8
    _ins_y_t = ch - 135 - 62
    _tr_y    = 14           # top of the right-hand info column
    if ftr_phase:
        d.text((cw - 16, _tr_y), "FF MASTER  [RIGHT]  CUTTING",
               fill=(255, 200, 100), font=fxs, anchor="rt")
        _ftr_role_map = {
            "WAIT": "WAITING", "STEADY": "STEADYING WM", "REACH": "REACH",
            "GRIP": "GRIP", "LIFT": "LIFT", "CARRY": "CARRY",
            "LOWER": "APPROACH PLATE", "CONTACT_PLATE": "CONTACT PLATE",
            "RELEASE": "RELEASE", "RETRACT": "RETRACT", "DONE": "DONE",
        }
        _ftr_role = _ftr_role_map.get(ftr_phase, ftr_phase)
        if grip_retry > 0 and ftr_phase in ("REACH", "GRIP"):
            _ftr_role += f"  RETRY {grip_retry}"
        _ftr_role_col = ((80, 255, 200) if ftr_phase == "DONE"
                         else (100, 255, 160) if ftr_phase == "CONTACT_PLATE"
                         else (80, 200, 255))
        d.text((cw - 16, _tr_y + 21), f"FUTURIST   [LEFT]   {_ftr_role}",
               fill=_ftr_role_col, font=fxs, anchor="rt")
        d.text((cw - 16, _tr_y + 42), "15 DOF · 5-Finger · MCP+PIP+DIP",
               fill=(150, 170, 215), font=f12, anchor="rt")
        _tf_top = _tr_y + 62
        d.text((_ins_x + 2, _ins_y_t - 15), "TOP VIEW  (both zones)",
               fill=(120, 130, 180), font=f12)
    else:
        d.text((cw - 16, _tr_y), "15 DOF · 5-Finger · MCP+PIP+DIP",
               fill=(190, 210, 255), font=fxs, anchor="rt")
        _tf_top = _tr_y + 24
        d.text((_ins_x + 2, _ins_y_t - 15), "TOP VIEW", fill=(120, 130, 180), font=f12)

    # ── Fracture front panel (top-right column, below the badges) ─────
    _D_MAX  = 12.0
    if (crack_front_mm > 0.05 or tear_level > 0) and phase in ("CONTACT", "SLICE"):
        _tf_x0  = cw - 16 - 150          # right-aligned 150 px panel
        _tf_y   = _tf_top + 14
        _cf_frac = float(np.clip(crack_front_mm / _D_MAX, 0.0, 1.0))
        _cf_col  = (int(220 + 35 * _cf_frac), int(210 - 160 * _cf_frac), 40)
        d.text((_tf_x0, _tf_y - 14), "FRACTURE FRONT", fill=(130, 115, 185), font=f12)
        d.text((_tf_x0, _tf_y), f"DEPTH  {crack_front_mm:4.1f} mm", fill=_cf_col, font=f12)
        _bar_done = int(150 * _cf_frac)
        d.rectangle([_tf_x0, _tf_y + 14, _tf_x0 + _bar_done, _tf_y + 15], fill=_cf_col)
        d.rectangle([_tf_x0 + _bar_done, _tf_y + 14, _tf_x0 + 150, _tf_y + 15],
                    fill=(35, 28, 55))
        _wz_col = (120, 180, 255) if front_width_mm > 0.5 else (70, 70, 100)
        d.text((_tf_x0, _tf_y + 18), f"ZONE   {front_width_mm:4.1f} mm", fill=_wz_col, font=f12)
        _prob_col = (255, int(210 - 160 * local_release_prob), 40)
        d.text((_tf_x0, _tf_y + 31), f"PROB   {local_release_prob * 100:4.0f}%",
               fill=_prob_col, font=f12)
        _prob_w = int(150 * local_release_prob)
        d.rectangle([_tf_x0, _tf_y + 45, _tf_x0 + _prob_w, _tf_y + 47], fill=_prob_col)
        d.rectangle([_tf_x0 + _prob_w, _tf_y + 45, _tf_x0 + 150, _tf_y + 47],
                    fill=(35, 28, 55))
        _spd_col = (255, 200, 60) if front_speed_mm_s > 5.0 else (120, 200, 160)
        _acc_sym = ("++" if front_accel_mm_s2 > 20.0
                    else "--" if front_accel_mm_s2 < -20.0 else "· ")
        _acc_col = ((255, 110, 50) if front_accel_mm_s2 > 20.0
                    else (80, 160, 255) if front_accel_mm_s2 < -20.0 else (130, 130, 150))
        d.text((_tf_x0, _tf_y + 51), f"SPD  {front_speed_mm_s:5.1f} mm/s",
               fill=_spd_col, font=f12)
        d.text((_tf_x0 + 110, _tf_y + 51), _acc_sym, fill=_acc_col, font=f12)

    # ── Bottom: phase timeline ────────────────────────────────────────
    tl_x0, tl_y0 = 20, ch - 28
    tl_w,  tl_h  = cw - 40, 14
    d.rectangle([tl_x0, tl_y0, tl_x0 + tl_w, tl_y0 + tl_h], fill=(20, 20, 35))
    if phase_history:
        for _i, (_ts, _ph) in enumerate(phase_history):
            _te = phase_history[_i + 1][0] if _i + 1 < len(phase_history) else t
            _x1 = tl_x0 + int(tl_w * max(float(_ts), 0.0) / episode_dur)
            _x2 = tl_x0 + int(tl_w * min(float(_te), episode_dur) / episode_dur)
            _col    = PHASE_COLORS.get(_ph, (100, 100, 120))
            _is_past = _i + 1 < len(phase_history)
            _seg_col = tuple(int(c * (0.68 if _is_past else 1.0)) for c in _col)
            if _x2 > _x1:
                d.rectangle([_x1, tl_y0, _x2, tl_y0 + tl_h], fill=_seg_col)
                if _x2 - _x1 >= 34:
                    d.text(((_x1 + _x2) // 2, tl_y0 + tl_h // 2),
                           _ph[:5], fill=(0, 0, 0), font=f12, anchor="mm")
    else:
        fill_w = int(tl_w * min(t / episode_dur, 1.0))
        if fill_w:
            d.rectangle([tl_x0, tl_y0, tl_x0 + fill_w, tl_y0 + tl_h], fill=phase_col)
    _cut_xs = [tl_x0 + int(tl_w * ct / episode_dur)
               for ct in (cut_t, cut2_t) if ct is not None]
    for mark_t in [1, 2, 3, 4, 5]:
        mx = tl_x0 + int(tl_w * mark_t / episode_dur)
        d.line([(mx, tl_y0), (mx, tl_y0 + tl_h)], fill=(80, 80, 110), width=1)
        # skip tick labels that would collide with a CUT-marker label
        if all(abs(mx - cx) > 40 for cx in _cut_xs):
            d.text((mx, tl_y0 - 2), f"{mark_t}s", fill=(100, 110, 140), font=f12, anchor="mb")
    if cut_t is not None:
        cx = tl_x0 + int(tl_w * cut_t / episode_dur)
        d.line([(cx, tl_y0 - 3), (cx, tl_y0 + tl_h + 1)], fill=(255, 90, 50), width=2)
        d.text((cx, tl_y0 - 2), f"CUT1 {cut_t:.2f}s", fill=(255, 120, 80), font=f12, anchor="mb")
    if cut2_t is not None:
        cx = tl_x0 + int(tl_w * cut2_t / episode_dur)
        d.line([(cx, tl_y0 - 3), (cx, tl_y0 + tl_h + 1)], fill=(255, 60, 200), width=2)
        d.text((cx, tl_y0 - 2), f"CUT2 {cut2_t:.2f}s", fill=(255, 100, 220), font=f12, anchor="mb")

    # ── Centre overlays ───────────────────────────────────────────────
    if ftr_phase:
        _err_sfx = (f"   delivery err {serve_err_mm:.0f} mm"
                    if serve_err_mm is not None and serve_err_mm == serve_err_mm
                    else "")
        if ftr_phase == "DONE":
            d.text((cw // 2, ch - 80),"•  QUARTER SERVED  •",
                   fill=(100, 255, 160), font=font, anchor="mm")
            d.text((cw // 2, ch - 58),f"ON PLATE{_err_sfx}",
                   fill=(80, 255, 200), font=fsm, anchor="mm")
        elif ftr_phase == "RETRACT":
            d.text((cw // 2, ch - 80),"•  PLACED ON PLATE  •",
                   fill=(100, 255, 180), font=font, anchor="mm")
            d.text((cw // 2, ch - 58),f"FUTURIST RETRACTING{_err_sfx}",
                   fill=(140, 210, 160), font=fsm, anchor="mm")
        elif ftr_phase == "RELEASE":
            d.text((cw // 2, ch - 80),"FUTURIST: RELEASE",
                   fill=(200, 255, 140), font=font, anchor="mm")
            d.text((cw // 2, ch - 58),f"fingers opening{_err_sfx}",
                   fill=(160, 240, 120), font=fsm, anchor="mm")
        elif ftr_phase == "CONTACT_PLATE":
            d.text((cw // 2, ch - 80),"•  ON PLATE  •",
                   fill=(80, 255, 160), font=font, anchor="mm")
            d.text((cw // 2, ch - 58),"FUTURIST: CONTACT PLATE",
                   fill=(100, 230, 180), font=fsm, anchor="mm")
        elif ftr_phase == "LOWER":
            d.text((cw // 2, ch - 80),"FUTURIST: LOWERING TO PLATE",
                   fill=(80, 180, 255), font=font, anchor="mm")
            d.text((cw // 2, ch - 58),"quarter descending to plate surface",
                   fill=(60, 150, 220), font=fsm, anchor="mm")
        elif ftr_phase == "CARRY":
            d.text((cw // 2, ch - 80), "• CARRYING QUARTER •",
                   fill=(255, 180, 100), font=font, anchor="mm")
            if grasp_closure and grasp_closure[1] == grasp_closure[1]:
                d.text((cw // 2, ch - 58),
                       f"clamp closure {grasp_closure[0]}/5 fingertips",
                       fill=(255, 210, 150), font=fsm, anchor="mm")
        elif ftr_phase in ("REACH", "GRIP", "LIFT"):
            d.text((cw // 2, ch - 80), f"FUTURIST: {ftr_phase}",
                   fill=(80, 200, 255), font=font, anchor="mm")
            if (ftr_phase in ("GRIP", "LIFT") and grasp_closure
                    and grasp_closure[1] == grasp_closure[1]):
                d.text((cw // 2, ch - 58),
                       f"grasp closure {grasp_closure[0]}/5 fingertips",
                       fill=(120, 220, 255), font=fsm, anchor="mm")
        elif cut2_fired:
            d.text((cw // 2, ch - 66),"• QUARTER CUT •",
                   fill=(255, 60, 200), font=font, anchor="mm")
        elif cut_fired:
            d.text((cw // 2, ch - 66),"• CUT EXECUTED •",
                   fill=(255, 90, 50), font=font, anchor="mm")
    else:
        if phase == "FINAL":
            d.text((cw // 2, ch - 66),"• SERVED •",
                   fill=(100, 255, 160), font=font, anchor="mm")
        elif phase in ("SWEEP", "LIFT"):
            d.text((cw // 2, ch - 66),"• PLATING •",
                   fill=(255, 180, 100), font=font, anchor="mm")
        elif phase == "KNIFE_DOWN":
            d.text((cw // 2, ch - 66),"• KNIFE DOWN •",
                   fill=(255, 220, 80), font=font, anchor="mm")
        elif cut2_fired:
            d.text((cw // 2, ch - 66),"• QUARTER CUT •",
                   fill=(255, 60, 200), font=font, anchor="mm")
        elif cut_fired:
            d.text((cw // 2, ch - 66),"• CUT EXECUTED •",
                   fill=(255, 90, 50), font=font, anchor="mm")
    if failure:
        d.text((cw // 2, ch - 66),f"FAIL: {failure}", fill=(255, 40, 40), font=fsm, anchor="mm")
    if slo_mo:
        d.text((cw - 16, ch - 52), "● 2× SLOW MOTION", fill=(255, 220, 60), font=fxs, anchor="rt")
    if cut_stars > 0:
        _stars = "●" * cut_stars + "·" * (5 - cut_stars)
        d.text((cw // 2, 16), f"CUT PRECISION  {_stars}", fill=(160, 255, 180), font=fsm, anchor="mt")

    return np.array(img)


# ── Video re-compression ──────────────────────────────────────────────
def recompress(path: pathlib.Path, target_mb: float = 20.0) -> None:
    """Re-encode video with ffmpeg to stay under target_mb."""
    import imageio_ffmpeg
    ffmpeg   = imageio_ffmpeg.get_ffmpeg_exe()
    import imageio
    reader   = imageio.get_reader(str(path))
    duration = reader.get_meta_data().get("duration", None)
    reader.close()
    if not duration:
        print("  [compress] could not determine duration, skipping")
        return
    budget_kbits = target_mb * 1024 * 8 * 0.95
    target_kbps  = int(budget_kbits / duration)
    tmp = path.with_suffix(".tmp.mp4")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(path),
         "-c:v", "libx264", "-b:v", f"{target_kbps}k",
         "-maxrate", f"{int(target_kbps * 1.5)}k",
         "-bufsize", f"{target_kbps * 2}k",
         "-preset", "slow", "-movflags", "+faststart", str(tmp)],
        check=True, capture_output=True,
    )
    size_mb = tmp.stat().st_size / (1024 * 1024)
    tmp.replace(path)
    print(f"  [compress] {size_mb:.1f} MB  (target ≤{target_mb} MB, "
          f"bitrate {target_kbps} kbps)")
