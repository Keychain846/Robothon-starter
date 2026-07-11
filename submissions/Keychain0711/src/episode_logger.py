"""
EpisodeLogger — per-episode stats and failure detection for the robot cutting demo.

Failure conditions checked each step:
  timeout        cut not fired by t >= cut_timeout
  excess_force   blade touch force exceeds max_touch_force (N)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    cut_timeout:     float = 5.0     # s — failure if cut not fired by this time
    max_touch_force: float = 500.0  # N — failure if blade impact force exceeds this


class EpisodeLogger:
    def __init__(self, ep_id: int, cfg: LogConfig | None = None):
        self.ep_id = ep_id
        self._cfg = cfg or LogConfig()
        self.success = False
        self.cut_time: float | None = None
        self.failure_reason: str | None = None
        self.min_blade_dist = float("inf")
        self.max_force: float = 0.0
        self._posture_sum: float = 0.0   # accumulate per-step posture RMS
        self._posture_n:   int   = 0
        self._cut_quality_sum: float = 0.0  # composite cut quality (0–100)
        self._cut_quality_n:   int   = 0
        self._phases_seen: list[str] = []
        self._last_phase: str = ""

    # ------------------------------------------------------------------
    def mark_success(self, cut_time: float):
        """Call immediately when the cut fires to disable further failure checks."""
        self.success  = True
        self.cut_time = cut_time

    def step(self, t: float, phase: str,
             blade_dist: float, touch_force: float,
             posture_rms: float = 0.0,
             cut_quality: float = 0.0) -> str | None:
        """
        Record one simulation step.
        posture_rms: RMS joint-angle deviation from home for waist+head+left-arm.
        cut_quality: 0–100 composite score (tilt, gyro, force, speed); tracked during SLICE.
        Returns a failure reason string if a new failure is detected, else None.
        """
        if phase != self._last_phase:
            self._phases_seen.append(phase)
            self._last_phase = phase

        self.min_blade_dist  = min(self.min_blade_dist, blade_dist)
        self.max_force       = max(self.max_force, touch_force)
        self._posture_sum   += posture_rms
        self._posture_n     += 1
        if phase in ("SLICE", "SLICE2") and cut_quality > 0:
            self._cut_quality_sum += cut_quality
            self._cut_quality_n   += 1

        if (touch_force > self._cfg.max_touch_force
                and not self.success and self.failure_reason is None):
            self.failure_reason = "excess_force"
            return self.failure_reason

        if (t >= self._cfg.cut_timeout and not self.success
                and self.failure_reason is None):
            self.failure_reason = "timeout"
            return self.failure_reason

        return None

    def finalize(self, success: bool, cut_time: float | None = None):
        if not self.success:
            self.success  = success
            self.cut_time = cut_time

    @property
    def posture_rms_mean(self) -> float:
        return self._posture_sum / self._posture_n if self._posture_n else 0.0

    @property
    def cut_quality_mean(self) -> float:
        return self._cut_quality_sum / self._cut_quality_n if self._cut_quality_n else 0.0

    def to_dict(self) -> dict:
        return {
            "ep":              self.ep_id + 1,
            "success":         "YES" if self.success else "NO",
            "cut_time_s":      f"{self.cut_time:.3f}" if self.cut_time is not None else "—",
            "min_blade_mm":    f"{self.min_blade_dist * 1000:.1f}",
            "max_force_N":     f"{self.max_force:.1f}",
            "posture_rms_rad": f"{self.posture_rms_mean:.4f}",
            "cut_quality_pct": f"{self.cut_quality_mean:.1f}",
            "failure":         self.failure_reason or "—",
        }


class RunLogger:
    """Aggregates EpisodeLogger results across a full run."""

    def __init__(self):
        self._episodes: list[EpisodeLogger] = []

    def add(self, ep: EpisodeLogger):
        self._episodes.append(ep)

    def print_summary(self):
        n     = len(self._episodes)
        n_ok  = sum(1 for e in self._episodes if e.success)
        times = [e.cut_time for e in self._episodes if e.cut_time is not None]

        print(f"\n{'=' * 68}")
        print(f"  Run summary   {n_ok}/{n} episodes successful")
        print(f"{'=' * 68}")
        hdr = (f"{'Ep':>3}  {'OK':>3}  {'CutT(s)':>8}  "
               f"{'MinDist(mm)':>12}  {'MaxF(N)':>8}  "
               f"{'PostureRMS(rad)':>16}  {'Fail':>12}")
        print(hdr)
        print("-" * 68)
        for e in self._episodes:
            d = e.to_dict()
            print(f"  {d['ep']:>1}  {d['success']:>3}  {d['cut_time_s']:>8}  "
                  f"{d['min_blade_mm']:>12}  {d['max_force_N']:>8}  "
                  f"{d['posture_rms_rad']:>16}  {d['failure']:>12}")
        print("-" * 68)
        if times:
            avg_t = sum(times) / len(times)
            p_scores = [e.posture_rms_mean for e in self._episodes]
            avg_p = sum(p_scores) / len(p_scores)
            print(f"  Avg cut time: {avg_t:.3f} s  "
                  f"Avg posture RMS: {avg_p:.4f} rad  "
                  f"Success: {n_ok/n*100:.0f}%")
        print(f"{'=' * 68}\n")
