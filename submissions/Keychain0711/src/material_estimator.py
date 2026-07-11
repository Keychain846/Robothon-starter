"""
MaterialEstimator — online recursive least-squares identification of watermelon
mechanical stiffness from contact force feedback.

Contact model:  F ≈ k·d + b   (k = stiffness N/m, d = blade penetration depth m)

During the CONTACT phase the blade descends into the watermelon while force rises.
Each (depth, force) sample updates the RLS estimate with exponential forgetting
(λ = 0.96) so the estimate tracks within-episode variability.

The identified stiffness drives adaptive slice_dur scaling for the second cut:
softer watermelons require shorter follow-through; harder ones need longer.
"""

import numpy as np

_K_DEFAULT      = 2200.0    # N/m — nominal WM stiffness prior
_K_MIN, _K_MAX  = 400.0, 35000.0
_FORGETTING     = 0.96      # exponential forgetting factor λ


class MaterialEstimator:
    """
    Online RLS estimator: theta = [k, b] such that F ≈ k·d + b.

    confidence rises from 0 → 1 over the first 30 contact observations.
    slice_dur_mult() maps estimated k to a duration scaling factor [0.75, 1.35].
    """

    def __init__(self, forgetting: float = _FORGETTING):
        self._lambda = forgetting
        self._k      = _K_DEFAULT
        self._b      = 0.0
        self._P      = np.eye(2) * 5000.0
        self._n_obs  = 0

    # ------------------------------------------------------------------
    def update(self, depth_m: float, force_N: float) -> float:
        """Update RLS with one (depth, force) pair.  Returns current k estimate."""
        if depth_m < 1e-4 or force_N < 0.5:
            return self._k
        phi   = np.array([depth_m, 1.0])
        theta = np.array([self._k, self._b])
        Pphi  = self._P @ phi
        gamma = float(self._lambda + phi @ Pphi)
        K_rls = Pphi / gamma
        err   = force_N - float(phi @ theta)
        theta = theta + K_rls * err
        self._k     = float(np.clip(theta[0], _K_MIN, _K_MAX))
        self._b     = float(theta[1])
        self._P     = (self._P - np.outer(K_rls, Pphi)) / self._lambda
        self._n_obs += 1
        return self._k

    # ------------------------------------------------------------------
    @property
    def stiffness_Npm(self) -> float:
        return self._k

    @property
    def confidence(self) -> float:
        """0 = no data, 1 = fully confident (≥ 30 observations)."""
        return min(1.0, self._n_obs / 30.0)

    def slice_dur_multiplier(self) -> float:
        """
        Maps estimated stiffness → slice duration multiplier for the second cut.
          k < 1500 N/m (soft)   → 0.75× (shorter follow-through)
          k ≈ 2200 N/m (nominal) → 1.0×
          k > 3500 N/m (hard)   → 1.35× (longer dwell to complete cut)
        Returns 1.0 (nominal) when confidence < 0.25 (not enough data).
        """
        if self.confidence < 0.25:
            return 1.0
        return float(np.clip(self._k / _K_DEFAULT, 0.75, 1.35))

    def reset(self):
        """Hard reset — back to prior.  Use for the very first episode."""
        self._k      = _K_DEFAULT
        self._b      = 0.0
        self._P      = np.eye(2) * 5000.0
        self._n_obs  = 0

    def soft_reset(self):
        """
        Warm reset — keep current k estimate as prior for the next episode.
        Tightens covariance (P) to reflect accumulated knowledge, but resets
        observation count so confidence rebuilds from zero each episode.
        Cross-episode warm-starting implements trial-to-trial adaptive learning:
        the stiffness estimate from episode N seeds episode N+1.
        """
        self._b      = 0.0
        self._P      = np.eye(2) * 800.0   # tight prior: we trust last episode's k
        self._n_obs  = 0
        # _k intentionally kept from previous episode
