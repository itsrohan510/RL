"""
sat_env.py — Gymnasium environment for satellite collision avoidance.

Observation (256-dim):  8 satellites × 32 features each
  Per satellite: [fuel_ratio, altitude_norm, vx, vy, vz]  (5)
                 + 3 nearest threats × [rel_pos(3), rel_vel(3), dist, miss, tca]  (27)

Action (24-dim):  8 satellites × 3 delta-v components, each in [-1, 1] scaled by MAX_DV
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_engine import (OrbitalSimulation, N_SATS, N_DEBRIS,
                        CAUTION_R, CRITICAL_R, WARNING_R)

K_THREATS   = 3
OBS_PER_SAT = 5 + K_THREATS * 9   # 32
OBS_DIM     = N_SATS * OBS_PER_SAT  # 256
ACT_DIM     = N_SATS * 3            # 24
MAX_DV      = 0.04
MAX_STEPS   = 1500
DT          = 0.2


class SatelliteAvoidanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.sim = OrbitalSimulation()
        self.observation_space = spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.steps = 0

    # ── reset ──
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.sim.reset()
        self.steps = 0
        threats = self.sim._compute_threats()
        return self._build_obs(threats), {}

    # ── step ──
    def step(self, action):
        self.steps += 1
        actions = np.reshape(action, (N_SATS, 3)) * MAX_DV
        total_dv = self.sim.apply_actions(actions)
        threats = self.sim.step(DT)
        reward = self._reward(threats, total_dv)
        terminated = self.steps >= MAX_STEPS
        info = {
            "collisions": self.sim.collisions,
            "n_critical": threats["n_critical"],
            "fuel_mean": float(np.mean(self.sim.fuel)),
        }
        return self._build_obs(threats), reward, terminated, False, info

    # ── observation ──
    def _build_obs(self, t):
        obs = np.zeros(OBS_DIM, np.float32)
        for i in range(N_SATS):
            base = i * OBS_PER_SAT
            r = np.linalg.norm(self.sim.sat_pos[i])
            obs[base]     = self.sim.fuel[i] / self.sim.max_fuel
            obs[base + 1] = r / 10.0
            obs[base + 2] = self.sim.sat_vel[i, 0] / 0.8
            obs[base + 3] = self.sim.sat_vel[i, 1] / 0.8
            obs[base + 4] = self.sim.sat_vel[i, 2] / 0.8

            # merge debris + sat threats, sort by distance, take top K
            all_dist = np.concatenate([t["sd_dist"][i], t["ss_dist"][i]])
            all_dp   = np.concatenate([t["sd_dp"][i],   t["ss_dp"][i]])
            all_dv   = np.concatenate([t["sd_dv"][i],   t["ss_dv"][i]])
            all_miss = np.concatenate([t["sd_miss"][i], t["ss_miss"][i]])
            all_tca  = np.concatenate([t["sd_tca"][i],  t["ss_tca"][i]])

            idxs = np.argsort(all_dist)[:K_THREATS]
            for k, idx in enumerate(idxs):
                if all_dist[idx] > CAUTION_R * 2:
                    break
                off = base + 5 + k * 9
                obs[off:off+3] = all_dp[idx] / CAUTION_R
                obs[off+3:off+6] = all_dv[idx] / 0.8
                obs[off + 6] = all_dist[idx] / CAUTION_R
                obs[off + 7] = all_miss[idx] / CAUTION_R
                obs[off + 8] = min(all_tca[idx] / 30.0, 1.0)
        return obs

    # ── reward ──
    def _reward(self, t, total_dv):
        r = 0.1                                       # survival bonus
        r -= 500.0 * t["new_collisions"]              # collision penalty
        r -= 5.0 * total_dv                           # fuel cost
        r -= 3.0 * np.sum(t["sd_miss"] < CRITICAL_R)  # proximity penalty
        r -= 0.5 * np.sum((t["sd_miss"] >= CRITICAL_R) & (t["sd_miss"] < WARNING_R))
        for i in range(N_SATS):
            for j in range(i + 1, N_SATS):
                if t["ss_miss"][i, j] < CRITICAL_R:
                    r -= 3.0
                elif t["ss_miss"][i, j] < WARNING_R:
                    r -= 0.5
        return float(r)
