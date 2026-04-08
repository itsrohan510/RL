"""
sim_engine.py — NumPy-vectorized port of simulation.js for fast headless training.
Physics are identical: same GM, radii, thresholds, Euler integrator.
"""

import numpy as np

EARTH_RADIUS = 5.0
GM = 25
COLLISION_R = 0.08
CRITICAL_R = 0.35
WARNING_R = 0.7
CAUTION_R = 1.2
N_SATS = 8
N_DEBRIS = 30

SAT_CONFIGS = [
    {"r": 7.0, "inc": 51,  "raan": 0,   "ph": 0},
    {"r": 7.0, "inc": 51,  "raan": 0,   "ph": 180},
    {"r": 7.8, "inc": 28,  "raan": 45,  "ph": 90},
    {"r": 7.8, "inc": 28,  "raan": 45,  "ph": 270},
    {"r": 8.5, "inc": 97,  "raan": 120, "ph": 60},
    {"r": 6.8, "inc": 5,   "raan": 200, "ph": 150},
    {"r": 9.0, "inc": 65,  "raan": 80,  "ph": 30},
    {"r": 8.2, "inc": 42,  "raan": 300, "ph": 210},
]


def circular_orbit_state(radius, inc_deg, raan_deg, phase_deg):
    inc, raan, ph = np.radians(inc_deg), np.radians(raan_deg), np.radians(phase_deg)
    v = np.sqrt(GM / radius)
    xo, yo = radius * np.cos(ph), radius * np.sin(ph)
    vxo, vyo = -v * np.sin(ph), v * np.cos(ph)
    ci, si = np.cos(inc), np.sin(inc)
    cr, sr = np.cos(raan), np.sin(raan)
    pos = np.array([cr*xo - sr*ci*yo, sr*xo + cr*ci*yo, si*yo])
    vel = np.array([cr*vxo - sr*ci*vyo, sr*vxo + cr*ci*vyo, si*vyo])
    return pos, vel


class OrbitalSimulation:
    def __init__(self):
        self.sat_pos = np.zeros((N_SATS, 3))
        self.sat_vel = np.zeros((N_SATS, 3))
        self.deb_pos = np.zeros((N_DEBRIS, 3))
        self.deb_vel = np.zeros((N_DEBRIS, 3))
        self.fuel = np.full(N_SATS, 100.0)
        self.max_fuel = 100.0
        self.collisions = 0
        self.time = 0.0
        self.step_count = 0
        self.reset()

    def reset(self):
        self.time = 0.0
        self.step_count = 0
        self.collisions = 0
        self.fuel[:] = self.max_fuel
        for i, cfg in enumerate(SAT_CONFIGS):
            p, v = circular_orbit_state(cfg["r"], cfg["inc"], cfg["raan"], cfg["ph"])
            self.sat_pos[i], self.sat_vel[i] = p, v
        self._randomize_debris(np.arange(N_DEBRIS))

    def _randomize_debris(self, indices):
        for i in indices:
            r = 6.4 + np.random.random() * 3.6
            p, v = circular_orbit_state(r, np.random.random()*120,
                                        np.random.random()*360, np.random.random()*360)
            self.deb_pos[i], self.deb_vel[i] = p, v

    # ── apply delta-v burns ──
    def apply_actions(self, actions):
        """actions: (N_SATS, 3) delta-v array. Returns total |dv| applied."""
        total_dv = 0.0
        for i in range(N_SATS):
            dv_mag = np.linalg.norm(actions[i])
            if dv_mag < 1e-6:
                continue
            cost = dv_mag * 30.0
            if self.fuel[i] < cost * 0.1:
                continue
            self.sat_vel[i] += actions[i]
            self.fuel[i] = max(0.0, self.fuel[i] - cost)
            total_dv += dv_mag
        return total_dv

    # ── physics step ──
    def step(self, dt=0.2):
        self.time += dt
        self.step_count += 1

        all_pos = np.concatenate([self.sat_pos, self.deb_pos])   # (38, 3)
        all_vel = np.concatenate([self.sat_vel, self.deb_vel])

        r = np.linalg.norm(all_pos, axis=1, keepdims=True).clip(min=1e-6)
        acc = -GM / (r ** 3) * all_pos
        all_vel = all_vel + acc * dt
        all_pos = all_pos + all_vel * dt

        self.sat_pos, self.sat_vel = all_pos[:N_SATS], all_vel[:N_SATS]
        self.deb_pos, self.deb_vel = all_pos[N_SATS:], all_vel[N_SATS:]

        # Respawn re-entered debris
        deb_r = np.linalg.norm(self.deb_pos, axis=1)
        bad = np.where(deb_r < EARTH_RADIUS * 0.9)[0]
        if len(bad):
            self._randomize_debris(bad)

        return self._compute_threats()

    # ── vectorized threat computation ──
    def _compute_threats(self):
        # sat ↔ debris  (N_SATS, N_DEBRIS, 3)
        sd_dp = self.deb_pos[None, :, :] - self.sat_pos[:, None, :]
        sd_dv = self.deb_vel[None, :, :] - self.sat_vel[:, None, :]
        sd_dist = np.linalg.norm(sd_dp, axis=2)

        sd_dp_dv = np.sum(sd_dp * sd_dv, axis=2)
        sd_dv_dv = np.sum(sd_dv * sd_dv, axis=2)
        sd_tca = np.where(sd_dv_dv > 1e-10, -sd_dp_dv / sd_dv_dv, 0.0)
        sd_tca = np.maximum(sd_tca, 0.0)
        sd_miss = np.linalg.norm(sd_dp + sd_dv * sd_tca[:, :, None], axis=2)

        # sat ↔ sat  (N_SATS, N_SATS, 3)
        ss_dp = self.sat_pos[None, :, :] - self.sat_pos[:, None, :]
        ss_dv = self.sat_vel[None, :, :] - self.sat_vel[:, None, :]
        ss_dist = np.linalg.norm(ss_dp, axis=2)
        np.fill_diagonal(ss_dist, np.inf)

        ss_dp_dv = np.sum(ss_dp * ss_dv, axis=2)
        ss_dv_dv = np.sum(ss_dv * ss_dv, axis=2)
        ss_tca = np.where(ss_dv_dv > 1e-10, -ss_dp_dv / ss_dv_dv, 0.0)
        ss_tca = np.maximum(ss_tca, 0.0)
        ss_miss = np.linalg.norm(ss_dp + ss_dv * ss_tca[:, :, None], axis=2)
        np.fill_diagonal(ss_miss, np.inf)

        n_coll = int(np.sum(sd_miss < COLLISION_R)
                     + np.sum(np.triu(ss_miss < COLLISION_R, k=1)))
        self.collisions += n_coll

        return dict(
            sd_dp=sd_dp, sd_dv=sd_dv, sd_dist=sd_dist, sd_miss=sd_miss, sd_tca=sd_tca,
            ss_dp=ss_dp, ss_dv=ss_dv, ss_dist=ss_dist, ss_miss=ss_miss, ss_tca=ss_tca,
            new_collisions=n_coll,
            n_critical=int(np.sum(sd_miss < CRITICAL_R) + np.sum(np.triu(ss_miss < CRITICAL_R, k=1))),
            n_warning=int(np.sum(sd_miss < WARNING_R) + np.sum(np.triu(ss_miss < WARNING_R, k=1))),
        )
