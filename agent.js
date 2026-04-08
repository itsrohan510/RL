/* ═══════════════════════════════════════════════════
   agent.js — Collision Avoidance Agent
   ═══════════════════════════════════════════════════ */

import { vec3Len, vec3Sub, vec3Add, vec3Scale, vec3Cross, vec3Norm, vec3Dot,
         CRITICAL_R, WARNING_R, CAUTION_R } from './simulation.js';

const BURN_TCA_THRESHOLD   = 35;   // seconds — plan burn if TCA < this
const BURN_MISS_THRESHOLD  = 0.38; // scene units — act if predicted miss < this
const MAX_DV               = 0.09; // max delta-v per burn
const MIN_FUEL_RESERVE     = 8;    // don't burn below this fuel level
const COOLDOWN_STEPS       = 30;   // steps between burns for same pair

export class CollisionAvoidanceAgent {
  constructor() {
    this.actionQueue = [];
    this.recentBurns = new Map();   // "satId|objId" → stepCount of last burn
    this.totalBurns = 0;
  }

  evaluate(observation) {
    this.actionQueue = [];

    const { danger_pairs, satellites, stepCount } = observation;
    if (!danger_pairs || danger_pairs.length === 0) return this.actionQueue;

    // Sort by urgency: lowest miss distance first
    const sorted = [...danger_pairs]
      .filter(dp => dp.level === 'CRITICAL' || dp.level === 'WARNING')
      .sort((a, b) => a.miss_distance - b.miss_distance);

    for (const dp of sorted) {
      // Skip if on cooldown
      const pairKey = `${dp.sat_id}|${dp.obj_id}`;
      const lastBurn = this.recentBurns.get(pairKey) || -Infinity;
      if (stepCount - lastBurn < COOLDOWN_STEPS) continue;

      // Only act if TCA is soon enough and miss distance is dangerous
      if (dp.tca > BURN_TCA_THRESHOLD && dp.distance > WARNING_R) continue;
      if (dp.miss_distance > BURN_MISS_THRESHOLD && dp.level !== 'CRITICAL') continue;

      const sat = satellites.find(s => s.id === dp.sat_id);
      if (!sat || sat.fuel < MIN_FUEL_RESERVE) continue;

      // Compute avoidance burn
      const action = this._computeBurn(sat, dp);
      if (action) {
        this.actionQueue.push(action);
        this.recentBurns.set(pairKey, stepCount);
        this.totalBurns++;
      }
    }

    return this.actionQueue;
  }

  _computeBurn(sat, dangerPair) {
    // Find the threatening object's relative position & velocity
    // We compute the burn to push the satellite AWAY from the closest approach
    const dp = dangerPair;

    // Relative position vector from satellite to object
    // (we approximate from midpoint data)
    const satPos = [sat.x, sat.y, sat.z];
    const satVel = [sat.vx, sat.vy, sat.vz];

    // Direction from midpoint to satellite — push along this
    const toSat = vec3Norm([
      sat.x - dp.midpoint[0],
      sat.y - dp.midpoint[1],
      sat.z - dp.midpoint[2],
    ]);

    // Also compute radial (away from Earth) component
    const radial = vec3Norm(satPos);

    // Combine: push perpendicular to velocity, away from threat
    // Use cross product to find optimal perpendicular direction
    const velDir = vec3Norm(satVel);
    let burnDir = vec3Cross(velDir, toSat);
    const burnDirLen = vec3Len(burnDir);

    if (burnDirLen < 0.01) {
      // Threat is along velocity — burn radially
      burnDir = radial;
    } else {
      burnDir = vec3Norm(burnDir);
      // Ensure burn pushes AWAY from threat, not toward it
      if (vec3Dot(burnDir, toSat) < 0) {
        burnDir = vec3Scale(burnDir, -1);
      }
    }

    // Scale burn magnitude by urgency
    let magnitude = MAX_DV;
    if (dp.level === 'CRITICAL') {
      magnitude = MAX_DV;
    } else if (dp.level === 'WARNING') {
      magnitude = MAX_DV * 0.6;
    } else {
      magnitude = MAX_DV * 0.3;
    }

    // Reduce if low fuel
    if (sat.fuel < 30) magnitude *= 0.5;

    const deltaV = vec3Scale(burnDir, magnitude);

    return {
      satellite_id: sat.id,
      delta_v: deltaV,
      metadata: {
        target_id: dp.obj_id,
        level: dp.level,
        tca: dp.tca,
        miss_distance: dp.miss_distance,
        burn_magnitude: magnitude,
        reason: `Avoiding ${dp.obj_id} (${dp.level}, miss=${dp.miss_distance.toFixed(3)})`
      }
    };
  }

  getStats() {
    return {
      totalBurns: this.totalBurns,
      pendingActions: this.actionQueue.length
    };
  }
}


/* ═══════════════════════════════════════════════════
   NeuralAgent — Runs a trained PPO policy (exported as JSON weights)
   in the browser via raw matrix multiplication. No ML framework needed.
   ═══════════════════════════════════════════════════ */

export class NeuralAgent {
  constructor(data) {
    this.layers = data.layers;
    this.norm   = data.normalization;
    this.cfg    = data.config;
    this.totalBurns = 0;
    this.actionQueue = [];
    this.recentBurns = new Map();
  }

  static async load(url = './trained_weights.json') {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    return new NeuralAgent(await resp.json());
  }

  evaluate(observation) {
    const obs  = this._buildObs(observation);
    const norm = this._normalize(obs);
    const raw  = this._forward(norm);

    // raw: [24] — 8 sats × 3 delta-v components
    this.actionQueue = [];
    const maxDv = this.cfg.max_dv;
    const sats  = observation.satellites;

    for (let i = 0; i < this.cfg.n_sats; i++) {
      const dv = [
        Math.tanh(raw[i*3])   * maxDv,
        Math.tanh(raw[i*3+1]) * maxDv,
        Math.tanh(raw[i*3+2]) * maxDv,
      ];
      const mag = Math.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2);
      if (mag < 0.002 || !sats[i]) continue;  // ignore trivial burns
      this.totalBurns++;
      this.actionQueue.push({
        satellite_id: sats[i].id,
        delta_v: dv,
        metadata: {
          burn_magnitude: mag,
          reason: `Neural policy (mag=${mag.toFixed(4)})`,
          level: 'WARNING',
          tca: 0, miss_distance: 0
        }
      });
    }
    return this.actionQueue;
  }

  /* ── build observation vector (must match Python sat_env._build_obs) ── */
  _buildObs(obs) {
    const { satellites, debris } = obs;
    const K   = this.cfg.k_threats;
    const OPS = this.cfg.obs_per_sat;
    const CR  = this.cfg.caution_r;
    const vec = new Array(OPS * satellites.length).fill(0);

    // pre-gather all object positions/velocities
    const objPos = [], objVel = [];
    for (const d of debris)     { objPos.push([d.x,d.y,d.z]); objVel.push([d.vx,d.vy,d.vz]); }
    for (const s of satellites) { objPos.push([s.x,s.y,s.z]); objVel.push([s.vx,s.vy,s.vz]); }

    for (let i = 0; i < satellites.length; i++) {
      const s = satellites[i], base = i * OPS;
      const r = Math.sqrt(s.x**2 + s.y**2 + s.z**2);
      vec[base]   = s.fuel / s.maxFuel;
      vec[base+1] = r / 10.0;
      vec[base+2] = s.vx / 0.8;
      vec[base+3] = s.vy / 0.8;
      vec[base+4] = s.vz / 0.8;

      // compute distances to all objects
      const threats = [];
      for (let j = 0; j < objPos.length; j++) {
        if (j === debris.length + i) continue;  // skip self
        const dp = [objPos[j][0]-s.x, objPos[j][1]-s.y, objPos[j][2]-s.z];
        const dv = [objVel[j][0]-s.vx, objVel[j][1]-s.vy, objVel[j][2]-s.vz];
        const dist = Math.sqrt(dp[0]**2+dp[1]**2+dp[2]**2);
        const dpDv = dp[0]*dv[0]+dp[1]*dv[1]+dp[2]*dv[2];
        const dvDv = dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2];
        let tca = dvDv > 1e-10 ? -dpDv/dvDv : 0;
        tca = Math.max(tca, 0);
        const pa = [dp[0]+dv[0]*tca, dp[1]+dv[1]*tca, dp[2]+dv[2]*tca];
        const miss = Math.sqrt(pa[0]**2+pa[1]**2+pa[2]**2);
        if (Number.isFinite(dist)) {
          threats.push({ dp, dv, dist, miss: Number.isFinite(miss) ? miss : dist, tca });
        }
      }
      threats.sort((a,b) => a.dist - b.dist);

      for (let k = 0; k < Math.min(K, threats.length); k++) {
        const t = threats[k];
        if (t.dist > CR * 2) break;
        const off = base + 5 + k * 9;
        vec[off]   = t.dp[0]/CR;  vec[off+1] = t.dp[1]/CR;  vec[off+2] = t.dp[2]/CR;
        vec[off+3] = t.dv[0]/0.8; vec[off+4] = t.dv[1]/0.8; vec[off+5] = t.dv[2]/0.8;
        vec[off+6] = t.dist/CR;   vec[off+7] = t.miss/CR;
        vec[off+8] = Math.min(t.tca/30, 1);
      }
    }
    return vec;
  }

  _normalize(obs) {
    if (!this.norm) return obs;
    return obs.map((v, i) => {
      const n = (v - this.norm.obs_mean[i]) / this.norm.obs_std[i];
      return Math.max(-this.norm.clip_obs, Math.min(this.norm.clip_obs, n));
    });
  }

  _forward(x) {
    for (const layer of this.layers) {
      const W = layer.weight, b = layer.bias;
      const out = new Array(b.length);
      for (let i = 0; i < b.length; i++) {
        let sum = b[i];
        for (let j = 0; j < x.length; j++) sum += W[i][j] * x[j];
        out[i] = sum;
      }
      x = layer.activation === 'tanh' ? out.map(Math.tanh) : out;
    }
    return x;
  }

  getStats() { return { totalBurns: this.totalBurns, pendingActions: this.actionQueue.length }; }
}
