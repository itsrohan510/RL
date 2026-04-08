/* ═══════════════════════════════════════════════════
   simulation.js — Orbital Mechanics & Collision Engine
   ═══════════════════════════════════════════════════ */

const EARTH_RADIUS  = 5;          // scene units
const GM            = 2.5;        // gravitational parameter (scene-units³/s²)
const COLLISION_R   = 0.08;       // collision distance
const CRITICAL_R    = 0.35;
const WARNING_R     = 0.7;
const CAUTION_R     = 1.2;

// ─── Utility vectors ───
function vec3Len(v) { return Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
function vec3Dot(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function vec3Sub(a,b) { return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function vec3Add(a,b) { return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]; }
function vec3Scale(v,s) { return [v[0]*s,v[1]*s,v[2]*s]; }
function vec3Cross(a,b) {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function vec3Norm(v) { const l=vec3Len(v)||1; return [v[0]/l,v[1]/l,v[2]/l]; }

// ─── Create body in circular orbit ───
function circularOrbitState(radius, incDeg, raanDeg, phaseDeg) {
  const inc  = incDeg  * Math.PI/180;
  const raan = raanDeg * Math.PI/180;
  const ph   = phaseDeg* Math.PI/180;
  const v    = Math.sqrt(GM/radius);

  // position in orbital plane
  const xo = radius * Math.cos(ph);
  const yo = radius * Math.sin(ph);
  // velocity in orbital plane (circular, perpendicular to position)
  const vxo = -v * Math.sin(ph);
  const vyo =  v * Math.cos(ph);

  // rotate by inclination then RAAN
  const ci=Math.cos(inc), si=Math.sin(inc);
  const cr=Math.cos(raan), sr=Math.sin(raan);

  return {
    x:  cr*xo - sr*ci*yo,
    y:  sr*xo + cr*ci*yo,
    z:  si*yo,
    vx: cr*vxo - sr*ci*vyo,
    vy: sr*vxo + cr*ci*vyo,
    vz: si*vyo
  };
}

let _bodyId = 0;
function createBody(type, name, radius, incDeg, raanDeg, phaseDeg, fuelCapacity=100) {
  const s = circularOrbitState(radius, incDeg, raanDeg, phaseDeg);
  return {
    id: `${type.toUpperCase()}-${String(++_bodyId).padStart(type==='sat'?2:3,'0')}`,
    name,
    type,
    x: s.x, y: s.y, z: s.z,
    vx: s.vx, vy: s.vy, vz: s.vz,
    fuel: fuelCapacity,
    maxFuel: fuelCapacity,
    status: 'SAFE',
    trail: []       // recent positions for rendering
  };
}

// ─── Simulation Engine ───
export class SimulationEngine {
  constructor() {
    this.satellites = [];
    this.debris = [];
    this.dangerPairs = [];
    this.time = 0;
    this.stepCount = 0;
    this.collisionsAvoided = 0;
    this.collisions = 0;
    this.totalFuelUsed = 0;
    this.reward = 0;
    this.episodeId = this._genEpisodeId();
    this.done = false;
    this.message = '';
    this.recentEvents = [];
    this._init();
  }

  _genEpisodeId() { return 'EP-' + Math.random().toString(36).slice(2,8).toUpperCase(); }

  _init() {
    _bodyId = 0;
    this.satellites = [];
    this.debris = [];

    // ── 8 Satellites in varied orbits ──
    const satConfigs = [
      { name:'Sentinel-A', r:7.0, inc:51,  raan:0,   ph:0  },
      { name:'Sentinel-B', r:7.0, inc:51,  raan:0,   ph:180},
      { name:'Horizon-1',  r:7.8, inc:28,  raan:45,  ph:90 },
      { name:'Horizon-2',  r:7.8, inc:28,  raan:45,  ph:270},
      { name:'Polar-Eye',  r:8.5, inc:97,  raan:120, ph:60 },
      { name:'Equator-X',  r:6.8, inc:5,   raan:200, ph:150},
      { name:'Vanguard',   r:9.0, inc:65,  raan:80,  ph:30 },
      { name:'Apex',       r:8.2, inc:42,  raan:300, ph:210},
    ];
    for (const c of satConfigs) {
      this.satellites.push(createBody('sat', c.name, c.r, c.inc, c.raan, c.ph));
    }

    // ── 30 Debris ──
    _bodyId = 0;
    for (let i = 0; i < 30; i++) {
      const r    = 6.4 + Math.random() * 3.6;
      const inc  = Math.random() * 120;
      const raan = Math.random() * 360;
      const ph   = Math.random() * 360;
      this.debris.push(createBody('deb', `Debris`, r, inc, raan, ph, 0));
    }
  }

  reset() {
    this.time = 0;
    this.stepCount = 0;
    this.collisionsAvoided = 0;
    this.collisions = 0;
    this.totalFuelUsed = 0;
    this.reward = 0;
    this.done = false;
    this.message = '';
    this.episodeId = this._genEpisodeId();
    this.recentEvents = [];
    this._init();
  }

  // Apply a delta-v burn to a satellite
  applyAction(action) {
    const sat = this.satellites.find(s => s.id === action.satellite_id);
    if (!sat || sat.fuel <= 0) return false;

    const dv = action.delta_v;
    const dvMag = vec3Len(dv);
    const fuelCost = dvMag * 30;

    if (sat.fuel < fuelCost * 0.1) return false; // not enough fuel even for a tiny burn

    sat.vx += dv[0];
    sat.vy += dv[1];
    sat.vz += dv[2];
    sat.fuel = Math.max(0, sat.fuel - fuelCost);
    this.totalFuelUsed += fuelCost;
    this.reward -= dvMag * 5; // fuel penalty
    return true;
  }

  step(dt) {
    this.time += dt;
    this.stepCount++;
    this.recentEvents = [];

    const allBodies = [...this.satellites, ...this.debris];

    // ── Propagate all bodies (symplectic Euler) ──
    for (const b of allBodies) {
      // Store trail point (every 4th step to save memory)
      if (this.stepCount % 4 === 0) {
        b.trail.push([b.x, b.y, b.z]);
        if (b.trail.length > 200) b.trail.shift();
      }

      // Gravitational acceleration
      const r = Math.sqrt(b.x*b.x + b.y*b.y + b.z*b.z);
      if (r < EARTH_RADIUS * 0.9) {
        // re-entered — respawn debris, mark satellite as lost
        if (b.type === 'deb') this._respawnDebris(b);
        continue;
      }
      const aG = -GM / (r * r * r); // factor for acceleration (includes 1/r for direction)
      b.vx += aG * b.x * dt;
      b.vy += aG * b.y * dt;
      b.vz += aG * b.z * dt;

      b.x += b.vx * dt;
      b.y += b.vy * dt;
      b.z += b.vz * dt;
    }

    // ── Collision / proximity detection ──
    this.dangerPairs = [];
    for (let i = 0; i < this.satellites.length; i++) {
      const sat = this.satellites[i];
      sat.status = 'SAFE';

      // vs debris
      for (let j = 0; j < this.debris.length; j++) {
        this._checkPair(sat, this.debris[j]);
      }
      // vs other satellites
      for (let k = i+1; k < this.satellites.length; k++) {
        this._checkPair(sat, this.satellites[k]);
      }
    }

    // Build observation
    return {
      done: this.done,
      reward: this.reward,
      satellites: this.satellites.map(s => ({
        id: s.id, name: s.name,
        x: s.x, y: s.y, z: s.z,
        vx: s.vx, vy: s.vy, vz: s.vz,
        fuel: s.fuel, maxFuel: s.maxFuel,
        status: s.status, trail: s.trail
      })),
      debris: this.debris.map(d => ({
        id: d.id,
        x: d.x, y: d.y, z: d.z,
        vx: d.vx, vy: d.vy, vz: d.vz,
        trail: d.trail
      })),
      danger_pairs: this.dangerPairs,
      message: this.message,
      time: this.time,
      stepCount: this.stepCount,
      episodeId: this.episodeId,
      collisionsAvoided: this.collisionsAvoided,
      collisions: this.collisions,
      totalFuelUsed: this.totalFuelUsed,
      reward: this.reward,
      recentEvents: this.recentEvents
    };
  }

  _checkPair(a, b) {
    const dp = [b.x-a.x, b.y-a.y, b.z-a.z];
    const dist = vec3Len(dp);

    if (dist > CAUTION_R) return;

    // TCA (linear approximation)
    const dv = [b.vx-a.vx, b.vy-a.vy, b.vz-a.vz];
    const dvLen2 = vec3Dot(dv, dv);
    let tca = 0, missDistance = dist;
    if (dvLen2 > 1e-10) {
      tca = -vec3Dot(dp, dv) / dvLen2;
      if (tca < 0) tca = 0; // already past closest approach in this direction
      const posAtTca = vec3Add(dp, vec3Scale(dv, tca));
      missDistance = vec3Len(posAtTca);
    }

    let level = 'CAUTION';
    if (missDistance < COLLISION_R) {
      level = 'COLLISION';
      this.collisions++;
      this.reward -= 500;
      this.recentEvents.push({ type: 'collision', pair: [a.id, b.id] });
      a.status = 'DANGER';
    } else if (missDistance < CRITICAL_R || (dist < CRITICAL_R)) {
      level = 'CRITICAL';
      a.status = 'DANGER';
    } else if (missDistance < WARNING_R || (dist < WARNING_R)) {
      level = 'WARNING';
      if (a.status !== 'DANGER') a.status = 'CAUTION';
    } else {
      if (a.status === 'SAFE') a.status = a.status; // keep safe
    }

    this.dangerPairs.push({
      sat_id: a.id,
      obj_id: b.id,
      distance: dist,
      miss_distance: missDistance,
      tca,
      level,
      midpoint: [(a.x+b.x)/2, (a.y+b.y)/2, (a.z+b.z)/2]
    });
  }

  _respawnDebris(d) {
    const r    = 6.4 + Math.random() * 3.6;
    const inc  = Math.random() * 120;
    const raan = Math.random() * 360;
    const ph   = Math.random() * 360;
    const s = circularOrbitState(r, inc, raan, ph);
    d.x = s.x; d.y = s.y; d.z = s.z;
    d.vx = s.vx; d.vy = s.vy; d.vz = s.vz;
    d.trail = [];
  }

  markAvoided(satId, objId) {
    this.collisionsAvoided++;
    this.reward += 100;
    this.recentEvents.push({ type: 'avoided', pair: [satId, objId] });
  }

  getState() {
    return {
      episodeId: this.episodeId,
      stepCount: this.stepCount,
      timeElapsed: this.time,
      collisionsAvoided: this.collisionsAvoided,
    };
  }
}

export { EARTH_RADIUS, CRITICAL_R, WARNING_R, CAUTION_R, COLLISION_R };
export { vec3Len, vec3Sub, vec3Add, vec3Scale, vec3Cross, vec3Norm, vec3Dot };
