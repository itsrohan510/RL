/* ═══════════════════════════════════════════════════
   main.js — Bootstrap & Game Loop
   ═══════════════════════════════════════════════════ */

import { SimulationEngine }        from './simulation.js';
import { CollisionAvoidanceAgent, NeuralAgent } from './agent.js';
import { SceneManager }            from './renderer.js';
import { UIManager, SoundEngine }  from './ui.js';

let sim, agent, scene, ui, sound;
let lastTime = 0;
let alertCooldowns = new Map();

async function init() {
  sim   = new SimulationEngine();

  // Try loading trained neural agent; fall back to heuristic
  agent = await NeuralAgent.load('./trained_weights.json');
  const agentType = agent ? 'Neural PPO Policy' : 'Heuristic';
  if (!agent) agent = new CollisionAvoidanceAgent();
  scene = new SceneManager(document.getElementById('canvas-container'));
  ui    = new UIManager();
  sound = new SoundEngine();

  // Sound toggle
  document.getElementById('sound-btn').addEventListener('click', () => {
    const on = sound.toggle();
    document.getElementById('sound-btn').textContent = on ? '🔊' : '🔇';
  });

  // Initialize sound on first click anywhere (autoplay policy)
  document.addEventListener('click', () => sound._ensure(), { once: true });

  await scene.init();

  ui.addLog(`[SYS] Orbital Guardian initialized — Episode ${sim.episodeId}`, 'info');
  ui.addLog(`[SYS] Agent: ${agentType}`, agentType.includes('Neural') ? 'avoid' : 'info');
  ui.addLog(`[SYS] Tracking ${sim.satellites.length} satellites, ${sim.debris.length} debris objects`, 'info');

  ui.hideLoading();

  requestAnimationFrame(loop);
}

function loop(timestamp) {
  requestAnimationFrame(loop);

  const rawDt = Math.min((timestamp - lastTime) / 1000, 0.05); // cap at 50ms
  lastTime = timestamp;

  const warp = ui.getTimeWarp();
  if (warp === 0) {
    scene.tick();
    return; // paused
  }

  const dt = rawDt * warp;

  // ── 1. Step simulation ──
  const obs = sim.step(dt);

  // ── 2. Agent evaluates & acts (Throttled for NeuralAgent to match 0.2s training) ──
  let actions = [];
  if (agent instanceof NeuralAgent) {
    agent.accumulator = (agent.accumulator || 0) + dt;
    if (agent.accumulator >= 0.2) {
      actions = agent.evaluate(obs);
      agent.accumulator -= 0.2;
    }
  } else {
    actions = agent.evaluate(obs);
  }

  for (const action of actions) {
    const ok = sim.applyAction(action);
    if (ok) {
      const m = action.metadata;

      // Thruster visual + sound
      const screenPos = scene.showManeuver(action.satellite_id, action.delta_v);
      sound.playThruster();

      // Log
      const dv = action.delta_v.map(v => v.toFixed(3));
      const timeStr = formatTime(obs.time);
      ui.addLog(
        `[${timeStr}] ${action.satellite_id}: Δv [${dv}] — ${m.reason}`,
        'burn'
      );

      if (screenPos) {
        ui.showManeuverPopup(screenPos, `BURN — ${action.satellite_id}`);
      }
    }
  }

  // ── 3. Handle events ──
  for (const evt of obs.recentEvents) {
    const timeStr = formatTime(obs.time);
    if (evt.type === 'avoided') {
      ui.addLog(`[${timeStr}] ✓ COLLISION AVOIDED — ${evt.pair[0]} ↔ ${evt.pair[1]}`, 'avoid');
      sound.playSuccess();
    } else if (evt.type === 'collision') {
      ui.addLog(`[${timeStr}] ✗ COLLISION — ${evt.pair[0]} ↔ ${evt.pair[1]}`, 'alert');
    }
  }

  // ── 4. Alerts for critical threats ──
  for (const dp of obs.danger_pairs) {
    if (dp.level === 'CRITICAL') {
      const key = `${dp.sat_id}|${dp.obj_id}`;
      const lastAlert = alertCooldowns.get(key) || 0;
      if (timestamp - lastAlert > 5000) {
        ui.showAlert('CRITICAL', `${dp.sat_id} ↔ ${dp.obj_id} — Miss: ${dp.miss_distance.toFixed(3)} — TCA: ${dp.tca.toFixed(1)}s`);
        sound.playAlarm('CRITICAL');
        sound.playHeartbeat();
        alertCooldowns.set(key, timestamp);
      }
    }
  }

  // ── 5. Mark avoidances (pair was critical, now resolved by distance) ──
  // Simple heuristic: if agent burned for a pair and distance is now increasing
  for (const dp of obs.danger_pairs) {
    if (dp.level === 'WARNING' || dp.level === 'CRITICAL') {
      const key = `${dp.sat_id}|${dp.obj_id}`;
      if (agent.recentBurns.has(key) && dp.tca <= 0.1 && dp.miss_distance > 0.1) {
        sim.markAvoided(dp.sat_id, dp.obj_id);
        agent.recentBurns.delete(key);
      }
    }
  }

  // ── 6. Update renderer & UI ──
  scene.updateBodies(obs.satellites, obs.debris);
  scene.updateDangerZones(obs.danger_pairs);
  scene.tick();

  // Throttle UI updates (every 3rd frame)
  if (obs.stepCount % 3 === 0) {
    ui.updateClock(obs.time, obs.episodeId);
    ui.updateRoster(obs.satellites);
    ui.updateThreats(obs.danger_pairs);
    ui.updateStats(obs);
  }
}

function formatTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const ms = Math.floor((t % 1) * 10);
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${ms}`;
}

init().catch(err => {
  console.error('Fatal init error:', err);
  document.getElementById('loading-screen').querySelector('.loader-subtext').textContent =
    'Error initializing — check console';
});
