/* main.js — Improved & Fixed Game Loop */
import { SimulationEngine } from './simulation.js';
import { CollisionAvoidanceAgent, NeuralAgent } from './agent.js';
import { SceneManager } from './renderer.js';
import { UIManager, SoundEngine } from './ui.js';

let sim, agent, scene, ui, sound;
let lastTime = 0;
let alertCooldowns = new Map();

async function init() {
  sim = new SimulationEngine();

  // Load neural agent or fallback
  agent = await NeuralAgent.load('./trained_weights.json').catch(() => null);
  agent = null;
  const agentType = agent ? 'Neural PPO' : 'Heuristic Agent';
  if (!agent) agent = new CollisionAvoidanceAgent();

  scene = new SceneManager(document.getElementById('canvas-container'));
  ui = new UIManager();
  sound = new SoundEngine();

  document.getElementById('sound-btn').addEventListener('click', () => {
    const on = sound.toggle();
    document.getElementById('sound-btn').textContent = on ? '🔊' : '🔇';
  });

  document.addEventListener('click', () => sound._ensure(), { once: true });

  await scene.init();

  ui.addLog(`[SYS] Orbital Guardian started — Episode ${sim.episodeId}`, 'info');
  ui.addLog(`[SYS] Agent: ${agentType}`, agentType.includes('Neural') ? 'avoid' : 'info');
  ui.addLog(`[SYS] ${sim.satellites.length} satellites + ${sim.debris.length} debris`, 'info');

  ui.hideLoading();
  requestAnimationFrame(loop);
}

function loop(timestamp) {
  requestAnimationFrame(loop);

  const rawDt = Math.min((timestamp - lastTime) / 1000, 0.05);
  lastTime = timestamp;

  const warp = ui.getTimeWarp();
  if (warp === 0) {
    scene.tick();   // still allow camera movement when paused
    return;
  }

  const dt = rawDt * warp * 2;   // ← Increased effective speed (8×) so orbits are visible

  const MAX_DT = 0.15;
  const substeps = Math.ceil(dt / MAX_DT);
  const subDt = dt / substeps;
  let obs;
  for (let i = 0; i < substeps; i++) obs = sim.step(subDt);

  // Agent actions
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
    if (sim.applyAction(action)) {
      const screenPos = scene.showManeuver(action.satellite_id, action.delta_v);
      sound.playThruster();

      const dv = action.delta_v.map(v => v.toFixed(3));
      ui.addLog(`[${formatTime(obs.time)}] ${action.satellite_id}: Δv [${dv}] — ${action.metadata.reason}`, 'burn');

      if (screenPos) ui.showManeuverPopup(screenPos, `BURN — ${action.satellite_id}`);
    }
  }

  // Events & alerts (same as before)
  for (const evt of obs.recentEvents) {
    const t = formatTime(obs.time);
    if (evt.type === 'avoided') {
      ui.addLog(`[${t}] ✓ AVOIDED — ${evt.pair[0]} ↔ ${evt.pair[1]}`, 'avoid');
      sound.playSuccess();
    } else if (evt.type === 'collision') {
      ui.addLog(`[${t}] ✗ COLLISION — ${evt.pair[0]} ↔ ${evt.pair[1]}`, 'alert');
    }
  }

  for (const dp of obs.danger_pairs) {
    if ((dp.level === 'WARNING' || dp.level === 'CRITICAL') && 
        dp.tca <= 0.5 && 
        dp.miss_distance > 0.12) {           // increased threshold

      const key = `${dp.sat_id}|${dp.obj_id}`;
      
      // If agent recently burned for this pair → count as avoided
      if (agent.recentBurns && agent.recentBurns.has(key)) {
        sim.markAvoided(dp.sat_id, dp.obj_id);
        agent.recentBurns.delete(key);
        ui.addLog(`[${formatTime(obs.time)}] ✓ COLLISION SUCCESSFULLY AVOIDED — ${dp.sat_id} ↔ ${dp.obj_id}`, 'avoid');
      }
    }
  }

  // Update visuals
  scene.updateBodies(obs.satellites, obs.debris);
  scene.updateDangerZones(obs.danger_pairs);
  scene.tick();

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

init().catch(err => console.error(err));