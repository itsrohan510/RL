/* ═══════════════════════════════════════════════════
   ui.js — HUD Manager & Sound Engine
   ═══════════════════════════════════════════════════ */

/* ─── Sound Engine (Web Audio API) ─── */
export class SoundEngine {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.enabled = true;
    this._started = false;
  }

  _ensure() {
    if (this._started) return true;
    try {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.25;
      this.master.connect(this.ctx.destination);
      this._started = true;
      return true;
    } catch { return false; }
  }

  toggle() {
    this.enabled = !this.enabled;
    if (this.master) this.master.gain.value = this.enabled ? 0.25 : 0;
    return this.enabled;
  }

  playThruster(duration = 1.2) {
    if (!this.enabled || !this._ensure()) return;
    const buf = this.ctx.createBuffer(1, this.ctx.sampleRate * duration, this.ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1);

    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass'; bp.frequency.value = 180; bp.Q.value = 3;
    const g = this.ctx.createGain();
    const now = this.ctx.currentTime;
    g.gain.setValueAtTime(0, now);
    g.gain.linearRampToValueAtTime(0.35, now + 0.08);
    g.gain.exponentialRampToValueAtTime(0.001, now + duration);
    src.connect(bp); bp.connect(g); g.connect(this.master);
    src.start(); src.stop(now + duration);
  }

  playAlarm(level = 'WARNING') {
    if (!this.enabled || !this._ensure()) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'square';

    if (level === 'CRITICAL') {
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.setValueAtTime(660, now + 0.1);
      osc.frequency.setValueAtTime(880, now + 0.2);
      g.gain.setValueAtTime(0.12, now);
      g.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      osc.connect(g); g.connect(this.master);
      osc.start(); osc.stop(now + 0.35);
    } else {
      osc.frequency.value = level === 'WARNING' ? 440 : 330;
      g.gain.setValueAtTime(0.08, now);
      g.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
      osc.connect(g); g.connect(this.master);
      osc.start(); osc.stop(now + 0.4);
    }
  }

  playSuccess() {
    if (!this.enabled || !this._ensure()) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(523, now);
    osc.frequency.setValueAtTime(659, now + 0.12);
    osc.frequency.setValueAtTime(784, now + 0.24);
    g.gain.setValueAtTime(0.15, now);
    g.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
    osc.connect(g); g.connect(this.master);
    osc.start(); osc.stop(now + 0.5);
  }

  playHeartbeat() {
    if (!this.enabled || !this._ensure()) return;
    const now = this.ctx.currentTime;
    for (let i = 0; i < 2; i++) {
      const osc = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      osc.type = 'sine'; osc.frequency.value = 60;
      const t = now + i * 0.18;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(0.2, t + 0.04);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
      osc.connect(g); g.connect(this.master);
      osc.start(t); osc.stop(t + 0.15);
    }
  }
}

/* ─── UI Manager ─── */
export class UIManager {
  constructor() {
    this.timeWarp = 1;
    this.paused = false;
    this._rosterEl  = document.getElementById('satellite-roster');
    this._threatEl  = document.getElementById('threat-feed');
    this._logEl     = document.getElementById('action-log');
    this._clockEl   = document.getElementById('clock-value');
    this._episodeEl = document.getElementById('episode-value');
    this._threatsCountEl = document.getElementById('active-threats');
    this._statAvoided = document.getElementById('stat-avoided');
    this._statFuel    = document.getElementById('stat-fuel');
    this._statReward  = document.getElementById('stat-reward');
    this._statStep    = document.getElementById('stat-step');
    this._alertOverlay = document.getElementById('alert-overlay');
    this._alertTitle   = document.getElementById('alert-title');
    this._alertMessage = document.getElementById('alert-message');
    this._maneuverPopup = document.getElementById('maneuver-popup');
    this._pauseBtn = document.getElementById('pause-btn');
    this._alertTimeout = null;
    this._logEntries = [];

    this._setupWarpButtons();
    this._setupPauseButton();
  }

  _setupWarpButtons() {
    const btns = document.querySelectorAll('.warp-btn');
    btns.forEach(btn => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.timeWarp = parseInt(btn.dataset.warp);
      });
    });
  }

  _setupPauseButton() {
    this._pauseBtn.addEventListener('click', () => {
      this.paused = !this.paused;
      this._pauseBtn.textContent = this.paused ? '▶' : '⏸';
    });
    this._pauseBtn.textContent = '⏸';
  }

  getTimeWarp()  { return this.paused ? 0 : this.timeWarp; }

  /* ── Mission Clock ── */
  updateClock(timeSeconds, episodeId) {
    const h = Math.floor(timeSeconds / 3600);
    const m = Math.floor((timeSeconds % 3600) / 60);
    const s = Math.floor(timeSeconds % 60);
    const ms = Math.floor((timeSeconds % 1) * 10);
    this._clockEl.textContent =
      `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${ms}`;
    this._episodeEl.textContent = `#${episodeId?.slice(-3) || '001'}`;
  }

  /* ── Satellite Roster ── */
  updateRoster(satellites) {
    let html = '';
    for (const sat of satellites) {
      const st = sat.status.toLowerCase();
      const fuelPct = (sat.fuel / sat.maxFuel * 100).toFixed(0);
      const fuelClass = fuelPct < 15 ? 'critical' : fuelPct < 40 ? 'low' : '';
      const alt = Math.sqrt(sat.x**2 + sat.y**2 + sat.z**2).toFixed(2);
      const spd = Math.sqrt(sat.vx**2 + sat.vy**2 + sat.vz**2).toFixed(3);
      html += `
        <div class="sat-card" data-id="${sat.id}">
          <div class="sat-card-header">
            <span class="sat-status-dot ${st}"></span>
            <span class="sat-name">${sat.id}</span>
            <span class="sat-status-label ${st}">${sat.status}</span>
          </div>
          <div class="sat-details">
            <span>ALT ${alt}</span>
            <span>SPD ${spd}</span>
            <span>FUEL ${fuelPct}%</span>
          </div>
          <div class="sat-fuel-bar"><div class="sat-fuel-fill ${fuelClass}" style="width:${fuelPct}%"></div></div>
        </div>`;
    }
    this._rosterEl.innerHTML = html;
  }

  /* ── Threat Feed ── */
  updateThreats(dangerPairs) {
    const important = dangerPairs
      .filter(dp => dp.level !== 'CAUTION')
      .sort((a,b) => a.miss_distance - b.miss_distance);

    this._threatsCountEl.textContent = `${important.length} ACTIVE`;

    let html = '';
    for (const dp of important.slice(0, 10)) {
      const lvl = dp.level.toLowerCase();
      html += `
        <div class="threat-card ${lvl}">
          <div class="threat-header">
            <span class="threat-pair">${dp.sat_id} ↔ ${dp.obj_id}</span>
            <span class="threat-badge ${lvl}">${dp.level}</span>
          </div>
          <div class="threat-details">
            <span>DIST <b>${dp.distance.toFixed(3)}</b></span>
            <span>MISS <b>${dp.miss_distance.toFixed(3)}</b></span>
            <span class="threat-tca">TCA ${dp.tca.toFixed(1)}s</span>
          </div>
        </div>`;
    }
    this._threatEl.innerHTML = html || '<div style="padding:12px;color:var(--color-text-dim);font-size:10px;text-align:center;">No active threats</div>';
  }

  /* ── Action Log ── */
  addLog(text, type = 'info') {
    const cls = type === 'burn' ? 'log-burn' : type === 'avoid' ? 'log-avoid' : type === 'alert' ? 'log-alert' : '';
    this._logEntries.push(`<div class="log-entry"><span class="${cls}">${text}</span></div>`);
    if (this._logEntries.length > 50) this._logEntries.shift();
    this._logEl.innerHTML = this._logEntries.join('');
    this._logEl.scrollTop = this._logEl.scrollHeight;
  }

  /* ── Stats ── */
  updateStats(obs) {
    this._statAvoided.textContent = obs.collisionsAvoided;
    this._statFuel.textContent = obs.totalFuelUsed.toFixed(1);
    this._statReward.textContent = Math.round(obs.reward);
    this._statStep.textContent = obs.stepCount;
  }

  /* ── Alert ── */
  showAlert(level, message) {
    this._alertTitle.textContent = level === 'CRITICAL' ? '⚠ COLLISION WARNING' : 'PROXIMITY ALERT';
    this._alertMessage.textContent = message;
    this._alertOverlay.classList.remove('hidden');
    if (this._alertTimeout) clearTimeout(this._alertTimeout);
    this._alertTimeout = setTimeout(() => this._alertOverlay.classList.add('hidden'), 2500);
  }

  /* ── Maneuver Popup ── */
  showManeuverPopup(screenPos, text) {
    this._maneuverPopup.style.left = `${screenPos.x}px`;
    this._maneuverPopup.style.top = `${screenPos.y}px`;
    this._maneuverPopup.querySelector('#maneuver-text').textContent = text;
    this._maneuverPopup.classList.remove('hidden');
    // Clone to restart animation
    const clone = this._maneuverPopup.cloneNode(true);
    this._maneuverPopup.replaceWith(clone);
    this._maneuverPopup = clone;
    setTimeout(() => clone.classList.add('hidden'), 2100);
  }

  /* ── Loading Screen ── */
  hideLoading() {
    const el = document.getElementById('loading-screen');
    if (el) {
      el.classList.add('fade-out');
      setTimeout(() => el.remove(), 900);
    }
  }
}
