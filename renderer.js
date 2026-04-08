/* ═══════════════════════════════════════════════════
   renderer.js — Three.js 3D Orbital Scene
   ═══════════════════════════════════════════════════ */

import * as THREE from 'three';
import { OrbitControls }   from 'three/addons/controls/OrbitControls.js';
import { EffectComposer }  from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { EARTH_RADIUS }    from './simulation.js';

const SAT_COLOR     = new THREE.Color(0x00f5ff);
const DEBRIS_COLOR  = new THREE.Color(0xff6b35);
const DANGER_COLOR  = new THREE.Color(0xff0040);
const SAFE_COLOR    = new THREE.Color(0x00ff88);
const TRAIL_SAT     = new THREE.Color(0x00a0cc);
const TRAIL_DEB     = new THREE.Color(0x993d1a);

const TRAIL_MAX = 15;

export class SceneManager {
  constructor(container) {
    this.container = container;
    this.satMeshes = new Map();
    this.debMeshes = new Map();
    this.satTrails = new Map();
    this.debTrails = new Map();
    this.dangerVisuals = [];
    this.thrusterEffects = [];
    this.maneuverCallbacks = [];
    this._clock = new THREE.Clock();
  }

  async init() {
    const w = window.innerWidth, h = window.innerHeight;

    // ── Scene ──
    this.scene = new THREE.Scene();

    // ── Camera ──
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 500);
    this.camera.position.set(12, 8, 14);

    // ── Renderer ──
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.container.appendChild(this.renderer.domElement);

    // ── Controls ──
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 7;
    this.controls.maxDistance = 50;

    // ── Post-processing ──
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 1.2, 0.5, 0.15);
    this.composer.addPass(bloom);

    // ── Build scene ──
    this._createStarfield();
    await this._createEarth();
    this._createAmbientLight();

    window.addEventListener('resize', () => this._onResize());
  }

  /* ═══════ EARTH ═══════ */
  async _createEarth() {
    const loader = new THREE.TextureLoader();
    const R = EARTH_RADIUS;

    // Try NASA Blue Marble, fallback to procedural
    let dayMap;
    try {
      dayMap = await new Promise((res, rej) => {
        loader.load(
          'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
          res, undefined, rej
        );
      });
    } catch {
      dayMap = this._proceduralEarthTexture();
    }

    const earthGeo = new THREE.SphereGeometry(R, 64, 64);
    const earthMat = new THREE.MeshStandardMaterial({
      map: dayMap,
      roughness: 0.85,
      metalness: 0.05,
    });
    this.earthMesh = new THREE.Mesh(earthGeo, earthMat);
    this.scene.add(this.earthMesh);

    // ── Atmosphere glow ──
    const atmosGeo = new THREE.SphereGeometry(R * 1.025, 64, 64);
    const atmosMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewPos;
        void main(){
          vNormal = normalize(normalMatrix * normal);
          vViewPos = (modelViewMatrix * vec4(position,1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
        }`,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vViewPos;
        void main(){
          float rim = 1.0 - abs(dot(vNormal, normalize(-vViewPos)));
          float intensity = pow(rim, 3.0) * 0.9;
          gl_FragColor = vec4(0.35, 0.65, 1.0, intensity);
        }`,
      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.FrontSide,
      depthWrite: false,
    });
    this.atmosphere = new THREE.Mesh(atmosGeo, atmosMat);
    this.scene.add(this.atmosphere);
  }

  _proceduralEarthTexture() {
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size; canvas.height = size / 2;
    const ctx = canvas.getContext('2d');
    // Ocean gradient
    const g = ctx.createLinearGradient(0, 0, 0, canvas.height);
    g.addColorStop(0, '#1a3a5c');
    g.addColorStop(0.5, '#1e5080');
    g.addColorStop(1, '#1a3a5c');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, canvas.height);

    // Rough continent shapes with noise
    ctx.fillStyle = '#2d6b3f';
    for (let i = 0; i < 120; i++) {
      const x = Math.random() * size;
      const y = Math.random() * canvas.height;
      const r = 5 + Math.random() * 25;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    const tex = new THREE.CanvasTexture(canvas);
    return tex;
  }

  /* ═══════ STARFIELD ═══════ */
  _createStarfield() {
    const count = 4000;
    const positions = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 80 + Math.random() * 120;
      positions[i*3]   = r * Math.sin(phi) * Math.cos(theta);
      positions[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i*3+2] = r * Math.cos(phi);
      sizes[i] = 0.3 + Math.random() * 1.2;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
    const mat = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.15,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.8,
    });
    this.scene.add(new THREE.Points(geo, mat));
  }

  /* ═══════ LIGHTING ═══════ */
  _createAmbientLight() {
    this.scene.add(new THREE.AmbientLight(0x334466, 0.6));
    const sun = new THREE.DirectionalLight(0xfff5e0, 2.0);
    sun.position.set(30, 15, 20);
    this.scene.add(sun);
  }

  /* ═══════ UPDATE BODIES ═══════ */
  updateBodies(satellites, debris) {
    // ── Satellites ──
    for (const sat of satellites) {
      let mesh = this.satMeshes.get(sat.id);
      if (!mesh) {
        mesh = this._createSatMesh();
        this.scene.add(mesh);
        this.satMeshes.set(sat.id, mesh);
      }
      mesh.position.set(sat.x, sat.y, sat.z);

      // color by status
      if (sat.status === 'DANGER') {
        mesh.material.emissive.copy(DANGER_COLOR);
        mesh.material.color.copy(DANGER_COLOR);
      } else if (sat.status === 'CAUTION') {
        mesh.material.emissive.lerp(SAT_COLOR, 0.5);
        mesh.material.color.copy(SAT_COLOR);
      } else {
        mesh.material.emissive.copy(SAT_COLOR);
        mesh.material.color.copy(SAT_COLOR);
      }

      // Trail
      this._updateTrail(sat.id, sat.trail, this.satTrails, TRAIL_SAT);
    }

    // ── Debris ──
    for (const deb of debris) {
      let mesh = this.debMeshes.get(deb.id);
      if (!mesh) {
        mesh = this._createDebrisMesh();
        this.scene.add(mesh);
        this.debMeshes.set(deb.id, mesh);
      }
      mesh.position.set(deb.x, deb.y, deb.z);
      mesh.rotation.x += 0.02;
      mesh.rotation.y += 0.03;

      this._updateTrail(deb.id, deb.trail, this.debTrails, TRAIL_DEB);
    }
  }

  _createSatMesh() {
    const geo = new THREE.OctahedronGeometry(0.09, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: SAT_COLOR,
      emissive: SAT_COLOR,
      emissiveIntensity: 2.5,
      roughness: 0.3,
      metalness: 0.7,
    });
    return new THREE.Mesh(geo, mat);
  }

  _createDebrisMesh() {
    const geo = new THREE.IcosahedronGeometry(0.04, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: DEBRIS_COLOR,
      emissive: DEBRIS_COLOR,
      emissiveIntensity: 1.5,
      roughness: 0.6,
      metalness: 0.4,
    });
    return new THREE.Mesh(geo, mat);
  }

  /* ═══════ TRAILS ═══════ */
  _updateTrail(id, trailData, trailMap, color) {
    if (!trailData || trailData.length < 2) return;

    let line = trailMap.get(id);
    const maxPts = Math.min(trailData.length, TRAIL_MAX);
    const positions = new Float32Array(maxPts * 3);
    const colors = new Float32Array(maxPts * 3);

    for (let i = 0; i < maxPts; i++) {
      const p = trailData[trailData.length - maxPts + i];
      positions[i*3]   = p[0];
      positions[i*3+1] = p[1];
      positions[i*3+2] = p[2];

      const alpha = i / maxPts;     // 0=old, 1=newest
      colors[i*3]   = color.r * alpha;
      colors[i*3+1] = color.g * alpha;
      colors[i*3+2] = color.b * alpha;
    }

    if (!line) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7 });
      line = new THREE.Line(geo, mat);
      this.scene.add(line);
      trailMap.set(id, line);
    } else {
      const geo = line.geometry;
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geo.attributes.position.needsUpdate = true;
      geo.attributes.color.needsUpdate = true;
    }
  }

  /* ═══════ DANGER ZONES ═══════ */
  updateDangerZones(dangerPairs) {
    // Remove old visuals
    for (const vis of this.dangerVisuals) {
      this.scene.remove(vis.sphere);
      this.scene.remove(vis.line);
      vis.sphere.geometry.dispose();
      vis.sphere.material.dispose();
      vis.line.geometry.dispose();
      vis.line.material.dispose();
    }
    this.dangerVisuals = [];

    const t = performance.now() * 0.003;

    for (const dp of dangerPairs) {
      if (dp.level === 'CAUTION') continue; // only show WARNING+

      const pulse = 0.5 + 0.5 * Math.sin(t * (dp.level === 'CRITICAL' ? 5 : 2));
      const radius = dp.level === 'CRITICAL' ? 0.25 : 0.15;

      // Pulsating sphere at midpoint
      const sphereGeo = new THREE.SphereGeometry(radius * (0.8 + 0.4 * pulse), 16, 16);
      const sphereMat = new THREE.MeshBasicMaterial({
        color: DANGER_COLOR,
        transparent: true,
        opacity: 0.15 + 0.15 * pulse,
        depthWrite: false,
      });
      const sphere = new THREE.Mesh(sphereGeo, sphereMat);
      sphere.position.set(dp.midpoint[0], dp.midpoint[1], dp.midpoint[2]);
      this.scene.add(sphere);

      // Connecting line between the two objects
      const satMesh = this.satMeshes.get(dp.sat_id);
      const objMesh = this.satMeshes.get(dp.obj_id) || this.debMeshes.get(dp.obj_id);
      const linePositions = new Float32Array(6);
      if (satMesh && objMesh) {
        linePositions[0] = satMesh.position.x;
        linePositions[1] = satMesh.position.y;
        linePositions[2] = satMesh.position.z;
        linePositions[3] = objMesh.position.x;
        linePositions[4] = objMesh.position.y;
        linePositions[5] = objMesh.position.z;
      }
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: dp.level === 'CRITICAL' ? 0xff0040 : 0xff6b35,
        transparent: true,
        opacity: 0.4 + 0.4 * pulse,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      this.scene.add(line);

      this.dangerVisuals.push({ sphere, line });
    }
  }

  /* ═══════ THRUSTER EFFECT ═══════ */
  showManeuver(satelliteId, deltaV) {
    const mesh = this.satMeshes.get(satelliteId);
    if (!mesh) return;

    // Particle burst in opposite direction of burn
    const burstDir = [-deltaV[0], -deltaV[1], -deltaV[2]];
    const particleCount = 40;
    const positions = new Float32Array(particleCount * 3);
    const velocities = [];

    for (let i = 0; i < particleCount; i++) {
      positions[i*3]   = mesh.position.x;
      positions[i*3+1] = mesh.position.y;
      positions[i*3+2] = mesh.position.z;
      velocities.push([
        burstDir[0] * (0.3 + Math.random() * 0.7) + (Math.random()-0.5)*0.15,
        burstDir[1] * (0.3 + Math.random() * 0.7) + (Math.random()-0.5)*0.15,
        burstDir[2] * (0.3 + Math.random() * 0.7) + (Math.random()-0.5)*0.15,
      ]);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0x4488ff,
      size: 0.06,
      transparent: true,
      opacity: 1.0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const points = new THREE.Points(geo, mat);
    this.scene.add(points);
    this.thrusterEffects.push({ points, velocities, age: 0, maxAge: 1.5 });

    // Briefly flash the satellite green
    const origEmissive = mesh.material.emissive.clone();
    mesh.material.emissive.copy(SAFE_COLOR);
    setTimeout(() => { mesh.material.emissive.copy(origEmissive); }, 400);

    // Return screen position for UI popup
    return this._toScreenPos(mesh.position);
  }

  _toScreenPos(worldPos) {
    const v = worldPos.clone().project(this.camera);
    return {
      x: (v.x * 0.5 + 0.5) * window.innerWidth,
      y: (-(v.y * 0.5) + 0.5) * window.innerHeight,
    };
  }

  /* ═══════ ANIMATION TICK ═══════ */
  tick() {
    const delta = this._clock.getDelta();

    // Rotate Earth slowly
    if (this.earthMesh) this.earthMesh.rotation.y += delta * 0.03;

    // Animate thruster effects
    for (let i = this.thrusterEffects.length - 1; i >= 0; i--) {
      const fx = this.thrusterEffects[i];
      fx.age += delta;
      if (fx.age > fx.maxAge) {
        this.scene.remove(fx.points);
        fx.points.geometry.dispose();
        fx.points.material.dispose();
        this.thrusterEffects.splice(i, 1);
        continue;
      }
      const posArr = fx.points.geometry.attributes.position.array;
      for (let j = 0; j < fx.velocities.length; j++) {
        posArr[j*3]   += fx.velocities[j][0] * delta * 2;
        posArr[j*3+1] += fx.velocities[j][1] * delta * 2;
        posArr[j*3+2] += fx.velocities[j][2] * delta * 2;
      }
      fx.points.geometry.attributes.position.needsUpdate = true;
      fx.points.material.opacity = 1.0 - (fx.age / fx.maxAge);
    }

    this.controls.update();
    this.composer.render();
  }

  _onResize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
  }
}
