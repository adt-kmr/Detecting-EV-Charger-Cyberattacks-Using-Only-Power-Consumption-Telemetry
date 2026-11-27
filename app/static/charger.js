// Clay-styled 3D EV charger driven by live model output.
// If app/static/models/ev_charger.glb exists it is loaded instead of the
// procedural body; LEDs, power particles and the alert halo are kept.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const PURPLE = new THREE.Color("#7c5cff");
const PURPLE_SOFT = new THREE.Color("#cfc2ff");
const YELLOW = new THREE.Color("#ffc933");

function clayTexture(size = 256) {
  // fingerprint-ish noise used as bump + roughness to read as modelling clay
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  const img = g.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const x = i % size, y = (i / size) | 0;
    const n = Math.random() * 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(x * 0.09 + Math.sin(y * 0.05) * 3));
    const v = 150 + n * 90;
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(2, 2);
  return t;
}

export class ChargerScene {
  constructor(el) {
    this.el = el;
    this.pAttack = 0;
    this.targetP = 0;
    this.flow = 0.4;
    this.screenData = { p: null, label: "--", group: "--", power: "--", time: "" };

    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }));
    r.setPixelRatio(Math.min(devicePixelRatio, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    el.appendChild(r.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    this.camera.position.set(4.6, 3.2, 6.2);

    const ctl = (this.controls = new OrbitControls(this.camera, r.domElement));
    ctl.target.set(0.3, 1.35, 0);
    ctl.enableDamping = true;
    ctl.enablePan = false;
    ctl.minDistance = 4.5;
    ctl.maxDistance = 11;
    ctl.maxPolarAngle = Math.PI * 0.47;
    ctl.autoRotate = true;
    ctl.autoRotateSpeed = 0.6;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xe8e1ff, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(4, 8, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = 6;
    key.shadow.camera.left = key.shadow.camera.bottom = -5;
    key.shadow.camera.right = key.shadow.camera.top = 5;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xfff1c2, 0.9);
    rim.position.set(-5, 3, -4);
    this.scene.add(rim);

    this.tex = clayTexture();
    this.clay = (color, extra = {}) =>
      new THREE.MeshStandardMaterial({
        color, roughness: 0.92, metalness: 0, bumpMap: this.tex, bumpScale: 0.6,
        roughnessMap: this.tex, ...extra,
      });

    this.buildGround();
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.buildCharger();
    this.buildCable();
    this.buildHalo();
    this.tryGLB();

    this.clock = new THREE.Clock();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    r.domElement.addEventListener("pointerdown", () => (ctl.autoRotate = false));
    this.loop();
  }

  buildGround() {
    const g = new THREE.Mesh(new THREE.CylinderGeometry(3.6, 3.7, 0.25, 96), this.clay("#ffffff", { bumpScale: 0.25, roughnessMap: null, emissive: "#fbf8ff", emissiveIntensity: 0.35 }));
    g.position.y = -0.125;
    g.receiveShadow = true;
    this.scene.add(g);
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, 0.04, 96), this.clay("#f3efff"));
    pad.position.y = 0.02;
    pad.receiveShadow = true;
    this.scene.add(pad);
    // parking bay stripes
    for (const x of [1.2, 2.6]) {
      const s = new THREE.Mesh(new RoundedBoxGeometry(0.12, 0.05, 2.0, 2, 0.03), this.clay("#ffe17a"));
      s.position.set(x, 0.04, 0.6);
      s.receiveShadow = true;
      this.scene.add(s);
    }
    // scan rings that pulse with each scored sample
    this.rings = [];
    for (let i = 0; i < 3; i++) {
      const m = new THREE.Mesh(
        new THREE.RingGeometry(0.95, 1.02, 96),
        new THREE.MeshBasicMaterial({ color: PURPLE, transparent: true, opacity: 0, side: THREE.DoubleSide })
      );
      m.rotation.x = -Math.PI / 2;
      m.position.y = 0.05;
      this.scene.add(m);
      this.rings.push(m);
    }
    this.ringIdx = 0;
  }

  buildCharger() {
    const body = (this.body = new THREE.Group());
    this.root.add(body);
    const add = (geo, mat, x, y, z, parent = body) => {
      const m = new THREE.Mesh(geo, mat);
      m.position.set(x, y, z);
      m.castShadow = m.receiveShadow = true;
      parent.add(m);
      return m;
    };
    // plinth
    add(new RoundedBoxGeometry(1.5, 0.22, 1.0, 4, 0.08), this.clay("#e8e1ff"), 0, 0.11, 0);
    // main pedestal
    this.shell = add(new RoundedBoxGeometry(1.2, 2.5, 0.78, 8, 0.22), this.clay("#ffffff"), 0, 1.47, 0);
    // purple side cheeks
    add(new RoundedBoxGeometry(0.14, 2.3, 0.82, 4, 0.06), this.clay("#a68fff"), -0.6, 1.47, 0);
    add(new RoundedBoxGeometry(0.14, 2.3, 0.82, 4, 0.06), this.clay("#a68fff"), 0.6, 1.47, 0);
    // canopy
    add(new RoundedBoxGeometry(1.5, 0.18, 1.05, 4, 0.08), this.clay("#7c5cff"), 0, 2.8, 0.05);
    // front bezel + screen
    add(new RoundedBoxGeometry(0.92, 0.72, 0.06, 4, 0.05), this.clay("#2a2240", { roughness: 0.6 }), 0, 1.98, 0.39);
    this.screenCanvas = document.createElement("canvas");
    this.screenCanvas.width = 512;
    this.screenCanvas.height = 384;
    this.screenTex = new THREE.CanvasTexture(this.screenCanvas);
    this.screenTex.colorSpace = THREE.SRGBColorSpace;
    const scr = new THREE.Mesh(new THREE.PlaneGeometry(0.82, 0.62), new THREE.MeshBasicMaterial({ map: this.screenTex, toneMapped: false }));
    scr.position.set(0, 1.98, 0.425);
    body.add(scr);
    // status LED bar
    this.ledMat = new THREE.MeshStandardMaterial({ color: PURPLE_SOFT, emissive: PURPLE, emissiveIntensity: 1.4, roughness: 0.4 });
    add(new THREE.CapsuleGeometry(0.045, 0.6, 6, 16), this.ledMat, 0, 1.35, 0.4).rotation.z = Math.PI / 2;
    // dial / button cluster
    for (let i = 0; i < 3; i++) {
      add(new THREE.CylinderGeometry(0.06, 0.06, 0.05, 24), this.clay(i === 1 ? "#ffc933" : "#e8e1ff"), -0.2 + i * 0.2, 1.12, 0.4).rotation.x = Math.PI / 2;
    }
    // vents
    for (let i = 0; i < 4; i++) {
      add(new RoundedBoxGeometry(0.6, 0.035, 0.02, 2, 0.01), this.clay("#e8e1ff"), 0, 0.55 + i * 0.09, 0.4);
    }
    // holster on the right cheek
    add(new RoundedBoxGeometry(0.22, 0.34, 0.26, 4, 0.07), this.clay("#7c5cff"), 0.78, 1.45, 0.12);
    // connector handle parked in holster
    const plug = (this.plug = new THREE.Group());
    plug.position.set(0.86, 1.52, 0.18);
    body.add(plug);
    add(new THREE.CapsuleGeometry(0.075, 0.28, 6, 16), this.clay("#2a2240", { roughness: 0.7 }), 0, 0, 0, plug);
    add(new THREE.CylinderGeometry(0.09, 0.09, 0.1, 24), this.clay("#ffc933"), 0, 0.2, 0, plug);
    plug.rotation.z = -0.35;
  }

  buildCable() {
    const pts = [
      new THREE.Vector3(0.62, 0.55, 0.15),
      new THREE.Vector3(1.05, 0.25, 0.35),
      new THREE.Vector3(1.4, 0.12, 0.6),
      new THREE.Vector3(1.55, 0.35, 0.4),
      new THREE.Vector3(1.25, 0.95, 0.25),
      new THREE.Vector3(0.95, 1.38, 0.2),
    ];
    this.curve = new THREE.CatmullRomCurve3(pts);
    const tube = new THREE.Mesh(new THREE.TubeGeometry(this.curve, 120, 0.045, 14), this.clay("#2a2240", { roughness: 0.75 }));
    tube.castShadow = true;
    this.root.add(tube);
    // energy particles riding the cable
    this.particles = [];
    const pm = new THREE.MeshStandardMaterial({ color: YELLOW, emissive: YELLOW, emissiveIntensity: 1.6 });
    for (let i = 0; i < 18; i++) {
      const s = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12), pm);
      s.userData.t = i / 18;
      this.root.add(s);
      this.particles.push(s);
    }
    this.particleMat = pm;
  }

  buildHalo() {
    // shield halo: calm purple when benign, flaring yellow under attack
    this.haloMat = new THREE.MeshBasicMaterial({ color: PURPLE, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false });
    this.halo = new THREE.Mesh(new THREE.TorusGeometry(1.25, 0.035, 12, 120), this.haloMat);
    this.halo.rotation.x = Math.PI / 2;
    this.halo.position.y = 1.5;
    this.root.add(this.halo);
    this.halo2 = this.halo.clone();
    this.halo2.material = this.haloMat;
    this.halo2.position.y = 2.2;
    this.halo2.scale.setScalar(0.85);
    this.root.add(this.halo2);
  }

  tryGLB() {
    new GLTFLoader().load(
      "/static/models/ev_charger.glb",
      (gltf) => {
        const m = gltf.scene;
        const box = new THREE.Box3().setFromObject(m);
        const size = box.getSize(new THREE.Vector3());
        const s = 2.9 / size.y;
        m.scale.setScalar(s);
        box.setFromObject(m);
        m.position.y -= box.min.y;
        m.position.x -= (box.min.x + box.max.x) / 2;
        m.position.z -= (box.min.z + box.max.z) / 2;
        m.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = true; } });
        this.body.visible = false;
        this.root.add(m);
        this.glb = m;
      },
      undefined,
      () => {} // no custom model supplied: keep the procedural charger
    );
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  drawScreen() {
    const g = this.screenCanvas.getContext("2d");
    const d = this.screenData, att = d.p != null && d.p >= 0.5;
    const W = 512, H = 384;
    g.fillStyle = "#1e1833";
    g.fillRect(0, 0, W, H);
    g.fillStyle = att ? "#ffc933" : "#a68fff";
    g.fillRect(0, 0, W, 10);
    g.font = "700 26px Nunito, sans-serif";
    g.fillStyle = "#cfc2ff";
    g.fillText("POWER SENTINEL", 28, 52);
    g.font = "600 20px Nunito, sans-serif";
    g.fillStyle = "#8c84a8";
    g.fillText(d.time || "", 28, 84);
    g.font = "900 64px Nunito, sans-serif";
    g.fillStyle = att ? "#ffc933" : "#ffffff";
    g.fillText(d.p == null ? "--" : att ? "ATTACK" : "BENIGN", 28, 170);
    g.font = "700 24px Nunito, sans-serif";
    g.fillStyle = "#cfc2ff";
    g.fillText(`P(attack)  ${d.p == null ? "--" : (d.p * 100).toFixed(1) + "%"}`, 28, 222);
    g.fillText(`Group      ${d.group}`, 28, 258);
    g.fillText(`Power      ${d.power}`, 28, 294);
    // probability bar
    g.fillStyle = "#2f2750";
    g.fillRect(28, 322, W - 56, 22);
    g.fillStyle = att ? "#ffc933" : "#7c5cff";
    g.fillRect(28, 322, (W - 56) * (d.p ?? 0), 22);
    this.screenTex.needsUpdate = true;
  }

  /** Feed one scored sample from the backend. */
  update(row) {
    this.targetP = row.p_attack ?? 0;
    this.flow = Math.min(2.5, Math.max(0.2, (row.signals.current_mA || 0) / 600));
    this.screenData = {
      p: row.p_attack,
      group: row.pred_group,
      power: `${(row.signals.power_mW / 1000).toFixed(2)} W`,
      time: `${row.time}  |  ${row.state}`,
    };
    this.drawScreen();
    this.pulseRing(row.p_attack >= 0.5);
  }

  pulseRing(att) {
    const ring = this.rings[this.ringIdx++ % this.rings.length];
    ring.material.color.copy(att ? YELLOW : PURPLE);
    const o = { s: 1, a: 0.55 };
    if (window.anime) {
      anime.remove(o);
      anime({
        targets: o, s: 3.3, a: 0, duration: 1400, easing: "easeOutQuad",
        update: () => { ring.scale.setScalar(o.s); ring.material.opacity = o.a; },
      });
    }
  }

  loop() {
    requestAnimationFrame(() => this.loop());
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    this.pAttack += (this.targetP - this.pAttack) * Math.min(1, dt * 4);
    const p = this.pAttack;
    const col = PURPLE.clone().lerp(YELLOW, p);

    this.ledMat.emissive.copy(col);
    this.ledMat.emissiveIntensity = 1.1 + (p > 0.5 ? Math.sin(t * 10) * 0.6 + 0.6 : Math.sin(t * 2) * 0.2);
    this.haloMat.color.copy(col);
    this.haloMat.opacity = 0.1 + p * 0.45;
    this.halo.rotation.z += dt * (0.3 + p * 2.5);
    this.halo2.rotation.z -= dt * (0.25 + p * 2);
    const hs = 1 + p * 0.08 * Math.sin(t * 8);
    this.halo.scale.setScalar(hs);

    // subtle glitch shake under attack
    const shake = p > 0.5 ? (p - 0.5) * 0.02 : 0;
    this.root.position.x = (Math.random() - 0.5) * shake;
    this.root.position.z = (Math.random() - 0.5) * shake;

    this.particleMat.color.copy(col);
    this.particleMat.emissive.copy(col);
    for (const s of this.particles) {
      s.userData.t = (s.userData.t + dt * 0.12 * this.flow) % 1;
      s.position.copy(this.curve.getPointAt(s.userData.t));
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
