// Reusable microduck viewer: MuJoCo-WASM physics + JS policy + three.js render.
// Physics and rendering both driven from MuJoCo state (data.geom_xpos/xmat), so
// there is no separate kinematics rig. Framework-free; a route mounts it onto a
// canvas. Adapted from pollen-robotics/microduck-simulator (Apache-2.0).
import loadMujoco from '../vendor/mujoco/mujoco.js';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { loadPolicy } from './policy.js';
import {
  JOINT_NAMES, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE, CMD_SIZE, ACTION_SCALE, DECIMATION,
} from './constants.js';

const asset = (p) => new URL(`../assets/${p}`, import.meta.url).href;

async function loadGlbGeometries() {
  const gltf = await new GLTFLoader().loadAsync(asset('microduck.glb'));
  const map = new Map();
  gltf.scene.traverse((o) => {
    if (!o.isMesh || !o.geometry) return;
    const name = o.userData.meshFile || o.name || o.geometry.name;
    if (name && !map.has(name)) map.set(name, o.geometry);
  });
  return map;
}

// opts: { canvas, onStatus?: fn, background?: number }
export async function createViewer({ canvas, onStatus = () => {}, background = 0x0e1116 }) {
  onStatus('loading physics…');
  const mujoco = await loadMujoco({
    locateFile: (p) => p.endsWith('.wasm')
      ? new URL('../vendor/mujoco/mujoco.wasm', import.meta.url).href : p,
  });

  // Robot MJCF has no floor/keyframe (the source app injects the arena at
  // runtime). Add a ground plane; set the spawn pose ourselves.
  let xml = await (await fetch(asset('robot_allcollisions.xml'))).text();
  xml = xml.replace('</worldbody>',
    '<geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" condim="3" friction="1 0.005 0.0001"/></worldbody>');
  const meshNames = [...new Set([...xml.matchAll(/<mesh file="([^"]+)"/g)].map((m) => m[1]))];
  const vfs = new mujoco.MjVFS();
  for (const f of meshNames) {
    vfs.addBuffer('assets/' + f, new Uint8Array(await (await fetch(asset('meshes/' + f))).arrayBuffer()));
  }
  const model = mujoco.MjModel.from_xml_string(xml, vfs);
  const data = new mujoco.MjData(model);

  const qposAdr = JOINT_NAMES.map((n) => model.jnt(n).qposadr);
  const dofAdr = JOINT_NAMES.map((n) => model.jnt(n).dofadr);
  const gyroAdr = model.sensor('imu_ang_vel').adr;
  const freeAdr = model.jnt('trunk_base_freejoint').qposadr;
  const trunkId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, 'trunk_base');

  function resetPose() {
    const q = data.qpos, v = data.qvel;
    for (let i = 0; i < v.length; i++) v[i] = 0;
    q[freeAdr] = 0; q[freeAdr + 1] = 0; q[freeAdr + 2] = 0.12;
    q[freeAdr + 3] = 1; q[freeAdr + 4] = 0; q[freeAdr + 5] = 0; q[freeAdr + 6] = 0;
    for (let j = 0; j < NUM_JOINTS; j++) q[qposAdr[j]] = DEFAULT_POSE[j];
    lastAction.fill(0);
    mujoco.mj_forward(model, data);
  }

  onStatus('loading policy…');
  const meta = await (await fetch(asset('policy.meta.json'))).json();
  const forward = loadPolicy(await (await fetch(asset('policy.bin'))).arrayBuffer(), meta);

  onStatus('building scene…');
  const glb = await loadGlbGeometries();
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(background);
  const MESH = mujoco.mjtGeom.mjGEOM_MESH.value;
  const matCache = new Map();
  const matFor = (i) => {
    const mid = model.geom_matid[i];
    const [r, g, b] = mid >= 0
      ? [model.mat_rgba[mid * 4], model.mat_rgba[mid * 4 + 1], model.mat_rgba[mid * 4 + 2]]
      : [model.geom_rgba[i * 4], model.geom_rgba[i * 4 + 1], model.geom_rgba[i * 4 + 2]];
    const key = `${r.toFixed(3)},${g.toFixed(3)},${b.toFixed(3)}`;
    if (!matCache.has(key)) matCache.set(key,
      new THREE.MeshStandardMaterial({ color: new THREE.Color(r, g, b), roughness: 0.65, metalness: 0.1 }));
    return matCache.get(key);
  };
  const meshGeoms = [];
  for (let i = 0; i < model.ngeom; i++) {
    if (model.geom_type[i] !== MESH || model.geom_group[i] !== 2) continue;  // visual only
    const dataid = model.geom_dataid[i];
    const mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH.value, dataid);
    const geo = glb.get(mname + '.stl') || glb.get(mname);
    if (!geo) continue;
    if (!geo.getAttribute('normal')) geo.computeVertexNormals();
    const m = new THREE.Mesh(geo, matFor(i));
    m.matrixAutoUpdate = false;
    scene.add(m); meshGeoms.push({ i, m });
  }

  const cam = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  cam.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.1));
  const dl = new THREE.DirectionalLight(0xffffff, 1.4); dl.position.set(1, -1, 2); scene.add(dl);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));

  function resize() {
    const w = canvas.clientWidth || 640, h = canvas.clientHeight || 480;
    renderer.setSize(w, h, false);
    cam.aspect = w / h; cam.updateProjectionMatrix();
  }

  // Orbit the camera around the (moving) trunk so the duck stays framed.
  let camAngle = -0.6;
  function aimCamera() {
    const cx = data.xpos[trunkId * 3], cy = data.xpos[trunkId * 3 + 1], cz = data.xpos[trunkId * 3 + 2];
    const R = 0.95;
    cam.position.set(cx + R * Math.cos(camAngle), cy + R * Math.sin(camAngle), cz + 0.11);
    cam.lookAt(cx, cy, cz + 0.02);
  }

  const obs = new Float32Array(OBS_SIZE);
  const cmd = new Float32Array(CMD_SIZE);      // zero command = balance/stand
  const lastAction = new Float32Array(NUM_JOINTS);
  function buildObs() {
    const sens = data.sensordata, qp = data.qpos, qv = data.qvel; let i = 0;
    for (let a = 0; a < 3; a++) obs[i++] = sens[gyroAdr + a];
    const w = qp[freeAdr + 3], x = qp[freeAdr + 4], y = qp[freeAdr + 5], z = qp[freeAdr + 6];
    const cx = -x, cy = -y, cz = -z, gz = -1;
    const tx = 2 * (cy * gz), ty = 2 * (-cx * gz), tz = 0;
    obs[i++] = w * tx + (cy * tz - cz * ty);
    obs[i++] = w * ty + (cz * tx - cx * tz);
    obs[i++] = gz + w * tz + (cx * ty - cy * tx);
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qp[qposAdr[j]] - DEFAULT_POSE[j];
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qv[dofAdr[j]];
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = lastAction[j];
    for (let c = 0; c < CMD_SIZE; c++) obs[i++] = cmd[c];
    return obs;
  }

  function controlStep() {
    const act = forward(buildObs());
    lastAction.set(act);
    const ctrl = data.ctrl;
    for (let j = 0; j < NUM_JOINTS; j++) ctrl[j] = DEFAULT_POSE[j] + act[j] * ACTION_SCALE;
    for (let s = 0; s < DECIMATION; s++) mujoco.mj_step(model, data);
  }
  function syncMeshes() {
    const xp = data.geom_xpos, xm = data.geom_xmat;
    for (const { i, m } of meshGeoms) {
      const p = i * 3, r = i * 9;
      m.matrix.set(xm[r], xm[r + 1], xm[r + 2], xp[p],
        xm[r + 3], xm[r + 4], xm[r + 5], xp[p + 1],
        xm[r + 6], xm[r + 7], xm[r + 8], xp[p + 2], 0, 0, 0, 1);
    }
  }

  resetPose();
  resize();
  let raf = 0, running = false;
  function frame() {
    if (!running) return;
    controlStep();
    camAngle += 0.0015;
    aimCamera(); syncMeshes();
    renderer.render(scene, cam);
    raf = requestAnimationFrame(frame);
  }
  const onResize = () => resize();
  globalThis.addEventListener('resize', onResize);
  onStatus('ready');

  return {
    start() { if (!running) { running = true; raf = requestAnimationFrame(frame); } },
    stop() { running = false; cancelAnimationFrame(raf); },
    push(vx = 0.8, vy = 0.5) { const v = data.qvel; v[0] += vx; v[1] += vy; },
    reset() { resetPose(); },
    setBackground(hex) { scene.background = new THREE.Color(hex); },
    trunkZ: () => data.xpos[trunkId * 3 + 2],
    resize,
    dispose() {
      running = false; cancelAnimationFrame(raf);
      globalThis.removeEventListener('resize', onResize);
      renderer.dispose();
    },
  };
}
