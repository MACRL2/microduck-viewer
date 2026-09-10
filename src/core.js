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

// opts: { canvas, onStatus?: fn, background?: number, ground?: 'circle'|'plane'|'none' }
//   ground 'circle' (default) — a soft blob shadow under the feet that fades out
//   ground 'plane'            — a full floor + fog fading into the backdrop
//   ground 'none'             — no floor at all
export async function createViewer({ canvas, onStatus = () => {}, background = 0x0e1116, ground = 'circle' }) {
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
    // MuJoCo recenters/reorients mesh assets on compile (mesh_pos/mesh_quat);
    // the GLB carries original vertices, so undo it: world = geom · inv(meshLocal).
    // Without this, offset assemblies (feet/soles/ankles) render staggered and
    // rotated ~90° (pads sideways) even though the physics is correct.
    const meshLocalInv = new THREE.Matrix4().compose(
      new THREE.Vector3(model.mesh_pos[dataid * 3], model.mesh_pos[dataid * 3 + 1], model.mesh_pos[dataid * 3 + 2]),
      new THREE.Quaternion(model.mesh_quat[dataid * 4 + 1], model.mesh_quat[dataid * 4 + 2],
        model.mesh_quat[dataid * 4 + 3], model.mesh_quat[dataid * 4]),   // wxyz -> xyzw
      new THREE.Vector3(1, 1, 1)).invert();
    scene.add(m); meshGeoms.push({ i, m, meshLocalInv });
  }

  const cam = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  cam.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.0));
  const dl = new THREE.DirectionalLight(0xffffff, 1.3);
  dl.position.set(0.6, -0.8, 1.8); scene.add(dl); scene.add(dl.target);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));

  // Ground treatments (see opts.ground). Both 'plane' and 'circle' cast a REAL
  // PCF shadow from the duck's geometry; they differ only in the receiver.
  let groundMat = null, shadowRx = null;
  if (ground !== 'none') {
    dl.castShadow = true; dl.shadow.mapSize.set(2048, 2048); dl.shadow.bias = -0.0015;
    dl.shadow.radius = 4;   // soften the PCF edge so the contact shadow reads as "fading"
    Object.assign(dl.shadow.camera, { left: -0.4, right: 0.4, top: 0.4, bottom: -0.4, near: 0.05, far: 4 });
    for (const { m } of meshGeoms) m.castShadow = true;
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  }
  if (ground === 'plane') {
    // Full floor: same color as the backdrop, with fog fading the far edge into
    // the background so there's no hard horizon line.
    groundMat = new THREE.MeshStandardMaterial({ color: background, roughness: 1, metalness: 0 });
    const g = new THREE.Mesh(new THREE.CircleGeometry(2.5, 64), groundMat);
    g.receiveShadow = true; scene.add(g);
    scene.fog = new THREE.Fog(background, 0.8, 2.8);
  } else if (ground === 'circle') {
    // Local "hovering" contact shadow: a ShadowMaterial receiver disc is fully
    // transparent EXCEPT where the duck's real shadow falls, so the true shadow
    // shape floats under the feet with no visible floor. Follows the trunk.
    shadowRx = new THREE.Mesh(new THREE.CircleGeometry(0.7, 48),
      new THREE.ShadowMaterial({ opacity: 0.32, transparent: true }));
    shadowRx.receiveShadow = true; shadowRx.position.z = 0.002; scene.add(shadowRx);
  }

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
  const _geom = new THREE.Matrix4();
  function syncMeshes() {
    const xp = data.geom_xpos, xm = data.geom_xmat;   // geom_xmat is row-major 3x3
    for (const { i, m, meshLocalInv } of meshGeoms) {
      const p = i * 3, r = i * 9;
      _geom.set(xm[r], xm[r + 1], xm[r + 2], xp[p],
        xm[r + 3], xm[r + 4], xm[r + 5], xp[p + 1],
        xm[r + 6], xm[r + 7], xm[r + 8], xp[p + 2], 0, 0, 0, 1);
      m.matrix.multiplyMatrices(_geom, meshLocalInv);
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
    if (shadowRx) { shadowRx.position.x = data.xpos[trunkId * 3]; shadowRx.position.y = data.xpos[trunkId * 3 + 1]; }
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
    setBackground(hex) {
      const c = new THREE.Color(hex);
      scene.background = c;
      if (scene.fog) scene.fog.color = c;
      if (groundMat) groundMat.color = c;
    },
    trunkZ: () => data.xpos[trunkId * 3 + 2],
    resize,
    dispose() {
      running = false; cancelAnimationFrame(raf);
      globalThis.removeEventListener('resize', onResize);
      renderer.dispose();
    },
  };
}
