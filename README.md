# microduck-viewer

In-browser physics simulations of Pollen Robotics' **microduck** — a small
articulated balancing robot — built to be **embedded in the
[MACRL2](https://macrl2.github.io/) interactive textbook** as iframes, kept in a
separate repo so the course content stays pure markdown.

MuJoCo compiled to WebAssembly steps the contact physics; a trained RL
locomotion policy runs in **plain JavaScript** at 50 Hz (no onnxruntime); three.js
renders directly from MuJoCo state. **No server, no build step.**

## Routes

| Route | What it shows |
|-------|---------------|
| [`/balance/`](./balance/) | The learned policy keeps the duck upright and recovers from shoves (Push button). |

Each route runs full-viewport and is embed-friendly:
- `?theme=light` switches to the light palette.
- `postMessage({type})` accepts `microduck:push`, `microduck:reset`, `microduck:theme`.

## Layout

```
index.html          landing page + route list
balance/index.html  the /balance/ route (canvas + controls)
src/
  core.js           reusable viewer: mujoco boot + policy + render + loop
  policy.js         plain-JS MLP inference (the onnxruntime replacement)
  constants.js      joint names, default pose, obs/action dims
vendor/
  mujoco/           @mujoco/mujoco WASM build (Apache-2.0)
  three/            three.js core + GLTFLoader/BufferGeometryUtils (MIT)
assets/
  microduck.glb     robot meshes for rendering
  robot_allcollisions.xml   MJCF (floor + spawn injected at load)
  meshes/*.stl      collision/visual meshes for the MuJoCo VFS
  policy.bin + policy.meta.json   re-exported policy weights
tests/smoke.mjs     headless balance+push smoke (needs puppeteer)
```

## Develop

```bash
# serve statically (any static server), e.g.
python3 -m http.server 8080
# open http://localhost:8080/balance/

# headless smoke (uses a puppeteer install on PATH / sibling node_modules)
node tests/smoke.mjs
```

## Attribution

Robot model/meshes ([pollen-robotics/microduck](https://github.com/pollen-robotics/microduck))
and RL policy ([pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl))
are **Apache-2.0**. The sim loop is adapted from Pollen's
[microduck-simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator).
See [`NOTICE`](./NOTICE) and [`LICENSE`](./LICENSE). Policy weights are re-exported
unmodified; only the runtime (onnxruntime → JS) changed.

## Optimization backlog

- Rebuild the MuJoCo mesh VFS from `microduck.glb` at load (as the source app
  does) to drop the ~5 MB of `assets/meshes/*.stl`.
