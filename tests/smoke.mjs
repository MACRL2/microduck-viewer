// Headless smoke: serve the viewer, load /balance/, let it run, assert the duck
// balances (and survives a push), and save a screenshot. Dev-only; needs the
// puppeteer install from the sibling course repo's node_modules.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from '/workspace/node_modules/puppeteer/lib/esm/puppeteer/puppeteer.js';

const ROOT = normalize(join(fileURLToPath(import.meta.url), '../..'));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.json': 'application/json', '.glb': 'model/gltf-binary',
  '.bin': 'application/octet-stream', '.stl': 'application/octet-stream', '.xml': 'application/xml' };
const server = http.createServer(async (req, res) => {
  try {
    let p = decodeURIComponent(req.url.split('?')[0]); if (p.endsWith('/')) p += 'index.html';
    const file = normalize(join(ROOT, p));
    if (!file.startsWith(ROOT)) return void res.writeHead(403).end();
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': MIME[extname(file)] || 'application/octet-stream' }); res.end(body);
  } catch { res.writeHead(404).end('nf'); }
});
await new Promise((r) => server.listen(0, r));
const port = server.address().port;

const browser = await puppeteer.launch({ headless: 'new',
  args: ['--no-sandbox', '--disable-setuid-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.setViewport({ width: 720, height: 560 });
const errs = [];
page.on('pageerror', (e) => errs.push(e.message));
page.on('requestfailed', (r) => { if (!r.url().endsWith('/favicon.ico')) errs.push('reqfail ' + r.url()); });
page.on('response', (r) => { if (r.status() >= 400 && !r.url().endsWith('/favicon.ico')) errs.push(`${r.status()} ${r.url()}`); });
await page.goto(`http://localhost:${port}/balance/`, { waitUntil: 'load' });
await page.waitForFunction(() => window.__viewer, { timeout: 120000 });
await new Promise((r) => setTimeout(r, 1500));                 // settle + balance
const z0 = await page.evaluate(() => window.__viewer.trunkZ());
await page.screenshot({ path: join(ROOT, 'tests', 'settle.png') });
await page.evaluate(() => window.__viewer.push(1.0, 0.6));     // shove
await new Promise((r) => setTimeout(r, 1500));                 // recover
const z1 = await page.evaluate(() => window.__viewer.trunkZ());
await page.screenshot({ path: join(ROOT, 'tests', 'smoke.png') });
await browser.close(); server.close();

console.log(`trunkZ: settle=${z0.toFixed(3)} after-push=${z1.toFixed(3)}`);
if (errs.length) console.log('page errors:\n  ' + errs.slice(0, 5).join('\n  '));
const ok = z0 > 0.10 && z1 > 0.08 && errs.length === 0;
console.log(ok ? '✅ SMOKE PASS — balances + recovers, no errors' : '❌ SMOKE FAIL');
process.exit(ok ? 0 : 1);
