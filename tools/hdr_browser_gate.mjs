#!/usr/bin/env node
// Browser HDR gate for the agate hdr-showcase (finite 240-frame demo).
//
// Runs the ALREADY-BUILT emscripten page (agate/zig-out/web/hdr-showcase.html)
// in real headless Chrome on Metal/WebGPU and validates four legs:
//
//   msaa1-unorm   ?frames=240&msaa=1          (UNORM single-encode proof)
//   msaa4-unorm   ?frames=240&msaa=4          (4x MSAA HDR path)
//   msaa1-srgb    ?frames=240&msaa=1&srgb=1  (sRGB backbuffer, HW encode)
//   msaa2-policy  ?frames=240&msaa=2          (WebGPU 2x unsupported -> 1x)
//
// Per leg the gate requires: setup backend WGPU, verdict PASS with
// failures=0 log_errors=0 sokol_live_allocations=0, checks above a leg
// minimum (a failed named check prints "hdr-showcase: FAIL" and fails the
// verdict, so failures==0 IS the required-checks coverage), the frame-60
// line reporting the expected effective sample count, a real (non-fallback,
// non-SwiftShader) WebGPU adapter, and a non-empty frame-60 canvas
// screenshot with dark-studio + HDR-highlight distribution.
//
// Screenshot comparison UNORM-vs-sRGB is COARSE only (mean + p99 of the
// absolute luma delta): exact pixel equality needs the same encoded frame,
// and the two legs intentionally use different output encodes whose match is
// the engine's exactly-one-encode contract (see postprocess/hdr.zig
// displayColor, unit-tested). The showcase uses a fixed dt=1/60 inside the
// finite gate so both captures run the same animation frame.
//
// Visual proof is screenshot-only: this gate NEVER claims GPU radiance
// readback. HDR 1/4/16 separation has CPU math goldens; here the
// highlight-fraction + cross-encode agreement is the render evidence.
//
// Usage:
//   node hdr_browser_gate.mjs [--out <dir>] [--serve-dir <dir>]
//     [--chrome <path>] [--timeout-ms <n>] [--frames <n>]
//
// Defaults: --out a fresh $TMPDIR/opencode/hdr-browser-* directory when
// available, --serve-dir ../zig-out/web relative to this
// file, --timeout-ms 150000 per leg. Writes NOTHING into the repo: http
// server is read-only, Chrome profile and all artifacts live under --out /
// the system temp dir. Chrome is always cleaned up (Browser.close, then
// SIGTERM/SIGKILL to the process group), even on failure: no orphan browser.

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { writeFileSync, mkdirSync, mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateSync } from 'node:zlib';
import net from 'node:net';

const HERE = dirname(fileURLToPath(import.meta.url));
const CHROME_DEFAULT = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const LEGS = [
  { name: 'msaa1-unorm', msaa: 1, srgb: false, expectSamples: 1, minChecks: 1000 },
  { name: 'msaa4-unorm', msaa: 4, srgb: false, expectSamples: 4, minChecks: 700 },
  { name: 'msaa1-srgb', msaa: 1, srgb: true, expectSamples: 1, minChecks: 1000 },
  { name: 'msaa2-policy', msaa: 2, srgb: false, expectSamples: 1, minChecks: 1000 },
];

// Emscripten routes ordinary stdout/stderr diagnostics through console.error,
// so the type alone means nothing. Classification runs on ANSI-stripped text:
// hard failures always fail; the ONLY allowances are the gate's own
// informational lines (setup / frame progress / PASS verdict) and engine
// `warning:` diagnostics (warn-once policy notes). A FAIL verdict, a GPU
// validation error, a panic, or any other console.error still fails the leg.
const ANSI_RE = /\x1b\[[0-9;]*m/g;
const stripAnsi = (s) => s.replace(ANSI_RE, '');
const HDR_INFO_RE = /^\s*hdr-showcase: (setup backend=\S+|frame \d+ draws=|frames=\d+ checks=\d+ .* verdict=PASS)/;
const HARD_FAIL_RE =
  /GPU(?:Validation|Internal|OutOfMemory)Error|uncaptured(?: GPU)? error|\[(?:sg|sokol|sapp)[^\]]*\]\s*\[(?:error|panic)\]|VALIDATE_[A-Z_]+|validation (?:failed|error)|\bpanic\b|\bfatal\b|^hdr-showcase: FAIL/i;
const SHELL_WARN_RE = /^\s*warning:/;

const VERDICT_RE =
  /hdr-showcase: frames=(\d+) checks=(\d+) failures=(\d+) log_errors=(\d+) sokol_live_allocations=(\d+) verdict=(\w+)/;
const SETUP_RE = /hdr-showcase: setup backend=(\S+) msaa=(\d+) srgb=(\S+) env_color_fmt=(\S+) swapchain_fmt=(\S+)/;
const FRAME_RE = /hdr-showcase: frame (\d+) draws=(\d+).*samples=(\d+) srgb=(\S+) size=(\d+)x(\d+)/;

// Coarse visual thresholds (screenshot evidence only, not radiance proof).
const VIS = {
  minUnique: 100, // distinct colors in a 1/16 stride sample
  minDarkFrac: 0.2, // dark studio interior dominates
  minHiFrac: 0.0005, // HDR rails/cards survive tonemap as highlights
  minStd: 0.005, // frame is not a flat clear color
};
const XENCODE = { maxMeanAbs: 0.03, maxP99Abs: 0.15 }; // UNORM vs sRGB coarse agreement

function arg(name, fallback) {
    const hit = process.argv.find((a) => a.startsWith(name + '='));
    if (hit) return hit.slice(name.length + 1);
    const index = process.argv.indexOf(name);
    return index >= 0 ? process.argv[index + 1] ?? fallback : fallback;
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.on('error', reject);
    s.listen(0, '127.0.0.1', () => {
      const p = s.address().port;
      s.close(() => resolve(p));
    });
  });
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.json': 'application/json',
  '.data': 'application/octet-stream',
};

function startStaticServer(root) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://x');
      let path = decodeURIComponent(url.pathname);
      if (path === '/') path = '/hdr-showcase.html';
      if (path === '/favicon.ico') {
        res.writeHead(204);
        res.end();
        return;
      }
      if (path.includes('..')) {
        res.writeHead(400);
        res.end('bad path');
        return;
      }
      const body = await readFile(join(root, path));
      res.writeHead(200, { 'Content-Type': MIME[extname(path)] || 'application/octet-stream' });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end('not found');
    }
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve(server));
  });
}

// Minimal PNG decoder (stdlib only): 8-bit non-interlaced RGB/RGBA.
function decodePNG(buf) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (!buf.subarray(0, 8).equals(sig)) throw new Error('not a PNG');
  let pos = 8;
  let width = 0;
  let height = 0;
  let colorType = 0;
  let bitDepth = 0;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      const interlace = data[12];
      if (bitDepth !== 8 || (colorType !== 2 && colorType !== 6) || interlace !== 0)
        throw new Error(`unsupported PNG ct=${colorType} depth=${bitDepth} interlace=${interlace}`);
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    pos += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const ch = colorType === 6 ? 4 : 3;
  const stride = width * ch;
  const px = Buffer.alloc(width * height * 4);
  let p = 0;
  const prev = Buffer.alloc(stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[p++];
    const cur = Buffer.alloc(stride);
    raw.copy(cur, 0, p, p + stride);
    p += stride;
    for (let i = 0; i < stride; i++) {
      const a = i >= ch ? cur[i - ch] : 0;
      const b = prev[i];
      const c = i >= ch ? prev[i - ch] : 0;
      let v = cur[i];
      if (filter === 1) v = (v + a) & 0xff;
      else if (filter === 2) v = (v + b) & 0xff;
      else if (filter === 3) v = (v + ((a + b) >> 1)) & 0xff;
      else if (filter === 4) {
        const pp = a + b - c;
        const pa = Math.abs(pp - a);
        const pb = Math.abs(pp - b);
        const pc = Math.abs(pp - c);
        const pr = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        v = (v + pr) & 0xff;
      }
      cur[i] = v;
    }
    cur.copy(prev);
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      px[o] = cur[x * ch];
      px[o + 1] = cur[x * ch + 1];
      px[o + 2] = cur[x * ch + 2];
      px[o + 3] = ch === 4 ? cur[x * ch + 3] : 255;
    }
  }
  return { width, height, px };
}

function luma(r, g, b) {
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

function imageStats(img) {
  const n = img.width * img.height;
  let sum = 0;
  let sum2 = 0;
  let dark = 0;
  let hi = 0;
  const uniq = new Set();
  for (let i = 0; i < n; i++) {
    const L = luma(img.px[i * 4], img.px[i * 4 + 1], img.px[i * 4 + 2]);
    sum += L;
    sum2 += L * L;
    if (L < 0.06) dark++;
    if (L > 0.85) hi++;
    if ((i & 15) === 0)
      uniq.add((img.px[i * 4] << 16) | (img.px[i * 4 + 1] << 8) | img.px[i * 4 + 2]);
  }
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sum2 / n - mean * mean));
  return { width: img.width, height: img.height, mean, std, darkFrac: dark / n, hiFrac: hi / n, unique: uniq.size };
}

function compareCoarse(a, b) {
  if (a.width !== b.width || a.height !== b.height) return { sameSize: false };
  const n = a.width * a.height;
  const diffs = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(
      luma(a.px[i * 4], a.px[i * 4 + 1], a.px[i * 4 + 2]) -
        luma(b.px[i * 4], b.px[i * 4 + 1], b.px[i * 4 + 2]),
    );
    diffs[i] = d;
    sum += d;
  }
  const sorted = Array.from(diffs).sort((x, y) => x - y);
  return {
    sameSize: true,
    meanAbs: sum / n,
    p99Abs: sorted[Math.min(n - 1, Math.floor(n * 0.99))],
    maxAbs: sorted[n - 1],
  };
}

const chromeChildren = [];
function killChromeChildren(sig) {
  for (const child of chromeChildren) {
    try {
      process.kill(-child.pid, sig);
    } catch {
      /* already gone */
    }
  }
}

async function runLeg(browserWsUrl, httpPort, leg, outDir, timeoutMs) {
  const transcript = [];
  const problems = [];
  let setup = null;
  let verdict = null;
  let frame60 = null;
  let adapter = null;
  let shotBuf = null;
  let shotAfterFrame = null;
  let lastFrameLog = 0;
  const log = (line) => transcript.push(line);

  const pageUrl =
    `http://127.0.0.1:${httpPort}/hdr-showcase.html` +
    `?frames=${FRAMES}&msaa=${leg.msaa}${leg.srgb ? '&srgb=1' : ''}&_=${Date.now()}`;

  const ws = new WebSocket(browserWsUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error('cdp connect failed'));
  });

  let nextId = 1;
  const pending = new Map();
  const send = (method, params = {}, sessionId = undefined) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      const msg = { id, method, params };
      if (sessionId) msg.sessionId = sessionId;
      ws.send(JSON.stringify(msg));
    });

  // The dispatcher MUST be installed before the first send: every reply
  // arrives here, and awaiting createTarget with no handler hangs forever.
  let onEvent = null;
  ws.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    if (msg.id && pending.has(msg.id)) {
      const { resolve: r, reject: rej } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) rej(new Error(msg.error.message));
      else r(msg.result);
    }
    if (onEvent) {
      try {
        onEvent(msg);
      } catch (e) {
        problems.push(`dispatcher: ${e.message}`);
      }
    }
  };

  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  const sid = sessionId;

  // Screenshot trigger: first frame-60 console line -> clip the canvas ASAP.
  // No page pause (that would stall the finite gate); CDP round-trip lands
  // within a few frames of 60, recorded as shotAfterFrame.
  let shotP = null;
  const takeShot = async () => {
    const rectEval = await send(
      'Runtime.evaluate',
      {
        expression: `(()=>{const c=document.getElementById('canvas');if(!c)return null;const r=c.getBoundingClientRect();return {x:Math.max(0,Math.round(r.x)),y:Math.max(0,Math.round(r.y)),width:Math.round(r.width),height:Math.round(r.height)};})()`,
        returnByValue: true,
      },
      sid,
    );
    const rect = rectEval && rectEval.result && rectEval.result.value;
    const params = { format: 'png' };
    if (rect && rect.width >= 16 && rect.height >= 16)
      params.clip = { ...rect, scale: 1 };
    const { data } = await send('Page.captureScreenshot', params, sid);
    shotBuf = Buffer.from(data, 'base64');
    shotAfterFrame = lastFrameLog;
  };

  const done = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`leg ${leg.name}: timeout ${timeoutMs}ms waiting for verdict`)), timeoutMs);
    onEvent = (msg) => {
      if (msg.method === 'Runtime.consoleAPICalled' && msg.params) {
        const type = msg.params.type || 'log';
        const text = (msg.params.args || [])
          .map((a) => (a.value !== undefined ? String(a.value) : a.description || ''))
          .join(' ');
        log(`[console.${type}] ${text}`);
        const plain = stripAnsi(text);
        if (HARD_FAIL_RE.test(plain)) problems.push(`console: ${text.slice(0, 300)}`);
        else if (HDR_INFO_RE.test(plain)) { /* gate's own progress lines */ } else if (type === 'error' && !SHELL_WARN_RE.test(plain))
          problems.push(`console.error: ${text.slice(0, 300)}`);
        let m = plain.match(SETUP_RE);
        if (m)
          setup = { backend: m[1], msaa: +m[2], srgb: m[3], envFmt: m[4], swapFmt: m[5], line: text.trim() };
        m = plain.match(FRAME_RE);
        if (m) {
          lastFrameLog = +m[1];
          if (+m[1] === 60 && !shotP) {
            frame60 = { frame: +m[1], draws: +m[2], samples: +m[3], srgb: m[4], w: +m[5], h: +m[6] };
            shotP = takeShot().catch((e) => problems.push(`screenshot: ${e.message}`));
          }
        }
        m = plain.match(VERDICT_RE);
        if (m) {
          verdict = {
            frames: +m[1],
            checks: +m[2],
            failures: +m[3],
            logErrors: +m[4],
            allocs: +m[5],
            verdict: m[6],
            line: text.trim(),
          };
          clearTimeout(timer);
          resolve();
        }
      } else if (msg.method === 'Runtime.exceptionThrown') {
        const d = msg.params.exceptionDetails || {};
        const t = (d.exception && d.exception.description) || d.text || 'unknown exception';
        problems.push(`exception: ${String(t).slice(0, 300)}`);
        log(`[exception] ${t}`);
      } else if (msg.method === 'Log.entryAdded') {
        const e = msg.params.entry || {};
        log(`[log.${e.level}] [${e.source}] ${e.url || ''} ${e.text}`);
        if (e.level === 'error') problems.push(`log.error: ${String(e.text || '').slice(0, 300)}`);
      }
    };
  });

  try {
    await send('Runtime.enable', {}, sid);
    await send('Page.enable', {}, sid);
    await send('Log.enable', {}, sid);
    await send('Network.enable', {}, sid);
    await send('Network.setCacheDisabled', { cacheDisabled: true }, sid);
    await send(
      'Emulation.setDeviceMetricsOverride',
      { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false },
      sid,
    );
    // Real-GPU probe: a second adapter request purely to read its info.
    // Fails the leg on fallback / SwiftShader / llvmpipe / SwANGLE strings.
    await send('Page.navigate', { url: pageUrl }, sid);
    await done;
    if (shotP) await Promise.race([shotP, wait(15000)]);
    try {
      const r = await send(
        'Runtime.evaluate',
        {
          expression: `(async()=>{try{if(!navigator.gpu)return{ok:false,reason:'no-navigator.gpu'};const a=await navigator.gpu.requestAdapter();if(!a)return{ok:false,reason:'requestAdapter-null'};let i={};try{i=(a.requestAdapterInfo?await a.requestAdapterInfo():(a.info||{}))||{};}catch(e){i={infoError:String(e)};}return{ok:true,info:{vendor:i.vendor||null,architecture:i.architecture||null,device:i.device||null,description:i.description||null,backend:i.backend||null,type:i.type||null,isFallbackAdapter:!!i.isFallbackAdapter}};}catch(e){return{ok:false,reason:String(e)};}})()`,
          awaitPromise: true,
          returnByValue: true,
        },
        sid,
      );
      adapter = r && r.result && r.result.value;
    } catch (e) {
      problems.push(`adapter probe: ${e.message}`);
    }
  } finally {
    try {
      await send('Target.closeTarget', { targetId });
    } catch {
      /* gone */
    }
    try {
      ws.close();
    } catch {
      /* gone */
    }
  }

  // ---- verdicts ----
  const failures = [...problems];
  if (!setup) failures.push('missing setup line (no valid context? allocation failure?)');
  else if (setup.backend !== 'WGPU') failures.push(`backend is ${setup.backend}, need WGPU (real GPU gate)`);
  if (!verdict) failures.push('missing verdict line (finite gate never completed)');
  else {
    if (verdict.verdict !== 'PASS') failures.push(`verdict=${verdict.verdict}`);
    if (verdict.failures !== 0) failures.push(`failures=${verdict.failures}`);
    if (verdict.logErrors !== 0) failures.push(`log_errors=${verdict.logErrors}`);
    if (verdict.allocs !== 0) failures.push(`sokol_live_allocations=${verdict.allocs}`);
    if (verdict.frames !== FRAMES) failures.push(`frames=${verdict.frames}, need ${FRAMES}`);
    if (verdict.checks < leg.minChecks) failures.push(`checks=${verdict.checks} < ${leg.minChecks} (required coverage)`);
    if (verdict.checks === 0) failures.push('checks==0 (allocation or context unavailable?)');
  }
  if (!frame60) failures.push('missing frame-60 line (screenshot trigger lost)');
  else if (frame60.samples !== leg.expectSamples)
    failures.push(`effective samples=${frame60.samples}, policy needs ${leg.expectSamples}`);
  if (!shotBuf) failures.push('no screenshot captured');
  const adapterText = JSON.stringify(adapter || {});
  if (!adapter || !adapter.ok) failures.push(`no WebGPU adapter (${adapterText.slice(0, 200)})`);
  else if (adapter.info && (adapter.info.isFallbackAdapter || /swiftshader|swangle|llvmpipe|software/i.test(adapterText)))
    failures.push(`fallback/software adapter only (${adapterText.slice(0, 200)})`);

  let stats = null;
  if (shotBuf) {
    const img = decodePNG(shotBuf);
    stats = imageStats(img);
    if (stats.unique < VIS.minUnique) failures.push(`flat frame: unique=${stats.unique}`);
    if (stats.std < VIS.minStd) failures.push(`empty frame: std=${stats.std.toFixed(4)}`);
    if (stats.darkFrac < VIS.minDarkFrac) failures.push(`not a dark studio: darkFrac=${stats.darkFrac.toFixed(3)}`);
    if (stats.hiFrac < VIS.minHiFrac) failures.push(`no HDR highlights: hiFrac=${stats.hiFrac.toFixed(4)}`);
    writeFileSync(join(outDir, `${leg.name}.png`), shotBuf);
  }
  writeFileSync(join(outDir, `${leg.name}.log`), transcript.join('\n') + '\n');
  const result = {
    leg: leg.name,
    query: `?frames=${FRAMES}&msaa=${leg.msaa}${leg.srgb ? '&srgb=1' : ''}`,
    ok: failures.length === 0,
    failures,
    setup,
    verdict,
    frame60,
    shotAfterFrame,
    adapter,
    stats,
    consoleProblems: problems,
  };
  writeFileSync(join(outDir, `${leg.name}.json`), JSON.stringify(result, null, 2));
  return result;
}

const FRAMES = parseInt(arg('--frames', '240'), 10);

async function main() {
  const serveDir = arg('--serve-dir', join(HERE, '..', 'zig-out', 'web'));
  const chromePath = arg('--chrome', CHROME_DEFAULT);
  const timeoutMs = parseInt(arg('--timeout-ms', '150000'), 10);
  if (!Number.isInteger(FRAMES) || FRAMES < 240) throw new Error('--frames must be at least 240 for complete phase coverage');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('--timeout-ms must be positive');
  const scopedTmp = join(tmpdir(), 'opencode');
  const baseTmp = existsSync(scopedTmp) ? scopedTmp : tmpdir();
  const outDir = arg('--out', null) ?? mkdtempSync(join(baseTmp, 'hdr-browser-'));
  mkdirSync(outDir, { recursive: true });

  const server = await startStaticServer(serveDir);
  const httpPort = server.address().port;
  const cdpPort = await freePort();
  const profileDir = mkdtempSync(join(baseTmp, 'hdr-gate-profile-'));
  console.log(`hdr-browser-gate: serve=${serveDir} http=${httpPort} cdp=${cdpPort} out=${outDir}`);

  // Chrome's own stderr (CVDisplayLink headless noise, updater chatter) must
  // not pollute the gate log: ignore it. Page diagnostics arrive via CDP.
  const child = spawn(
    chromePath,
    [
      '--headless=new',
      `--remote-debugging-port=${cdpPort}`,
      '--enable-webgpu',
      '--use-angle=metal',
      '--window-size=1280,720',
      `--user-data-dir=${profileDir}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-background-networking',
      'about:blank',
    ],
    { stdio: 'ignore', detached: true },
  );
  chromeChildren.push(child);

  const killer = (sig) => killChromeChildren(sig);
  const onSignal = (s) => () => {
    killer(s === 'SIGTERM' ? 'SIGTERM' : 'SIGKILL');
    process.exit(1);
  };
  process.on('SIGINT', onSignal('SIGINT'));
  process.on('SIGTERM', onSignal('SIGTERM'));
  process.on('unhandledRejection', (e) => {
    console.error('[fatal] unhandledRejection:', (e && e.stack) || e);
    killer('SIGKILL');
    process.exit(1);
  });

  let browserWsUrl = null;
  try {
    for (let i = 0; i < 30; i++) {
      await wait(500);
      try {
        const res = await fetch(`http://127.0.0.1:${cdpPort}/json/version`);
        if (res.ok) {
          browserWsUrl = (await res.json()).webSocketDebuggerUrl;
          break;
        }
      } catch {
        /* not up yet */
      }
    }
    if (!browserWsUrl) throw new Error(`no Chrome on CDP port ${cdpPort}`);

    const results = [];
    for (const leg of LEGS) {
      console.log(`--- leg ${leg.name} (${leg.msaa}x srgb=${leg.srgb}) ---`);
      const r = await runLeg(browserWsUrl, httpPort, leg, outDir, timeoutMs);
      console.log(
        `${r.ok ? 'PASS' : 'FAIL'} ${leg.name}: ` +
          `backend=${r.setup && r.setup.backend} frames=${r.verdict && r.verdict.frames} ` +
          `checks=${r.verdict && r.verdict.checks} samples=${r.frame60 && r.frame60.samples} ` +
          `mean=${r.stats && r.stats.mean.toFixed(3)} dark=${r.stats && r.stats.darkFrac.toFixed(3)} ` +
          `hi=${r.stats && r.stats.hiFrac.toFixed(4)}` +
          (r.failures.length ? ` :: ${r.failures.join('; ')}` : ''),
      );
      results.push(r);
    }

    // Coarse UNORM-vs-sRGB agreement (same geometry, same fixed-dt frame).
    let xencode = null;
    try {
      const a = decodePNG(await readFile(join(outDir, 'msaa1-unorm.png')));
      const b = decodePNG(await readFile(join(outDir, 'msaa1-srgb.png')));
      xencode = compareCoarse(a, b);
      if (!xencode.sameSize || xencode.meanAbs > XENCODE.maxMeanAbs || xencode.p99Abs > XENCODE.maxP99Abs) {
        const r = results.find((x) => x.leg === 'msaa1-srgb');
        r.ok = false;
        r.failures.push(`xencode mismatch: ${JSON.stringify(xencode)}`);
      }
    } catch (e) {
      xencode = { error: e.message };
      for (const r of results) {
        r.ok = false;
        r.failures.push(`xencode compare failed: ${e.message}`);
      }
    }
    console.log(
      `xencode unorm-vs-srgb: ` +
        (xencode && xencode.sameSize
          ? `meanAbs=${xencode.meanAbs.toFixed(4)} p99Abs=${xencode.p99Abs.toFixed(4)} maxAbs=${xencode.maxAbs.toFixed(4)}`
          : JSON.stringify(xencode)),
    );

    const summary = {
      ok: results.every((r) => r.ok),
      frames: FRAMES,
      legs: results.map((r) => ({
        leg: r.leg,
        ok: r.ok,
        checks: r.verdict && r.verdict.checks,
        samples: r.frame60 && r.frame60.samples,
        setup: r.setup && r.setup.line,
        stats: r.stats,
        failures: r.failures,
      })),
      xencode,
    };
    writeFileSync(join(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
    console.log(`summary: ${summary.ok ? 'PASS' : 'FAIL'} -> ${join(outDir, 'summary.json')}`);
    process.exitCode = summary.ok ? 0 : 1;
  } catch (e) {
    console.error('[fatal]', (e && e.stack) || e);
    process.exitCode = 1;
  } finally {
    try {
      if (browserWsUrl) {
        const bws = new WebSocket(browserWsUrl);
        await new Promise((resolve) => {
          bws.onopen = resolve;
          setTimeout(resolve, 2000);
        });
        try {
          bws.send(JSON.stringify({ id: 1, method: 'Browser.close' }));
        } catch {
          /* fall through */
        }
        await wait(1500);
        try {
          bws.close();
        } catch {
          /* gone */
        }
      }
    } catch {
      /* fall through to signals */
    }
    killer('SIGTERM');
    await wait(1500);
    killer('SIGKILL');
    await new Promise((resolve) => server.close(resolve));
    try {
      rmSync(profileDir, { recursive: true, force: true });
    } catch {
      /* best effort */
    }
  }
}

main();
