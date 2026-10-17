#!/usr/bin/env node
// TAA pixel-quality temporal regression gate (graphics-roadmap C.3 slice).
//
// Runs the ALREADY-BUILT emscripten page (examples/zig-out/web/taa-quality.html
// or --serve-dir) in real headless Chrome on Metal/WebGPU across 10 legs.
// Test/reference pairs share the SAME encoded frame (same trajectory frame,
// fixed dt=1/60); cut/reset legs capture AT the cut/reset frame with a matched
// reference carrying the identical actual camera change:
//
//   ref-a      taa=0, f=150           instantaneous reference, scale 1
//   ref-dup    taa=0, f=150           duplicate reference (noise floor)
//   taa        TAA + reactive, f=150  warm-history temporal resolve, scale 1
//   noreact    TAA, reactive=0, f=150 NEGATIVE CONTROL (must be worse)
//   ref067     taa=0, f=150           instantaneous reference, scale 0.67
//   taa067     TAA + reactive, f=150  temporal resolve at render scale 0.67
//   cutcap     TAA + cut=120, f=120   capture AT the camera-cut frame
//   refcut     taa=0 + cut=120, f=120 matched cut reference (same camera snap)
//   resetcap   TAA + reset=130, f=130 capture AT the explicit-reset frame
//   refres     taa=0 + reset=130      matched reset reference
//
// Behind --classes (default OFF, base runs unchanged) 4 additive legs cover
// the knob-gated content classes (fixture skin=1 / inst=1, same TAAQ caps):
//
//   skinref    taa=0 + skin=1, f=150  matched skinned reference
//   skintaa    TAA + skin=1, f=150    skinned temporal resolve
//   instref    taa=0 + inst=1, f=150  matched instanced reference
//   insttaa    TAA + inst=1, f=150    instanced temporal resolve
//
// Every leg runs with hold=1: after the Nth frame the page issues NO update,
// prepare, render or commit — the swapchain keeps presenting the Nth encoded
// frame, so the captures are that actual motion frame (never a re-rendered
// stationary tail). The gate waits for "taa-quality: held frame=N", captures
// TWICE (byte-exact equality required), then sets window.tacaqFinish=1: the
// page quits itself and runs the normal cleanup, and the gate requires
// verdict=PASS with log_errors=0 and zero live allocations AFTER capture
// before closing the target (no leak-proof claims from a killed target).
//
// Metrics (screenshots only, 8-bit sRGB canvas): ghost residual of the test
// leg vs the exact same-frame instantaneous reference + an edge-sharpness
// (contrast-blur proxy) ratio, measured inside the sweep ROI (center band
// holding the full hero/glass travel; static borders excluded so they cannot
// dilute the ghost signal). Reference screenshots are real GPU renders of
// the real engine (scene.render), not CPU fakes.
//
// Verdicts: PASS / FAIL / INCONCLUSIVE. INCONCLUSIVE covers zero signal,
// size mismatch, reference drift, a silent negative control, or a
// non-identical held-capture pair. Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
//
// Usage:
//   node taa_browser_gate.mjs [--out <dir>] [--serve-dir <dir>]
//     [--chrome <path>] [--timeout-ms <n>] [--classes]
//
// --classes runs the 4 additive skinned/instanced legs (14 legs total);
// without it the gate is the base 10 legs, bit-identical to before.
//
// Writes NOTHING into the repo. Chrome profile + artifacts live under --out.

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import {
  writeFileSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, extname, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { inflateSync } from "node:zlib";
import net from "node:net";

const HERE = dirname(fileURLToPath(import.meta.url));
const CHROME_DEFAULT =
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PAGE = "taa-quality.html";

// Fixture-justified pixel thresholds (8-bit sRGB canvas luma, see metrics):
// the checker edge spans ~0.85 luma; the hero crosses center at ~8 px/frame,
// so one unshed history frame adds ~0.05+ edge residual while warm-history
// jitter shimmer sits ~0.01-0.02. PASS_CAP=0.06 separates clean resolves
// from ghosting without tuning to a particular run; the negative control
// must clear the resolve by SENS_MARGIN to prove the gate sees anything.
// Calibration (13.10.2026, WebGPU/Metal Chrome, 14-leg run, frozen):
//   max clean ROI residual 0.2463 (taa), min sharpness 0.604 (control),
//   max flat p95 0.0277 (taa067), control delta noreact-taa = 0.0015.
// Caps carry ~20-25% regression headroom over that baseline; they are a
// REGRESSION TRIPWIRE on the current resolve behavior (taa.blend 0.9 at
// ~8 px/frame hero sweep — absolute residuals are expected to be large),
// not an aesthetic verdict. The negative-control margin sits below the
// measured 0.0015 delta and above the ref/refdup noise floor (0).
export const TAAQ = {
  passEdgeMean: 0.3, // taa-on ROI ghost residual cap
  passFlatP95: 0.04, // ROI flat-region residual cap (no global smear)
  sharpLo: 0.55, // edge-sharpness ratio floor (no blur-out)
  sharpHi: 1.2, // edge-sharpness ratio cap (no ringing)
  sensMargin: 0.001, // reactiveOFF must exceed taa-on by this (sensitivity)
  noiseEdgeMean: 0.006, // ref-vs-refdup cap (determinism floor)
  edgeGrad: 0.15, // reference gradient threshold for the edge mask
  ghostLuma: 0.05, // |diff| floor counted as ghosted (for reporting)
  minUnique: 32, // distinct colors in a 1/8 stride sample (blank guard)
};

// Sweep ROI (relative coords): the hero travels x in [-1.1, 1.1] world with
// half-width 0.5 and the glass counter-sweeps across it; the camera is fixed,
// so all disocclusion lives in the center band. Static checker borders and
// the floor (which would dilute a global mean toward zero and fake a PASS)
// are excluded. Thresholds still key off the checker edge span (~0.85 luma),
// not off any measured run.
export const ROI = { x0: 0.2, y0: 0.15, x1: 0.8, y1: 0.85 };

// Expected realized render target for a leg scale at the 960x600 gate
// window: mirrors options.scaledRenderSize (round, clamp >= 1). The gate
// requires this EXACT target in the frame1/capture logs — a realized size,
// not merely the requested query knob.
export function expectedTarget(scale) {
  return {
    w: Math.max(1, Math.round(960 * scale)),
    h: Math.max(1, Math.round(600 * scale)),
  };
}

export function legTable(withClasses = false) {
  const H = (frames) => `frames=${frames}&hold=1`;
  const base = [
    {
      name: "ref-a",
      query: `${H(150)}&taa=0`,
      scale: 1.0,
      frames: 150,
      taa: false,
      expectMarkers: false,
      role: "ref",
    },
    {
      name: "ref-dup",
      query: `${H(150)}&taa=0`,
      scale: 1.0,
      frames: 150,
      taa: false,
      expectMarkers: false,
      role: "refdup",
    },
    {
      name: "taa",
      query: H(150),
      scale: 1.0,
      frames: 150,
      taa: true,
      expectMarkers: true,
      role: "test",
    },
    {
      name: "noreact",
      query: `${H(150)}&reactive=0`,
      scale: 1.0,
      frames: 150,
      taa: true,
      expectMarkers: false,
      role: "control",
    },
    {
      name: "ref067",
      query: `${H(150)}&taa=0&rscale=0.67`,
      scale: 0.67,
      frames: 150,
      taa: false,
      expectMarkers: false,
      role: "ref",
    },
    {
      name: "taa067",
      query: `${H(150)}&rscale=0.67`,
      scale: 0.67,
      frames: 150,
      taa: true,
      expectMarkers: true,
      role: "test",
    },
    {
      name: "cutcap",
      query: `${H(120)}&cut=120`,
      scale: 1.0,
      frames: 120,
      cut: 120,
      taa: true,
      expectMarkers: true,
      role: "test",
    },
    {
      name: "refcut",
      query: `${H(120)}&taa=0&cut=120`,
      scale: 1.0,
      frames: 120,
      cut: 120,
      taa: false,
      expectMarkers: false,
      role: "ref",
    },
    {
      name: "resetcap",
      query: `${H(130)}&reset=130`,
      scale: 1.0,
      frames: 130,
      reset: 130,
      taa: true,
      expectMarkers: true,
      role: "test",
    },
    {
      name: "refreset",
      query: `${H(130)}&taa=0&reset=130`,
      scale: 1.0,
      frames: 130,
      reset: 130,
      taa: false,
      expectMarkers: false,
      role: "ref",
    },
  ];
  if (!withClasses) return base;
  // Additive class legs: matched same-frame test/ref pairs at scale 1,
  // warm history 150 (same as the base resolve legs).
  return [
    ...base,
    {
      name: "skinref",
      query: `${H(150)}&taa=0&skin=1`,
      scale: 1.0,
      frames: 150,
      taa: false,
      expectMarkers: false,
      role: "ref",
    },
    {
      name: "skintaa",
      query: `${H(150)}&skin=1`,
      scale: 1.0,
      frames: 150,
      taa: true,
      expectMarkers: true,
      role: "test",
    },
    {
      name: "instref",
      query: `${H(150)}&taa=0&inst=1`,
      scale: 1.0,
      frames: 150,
      taa: false,
      expectMarkers: false,
      role: "ref",
    },
    {
      name: "insttaa",
      query: `${H(150)}&inst=1`,
      scale: 1.0,
      frames: 150,
      taa: true,
      expectMarkers: true,
      role: "test",
    },
    {
      name: "partref",
      query: `${H(150)}&taa=0&part=1`,
      scale: 1.0,
      frames: 150,
      taa: false,
      expectMarkers: false,
      role: "ref",
    },
    {
      name: "parttaa",
      query: `${H(150)}&part=1`,
      scale: 1.0,
      frames: 150,
      taa: true,
      expectMarkers: true,
      role: "test",
    },
  ];
}

// Class test/ref pairs covered behind --classes (same TAAQ caps as base).
export const CLASS_PAIRS = [
  ["skintaa", "skinref"],
  ["insttaa", "instref"],
  ["parttaa", "partref"],
];

function arg(name, fallback) {
  const hit = process.argv.find((a) => a.startsWith(name + "="));
  if (hit) return hit.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? (process.argv[index + 1] ?? fallback) : fallback;
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.on("error", reject);
    s.listen(0, "127.0.0.1", () => {
      const p = s.address().port;
      s.close(() => resolve(p));
    });
  });
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".json": "application/json",
  ".data": "application/octet-stream",
};

function startStaticServer(root) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, "http://x");
      let path = decodeURIComponent(url.pathname);
      if (path === "/") path = `/${PAGE}`;
      if (path === "/favicon.ico") {
        res.writeHead(204);
        res.end();
        return;
      }
      if (path.includes("..")) {
        res.writeHead(400);
        res.end("bad path");
        return;
      }
      const body = await readFile(join(root, path));
      res.writeHead(200, {
        "Content-Type": MIME[extname(path)] || "application/octet-stream",
      });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end("not found");
    }
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server));
  });
}

// Minimal PNG decoder (stdlib only): 8-bit non-interlaced RGB/RGBA.
export function decodePNG(buf) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (!buf.subarray(0, 8).equals(sig)) throw new Error("not a PNG");
  let pos = 8;
  let width = 0;
  let height = 0;
  let colorType = 0;
  let bitDepth = 0;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString("ascii", pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      const interlace = data[12];
      if (
        bitDepth !== 8 ||
        (colorType !== 2 && colorType !== 6) ||
        interlace !== 0
      )
        throw new Error(
          `unsupported PNG ct=${colorType} depth=${bitDepth} interlace=${interlace}`,
        );
    } else if (type === "IDAT") {
      idat.push(data);
    } else if (type === "IEND") {
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

export function lumaAt(img, x, y) {
  const o = (y * img.width + x) * 4;
  return (
    (0.2126 * img.px[o] + 0.7152 * img.px[o + 1] + 0.0722 * img.px[o + 2]) / 255
  );
}

// Ghost residual of test vs same-frame reference + edge-sharpness ratio,
// measured inside `roi` (default: whole frame). Edge mask comes from the
// REFERENCE gradient (central differences), so the mask cannot be gamed by
// blurring the test image. Returns null on size mismatch (caller reports
// INCONCLUSIVE, never a silent pass). changedPixels is ROI-scoped: a frozen
// frame reports zero signal even if borders differ.
export function ghostMetrics(ref, test, roi = null) {
  if (ref.width !== test.width || ref.height !== test.height) return null;
  const w = ref.width;
  const h = ref.height;
  const r = roi || { x0: 0, y0: 0, x1: 1, y1: 1 };
  const x0 = Math.max(0, Math.floor(r.x0 * w));
  const x1 = Math.min(w, Math.ceil(r.x1 * w));
  const y0 = Math.max(0, Math.floor(r.y0 * h));
  const y1 = Math.min(h, Math.ceil(r.y1 * h));
  let edgeSum = 0;
  let edgeN = 0;
  let edgeGradRef = 0;
  let edgeGradTest = 0;
  let ghostN = 0;
  // Global guard: mean |diff| over reference edge pixels of the FULL frame,
  // reported alongside (never a verdict input — the ROI carries that).
  let gEdgeSum = 0;
  let gEdgeN = 0;
  const flatDiffs = [];
  let changed = 0;
  const grad = (img, x, y) => {
    const xm = Math.max(0, x - 1);
    const xp = Math.min(w - 1, x + 1);
    const ym = Math.max(0, y - 1);
    const yp = Math.min(h - 1, y + 1);
    return Math.hypot(
      (lumaAt(img, xp, y) - lumaAt(img, xm, y)) / (xp - xm),
      (lumaAt(img, x, yp) - lumaAt(img, x, ym)) / (yp - ym),
    );
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const o = i * 4;
      const d = Math.abs(lumaAt(ref, x, y) - lumaAt(test, x, y));
      const g = grad(ref, x, y);
      if (g >= TAAQ.edgeGrad) {
        gEdgeSum += d;
        gEdgeN++;
      }
      if (x < x0 || x >= x1 || y < y0 || y >= y1) continue;
      let pxChanged = false;
      for (let c = 0; c < 4; c++)
        if (ref.px[o + c] !== test.px[o + c]) pxChanged = true;
      if (pxChanged) changed++;
      if (g >= TAAQ.edgeGrad) {
        edgeSum += d;
        edgeN++;
        edgeGradRef += g;
        edgeGradTest += grad(test, x, y);
        if (d >= TAAQ.ghostLuma) ghostN++;
      } else {
        flatDiffs.push(d);
      }
    }
  }
  flatDiffs.sort((a, b) => a - b);
  const flatP95 = flatDiffs.length
    ? flatDiffs[
        Math.min(flatDiffs.length - 1, Math.floor(flatDiffs.length * 0.95))
      ]
    : 0;
  return {
    edgeMean: edgeN ? edgeSum / edgeN : 0,
    edgeFrac: edgeN / Math.max(1, (x1 - x0) * (y1 - y0)),
    ghostFrac: edgeN ? ghostN / edgeN : 0,
    flatP95,
    sharpRatio: edgeGradRef > 0 ? edgeGradTest / edgeGradRef : 1,
    changedPixels: changed,
    globalEdgeMean: gEdgeN ? gEdgeSum / gEdgeN : 0,
    roi: { ...r },
  };
}

const ANSI_RE = /\x1b\[[0-9;]*m/g;
const stripAnsi = (s) => s.replace(ANSI_RE, "");
const HARD_FAIL_RE =
  /GPU(?:Validation|Internal|OutOfMemory)Error|uncaptured(?: GPU)? error|\[(?:sg|sokol|sapp)[^\]]*\]\s*\[(?:error|panic)\]|VALIDATE_[A-Z_]+|validation (?:failed|error)|\bpanic\b|\bfatal\b|^taa-quality FAIL|taa-quality: verdict=FAIL/i;
const SHELL_WARN_RE = /^\s*warning:/;
// Info lines are the fixture's own progress proof — narrowed so a FAIL
// verdict is NEVER informational (see classifyDiagnostic tests).
const TAAQ_INFO_RE =
  /^\s*taa-quality: (setup backend=|frame1 draws=|activity frame=|capture frame=|held frame=|held tick=|finish requested|teardown log_errors=|verdict=PASS)/;

export function classifyDiagnostic(text) {
  const plain = stripAnsi(text);
  if (HARD_FAIL_RE.test(plain)) return "hard-fail";
  if (TAAQ_INFO_RE.test(plain)) return "info";
  return "other";
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
  let frame1 = null;
  let activity = null;
  let capture = null;
  let heldFrame = 0;
  let verdict = null;
  let teardown = null;
  let adapter = null;
  let shotBuf = null;
  let shotDup = null;
  const log = (line) => transcript.push(line);

  const pageUrl = `http://127.0.0.1:${httpPort}/${PAGE}?${leg.query}&_=${Date.now()}`;
  const ws = new WebSocket(browserWsUrl);
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = () => reject(new Error("cdp connect failed"));
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

  const { targetId } = await send("Target.createTarget", {
    url: "about:blank",
  });
  const { sessionId } = await send("Target.attachToTarget", {
    targetId,
    flatten: true,
  });
  const sid = sessionId;

  const done = new Promise((resolve, reject) => {
    const timer = setTimeout(
      () =>
        reject(
          new Error(
            `leg ${leg.name}: timeout ${timeoutMs}ms waiting for held frame`,
          ),
        ),
      timeoutMs,
    );
    onEvent = (msg) => {
      if (msg.method === "Runtime.consoleAPICalled" && msg.params) {
        const type = msg.params.type || "log";
        const text = (msg.params.args || [])
          .map((a) =>
            a.value !== undefined ? String(a.value) : a.description || "",
          )
          .join(" ");
        log(`[console.${type}] ${text}`);
        const plain = stripAnsi(text);
        const cls = classifyDiagnostic(text);
        if (cls === "hard-fail")
          problems.push(`console: ${text.slice(0, 300)}`);
        else if (cls === "info") {
          /* fixture's own progress lines */
        } else if (type === "error" && !SHELL_WARN_RE.test(plain))
          problems.push(`console.error: ${text.slice(0, 300)}`);
        let m = plain.match(
          /taa-quality: setup backend=(\S+) taa=(\S+) reactive=(\S+) scale=([\d.]+) frames=(\d+) cut=(\d+) reset=(\d+)/,
        );
        if (m)
          setup = {
            backend: m[1],
            taa: m[2],
            reactive: m[3],
            scale: +m[4],
            frames: +m[5],
            cut: +m[6],
            reset: +m[7],
          };
        m = plain.match(
          /taa-quality: frame1 draws=(\d+) main_draws=(\d+) post_draws=(\d+) target=(\d+)x(\d+)/,
        );
        if (m)
          frame1 = {
            draws: +m[1],
            mainDraws: +m[2],
            postDraws: +m[3],
            targetW: +m[4],
            targetH: +m[5],
          };
        m = plain.match(
          /taa-quality: activity frame=30 draws=(\d+) main_draws=(\d+) post_draws=(\d+) velocity_sized=(\S+) markers_active=(\S+) hero_x=([-\d.]+) glass_x=([-\d.]+)/,
        );
        if (m)
          activity = {
            draws: +m[1],
            mainDraws: +m[2],
            postDraws: +m[3],
            velocitySized: m[4],
            markers: m[5],
            heroX: +m[6],
            glassX: +m[7],
          };
        m = plain.match(
          /taa-quality: capture frame=(\d+) draws=(\d+) main_draws=(\d+) post_draws=(\d+) markers_active=(\S+) taa_frame=(\d+) reset_pending=(\S+) cut_applied=(\S+) target=(\d+)x(\d+) hero_x=([-\d.]+) glass_x=([-\d.]+)/,
        );
        if (m)
          capture = {
            frame: +m[1],
            draws: +m[2],
            mainDraws: +m[3],
            postDraws: +m[4],
            markers: m[5],
            taaFrame: +m[6],
            resetPending: m[7],
            cutApplied: m[8],
            targetW: +m[9],
            targetH: +m[10],
            heroX: +m[11],
            glassX: +m[12],
          };
        m = plain.match(/taa-quality: held frame=(\d+)/);
        if (m) {
          heldFrame = +m[1];
          clearTimeout(timer);
          resolve();
        }
        m = plain.match(
          /taa-quality: teardown log_errors=(\d+) sokol_live_allocations=(\d+)/,
        );
        if (m) teardown = { logErrors: +m[1], allocs: +m[2] };
        m = plain.match(
          /taa-quality: verdict=(\w+) failures=(\d+) frames=(\d+)/,
        );
        if (m) verdict = { verdict: m[1], failures: +m[2], frames: +m[3] };
      } else if (msg.method === "Runtime.exceptionThrown") {
        const d = msg.params.exceptionDetails || {};
        const t =
          (d.exception && d.exception.description) ||
          d.text ||
          "unknown exception";
        problems.push(`exception: ${String(t).slice(0, 300)}`);
        log(`[exception] ${t}`);
      } else if (msg.method === "Log.entryAdded") {
        const e = msg.params.entry || {};
        log(`[log.${e.level}] [${e.source}] ${e.url || ""} ${e.text}`);
        if (e.level === "error")
          problems.push(`log.error: ${String(e.text || "").slice(0, 300)}`);
      }
    };
  });

  try {
    await send("Runtime.enable", {}, sid);
    await send("Page.enable", {}, sid);
    await send("Log.enable", {}, sid);
    await send("Network.enable", {}, sid);
    await send("Network.setCacheDisabled", { cacheDisabled: true }, sid);
    await send(
      "Emulation.setDeviceMetricsOverride",
      { width: 960, height: 600, deviceScaleFactor: 1, mobile: false },
      sid,
    );
    await send("Page.navigate", { url: pageUrl }, sid);
    await done;
    // Frozen frame: the page issues no render/commit anymore, so both
    // captures below must be byte-exact. Two captures separated in time
    // prove the hold (a re-rendering tail would advance TAA history and the
    // bytes would drift).
    const captureCanvas = async () => {
      const rectEval = await send(
        "Runtime.evaluate",
        {
          expression: `(()=>{const c=document.getElementById('canvas');if(!c)return null;const r=c.getBoundingClientRect();return {x:Math.max(0,Math.round(r.x)),y:Math.max(0,Math.round(r.y)),width:Math.round(r.width),height:Math.round(r.height)};})()`,
          returnByValue: true,
        },
        sid,
      );
      const rect = rectEval && rectEval.result && rectEval.result.value;
      const shotParams = { format: "png" };
      if (rect && rect.width >= 16 && rect.height >= 16)
        shotParams.clip = { ...rect, scale: 1 };
      const { data } = await send("Page.captureScreenshot", shotParams, sid);
      return Buffer.from(data, "base64");
    };
    shotBuf = await captureCanvas();
    await wait(750);
    shotDup = await captureCanvas();
    if (!shotBuf.equals(shotDup))
      problems.push(
        `held captures differ (${shotBuf.length}B vs ${shotDup.length}B — frame not frozen, history advanced during hold?)`,
      );
    // Controlled finish: the page quits itself and runs the normal cleanup,
    // so teardown/leak proof below is real (never claimed from a killed target).
    await send(
      "Runtime.evaluate",
      { expression: "window.tacaqFinish=1", returnByValue: true },
      sid,
    );
    const t0 = Date.now();
    while (!verdict) {
      if (Date.now() - t0 > 30000)
        throw new Error(
          `leg ${leg.name}: timeout waiting for verdict after finish`,
        );
      await wait(250);
    }
    try {
      const r = await send(
        "Runtime.evaluate",
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
      await send("Target.closeTarget", { targetId });
    } catch {
      /* gone */
    }
    try {
      ws.close();
    } catch {
      /* gone */
    }
  }

  const failures = [...problems];
  const expT = expectedTarget(leg.scale);
  if (!setup) failures.push("missing setup line");
  else {
    if (setup.backend !== "WGPU")
      failures.push(`backend is ${setup.backend}, need WGPU (real GPU gate)`);
    if (Math.abs(setup.scale - leg.scale) > 0.005)
      failures.push(
        `setup scale=${setup.scale} != leg scale=${leg.scale} (rscale knob lost?)`,
      );
    if (setup.frames !== leg.frames)
      failures.push(`setup frames=${setup.frames} != leg frames=${leg.frames}`);
    if (setup.cut !== (leg.cut || 0) || setup.reset !== (leg.reset || 0))
      failures.push(
        `setup cut/reset ${setup.cut}/${setup.reset} != leg (event mismatch?)`,
      );
  }
  const checkTarget = (label, t) => {
    if (!t) {
      failures.push(`missing ${label} target proof`);
      return;
    }
    if (t.targetW !== expT.w || t.targetH !== expT.h)
      failures.push(
        `${label} target=${t.targetW}x${t.targetH}, need realized ${expT.w}x${expT.h} for scale=${leg.scale} (not merely the query knob)`,
      );
  };
  checkTarget("frame1", frame1);
  if (!activity) failures.push("missing frame-30 activity proof");
  else {
    if (activity.draws === 0)
      failures.push("zero draws at frame 30 (fixture not rendering)");
    // Velocity activity is required ONLY on TAA legs: TAA-off references
    // have no motion-vector consumer (blur-only path), so an unsized
    // velocity target there is correct, not a failure.
    if (leg.taa && activity.velocitySized !== "true")
      failures.push(
        "velocity target unsized on a TAA leg (no motion vectors, gate is blind)",
      );
    // Requested-flag echo (the ACTUAL marker execution proof is the
    // main-draw delta taa-vs-noreact in decideGate, not this flag).
    if (activity.markers !== String(leg.expectMarkers))
      failures.push(
        `markers_active=${activity.markers}, need ${leg.expectMarkers} on ${leg.name}`,
      );
  }
  if (!capture) failures.push("missing capture-frame proof");
  else {
    if (capture.frame !== leg.frames)
      failures.push(
        `capture frame=${capture.frame}, need ${leg.frames} (same encoded frame)`,
      );
    if (capture.draws === 0 || capture.mainDraws === 0)
      failures.push("zero draws at capture (empty frame)");
    // History-reject frame parity: every TAA-active composite advances the
    // ping-pong exactly once, so taa_frame must equal the frame count (no
    // skipped/reused snapshots); TAA-off legs must sit at exactly zero.
    if (capture.taaFrame !== (leg.taa ? leg.frames : 0))
      failures.push(
        `taa_frame=${capture.taaFrame}, need ${leg.taa ? leg.frames : 0} (history parity broken)`,
      );
    if (capture.markers !== String(leg.expectMarkers))
      failures.push(
        `capture markers_active=${capture.markers}, need ${leg.expectMarkers}`,
      );
    if (setup && setup.cut > 0 && capture.cutApplied !== "true")
      failures.push("cut leg never applied the actual camera change");
    if (setup && setup.reset > 0 && capture.resetPending !== "false")
      failures.push(
        "reset request still pending at capture (reset never consumed)",
      );
    if (capture.targetW !== expT.w || capture.targetH !== expT.h)
      failures.push(
        `capture target=${capture.targetW}x${capture.targetH}, need realized ${expT.w}x${expT.h}`,
      );
  }
  if (!heldFrame)
    failures.push(
      "no held frame (finite gate never froze; capture would race teardown)",
    );
  else if (heldFrame !== leg.frames)
    failures.push(`held frame=${heldFrame}, need ${leg.frames}`);
  if (!shotBuf) failures.push("no screenshot captured");
  else if (shotBuf.length < 4096)
    failures.push(`screenshot suspiciously small (${shotBuf.length}B)`);
  // Controlled-finish proof: the page ran its normal cleanup AFTER capture.
  if (!verdict) failures.push("no verdict after controlled finish");
  else {
    if (verdict.verdict !== "PASS" || verdict.failures !== 0)
      failures.push(
        `verdict=${verdict.verdict} failures=${verdict.failures} (need PASS/0)`,
      );
  }
  if (!teardown) failures.push("no teardown proof after controlled finish");
  else {
    if (teardown.logErrors !== 0)
      failures.push(`log_errors=${teardown.logErrors} (need 0)`);
    if (teardown.allocs !== 0)
      failures.push(`sokol_live_allocations=${teardown.allocs} (need 0)`);
  }
  const adapterText = JSON.stringify(adapter || {});
  if (!adapter || !adapter.ok)
    failures.push(`no WebGPU adapter (${adapterText.slice(0, 200)})`);
  else if (
    adapter.info &&
    (adapter.info.isFallbackAdapter ||
      /swiftshader|swangle|llvmpipe|software/i.test(adapterText))
  )
    failures.push(
      `fallback/software adapter only (${adapterText.slice(0, 200)})`,
    );

  let stats = null;
  let metrics = null;
  if (shotBuf) {
    try {
      const img = decodePNG(shotBuf);
      stats = imageStats(img);
      if (stats.unique < TAAQ.minUnique)
        failures.push(`flat frame: unique=${stats.unique}`);
      if (stats.std < 0.005)
        failures.push(`empty frame: std=${stats.std.toFixed(4)}`);
      if (stats.mean < 0.01 || stats.mean > 0.6)
        failures.push(
          `implausible exposure: mean=${stats.mean.toFixed(3)} (fixture: dark room + checker)`,
        );
      writeFileSync(join(outDir, `${leg.name}.png`), shotBuf);
      if (shotDup) writeFileSync(join(outDir, `${leg.name}-b.png`), shotDup);
    } catch (e) {
      failures.push(`screenshot decode: ${e.message}`);
    }
  }
  writeFileSync(join(outDir, `${leg.name}.log`), transcript.join("\n") + "\n");
  const result = {
    leg: leg.name,
    role: leg.role,
    query: `?${leg.query}`,
    ok: failures.length === 0,
    failures,
    setup,
    frame1,
    activity,
    capture,
    heldFrame,
    verdict,
    teardown,
    adapter,
    stats,
    metrics,
    consoleProblems: problems,
  };
  writeFileSync(
    join(outDir, `${leg.name}.json`),
    JSON.stringify(result, null, 2),
  );
  return result;
}

export function imageStats(img) {
  const n = img.width * img.height;
  let sum = 0;
  let sum2 = 0;
  const uniq = new Set();
  for (let i = 0; i < n; i++) {
    const L = lumaAt(img, i % img.width, Math.floor(i / img.width));
    sum += L;
    sum2 += L * L;
    if ((i & 7) === 0)
      uniq.add(
        (img.px[i * 4] << 16) | (img.px[i * 4 + 1] << 8) | img.px[i * 4 + 2],
      );
  }
  const mean = sum / n;
  return {
    width: img.width,
    height: img.height,
    mean,
    std: Math.sqrt(Math.max(0, sum2 / n - mean * mean)),
    unique: uniq.size,
  };
}

// (per-leg frame counts live in legTable(); warm history needs >= 120.)

// Gate verdict over the base 10 leg screenshots plus, when present, the
// --classes pairs. Pure (unit-tested): takes the per-leg records with ROI
// metrics + capture proofs and returns PASS/FAIL/INCONCLUSIVE. Class legs
// are additive: a base-only byName decides exactly as before; a partial
// class set (some legs present, pair incomplete) is INCONCLUSIVE.
export function decideGate(byName) {
  const notes = [];
  for (const leg of legTable()) {
    const r = byName[leg.name];
    if (!r)
      return { verdict: "INCONCLUSIVE", notes: [`missing leg ${leg.name}`] };
    if (r.failures.length)
      return {
        verdict: "FAIL",
        notes: [`leg ${leg.name}: ${r.failures.join("; ")}`],
      };
  }
  const classLegs = CLASS_PAIRS.flat();
  const anyClass = classLegs.some((n) => byName[n]);
  if (anyClass) {
    for (const n of classLegs) {
      const r = byName[n];
      if (!r)
        return {
          verdict: "INCONCLUSIVE",
          notes: [`missing class leg ${n} (--classes run incomplete)`],
        };
      if (r.failures.length)
        return {
          verdict: "FAIL",
          notes: [`leg ${n}: ${r.failures.join("; ")}`],
        };
    }
  }
  const need = (name) => {
    const r = byName[name];
    if (!r.metrics) return { fail: `leg ${name}: no metrics (size mismatch?)` };
    if (!r.capture) return { fail: `leg ${name}: no capture proof` };
    return { metrics: r.metrics, capture: r.capture };
  };
  // Noise floor first: a drifting duplicate poisons every comparison below.
  const dup = ghostMetricsRefDup(byName["ref-a"], byName["ref-dup"]);
  if (dup) return dup;
  // Actual marker execution proof (not the requested flag): the ONLY
  // difference between the taa and noreact legs is the reactive marker lane,
  // whose glass redraw lands in main_draw_calls. Delta < 1 means the marker
  // path never executed — fail, never pass on flags alone.
  const mTaa = need("taa");
  const mNo = need("noreact");
  if (mTaa.fail || mNo.fail)
    return { verdict: "FAIL", notes: [mTaa.fail || mNo.fail] };
  const markerDelta = mTaa.capture.mainDraws - mNo.capture.mainDraws;
  if (markerDelta < 1)
    return {
      verdict: "FAIL",
      notes: [
        `marker draws never executed: taa main_draws=${mTaa.capture.mainDraws} vs noreact main_draws=${mNo.capture.mainDraws} (need delta >= 1 glass redraw)`,
      ],
    };
  notes.push(
    `markers executed: main_draws delta taa-vs-noreact = ${markerDelta}`,
  );
  const verdicts = [];
  const pairs = [
    ["taa", "ref-a"],
    ["taa067", "ref067"],
    ["cutcap", "refcut"],
    ["resetcap", "refreset"],
  ];
  // Class pairs share the same TAAQ caps; the marker-delta proof and the
  // negative control stay on the base legs only.
  if (anyClass) pairs.push(...CLASS_PAIRS);
  for (const [test, ref] of pairs) {
    const t = need(test);
    const f = need(ref);
    if (t.fail || f.fail)
      return { verdict: "INCONCLUSIVE", notes: [t.fail || f.fail] };
    // Same-frame parity pin: the pair must be the same encoded frame with
    // the same realized target, or the residual is meaningless.
    if (t.capture.frame !== f.capture.frame)
      return {
        verdict: "INCONCLUSIVE",
        notes: [
          `${test}/${ref}: frame parity broken (${t.capture.frame} vs ${f.capture.frame})`,
        ],
      };
    if (t.metrics.changedPixels === 0)
      return {
        verdict: "INCONCLUSIVE",
        notes: [
          `${test}: zero signal (ROI-identical to reference — fixture static?)`,
        ],
      };
    if (t.metrics.edgeMean > TAAQ.passEdgeMean)
      verdicts.push(
        `FAIL ${test}: ROI ghost edgeMean=${t.metrics.edgeMean.toFixed(4)} > ${TAAQ.passEdgeMean}`,
      );
    else if (t.metrics.flatP95 > TAAQ.passFlatP95)
      verdicts.push(
        `FAIL ${test}: ROI flat p95=${t.metrics.flatP95.toFixed(4)} > ${TAAQ.passFlatP95}`,
      );
    else if (
      t.metrics.sharpRatio < TAAQ.sharpLo ||
      t.metrics.sharpRatio > TAAQ.sharpHi
    )
      verdicts.push(
        `FAIL ${test}: sharpness=${t.metrics.sharpRatio.toFixed(3)} outside [${TAAQ.sharpLo}, ${TAAQ.sharpHi}]`,
      );
    else
      notes.push(
        `${test}: roiEdge=${t.metrics.edgeMean.toFixed(4)} globalEdge=${t.metrics.globalEdgeMean.toFixed(4)} ghostFrac=${t.metrics.ghostFrac.toFixed(3)} sharp=${t.metrics.sharpRatio.toFixed(3)}`,
      );
  }
  // Negative control: reactiveOFF must be measurably worse than the resolve,
  // proving the gate (and the marker path) actually sees history shedding.
  const res = need("taa");
  const ctl = need("noreact");
  if (ctl.metrics.edgeMean < res.metrics.edgeMean + TAAQ.sensMargin)
    return {
      verdict: "INCONCLUSIVE",
      notes: [
        `negative control silent: noreact=${ctl.metrics.edgeMean.toFixed(4)} vs taa=${res.metrics.edgeMean.toFixed(4)} (need +${TAAQ.sensMargin})`,
      ],
    };
  notes.push(
    `control: noreact=${ctl.metrics.edgeMean.toFixed(4)} vs taa=${res.metrics.edgeMean.toFixed(4)} (sensitive)`,
  );
  if (verdicts.length) return { verdict: "FAIL", notes: verdicts };
  return { verdict: "PASS", notes };
}

function ghostMetricsRefDup(a, b) {
  if (!a.metrics || !b.metrics)
    return {
      verdict: "INCONCLUSIVE",
      notes: ["duplicate reference has no metrics (size mismatch?)"],
    };
  if (b.metrics.changedPixels !== 0 || b.metrics.edgeMean > TAAQ.noiseEdgeMean)
    return {
      verdict: "INCONCLUSIVE",
      notes: [
        `reference noise floor drifted: changed=${b.metrics.changedPixels} edgeMean=${b.metrics.edgeMean.toFixed(4)} (nondeterministic capture)`,
      ],
    };
  return null;
}

async function main() {
  const serveDir = arg(
    "--serve-dir",
    join(HERE, "..", "examples", "zig-out", "web"),
  );
  const chromePath = arg("--chrome", CHROME_DEFAULT);
  const timeoutMs = parseInt(arg("--timeout-ms", "150000"), 10);
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0)
    throw new Error("--timeout-ms must be positive");
  const scopedTmp = join(tmpdir(), "opencode");
  const baseTmp = existsSync(scopedTmp) ? scopedTmp : tmpdir();
  const outDir =
    arg("--out", null) ?? mkdtempSync(join(baseTmp, "taa-browser-"));
  mkdirSync(outDir, { recursive: true });
  const useClasses = process.argv.includes("--classes");
  const LEGS = legTable(useClasses);

  const server = await startStaticServer(serveDir);
  const httpPort = server.address().port;
  const cdpPort = await freePort();
  const profileDir = mkdtempSync(join(baseTmp, "taa-gate-profile-"));
  console.log(
    `taa-browser-gate: serve=${serveDir} http=${httpPort} cdp=${cdpPort} out=${outDir} legs=${LEGS.length}`,
  );

  const child = spawn(
    chromePath,
    [
      "--headless=new",
      `--remote-debugging-port=${cdpPort}`,
      "--enable-webgpu",
      "--use-angle=metal",
      "--window-size=960,600",
      `--user-data-dir=${profileDir}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-background-networking",
      "about:blank",
    ],
    { stdio: "ignore", detached: true },
  );
  chromeChildren.push(child);

  const killer = (sig) => killChromeChildren(sig);
  const onSignal = (s) => () => {
    killer(s === "SIGTERM" ? "SIGTERM" : "SIGKILL");
    process.exit(1);
  };
  process.on("SIGINT", onSignal("SIGINT"));
  process.on("SIGTERM", onSignal("SIGTERM"));
  process.on("unhandledRejection", (e) => {
    console.error("[fatal] unhandledRejection:", (e && e.stack) || e);
    killer("SIGKILL");
    process.exit(1);
  });

  let browserWsUrl = null;
  let code = 1;
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
      console.log(`--- leg ${leg.name} (${leg.query}) ---`);
      const r = await runLeg(browserWsUrl, httpPort, leg, outDir, timeoutMs);
      console.log(
        `${r.ok ? "PASS" : "FAIL"} ${leg.name}: held=${r.heldFrame} draws=${r.capture && r.capture.draws} markers=${r.capture && r.capture.markers}` +
          (r.failures.length ? ` :: ${r.failures.join("; ")}` : ""),
      );
      results.push(r);
    }

    // Pixel metrics over the held captures (real GPU renders, same frame),
    // ROI-scoped so static borders cannot dilute the ghost signal.
    const byName = {};
    for (const r of results) byName[r.leg] = r;
    // Pairwise metrics against the matching-scale/frame reference.
    const pair = async (test, ref) => {
      const a = decodePNG(await readFile(join(outDir, `${ref}.png`)));
      const b = decodePNG(await readFile(join(outDir, `${test}.png`)));
      return ghostMetrics(a, b, ROI);
    };
    try {
      const pairs = [
        ["ref-dup", "ref-a"],
        ["taa", "ref-a"],
        ["noreact", "ref-a"],
        ["ref067", "ref067"],
        ["taa067", "ref067"],
        ["cutcap", "refcut"],
        ["refcut", "refcut"],
        ["resetcap", "refreset"],
        ["refreset", "refreset"],
        ["ref-a", "ref-a"],
      ];
      if (useClasses)
        for (const [t, r] of CLASS_PAIRS) pairs.push([t, r], [r, r]);
      for (const [test, ref] of pairs)
        byName[test].metrics = await pair(test, ref);
    } catch (e) {
      for (const r of results) {
        r.ok = false;
        r.failures.push(`metric decode failed: ${e.message}`);
      }
    }
    for (const r of results) {
      r.metrics = byName[r.leg].metrics || null;
      writeFileSync(join(outDir, `${r.leg}.json`), JSON.stringify(r, null, 2));
    }

    const gate = results.every((r) => r.ok)
      ? decideGate(byName)
      : {
          verdict: "FAIL",
          notes: results.flatMap((r) =>
            r.failures.map((f) => `${r.leg}: ${f}`),
          ),
        };
    for (const r of results)
      if (r.metrics)
        console.log(
          `metric ${r.leg}: roiEdge=${r.metrics.edgeMean.toFixed(4)} globalEdge=${r.metrics.globalEdgeMean.toFixed(4)} ghostFrac=${r.metrics.ghostFrac.toFixed(3)} flatP95=${r.metrics.flatP95.toFixed(4)} sharp=${r.metrics.sharpRatio.toFixed(3)} changed=${r.metrics.changedPixels}`,
        );
    console.log(`gate verdict: ${gate.verdict}`);
    for (const n of gate.notes) console.log(`  ${n}`);
    const summary = {
      verdict: gate.verdict,
      notes: gate.notes,
      legs: results.map((r) => ({
        leg: r.leg,
        ok: r.ok,
        metrics: r.metrics,
        failures: r.failures,
      })),
    };
    writeFileSync(
      join(outDir, "summary.json"),
      JSON.stringify(summary, null, 2),
    );
    console.log(`summary: ${gate.verdict} -> ${join(outDir, "summary.json")}`);
    code =
      gate.verdict === "PASS" ? 0 : gate.verdict === "INCONCLUSIVE" ? 2 : 1;
    process.exitCode = code;
  } catch (e) {
    console.error("[fatal]", (e && e.stack) || e);
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
          bws.send(JSON.stringify({ id: 1, method: "Browser.close" }));
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
    killer("SIGTERM");
    await wait(1500);
    killer("SIGKILL");
    await new Promise((resolve) => server.close(resolve));
    try {
      rmSync(profileDir, { recursive: true, force: true });
    } catch {
      /* best effort */
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  main();
