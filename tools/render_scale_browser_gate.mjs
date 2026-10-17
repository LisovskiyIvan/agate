#!/usr/bin/env node
// Render-scale + clustered point-light pixel gate (headless Chromium/WebGPU).
//
// Runs the already-built page (examples/zig-out/web/render-scale-gate.html)
// in real headless Chrome on Metal/WebGPU across three held-frame legs:
//
//   lit05    scale 0.5 + one point light   test (realized 480x300 target)
//   lit10    scale 1.0 + the same light    full-res control (960x600)
//   dark05   scale 0.5, light off          negative control (black wall)
//
// The fixture wall is otherwise unlit, so the point light's hotspot is the
// only signal. At scale 0.5 the clustered tiles must be built from the HDR
// TARGET pixels; if they are built from the window instead, the half-res
// fragment indexes a different tile row and the hotspot disappears. The gate
// checks that the hotspot exists, sits at the same normalized position at
// both scales, and is absent from the negative control.
//
// Every leg runs with hold=1: after the capture frame the page issues no
// update, prepare, render or commit, so two CDP captures must be byte-exact.
// The gate then sets window.rsgFinish=1; the page quits itself and runs the
// normal cleanup, and the gate requires verdict=PASS with log_errors=0 and
// zero live allocations AFTER capture before closing the target.
//
// Verdicts: PASS / FAIL / INCONCLUSIVE. INCONCLUSIVE covers a silent
// negative control, an empty control leg, or a non-identical held pair.
// Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
//
// Usage:
//   node render_scale_browser_gate.mjs [--out <dir>] [--serve-dir <dir>]
//     [--chrome <path>] [--timeout-ms <n>]

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { writeFileSync, mkdirSync, mkdtempSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, extname, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import net from "node:net";
import { decodePNG, lumaAt } from "./taa_browser_gate.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const CHROME_DEFAULT = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PAGE = "render-scale-gate.html";
const WINDOW_W = 960;
const WINDOW_H = 600;

// Thresholds are fixture-justified, not tuned to one run: the wall is black
// without the light (clear colour only) and the ACES-mapped hotspot is a
// broad, high-contrast blob, so a 0.15 luma floor separates on/off by a wide
// margin even after the scale-0.5 upscale.
export const RSG = {
  minRightMean: 0.2, // lit05 hotspot ROI mean floor
  minContrast: 0.15, // lit05 hotspot ROI must beat dark05 by this
  maxLeftMean: 0.02, // light must not smear over the unlit left side
  hotspotLuma: 0.15, // pixel counts as lit in the centroid
  hotspotX: [0.6, 0.76], // expected normalized hotspot x at both scales
  hotspotY: [0.34, 0.58],
  posTolerance: 0.03, // lit05 vs lit10 centroid drift
  brightnessRatio: [0.6, 1.6], // lit05/lit10 hotspot-ROI mean
  minUnique: 32,
  minStd: 0.01,
  maxMean: 0.85,
};

// Subject ROI: the light lives on the right half; the left side is the
// "no leak" control. The negative control compares the same right rect.
export const ROI = {
  // Tight box around the fixture hotspot. A broad right-half ROI would
  // dilute a small light and hide exactly the regression this gate targets;
  // the left box is the "no leak" control.
  right: { x0: 0.6, y0: 0.35, x1: 0.75, y1: 0.55 },
  left: { x0: 0.02, y0: 0.25, x1: 0.32, y1: 0.8 },
};

export function expectedTarget(scale) {
  return {
    w: Math.max(1, Math.round(WINDOW_W * scale)),
    h: Math.max(1, Math.round(WINDOW_H * scale)),
  };
}

export function legTable() {
  return [
    {
      name: "lit05",
      query: "frames=8&hold=1&rscale=0.5",
      scale: 0.5,
      lit: true,
      frames: 8,
      role: "test",
    },
    {
      name: "lit10",
      query: "frames=8&hold=1&rscale=1.0",
      scale: 1.0,
      lit: true,
      frames: 8,
      role: "control",
    },
    {
      name: "dark05",
      query: "frames=8&hold=1&rscale=0.5&light=0",
      scale: 0.5,
      lit: false,
      frames: 8,
      role: "negative",
    },
  ];
}

function arg(name, fallback) {
  const hit = process.argv.find((a) => a.startsWith(name + "="));
  if (hit) return hit.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? (process.argv[index + 1] ?? fallback) : fallback;
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// --- Pixel metrics (pure, unit-tested) ---

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

export function regionStats(img, roi) {
  const x0 = Math.max(0, Math.floor(roi.x0 * img.width));
  const x1 = Math.min(img.width, Math.ceil(roi.x1 * img.width));
  const y0 = Math.max(0, Math.floor(roi.y0 * img.height));
  const y1 = Math.min(img.height, Math.ceil(roi.y1 * img.height));
  let sum = 0;
  let n = 0;
  for (let y = y0; y < y1; y++)
    for (let x = x0; x < x1; x++) {
      sum += lumaAt(img, x, y);
      n++;
    }
  return { mean: n ? sum / n : 0, n };
}

// Luma-weighted centroid of pixels above the hotspot floor. Returns null when
// the frame has no lit pixels (the exact failure old tile mapping produced).
export function hotspotCentroid(img, threshold = RSG.hotspotLuma) {
  let sx = 0;
  let sy = 0;
  let sw = 0;
  let count = 0;
  for (let y = 0; y < img.height; y++)
    for (let x = 0; x < img.width; x++) {
      const L = lumaAt(img, x, y);
      if (L < threshold) continue;
      sx += x * L;
      sy += y * L;
      sw += L;
      count++;
    }
  if (!count || sw <= 0) return null;
  return {
    x: sx / sw / img.width,
    y: sy / sw / img.height,
    count,
    frac: count / (img.width * img.height),
  };
}

// Pure verdict over per-leg records. A broken control is INCONCLUSIVE (the
// gate cannot distinguish "regression" from "fixture never rendered"); a
// missing/wrong hotspot on the lit leg is FAIL.
export function decideGate(byName) {
  const notes = [];
  const need = (name) => {
    const r = byName[name];
    if (!r) return { fail: `missing leg ${name}`, inconclusive: true };
    if (r.failures.length) return { fail: `leg ${name}: ${r.failures.join("; ")}` };
    if (!r.metrics) return { fail: `leg ${name}: no metrics`, inconclusive: true };
    return { r };
  };
  for (const leg of legTable()) {
    const got = need(leg.name);
    if (got.fail)
      return {
        verdict: got.inconclusive ? "INCONCLUSIVE" : "FAIL",
        notes: [got.fail],
      };
  }
  const test = need("lit05").r;
  const control = need("lit10").r;
  const dark = need("dark05").r;
  const t = test.metrics;
  const c = control.metrics;
  const d = dark.metrics;

  if (!c.hotspot || c.right.mean < RSG.minRightMean)
    return {
      verdict: "INCONCLUSIVE",
      notes: [
        `full-res control has no usable light: right=${c.right.mean.toFixed(3)} hotspot=${JSON.stringify(c.hotspot)}`,
      ],
    };
  if (d.hotspot || d.right.mean > RSG.maxLeftMean)
    return {
      verdict: "INCONCLUSIVE",
      notes: [
        `negative control is not dark: right=${d.right.mean.toFixed(3)} hotspot=${JSON.stringify(d.hotspot)} (light=0 ignored?)`,
      ],
    };

  if (!t.hotspot)
    return {
      verdict: "FAIL",
      notes: [
        `scale 0.5 has no lit pixels (right=${t.right.mean.toFixed(3)}); clustered tiles missed the half-res target`,
      ],
    };
  if (t.right.mean < RSG.minRightMean)
    return {
      verdict: "FAIL",
      notes: [`scale 0.5 right ROI too dark: ${t.right.mean.toFixed(3)} < ${RSG.minRightMean}`],
    };
  if (t.right.mean - d.right.mean < RSG.minContrast)
    return {
      verdict: "FAIL",
      notes: [
        `scale 0.5 light contrast ${(t.right.mean - d.right.mean).toFixed(3)} < ${RSG.minContrast}`,
      ],
    };
  if (t.left.mean > RSG.maxLeftMean)
    return {
      verdict: "FAIL",
      notes: [`light leaked into the unlit left side: ${t.left.mean.toFixed(3)} > ${RSG.maxLeftMean}`],
    };
  const dx = Math.abs(t.hotspot.x - c.hotspot.x);
  const dy = Math.abs(t.hotspot.y - c.hotspot.y);
  if (dx > RSG.posTolerance || dy > RSG.posTolerance)
    return {
      verdict: "FAIL",
      notes: [
        `hotspot moved with scale: lit05=(${t.hotspot.x.toFixed(3)},${t.hotspot.y.toFixed(3)}) lit10=(${c.hotspot.x.toFixed(3)},${c.hotspot.y.toFixed(3)}) tolerance=${RSG.posTolerance}`,
      ],
    };
  if (
    t.hotspot.x < RSG.hotspotX[0] ||
    t.hotspot.x > RSG.hotspotX[1] ||
    t.hotspot.y < RSG.hotspotY[0] ||
    t.hotspot.y > RSG.hotspotY[1]
  )
    return {
      verdict: "FAIL",
      notes: [
        `hotspot outside the fixture ROI: (${t.hotspot.x.toFixed(3)},${t.hotspot.y.toFixed(3)})`,
      ],
    };
  const ratio = t.right.mean / c.right.mean;
  if (ratio < RSG.brightnessRatio[0] || ratio > RSG.brightnessRatio[1])
    return {
      verdict: "FAIL",
      notes: [
        `scale-0.5 brightness ratio ${ratio.toFixed(3)} outside [${RSG.brightnessRatio[0]}, ${RSG.brightnessRatio[1]}] (lit10=${c.right.mean.toFixed(3)})`,
      ],
    };
  notes.push(
    `lit05 right=${t.right.mean.toFixed(3)} hotspot=(${t.hotspot.x.toFixed(3)},${t.hotspot.y.toFixed(3)})`,
  );
  notes.push(
    `lit10 right=${c.right.mean.toFixed(3)} hotspot=(${c.hotspot.x.toFixed(3)},${c.hotspot.y.toFixed(3)})`,
  );
  notes.push(`contrast vs dark05=${(t.right.mean - d.right.mean).toFixed(3)}`);
  return { verdict: "PASS", notes };
}

// --- Headless runner ---

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".json": "application/json",
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

const ANSI_RE = /\x1b\[[0-9;]*m/g;
const stripAnsi = (s) => s.replace(ANSI_RE, "");
const HARD_FAIL_RE =
  /GPU(?:Validation|Internal|OutOfMemory)Error|uncaptured(?: GPU)? error|\[(?:sg|sokol|sapp)[^\]]*\]\s*\[(?:error|panic)\]|VALIDATE_[A-Z_]+|validation (?:failed|error)|\bpanic\b|\bfatal\b|^render-scale-gate FAIL|render-scale-gate: verdict=FAIL/i;
const INFO_RE =
  /^\s*render-scale-gate: (setup backend=|frame1 frame=|capture frame=|held frame=|finish requested|teardown log_errors=|verdict=PASS)/;
const SHELL_WARN_RE = /^\s*warning:/;

export function classifyDiagnostic(text) {
  const plain = stripAnsi(text);
  if (HARD_FAIL_RE.test(plain)) return "hard-fail";
  if (INFO_RE.test(plain)) return "info";
  return "other";
}

async function runLeg(browserWsUrl, httpPort, leg, outDir, timeoutMs) {
  const transcript = [];
  const problems = [];
  let setup = null;
  let frame1 = null;
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

  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
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
        if (cls === "hard-fail") problems.push(`console: ${text.slice(0, 300)}`);
        else if (cls !== "info" && type === "error" && !SHELL_WARN_RE.test(plain))
          problems.push(`console.error: ${text.slice(0, 300)}`);
        let m = plain.match(
          /render-scale-gate: setup backend=(\S+) scale=([\d.]+) lit=(\S+) frames=(\d+)/,
        );
        if (m)
          setup = {
            backend: m[1],
            scale: +m[2],
            lit: m[3] === "true",
            frames: +m[4],
          };
        m = plain.match(
          /render-scale-gate: frame1 frame=(\d+) target=(\d+)x(\d+) tiles=(\d+)x(\d+) win=(\d+)x(\d+) draws=(\d+) main=(\d+) rscale=([\d.]+) lit=(\S+)/,
        );
        if (m)
          frame1 = {
            frame: +m[1],
            targetW: +m[2],
            targetH: +m[3],
            tilesX: +m[4],
            tilesY: +m[5],
            winW: +m[6],
            winH: +m[7],
            draws: +m[8],
            mainDraws: +m[9],
            scale: +m[10],
            lit: m[11] === "true",
          };
        m = plain.match(
          /render-scale-gate: capture frame=(\d+) target=(\d+)x(\d+) tiles=(\d+)x(\d+) win=(\d+)x(\d+) draws=(\d+) main=(\d+) rscale=([\d.]+) lit=(\S+)/,
        );
        if (m)
          capture = {
            frame: +m[1],
            targetW: +m[2],
            targetH: +m[3],
            tilesX: +m[4],
            tilesY: +m[5],
            winW: +m[6],
            winH: +m[7],
            draws: +m[8],
            mainDraws: +m[9],
            scale: +m[10],
            lit: m[11] === "true",
          };
        m = plain.match(/render-scale-gate: held frame=(\d+)/);
        if (m) {
          heldFrame = +m[1];
          clearTimeout(timer);
          resolve();
        }
        m = plain.match(
          /render-scale-gate: teardown log_errors=(\d+) sokol_live_allocations=(\d+)/,
        );
        if (m) teardown = { logErrors: +m[1], allocs: +m[2] };
        m = plain.match(
          /render-scale-gate: verdict=(\w+) failures=(\d+) frames=(\d+)/,
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
      {
        width: WINDOW_W,
        height: WINDOW_H,
        deviceScaleFactor: 1,
        mobile: false,
      },
      sid,
    );
    await send("Page.navigate", { url: pageUrl }, sid);
    await done;

    // Frozen frame: no render/commit, so both captures must be byte-exact.
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
      problems.push("held captures differ (frame not frozen)");

    await send(
      "Runtime.evaluate",
      { expression: "window.rsgFinish=1", returnByValue: true },
      sid,
    );
    const t0 = Date.now();
    while (!verdict) {
      if (Date.now() - t0 > 30000)
        throw new Error(`leg ${leg.name}: timeout after finish`);
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
      failures.push(`setup scale=${setup.scale} != leg scale=${leg.scale}`);
    if (setup.lit !== leg.lit)
      failures.push(`setup lit=${setup.lit} != leg lit=${leg.lit}`);
    if (setup.frames !== leg.frames)
      failures.push(`setup frames=${setup.frames} != leg frames=${leg.frames}`);
  }
  const checkTarget = (label, t) => {
    if (!t) {
      failures.push(`missing ${label} target proof`);
      return;
    }
    if (t.targetW !== expT.w || t.targetH !== expT.h)
      failures.push(
        `${label} target=${t.targetW}x${t.targetH}, need realized ${expT.w}x${expT.h} for scale=${leg.scale}`,
      );
    // The tile grid is built from the same snapshot size as the target; a
    // window-sized grid at scale 0.5 is the regression in structural form.
    // No clustered lights means the tile path is gated off entirely.
    const needTx = leg.lit ? Math.ceil(expT.w / 64) : 0;
    const needTy = leg.lit ? Math.ceil(expT.h / 64) : 0;
    if (t.tilesX !== needTx || t.tilesY !== needTy)
      failures.push(
        `${label} tiles=${t.tilesX}x${t.tilesY}, need ${needTx}x${needTy} for ${expT.w}x${expT.h}`,
      );
  };
  checkTarget("frame1", frame1);
  checkTarget("capture", capture);
  if (!capture) failures.push("missing capture-frame proof");
  else {
    if (capture.frame !== leg.frames)
      failures.push(`capture frame=${capture.frame}, need ${leg.frames}`);
    if (capture.draws === 0 || capture.mainDraws === 0)
      failures.push("zero draws at capture (empty frame)");
    if (Math.abs(capture.scale - leg.scale) > 0.005)
      failures.push(`capture scale=${capture.scale} != ${leg.scale}`);
  }
  if (!heldFrame)
    failures.push("no held frame (gate would race teardown)");
  else if (heldFrame !== leg.frames)
    failures.push(`held frame=${heldFrame}, need ${leg.frames}`);
  if (!shotBuf) failures.push("no screenshot captured");
  else if (leg.lit && shotBuf.length < 4096)
    failures.push(`screenshot suspiciously small (${shotBuf.length}B)`);
  if (!verdict) failures.push("no verdict after controlled finish");
  else if (verdict.verdict !== "PASS" || verdict.failures !== 0)
    failures.push(`verdict=${verdict.verdict} failures=${verdict.failures}`);
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
    failures.push(`fallback/software adapter only (${adapterText.slice(0, 200)})`);

  let stats = null;
  let metrics = null;
  if (shotBuf) {
    try {
      const img = decodePNG(shotBuf);
      stats = imageStats(img);
      // A negative control is EXPECTED to be flat and small (black wall);
      // only lit legs must carry structure.
      if (leg.lit) {
        if (stats.unique < RSG.minUnique)
          failures.push(`flat frame: unique=${stats.unique}`);
        if (stats.std < RSG.minStd)
          failures.push(`empty frame: std=${stats.std.toFixed(4)}`);
      }
      if (stats.mean > RSG.maxMean)
        failures.push(`implausible exposure: mean=${stats.mean.toFixed(3)}`);
      metrics = {
        right: regionStats(img, ROI.right),
        left: regionStats(img, ROI.left),
        hotspot: hotspotCentroid(img),
        stats,
      };
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
    capture,
    heldFrame,
    verdict,
    teardown,
    adapter,
    stats,
    metrics,
  };
  writeFileSync(join(outDir, `${leg.name}.json`), JSON.stringify(result, null, 2));
  return result;
}

async function main() {
  const serveDir = arg("--serve-dir", join(HERE, "..", "examples", "zig-out", "web"));
  const chromePath = arg("--chrome", CHROME_DEFAULT);
  const timeoutMs = parseInt(arg("--timeout-ms", "120000"), 10);
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0)
    throw new Error("--timeout-ms must be positive");
  const scopedTmp = join(tmpdir(), "opencode");
  const baseTmp = existsSync(scopedTmp) ? scopedTmp : tmpdir();
  const outDir =
    arg("--out", null) ?? mkdtempSync(join(baseTmp, "rsg-browser-"));
  mkdirSync(outDir, { recursive: true });
  const LEGS = legTable();

  const server = await startStaticServer(serveDir);
  const httpPort = server.address().port;
  const cdpPort = await freePort();
  const profileDir = mkdtempSync(join(baseTmp, "rsg-gate-profile-"));
  console.log(
    `render-scale-browser-gate: serve=${serveDir} http=${httpPort} cdp=${cdpPort} out=${outDir} legs=${LEGS.length}`,
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
        `${r.ok ? "PASS" : "FAIL"} ${leg.name}: held=${r.heldFrame} target=${r.capture && `${r.capture.targetW}x${r.capture.targetH}`} hotspot=${r.metrics && r.metrics.hotspot ? `(${r.metrics.hotspot.x.toFixed(3)},${r.metrics.hotspot.y.toFixed(3)})` : "none"}` +
          (r.failures.length ? ` :: ${r.failures.join("; ")}` : ""),
      );
      results.push(r);
    }

    const byName = {};
    for (const r of results) byName[r.leg] = r;
    for (const r of results) {
      if (r.metrics) {
        console.log(
          `metric ${r.leg}: right=${r.metrics.right.mean.toFixed(3)} left=${r.metrics.left.mean.toFixed(3)} hotspot=${r.metrics.hotspot ? `(${r.metrics.hotspot.x.toFixed(3)},${r.metrics.hotspot.y.toFixed(3)}) frac=${r.metrics.hotspot.frac.toFixed(3)}` : "none"}`,
        );
      }
    }

    const gate = results.every((r) => r.ok)
      ? decideGate(byName)
      : {
          verdict: "FAIL",
          notes: results.flatMap((r) =>
            r.failures.map((f) => `${r.leg}: ${f}`),
          ),
        };
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
    writeFileSync(join(outDir, "summary.json"), JSON.stringify(summary, null, 2));
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
