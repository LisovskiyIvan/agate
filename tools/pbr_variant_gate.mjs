#!/usr/bin/env node
// PBR variant parity gate (headless Chromium/WebGPU).
//
// Runs the already-built `pbr-variant-gate.html` page three times with the
// same scene/material/lighting and the same held frame, changing only the
// vertex path that draws the box:
//
//   rigid     standard pbr.glsl
//   skinned   skinned_pbr.glsl (one identity bone)
//   instanced instanced_pbr.glsl (one identity instance)
//
// The floor, shadows, area/clustered/legacy lights and IBL are identical.
// A dedup mistake confined to the skinned or instanced fragment wiring
// changes that leg's pixels and fails the pairwise comparison; a shared
// mistake changes all three equally, so pass `--baseline <dir>` with a
// pre-refactor capture set to catch it too.
//
// Every leg uses hold=1 and must quit itself via window.pvgFinish=1; the
// gate requires a byte-identical held pair, realized 960x600, at least one
// shadow draw, verdict=PASS, zero log errors and zero live allocations.
//
// Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE.
//
// Usage:
//   node pbr_variant_gate.mjs [--out <dir>] [--baseline <dir>]
//     [--serve-dir <dir>] [--chrome <path>] [--timeout-ms <n>]

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { writeFileSync, mkdirSync, mkdtempSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, extname, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import net from "node:net";
import { decodePNG, lumaAt, imageStats } from "./taa_browser_gate.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const CHROME_DEFAULT = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PAGE = "pbr-variant-gate.html";
const WINDOW_W = 960;
const WINDOW_H = 600;

export const PVG = {
  nameTolerance: 0.0015, // mean |luma| difference between variants
  changedFraction: 0.01, // pixels with any channel diff > 2
  maxChannelDiff: 6, // hard per-channel ceiling
  minUnique: 64,
  minStd: 0.02,
  maxMean: 0.85,
  frames: 10,
};

export function legTable() {
  return [
    { name: "rigid", query: "variant=rigid", variant: "rigid", role: "test" },
    { name: "skinned", query: "variant=skinned", variant: "skinned", role: "test" },
    { name: "instanced", query: "variant=instanced", variant: "instanced", role: "test" },
  ].map((l) => ({
    ...l,
    frames: PVG.frames,
    query: `${l.query}&frames=${PVG.frames}&hold=1`,
  }));
}

function arg(name, fallback) {
  const hit = process.argv.find((a) => a.startsWith(name + "="));
  if (hit) return hit.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? (process.argv[index + 1] ?? fallback) : fallback;
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// Pairwise image comparison over 8-bit sRGB captures. `meanLuma` is the mean
// absolute luma difference; `changedFrac` counts pixels where any channel
// differs by more than `softChannel`; `maxChannel` is the worst channel delta.
export function diffMetrics(a, b, softChannel = 2) {
  if (a.width !== b.width || a.height !== b.height)
    return { sizeMismatch: true };
  let sum = 0;
  let changed = 0;
  let maxChannel = 0;
  const n = a.width * a.height;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    let pixelChanged = false;
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(a.px[o + c] - b.px[o + c]);
      if (d > maxChannel) maxChannel = d;
      if (d > softChannel) pixelChanged = true;
    }
    if (pixelChanged) changed++;
    sum += Math.abs(lumaAt(a, i % a.width, Math.floor(i / a.width)) - lumaAt(b, i % b.width, Math.floor(i / b.width)));
  }
  return {
    meanLuma: sum / n,
    changedFrac: changed / n,
    maxChannel,
    width: a.width,
    height: a.height,
  };
}

export function decideGate(byName, baseline = null) {
  const notes = [];
  for (const leg of legTable()) {
    const r = byName[leg.name];
    if (!r) return { verdict: "INCONCLUSIVE", notes: [`missing leg ${leg.name}`] };
    if (r.failures.length)
      return { verdict: "FAIL", notes: [`leg ${leg.name}: ${r.failures.join("; ")}`] };
    if (!r.image) return { verdict: "INCONCLUSIVE", notes: [`leg ${leg.name}: no image`] };
  }
  const pairs = [
    ["skinned", "rigid"],
    ["instanced", "rigid"],
    ["instanced", "skinned"],
  ];
  for (const [x, y] of pairs) {
    const d = diffMetrics(byName[x].image, byName[y].image);
    if (d.sizeMismatch) return { verdict: "INCONCLUSIVE", notes: [`${x}/${y}: size mismatch`] };
    if (
      d.meanLuma > PVG.nameTolerance ||
      d.changedFrac > PVG.changedFraction ||
      d.maxChannel > PVG.maxChannelDiff
    )
      return {
        verdict: "FAIL",
        notes: [
          `${x} vs ${y}: meanLuma=${d.meanLuma.toFixed(5)} changedFrac=${d.changedFrac.toFixed(4)} maxChannel=${d.maxChannel}`,
        ],
      };
    notes.push(
      `${x} vs ${y}: meanLuma=${d.meanLuma.toFixed(5)} changedFrac=${d.changedFrac.toFixed(4)} maxChannel=${d.maxChannel}`,
    );
  }
  if (baseline) {
    for (const leg of legTable()) {
      const before = baseline[leg.name];
      if (!before) {
        return { verdict: "INCONCLUSIVE", notes: [`baseline missing ${leg.name}.png`] };
      }
      const d = diffMetrics(byName[leg.name].image, before);
      if (d.sizeMismatch)
        return {
          verdict: "INCONCLUSIVE",
          notes: [`baseline ${leg.name}: captured size differs`],
        };
      if (
        d.meanLuma > PVG.nameTolerance ||
        d.changedFrac > PVG.changedFraction ||
        d.maxChannel > PVG.maxChannelDiff
      )
        return {
          verdict: "FAIL",
          notes: [
            `baseline ${leg.name}: meanLuma=${d.meanLuma.toFixed(5)} changedFrac=${d.changedFrac.toFixed(4)} maxChannel=${d.maxChannel}`,
          ],
        };
      notes.push(
        `baseline ${leg.name}: meanLuma=${d.meanLuma.toFixed(5)} changedFrac=${d.changedFrac.toFixed(4)} maxChannel=${d.maxChannel}`,
      );
    }
  }
  return { verdict: "PASS", notes };
}

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
      /* gone */
    }
  }
}

const ANSI_RE = /\x1b\[[0-9;]*m/g;
const stripAnsi = (s) => s.replace(ANSI_RE, "");
const HARD_FAIL_RE =
  /GPU(?:Validation|Internal|OutOfMemory)Error|uncaptured(?: GPU)? error|\[(?:sg|sokol|sapp)[^\]]*\]\s*\[(?:error|panic)\]|VALIDATE_[A-Z_]+|validation (?:failed|error)|\bpanic\b|\bfatal\b|^pbr-variant-gate FAIL|pbr-variant-gate: verdict=FAIL/i;
const INFO_RE =
  /^\s*pbr-variant-gate: (setup backend=|frame1 frame=|capture frame=|held frame=|finish requested|teardown log_errors=|verdict=PASS)/;
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
      () => reject(new Error(`leg ${leg.name}: timeout ${timeoutMs}ms waiting for held frame`)),
      timeoutMs,
    );
    onEvent = (msg) => {
      if (msg.method === "Runtime.consoleAPICalled" && msg.params) {
        const type = msg.params.type || "log";
        const text = (msg.params.args || [])
          .map((a) => (a.value !== undefined ? String(a.value) : a.description || ""))
          .join(" ");
        log(`[console.${type}] ${text}`);
        const plain = stripAnsi(text);
        const cls = classifyDiagnostic(text);
        if (cls === "hard-fail") problems.push(`console: ${text.slice(0, 300)}`);
        else if (cls !== "info" && type === "error" && !SHELL_WARN_RE.test(plain))
          problems.push(`console.error: ${text.slice(0, 300)}`);
        let m = plain.match(
          /pbr-variant-gate: setup backend=(\S+) variant=(\S+) frames=(\d+)/,
        );
        if (m) setup = { backend: m[1], variant: m[2], frames: +m[3] };
        m = plain.match(
          /pbr-variant-gate: frame1 frame=(\d+) variant=(\S+) target=(\d+)x(\d+) draws=(\d+) main=(\d+) shadow=(\d+)/,
        );
        if (m)
          frame1 = {
            frame: +m[1],
            variant: m[2],
            targetW: +m[3],
            targetH: +m[4],
            draws: +m[5],
            main: +m[6],
            shadow: +m[7],
          };
        m = plain.match(
          /pbr-variant-gate: capture frame=(\d+) variant=(\S+) target=(\d+)x(\d+) draws=(\d+) main=(\d+) shadow=(\d+)/,
        );
        if (m)
          capture = {
            frame: +m[1],
            variant: m[2],
            targetW: +m[3],
            targetH: +m[4],
            draws: +m[5],
            main: +m[6],
            shadow: +m[7],
          };
        m = plain.match(/pbr-variant-gate: held frame=(\d+)/);
        if (m) {
          heldFrame = +m[1];
          clearTimeout(timer);
          resolve();
        }
        m = plain.match(
          /pbr-variant-gate: teardown log_errors=(\d+) sokol_live_allocations=(\d+)/,
        );
        if (m) teardown = { logErrors: +m[1], allocs: +m[2] };
        m = plain.match(/pbr-variant-gate: verdict=(\w+) failures=(\d+) frames=(\d+)/);
        if (m) verdict = { verdict: m[1], failures: +m[2], frames: +m[3] };
      } else if (msg.method === "Runtime.exceptionThrown") {
        const d = msg.params.exceptionDetails || {};
        const t = (d.exception && d.exception.description) || d.text || "unknown exception";
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
      { width: WINDOW_W, height: WINDOW_H, deviceScaleFactor: 1, mobile: false },
      sid,
    );
    await send("Page.navigate", { url: pageUrl }, sid);
    await done;

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
    if (!shotBuf.equals(shotDup)) problems.push("held captures differ (frame not frozen)");

    await send("Runtime.evaluate", { expression: "window.pvgFinish=1", returnByValue: true }, sid);
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
  if (!setup) failures.push("missing setup line");
  else {
    if (setup.backend !== "WGPU") failures.push(`backend is ${setup.backend}, need WGPU`);
    if (setup.variant !== leg.variant)
      failures.push(`setup variant=${setup.variant} != ${leg.variant}`);
    if (setup.frames !== leg.frames)
      failures.push(`setup frames=${setup.frames} != ${leg.frames}`);
  }
  for (const [label, t] of [
    ["frame1", frame1],
    ["capture", capture],
  ]) {
    if (!t) {
      failures.push(`missing ${label} proof`);
      continue;
    }
    if (t.targetW !== WINDOW_W || t.targetH !== WINDOW_H)
      failures.push(`${label} target=${t.targetW}x${t.targetH}, need ${WINDOW_W}x${WINDOW_H}`);
    if (t.variant !== leg.variant)
      failures.push(`${label} variant=${t.variant} != ${leg.variant}`);
    if (t.draws < 2 || t.main < 2) failures.push(`${label} draws=${t.draws} main=${t.main} (empty box/floor)`);
    if (t.shadow < 1) failures.push(`${label} shadow=${t.shadow} (no shadow pass)`);
  }
  if (!capture) failures.push("missing capture-frame proof");
  else if (capture.frame !== leg.frames)
    failures.push(`capture frame=${capture.frame}, need ${leg.frames}`);
  if (!heldFrame) failures.push("no held frame");
  else if (heldFrame !== leg.frames) failures.push(`held frame=${heldFrame}, need ${leg.frames}`);
  if (!shotBuf) failures.push("no screenshot captured");
  if (!verdict) failures.push("no verdict after controlled finish");
  else if (verdict.verdict !== "PASS" || verdict.failures !== 0)
    failures.push(`verdict=${verdict.verdict} failures=${verdict.failures}`);
  if (!teardown) failures.push("no teardown proof");
  else {
    if (teardown.logErrors !== 0) failures.push(`log_errors=${teardown.logErrors}`);
    if (teardown.allocs !== 0) failures.push(`sokol_live_allocations=${teardown.allocs}`);
  }
  const adapterText = JSON.stringify(adapter || {});
  if (!adapter || !adapter.ok) failures.push(`no WebGPU adapter (${adapterText.slice(0, 160)})`);
  else if (
    adapter.info &&
    (adapter.info.isFallbackAdapter || /swiftshader|swangle|llvmpipe|software/i.test(adapterText))
  )
    failures.push(`fallback/software adapter only (${adapterText.slice(0, 160)})`);

  let image = null;
  if (shotBuf) {
    try {
      image = decodePNG(shotBuf);
      const stats = imageStats(image);
      if (stats.unique < PVG.minUnique) failures.push(`flat frame unique=${stats.unique}`);
      if (stats.std < PVG.minStd) failures.push(`empty frame std=${stats.std.toFixed(4)}`);
      if (stats.mean > PVG.maxMean) failures.push(`implausible exposure mean=${stats.mean.toFixed(3)}`);
      writeFileSync(join(outDir, `${leg.name}.png`), shotBuf);
      if (shotDup) writeFileSync(join(outDir, `${leg.name}-b.png`), shotDup);
    } catch (e) {
      failures.push(`screenshot decode: ${e.message}`);
    }
  }
  writeFileSync(join(outDir, `${leg.name}.log`), transcript.join("\n") + "\n");
  const result = {
    leg: leg.name,
    variant: leg.variant,
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
    image,
  };
  writeFileSync(join(outDir, `${leg.name}.json`), JSON.stringify({ ...result, image: undefined }, null, 2));
  return result;
}

async function main() {
  const serveDir = arg("--serve-dir", join(HERE, "..", "examples", "zig-out", "web"));
  const baselineDir = arg("--baseline", null);
  const chromePath = arg("--chrome", CHROME_DEFAULT);
  const timeoutMs = parseInt(arg("--timeout-ms", "120000"), 10);
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0)
    throw new Error("--timeout-ms must be positive");
  const scopedTmp = join(tmpdir(), "opencode");
  const baseTmp = existsSync(scopedTmp) ? scopedTmp : tmpdir();
  const outDir = arg("--out", null) ?? mkdtempSync(join(baseTmp, "pvg-browser-"));
  mkdirSync(outDir, { recursive: true });
  const LEGS = legTable();

  const server = await startStaticServer(serveDir);
  const httpPort = server.address().port;
  const cdpPort = await freePort();
  const profileDir = mkdtempSync(join(baseTmp, "pvg-gate-profile-"));
  console.log(
    `pbr-variant-browser-gate: serve=${serveDir} http=${httpPort} cdp=${cdpPort} out=${outDir} legs=${LEGS.length}` +
      (baselineDir ? ` baseline=${baselineDir}` : ""),
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
        `${r.ok ? "PASS" : "FAIL"} ${leg.name}: held=${r.heldFrame} target=${r.capture && `${r.capture.targetW}x${r.capture.targetH}`} draws=${r.capture && r.capture.draws}` +
          (r.failures.length ? ` :: ${r.failures.join("; ")}` : ""),
      );
      results.push(r);
    }

    const byName = {};
    for (const r of results) byName[r.leg] = r;

    const baseline = {};
    if (baselineDir) {
      for (const leg of LEGS) {
        try {
          baseline[leg.name] = decodePNG(await readFile(join(baselineDir, `${leg.name}.png`)));
        } catch {
          /* decideGate reports the missing leg */
        }
      }
    }

    const gate = results.every((r) => r.ok)
      ? decideGate(byName, baselineDir ? baseline : null)
      : { verdict: "FAIL", notes: results.flatMap((r) => r.failures.map((f) => `${r.leg}: ${f}`)) };
    console.log(`gate verdict: ${gate.verdict}`);
    for (const n of gate.notes) console.log(`  ${n}`);
    const summary = {
      verdict: gate.verdict,
      notes: gate.notes,
      baseline: baselineDir,
      legs: results.map((r) => ({
        leg: r.leg,
        ok: r.ok,
        stats: r.image ? imageStats(r.image) : null,
        failures: r.failures,
      })),
    };
    writeFileSync(join(outDir, "summary.json"), JSON.stringify(summary, null, 2));
    console.log(`summary: ${gate.verdict} -> ${join(outDir, "summary.json")}`);
    code = gate.verdict === "PASS" ? 0 : gate.verdict === "INCONCLUSIVE" ? 2 : 1;
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
