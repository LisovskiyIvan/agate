import test from "node:test";
import assert from "node:assert/strict";
import {
  PVG,
  legTable,
  diffMetrics,
  decideGate,
  classifyDiagnostic,
} from "./pbr_variant_gate.mjs";

function image(w = 320, h = 200) {
  const px = Buffer.alloc(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      px[o] = (x * 255) / w;
      px[o + 1] = (y * 255) / h;
      px[o + 2] = ((x + y) * 127) / (w + h);
      px[o + 3] = 255;
    }
  return { width: w, height: h, px };
}

function record(img, failures = []) {
  return { failures, image: img };
}

function set(over = {}) {
  const base = {
    rigid: record(image()),
    skinned: record(image()),
    instanced: record(image()),
  };
  return { ...base, ...over };
}

test("leg table pins three variants at the same held frame", () => {
  const legs = legTable();
  assert.deepEqual(legs.map((l) => l.variant), ["rigid", "skinned", "instanced"]);
  for (const l of legs) {
    assert.equal(l.frames, PVG.frames);
    assert.match(l.query, new RegExp(`frames=${PVG.frames}`));
    assert.match(l.query, /hold=1/);
  }
});

test("diff metrics are zero for identical captures and see a real change", () => {
  const a = image();
  const b = image();
  const same = diffMetrics(a, b);
  assert.equal(same.meanLuma, 0);
  assert.equal(same.changedFrac, 0);
  assert.equal(same.maxChannel, 0);
  // Change a 20x20 block by 40 codes.
  const changed = image();
  for (let y = 50; y < 70; y++)
    for (let x = 50; x < 70; x++) {
      const o = (y * changed.width + x) * 4;
      changed.px[o] = Math.min(255, changed.px[o] + 40);
    }
  const d = diffMetrics(a, changed);
  assert.ok(d.meanLuma > 0);
  assert.ok(d.changedFrac > 0.005);
  assert.equal(d.maxChannel, 40);
});

test("gate passes identical variants and accepts a matching baseline", () => {
  assert.equal(decideGate(set()).verdict, "PASS");
  const base = { rigid: image(), skinned: image(), instanced: image() };
  assert.equal(decideGate(set(), base).verdict, "PASS");
});

test("gate fails when one variant drifts", () => {
  const bad = image();
  for (let i = 0; i < bad.px.length; i += 4) bad.px[i] = Math.min(255, bad.px[i] + 20);
  const d = decideGate(set({ skinned: record(bad) }));
  assert.equal(d.verdict, "FAIL");
  assert.ok(d.notes.some((n) => n.includes("skinned vs rigid")));
});

test("gate fails when a baseline leg drifts", () => {
  const bad = image();
  for (let i = 0; i < bad.px.length; i += 4) bad.px[i + 1] = Math.min(255, bad.px[i + 1] + 20);
  const d = decideGate(set(), { rigid: bad, skinned: image(), instanced: image() });
  assert.equal(d.verdict, "FAIL");
  assert.ok(d.notes.some((n) => n.includes("baseline rigid")));
});

test("missing legs and missing baseline are INCONCLUSIVE", () => {
  const missing = set();
  delete missing.instanced;
  assert.equal(decideGate(missing).verdict, "INCONCLUSIVE");
  assert.equal(decideGate(set(), { rigid: image() }).verdict, "INCONCLUSIVE");
  const bad = set({ rigid: record(image(), ["no held frame"]) });
  assert.equal(decideGate(bad).verdict, "FAIL");
});

test("diagnostic classifier separates progress from hard failures", () => {
  assert.equal(
    classifyDiagnostic("pbr-variant-gate: verdict=PASS failures=0 frames=10"),
    "info",
  );
  assert.equal(classifyDiagnostic("pbr-variant-gate FAIL: empty box"), "hard-fail");
  assert.equal(classifyDiagnostic("warning: [mesh-vanish-probe] ARMED"), "other");
});
