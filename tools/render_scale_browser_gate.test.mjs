import test from "node:test";
import assert from "node:assert/strict";
import {
  RSG,
  ROI,
  expectedTarget,
  legTable,
  imageStats,
  regionStats,
  hotspotCentroid,
  decideGate,
  classifyDiagnostic,
} from "./render_scale_browser_gate.mjs";

function makeImage(w = 960, h = 600) {
  return { width: w, height: h, px: Buffer.alloc(w * h * 4) };
}

function setLuma(img, x, y, luma) {
  const o = (y * img.width + x) * 4;
  const v = Math.round(Math.max(0, Math.min(1, luma)) * 255);
  img.px[o] = v;
  img.px[o + 1] = v;
  img.px[o + 2] = v;
  img.px[o + 3] = 255;
}

// Solid black frame plus a bright disk at the fixture hotspot position.
function litImage(w = 960, h = 600, cx = 0.675, cy = 0.456, radius = 0.06) {
  const img = makeImage(w, h);
  const rx = radius * w;
  const ry = radius * h * (w / h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const dx = (x - cx * w) / rx;
      const dy = (y - cy * h) / ry;
      const d = Math.hypot(dx, dy);
      if (d <= 1) setLuma(img, x, y, 0.9 * (1 - d * d));
    }
  return img;
}

function record({ right = 0, left = 0, hotspot = null } = {}) {
  return {
    failures: [],
    metrics: {
      right: { mean: right, n: 100 },
      left: { mean: left, n: 100 },
      hotspot,
      stats: { mean: right * 0.2, std: 0.1, unique: 128 },
    },
  };
}

function gateSet(over = {}) {
  const base = {
    lit05: record({ right: 0.638, left: 0, hotspot: { x: 0.675, y: 0.456, frac: 0.03 } }),
    lit10: record({ right: 0.638, left: 0, hotspot: { x: 0.675, y: 0.456, frac: 0.03 } }),
    dark05: record({ right: 0, left: 0, hotspot: null }),
  };
  return { ...base, ...over };
}

test("expected target and tile grid follow the requested scale", () => {
  assert.deepEqual(expectedTarget(0.5), { w: 480, h: 300 });
  assert.deepEqual(expectedTarget(1.0), { w: 960, h: 600 });
  assert.deepEqual(expectedTarget(0.67), { w: 643, h: 402 });
  assert.equal(Math.ceil(480 / 64), 8); // scale-0.5 grid
  assert.equal(Math.ceil(960 / 64), 15); // scale-1 grid
});

test("leg table carries the three expected legs", () => {
  const legs = legTable();
  assert.deepEqual(legs.map((l) => l.name), ["lit05", "lit10", "dark05"]);
  assert.deepEqual(legs.map((l) => l.scale), [0.5, 1.0, 0.5]);
  assert.deepEqual(legs.map((l) => l.lit), [true, true, false]);
});

test("image stats and hotspot centroid see the lit disk", () => {
  const img = litImage();
  const stats = imageStats(img);
  assert.equal(stats.width, 960);
  assert.ok(stats.std > 0.05);
  const right = regionStats(img, ROI.right);
  assert.ok(right.mean > RSG.minRightMean, `right=${right.mean}`);
  assert.equal(regionStats(img, ROI.left).mean, 0);
  const hs = hotspotCentroid(img);
  assert.ok(hs);
  assert.ok(Math.abs(hs.x - 0.675) < 0.01, `x=${hs.x}`);
  assert.ok(Math.abs(hs.y - 0.456) < 0.01, `y=${hs.y}`);
  assert.equal(hotspotCentroid(makeImage()), null);
});

test("gate passes a clean lit/control/dark set", () => {
  const gate = decideGate(gateSet());
  assert.equal(gate.verdict, "PASS", gate.notes.join("; "));
});

test("gate fails when the scaled hotspot disappears", () => {
  const gate = decideGate(
    gateSet({ lit05: record({ right: 0, left: 0, hotspot: null }) }),
  );
  assert.equal(gate.verdict, "FAIL");
  assert.ok(gate.notes.some((n) => n.includes("no lit pixels")));
});

test("gate fails when the hotspot moves with scale", () => {
  const gate = decideGate(
    gateSet({
      lit05: record({
        right: 0.638,
        left: 0,
        hotspot: { x: 0.73, y: 0.456, frac: 0.03 },
      }),
    }),
  );
  assert.equal(gate.verdict, "FAIL");
  assert.ok(gate.notes.some((n) => n.includes("hotspot moved")));
});

test("gate fails when the light leaks to the unlit side", () => {
  const gate = decideGate(
    gateSet({
      lit05: record({
        right: 0.638,
        left: 0.2,
        hotspot: { x: 0.675, y: 0.456, frac: 0.5 },
      }),
    }),
  );
  assert.equal(gate.verdict, "FAIL");
  assert.ok(gate.notes.some((n) => n.includes("leaked")));
});

test("a broken control is INCONCLUSIVE, never a pass", () => {
  const noControl = decideGate(
    gateSet({
      lit10: record({ right: 0, left: 0, hotspot: null }),
    }),
  );
  assert.equal(noControl.verdict, "INCONCLUSIVE");
  const brightDark = decideGate(gateSet({ dark05: record({ right: 0.5, left: 0.5 }) }));
  assert.equal(brightDark.verdict, "INCONCLUSIVE");
});

test("a leg failure or missing leg fails closed", () => {
  const failed = gateSet();
  failed.lit05.failures = ["no held frame"];
  assert.equal(decideGate(failed).verdict, "FAIL");
  const missing = gateSet();
  delete missing.dark05;
  assert.equal(decideGate(missing).verdict, "INCONCLUSIVE");
});

test("diagnostic classifier separates fixture info from hard failures", () => {
  assert.equal(
    classifyDiagnostic("render-scale-gate: verdict=PASS failures=0 frames=8"),
    "info",
  );
  assert.equal(
    classifyDiagnostic("render-scale-gate FAIL: no draws"),
    "hard-fail",
  );
  assert.equal(classifyDiagnostic("warning: [mesh-vanish-probe] ARMED"), "other");
});
