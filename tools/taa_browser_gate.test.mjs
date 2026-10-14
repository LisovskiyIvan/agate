import test from "node:test";
import assert from "node:assert/strict";
import {
  ghostMetrics,
  classifyDiagnostic,
  legTable,
  decideGate,
  expectedTarget,
  CLASS_PAIRS,
  TAAQ,
  ROI,
  decodePNG,
} from "./taa_browser_gate.mjs";

const image = (rgba, width = 1) => ({
  width,
  height: rgba.length / (width * 4),
  px: Buffer.from(rgba),
});

// Flat-vs-flat: no edges, no residual, sharpness neutral.
test("ghost metrics on identical frames report zero residual", () => {
  const m = ghostMetrics(image([10, 20, 30, 255]), image([10, 20, 30, 255]));
  assert.equal(m.edgeMean, 0);
  assert.equal(m.changedPixels, 0);
  assert.equal(m.sharpRatio, 1);
});

// Checker pair: reference carries the edge mask so a blurred test loses sharpness.
test("ghost metrics catch blur as sharpness loss on reference edges", () => {
  const ref = image(
    [
      0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0,
      0, 255, 255, 255, 255, 255,
    ],
    3,
  );
  const flat = image(
    new Array(24).fill(128).map((v, i) => (i % 4 === 3 ? 255 : v)),
    3,
  );
  const m = ghostMetrics(ref, flat);
  assert.ok(m.edgeFrac > 0);
  assert.ok(m.edgeMean > 0.1);
  assert.ok(m.sharpRatio < 0.75);
});

test("ghost metrics return null on size mismatch (no silent pass)", () => {
  assert.equal(
    ghostMetrics(image([0, 0, 0, 255]), image([0, 0, 0, 255, 0, 0, 0, 255])),
    null,
  );
});

// Localized trail: a 2x2 ghost square INSIDE the ROI is detected...
test("ROI metrics detect a localized trail inside the sweep region", () => {
  // 8x2: left half black, right half white (one vertical edge at x=4).
  const row = (w) => {
    const px = [];
    for (let x = 0; x < w; x++) {
      const v = x < w / 2 ? 0 : 255;
      px.push(v, v, v, 255);
    }
    return px;
  };
  const ref = image([...row(8), ...row(8)], 8);
  const ghost = image([...row(8), ...row(8)], 8);
  // Smear the edge inside the ROI (x=3..4 of row 0): mid-grey trail pixels.
  for (const x of [3, 4]) {
    const o = x * 4;
    ghost.px[o] = ghost.px[o + 1] = ghost.px[o + 2] = 128;
  }
  const m = ghostMetrics(ref, ghost, { x0: 0.25, y0: 0, x1: 0.75, y1: 1 });
  assert.ok(m.edgeN === undefined || m.edgeMean > 0.05);
  assert.ok(m.changedPixels > 0);
});

// ...while the SAME square OUTSIDE the ROI is ignored (no border dilution).
test("ROI metrics ignore an identical trail outside the sweep region", () => {
  const row = (w) => {
    const px = [];
    for (let x = 0; x < w; x++) {
      const v = x < w / 2 ? 0 : 255;
      px.push(v, v, v, 255);
    }
    return px;
  };
  const ref = image([...row(8), ...row(8)], 8);
  const ghost = image([...row(8), ...row(8)], 8);
  // Smear far-left flat pixels (x=0..1, outside ROI, no reference edge).
  for (const x of [0, 1]) {
    const o = x * 4;
    ghost.px[o] = ghost.px[o + 1] = ghost.px[o + 2] = 128;
  }
  const m = ghostMetrics(ref, ghost, { x0: 0.25, y0: 0, x1: 0.75, y1: 1 });
  assert.equal(m.edgeMean, 0);
  assert.equal(m.changedPixels, 0);
  // ...but the global guard still sees it (reported, never verdict input).
  const g = ghostMetrics(ref, ghost);
  assert.ok(g.changedPixels > 0);
});

test("frozen frame reports zero ROI signal (no fake activity)", () => {
  const ref = image([0, 0, 0, 255, 255, 255, 255, 255], 2);
  const m = ghostMetrics(ref, ref, ROI);
  assert.equal(m.changedPixels, 0);
  assert.equal(m.edgeMean, 0);
});

test("fixture diagnostics classify without hiding hard failures", () => {
  assert.equal(classifyDiagnostic("taa-quality: held frame=150"), "info");
  assert.equal(
    classifyDiagnostic("taa-quality: setup backend=WGPU taa=true"),
    "info",
  );
  assert.equal(
    classifyDiagnostic("taa-quality: verdict=PASS failures=0 frames=150"),
    "info",
  );
  assert.equal(
    classifyDiagnostic("taa-quality: held frame=150 GPUValidationError"),
    "hard-fail",
  );
  assert.equal(classifyDiagnostic("taa-quality FAIL: ghosting"), "hard-fail");
  assert.equal(
    classifyDiagnostic("taa-quality: verdict=FAIL failures=2 frames=150"),
    "hard-fail",
  );
  assert.equal(classifyDiagnostic("some random log line"), "other");
});

test("expected target mirrors scaledRenderSize (realized, not queried)", () => {
  assert.deepEqual(expectedTarget(1.0), { w: 960, h: 600 });
  assert.deepEqual(expectedTarget(0.67), { w: 643, h: 402 });
});

test("leg table: 10 legs with per-leg frames, cut/reset pairs, marker roles", () => {
  const legs = legTable();
  assert.equal(legs.length, 10);
  const roles = legs.map((l) => l.role);
  assert.ok(
    roles.includes("ref") &&
      roles.includes("refdup") &&
      roles.includes("control"),
  );
  assert.equal(legs.filter((l) => l.role === "test").length, 4);
  assert.ok(legs.every((l) => l.query.includes("hold=1")));
  assert.ok(legs.every((l) => l.frames >= 120));
  const cut = legs.find((l) => l.name === "cutcap");
  const refcut = legs.find((l) => l.name === "refcut");
  assert.equal(cut.frames, 120);
  assert.ok(cut.query.includes("cut=120") && refcut.query.includes("cut=120"));
  assert.ok(refcut.query.includes("taa=0"));
  const reset = legs.find((l) => l.name === "resetcap");
  assert.equal(reset.frames, 130);
  assert.ok(reset.query.includes("reset=130"));
  assert.ok(legs.find((l) => l.name === "taa").expectMarkers);
  assert.ok(!legs.find((l) => l.name === "noreact").expectMarkers);
  assert.ok(!legs.find((l) => l.name === "ref-a").taa);
});

const shot = (m, mainDraws = 8, frame = 150) => ({
  leg: "x",
  failures: [],
  stats: { unique: 500, std: 0.1 },
  metrics: m,
  capture: { frame, mainDraws },
  setup: { backend: "WGPU" },
});
const refShot = (edgeMean, extra = {}, mainDraws = 8, frame = 150) =>
  shot(
    {
      edgeMean,
      edgeFrac: 0.2,
      ghostFrac: 0,
      flatP95: 0.01,
      sharpRatio: 1.0,
      changedPixels: 100,
      globalEdgeMean: edgeMean,
      ...extra,
    },
    mainDraws,
    frame,
  );

test("gate verdicts: pass, ghost fail, inconclusive noise and silent control", () => {
  const byName = (over) => ({
    "ref-a": refShot(0, {}, 8, 150),
    "ref-dup": refShot(0, { changedPixels: 0 }, 8, 150),
    taa: refShot(0.02, {}, 9, 150),
    noreact: refShot(0.04, {}, 8, 150),
    ref067: refShot(0, {}, 8, 150),
    taa067: refShot(0.02, {}, 9, 150),
    cutcap: refShot(0.02, {}, 9, 120),
    refcut: refShot(0, {}, 8, 120),
    resetcap: refShot(0.02, {}, 9, 130),
    refreset: refShot(0, {}, 8, 130),
    ...over,
  });
  assert.equal(decideGate(byName()).verdict, "PASS");
  const ghost = TAAQ.passEdgeMean + 0.05;
  assert.equal(
    decideGate(
      byName({
        taa: refShot(ghost, {}, 9, 150),
        noreact: refShot(ghost + 0.1, {}, 8, 150),
      }),
    ).verdict,
    "FAIL",
  );
  assert.equal(
    decideGate(byName({ "ref-dup": refShot(0.05, { changedPixels: 9 }) }))
      .verdict,
    "INCONCLUSIVE",
  );
  assert.equal(
    decideGate(
      byName({ noreact: refShot(0.02 + TAAQ.sensMargin * 0.5, {}, 8, 150) }),
    ).verdict,
    "INCONCLUSIVE",
  );
  assert.equal(
    decideGate(byName({ taa: refShot(0, { changedPixels: 0 }, 9, 150) }))
      .verdict,
    "INCONCLUSIVE",
  );
});

test("unexecuted markers fail (flags alone never pass)", () => {
  const byName = {
    "ref-a": refShot(0, {}, 8, 150),
    "ref-dup": refShot(0, { changedPixels: 0 }, 8, 150),
    taa: refShot(0.02, {}, 8, 150),
    noreact: refShot(0.04, {}, 8, 150),
    ref067: refShot(0, {}, 8, 150),
    taa067: refShot(0.02, {}, 8, 150),
    cutcap: refShot(0.02, {}, 8, 120),
    refcut: refShot(0, {}, 8, 120),
    resetcap: refShot(0.02, {}, 8, 130),
    refreset: refShot(0, {}, 8, 130),
  };
  const d = decideGate(byName);
  assert.equal(d.verdict, "FAIL");
  assert.ok(d.notes.some((n) => n.includes("marker draws never executed")));
});

test("broken frame parity is inconclusive, not a metric verdict", () => {
  const byName = {
    "ref-a": refShot(0, {}, 8, 150),
    "ref-dup": refShot(0, { changedPixels: 0 }, 8, 150),
    taa: refShot(0.02, {}, 9, 150),
    noreact: refShot(0.04, {}, 8, 150),
    ref067: refShot(0, {}, 8, 150),
    taa067: refShot(0.02, {}, 9, 150),
    cutcap: refShot(0.02, {}, 9, 121),
    refcut: refShot(0, {}, 8, 120),
    resetcap: refShot(0.02, {}, 9, 130),
    refreset: refShot(0, {}, 8, 130),
  };
  const d = decideGate(byName);
  assert.equal(d.verdict, "INCONCLUSIVE");
  assert.ok(d.notes.some((n) => n.includes("frame parity")));
});

test("leg failures force FAIL, never a metric pass", () => {
  const bad = refShot(0.02, {}, 9, 150);
  bad.failures = ["missing held frame"];
  const byName = {
    "ref-a": refShot(0, {}, 8, 150),
    "ref-dup": refShot(0, { changedPixels: 0 }, 8, 150),
    taa: bad,
    noreact: refShot(0.04, {}, 8, 150),
    ref067: refShot(0, {}, 8, 150),
    taa067: refShot(0.02, {}, 9, 150),
    cutcap: refShot(0.02, {}, 9, 120),
    refcut: refShot(0, {}, 8, 120),
    resetcap: refShot(0.02, {}, 9, 130),
    refreset: refShot(0, {}, 8, 130),
  };
  const d = decideGate(byName);
  assert.equal(d.verdict, "FAIL");
  assert.ok(d.notes.some((n) => n.includes("taa")));
});

test("leg table with classes adds matched skin/inst/part pairs (default 10 unchanged)", () => {
  assert.equal(legTable().length, 10);
  assert.equal(legTable(false).length, 10);
  const legs = legTable(true);
  assert.equal(legs.length, 16);
  // Base prefix bit-identical: same names, queries, frames.
  assert.deepEqual(
    legs.slice(0, 10).map((l) => l.name),
    legTable().map((l) => l.name),
  );
  assert.deepEqual(
    legs.slice(0, 10).map((l) => l.query),
    legTable().map((l) => l.query),
  );
  const extra = legs.slice(10);
  assert.deepEqual(
    extra.map((l) => l.name),
    ["skinref", "skintaa", "instref", "insttaa", "partref", "parttaa"],
  );
  assert.ok(extra.every((l) => l.query.includes("hold=1")));
  assert.ok(extra.every((l) => l.frames >= 120 && l.frames === 150));
  assert.ok(extra.every((l) => l.scale === 1.0));
  const skin = extra.filter((l) => l.name.includes("skin"));
  assert.ok(skin.every((l) => l.query.includes("skin=1")));
  const inst = extra.filter((l) => l.name.includes("inst"));
  assert.ok(inst.every((l) => l.query.includes("inst=1")));
  const part = extra.filter((l) => l.name.includes("part"));
  assert.ok(part.every((l) => l.query.includes("part=1")));
  for (const [test, ref] of CLASS_PAIRS) {
    const t = legs.find((l) => l.name === test);
    const f = legs.find((l) => l.name === ref);
    assert.ok(t.taa && !f.taa);
    assert.ok(t.query.includes("taa=0") === false);
    assert.ok(f.query.includes("taa=0"));
    assert.ok(t.expectMarkers && !f.expectMarkers);
    assert.equal(t.frames, f.frames);
  }
});

const withClasses = (over = {}) => ({
  "ref-a": refShot(0, {}, 8, 150),
  "ref-dup": refShot(0, { changedPixels: 0 }, 8, 150),
  taa: refShot(0.02, {}, 9, 150),
  noreact: refShot(0.04, {}, 8, 150),
  ref067: refShot(0, {}, 8, 150),
  taa067: refShot(0.02, {}, 9, 150),
  cutcap: refShot(0.02, {}, 9, 120),
  refcut: refShot(0, {}, 8, 120),
  resetcap: refShot(0.02, {}, 9, 130),
  refreset: refShot(0, {}, 8, 130),
  skinref: refShot(0, {}, 8, 150),
  skintaa: refShot(0.02, {}, 9, 150),
  instref: refShot(0, {}, 8, 150),
  insttaa: refShot(0.02, {}, 9, 150),
  partref: refShot(0, {}, 8, 150),
  parttaa: refShot(0.02, {}, 9, 150),
  ...over,
});

test("gate with classes: pass, class ghost fail, partial set inconclusive", () => {
  assert.equal(decideGate(withClasses()).verdict, "PASS");
  // Base-only input still decides exactly as before (additive legs absent).
  const { skinref, skintaa, instref, insttaa, partref, parttaa, ...baseOnly } =
    withClasses();
  assert.equal(decideGate(baseOnly).verdict, "PASS");
  const ghost = TAAQ.passEdgeMean + 0.05;
  assert.equal(
    decideGate(withClasses({ skintaa: refShot(ghost, {}, 9, 150) })).verdict,
    "FAIL",
  );
  assert.equal(
    decideGate(withClasses({ insttaa: refShot(ghost, {}, 9, 150) })).verdict,
    "FAIL",
  );
  // Partial class set: pair incomplete -> INCONCLUSIVE, never a pass.
  const { insttaa: _drop, ...partial } = withClasses();
  const d = decideGate(partial);
  assert.equal(d.verdict, "INCONCLUSIVE");
  assert.ok(d.notes.some((n) => n.includes("class leg")));
  // Class leg failure forces FAIL.
  const bad = refShot(0.02, {}, 9, 150);
  bad.failures = ["no held frame"];
  assert.equal(decideGate(withClasses({ skintaa: bad })).verdict, "FAIL");
});

test("PNG decoder rejects non-PNG input", () => {
  assert.throws(
    () => decodePNG(Buffer.from([1, 2, 3, 4, 5, 6, 7, 8])),
    /not a PNG/,
  );
  assert.ok(TAAQ.passEdgeMean > TAAQ.sensMargin);
});
