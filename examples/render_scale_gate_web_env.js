// Emscripten pre-run: none needed — render_scale_gate.zig reads the query
// string directly via emscripten_run_script_int. This file documents the
// accepted knobs so the gate and the fixture cannot drift apart:
//   frames=<n>      finite run length (default 8)
//   rscale=<f>      render scale (gate uses 0.5 and 1.0; the fixture reads it
//                   as fixed-point hundredths because emscripten_run_script_int
//                   truncates floats)
//   light=0         negative control: no point light, black wall
//   hold            freeze the final frame for a deterministic CDP capture
//                   (no quit, no teardown clear, no screenshot race)
Module["preRun"] = Module["preRun"] || [];
Module["preRun"].push(function () {
  const q = new URLSearchParams(location.search);
  if (!q.has("frames"))
    console.log(
      "render-scale-gate: web env defaults (frames=8, rscale=0.5, hold expected from gate)",
    );
});
