// Emscripten pre-run: none needed — pbr_variant_gate.zig reads the query
// string directly via emscripten_run_script_int. Accepted knobs:
//   frames=<n>      finite run length (default 10)
//   variant=rigid|skinned|instanced   which vertex path draws the box
//   hold            freeze the final frame for a deterministic CDP capture
//                   (no quit, no teardown clear, no screenshot race)
Module["preRun"] = Module["preRun"] || [];
Module["preRun"].push(function () {
  const q = new URLSearchParams(location.search);
  if (!q.has("frames"))
    console.log(
      "pbr-variant-gate: web env defaults (frames=10, variant=rigid, hold expected from gate)",
    );
});
