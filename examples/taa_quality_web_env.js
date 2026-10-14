// Emscripten pre-run: none needed — taa_quality.zig reads the query string
// directly via emscripten_run_script_int. This file only documents the
// accepted knobs so the gate and the fixture cannot drift apart:
//   frames=<n>      finite run length (default 150, gate minimum 120)
//   taa=0           instantaneous reference leg (TAA resolve off)
//   reactive=0      negative control (velocity-less history never sheds)
//   rscale=<f>      render scale (gate uses 1 and 0.67; the fixture reads it
//                   as fixed-point hundredths because emscripten_run_script_int
//                   truncates floats, so rscale=0.67 must NOT arrive as 0)
//   cut=<f>         camera-cut frame (history-reset leg)
//   reset=<f>        explicit resetTaa frame
//   skin=1          knob-gated skinned box + animated bones (gate --classes)
//   inst=1          knob-gated 32-box instanced batch (gate --classes)
//   hold            freeze the final frame for a deterministic CDP capture
//                   (no quit, no teardown clear, no screenshot race)
Module["preRun"] = Module["preRun"] || [];
Module["preRun"].push(function () {
  const q = new URLSearchParams(location.search);
  if (!q.has("frames"))
    console.log(
      "taa-quality: web env defaults (frames=150, hold expected from gate)",
    );
});
