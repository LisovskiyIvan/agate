# Loading performance

## Stage timings (caller-owned, opt-in)

`SceneLoader.LoadOptions.timings: ?*LoadTimings` collects per-stage wall
times in milliseconds. Default `null` keeps every existing call working
unchanged with zero clock overhead and no logging:

```zig
var t = @import("agate").LoadTimings{};
_ = try SceneLoader.appendGlbOptions(&scene, path, .{ .timings = &t });
// t.parse_ms / buffers_ms / meshopt_ms / textures_ms / materials_ms /
// geometry_ms / total_ms
```

Stages:

- `parse_ms` — `cgltf_parse_file`.
- `buffers_ms` — `cgltf_load_buffers`.
- `meshopt_ms` — meshopt buffer-view decode (no-op for files without
  `EXT_meshopt_compression`).
- `textures_ms` — parallel CPU decode (`decodeImagesInParallel`; 0 in
  async mode, where decode happens later on the upload queue).
- `materials_ms` — GPU upload + material wiring (`loadMaterials`; in async
  mode, slot wiring/registration — the queued decode+upload completes after
  return and is captured by no stage).
- `geometry_ms` — skins + mesh spawn + animations + lights/cameras.
- `total_ms` — whole `appendGlbOptions` call, including cleanup, also on error.

The caller's timings are reset on each call (including async/error paths),
so reusing one record never leaves values from the previous asset.
A stage that fails records time-to-failure, not a silent 0; stages never
reached stay 0. The extension gate, `validateTextureCoordinates`, and the
stager allocs run before the first timed section, so they appear in
`total_ms` only — per-stage values do not sum to the total.

Clock: the app-owned `sokol.time` timeline (same as `scene.elapsedMsSince`
and the profiler) — the host calls `sokol.time.setup()` once at startup;
the loader never calls setup itself. The default `null` path never reads
the clock.

## Observed profile (DamagedHelmet / Fox)

Parser stages are tiny; wall time concentrates in stb decode plus the CPU
mip chain (`buildRaw` L0 + `boxDownsampleU8` per level). That is why this
round only touched the texture seam:

1. Fused L0 sRGB convert-copy (`buildRawSrgbFused` in
   `texture/core.zig`): one pass over level 0 instead of
   convert-in-place + memcpy. Bit-identical bytes (same
   `color.srgbToLinearU8` LUT, alpha verbatim, same box chain).
2. Stride-4 convert loop (no per-byte `i % 4`).
3. Hoisted row bases in `boxDownsampleU8`'s generic NPOT path (same taps
   and `(sum + 2) >> 2` rounding).

Deliberately out of scope: changing image formats, color semantics,
filtering, disabling mips, or clone-only benchmark edits.

## Preservation tests

`texture/mip_tests.zig`:

- `buildRawSrgbFused is bit-identical to convert-in-place + buildRaw`
  (2x2, mixed colors/alpha, full chain compared level by level).
- `buildRawSrgbFused preserves 1px, NPOT and alpha edge cases`.
- Pre-existing sRGB/linear/gamma suites pin filter order, rounding,
  and NPOT tail behavior.

## Micro-benchmark (`tools/bench_texture.zig`, idle tree only)

`bench-texture` compares the old L0 path against the fused one on a
deterministic in-memory RGBA source (no stb decode, no GPU upload):

1. Bit-identity gate for 1x1, 3x1 (NPOT tail), 2x2 (fast path + alpha),
   and 2048x2048 — dimensions, level count, and every byte of every mip
   level must match, otherwise the tool exits non-zero before timing.
2. Timed fixture: 2048x2048 RGBA (16 MiB source, 12 levels). Warmup,
   then N alternating-order batches (classic, fused, fused, classic,
   ...) to cancel drift; medians + min/max printed in ms via `sokol.time`.

The classic side replicates the pre-optimization helper
(`color.convertSrgbToLinearInPlace`, same LUT, same per-byte `i % 4`
shape) verbatim — the `texture` facade only re-exports
`srgbToLinearU8`, so the old path is copied exactly, not guessed. The
per-rep scratch refresh (`memcpy`) is excluded from both timers, which
favors classic. Timed default batches use ~53 MiB (source + scratch + one
live chain, freed before the other path runs); the identity gate needs two
chains (~75 MiB). GLB mode also retains the decoded input images. The CLI
takes an optional rep count followed by
optional GLB paths (`zig build bench-texture -- 20
../bench/agate/assets/DamagedHelmet.glb ../bench/agate/assets/Fox.glb`,
default 10, clamped to 1..50; a non-numeric first arg is treated as a GLB
path). With GLB paths the timed sources are the real sRGB texture images
actually used as color slots (baseColor, emissive, sheen color when
present — the same slots the loader decodes with `srgb_to_linear`), unique
by image, each decoded once via `Texture.decodeMemory` with
`{ .gen_mipmaps = false, .srgb_to_linear = false }` outside every timer and
gated for full-chain bit-identity before timing. stb decode is never counted
as a conversion stage; non-embedded or non-RGBA images fail explicitly.

### Measured 03.10.2026 (native macOS ARM64, ReleaseFast)

Recorded on an idle tree; not re-run here — the unit tests re-verify
byte-identity only, and no new timed benchmark was performed in this change.

`zig build bench-texture -Doptimize=ReleaseFast -- 20`, idle tree:

| Convert + complete mip chain, 2048² RGBA8 | Median | Range |
|---|---:|---:|
| Classic | 6.739 ms | 6.680–6.788 ms |
| Fused | 5.072 ms | 5.040–5.114 ms |

Final rerun: 1.33× on this stage (24.7% less time), every mip byte equal. This excludes
stb decode, GLB parsing, GPU upload and first-frame shader compilation;
it is **not** a 25% whole-load or wasm speedup claim.

Real asset inputs, same native ReleaseFast / 20 alternating batches:

```sh
zig build bench-texture -Doptimize=ReleaseFast -- 20 \
  ../bench/agate/assets/DamagedHelmet.glb ../bench/agate/assets/Fox.glb
```

| Source (used color-slot image) | Classic median | Fused median | Less stage time |
|---|---:|---:|---:|
| DamagedHelmet image 1, 2048² | 6.753 ms | 5.076 ms | 24.8% |
| DamagedHelmet image 2, 2048² | 6.809 ms | 5.153 ms | 24.3% |
| Fox, 1024² | 1.656 ms | 1.303 ms | 21.3% |

Every mip byte is identical for all three real images. Decode is outside
these timers; both paths use the same current mip implementation, so this
A/B measures the L0 conversion/copy improvement, not a separate mip-loop gain.

### Real load-stage snapshot (wasm / WebGPU, not native)

The benchmark's opt-in `--load-timings` / `?loadtimings=1` records the
actual `appendGlbOptions` stages without changing import count or content.
One quiet run (`scene_main_current`, Chrome/macOS; raw values in
`bench/load_stage_measurements.json`):

| Stage | DamagedHelmet, 1 load | Fox, sum of 50 loads |
|---|---:|---:|
| Parse | 2.5 ms | 7.3 ms |
| Buffers | 0.2 ms | 0.1 ms |
| Meshopt | 0.1 ms | 0.0 ms |
| Texture decode + conversion + mips | 182.2 ms | 285.3 ms |
| Upload/material wiring | 9.0 ms | 40.1 ms |
| Geometry/animations/lights | 1.9 ms | 6.6 ms |
| Total (including cleanup) | 196.3 ms | 340.2 ms |

Reproduce from `bench/` with the HTTP server running:

```sh
node run_browser_bench.mjs --scene=main --tag=scene_main_loadtimings \
  --freeze --modes=off '--agate-extra=&loadtimings=1'
```

Both engines still import Fox **50 times**. This single snapshot is
diagnostic, not a before/after load-speed benchmark. The paired reports
remain unfavorable for total loading: Agate/Babylon first frame
557.7/275.7 ms, helmet load 197.3/58.3 ms. Native texture-stage savings
cannot be extrapolated to wasm or whole-load speedups.
