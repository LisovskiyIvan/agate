# Vendored sokol-zig (patched)

Snapshot of upstream **floooh/sokol-zig @fafdd96d1694f72fd3a261e25996d51c8b75f62e**
(`git+https://github.com/floooh/sokol-zig.git#fafdd96d`, package hash
`sokol-0.1.0-pb1HKwkTQQA_o-CH0MwmkeZw_5n6EdqNQHWuV45dnW3R`), vendored as a path
dependency of `agate` so that one local patch can be applied to the generated
bindings.

## Why

`src/sokol/gfx.zig` is machine generated ("machine generated, do not edit") and
declares every C entry point that takes a resource handle by value like this:

```zig
extern fn sg_apply_pipeline(Pipeline) void;   // Pipeline = extern struct { id: u32 }
```

Zig 0.16 lowers that 4-byte `extern struct` parameter to a widened `i64` LLVM
parameter and materialises it with an 8-byte load out of a 4-byte object:

```llvm
%slot = alloca %gfx.Pipeline, align 4
...
%v = load i64, ptr %slot, align 8     ; out-of-bounds -> UB
call void @sg_apply_pipeline(i64 %v)
```

Whenever the handle is memory-resident at the call site and SROA cannot promote
the slot (a phi between two pipelines, a value returned through `sret`, any
escaped alloca), that load survives. LLVM then treats it as undefined: in
`-Doptimize=ReleaseFast` it folded the whole call argument to `0`, so
`sg_apply_pipeline()` dereferenced a NULL pipeline slot (`EXC_BAD_ACCESS` at
offset `0x8` in `HighlightPass.renderMask`). `-OReleaseSafe`/`Debug` never inline
the wrapper the same way, which is why the crash only showed up in optimized
builds.

The same shape exists for `sg_query_pipeline_state` (the second `mov x0, #0x0`
site in the binary, `scene/viewport_clear.zig`) and for all other handle-taking
entry points (`sg_destroy_*`, `sg_update_buffer`, `sg_query_*_state`, ...).

Passing the handle as `u32` is ABI-equivalent on AArch64 — the C side reads
`w0` — and makes Zig load exactly 4 bytes, so the UB disappears.

## The patch

Applied by `tools/patch_sokol_handle_abi.py` (91 `extern fn` declarations plus
their wrapper bodies in `src/sokol/gfx.zig`):

```zig
-extern fn sg_apply_pipeline(Pipeline) void;
-pub fn applyPipeline(pip: Pipeline) void { sg_apply_pipeline(pip); }
+extern fn sg_apply_pipeline(u32) void;
+pub fn applyPipeline(pip: Pipeline) void { sg_apply_pipeline(pip.id); }
```

Only `gfx.zig` is patched because that is the only binding file `agate` compiles.
The sibling modules that also pass 4-byte handles by value (`cmdbuf.zig`,
`debugtext.zig`, `framebuffer.zig`, `gl.zig`, `fetch.zig`) carry the same latent
defect but are unused; pass them to the script explicitly if that ever changes.

## Refreshing from upstream

`agate` depends on sokol via a **path dependency** (`.sokol = .{ .path =
"vendor/sokol" }` in `build.zig.zon`), so a refresh must replace the
*contents* of `vendor/sokol` and leave `build.zig.zon` untouched.

> Do NOT use `zig fetch --save`: it rewrites `build.zig.zon` to a
> URL-dependency, destroying the path dependency (and the comment block
> above it documenting why sokol is vendored). Plain `zig fetch` only
> populates the global cache and prints the package hash.

```sh
# 1. Fetch into the global cache and note the printed hash, e.g.
#    sokol-0.1.0-pb1HKwkTQQA_o-CH0MwmkeZw_5n6EdqNQHWuV45dnW3R.
#    Tarballs land in <global_cache_dir>/p/ (`zig env` -> .global_cache_dir,
#    defaults to ~/.cache/zig).
zig fetch git+https://github.com/floooh/sokol-zig.git#<commit>

# 2. Replace vendor/sokol with the pristine upstream tree. The tarball
#    wraps everything in one top-level <hash>/ directory, hence
#    --strip-components=1. Keep this README: it is not part of upstream.
HASH=<hash-printed-by-zig-fetch>
CACHE="${ZIG_GLOBAL_CACHE_DIR:-$HOME/.cache/zig}"
cp vendor/sokol/README.agate.md /tmp/README.agate.md
rm -rf vendor/sokol
mkdir -p vendor/sokol
tar -xzf "$CACHE/p/$HASH.tar.gz" -C vendor/sokol --strip-components=1
cp /tmp/README.agate.md vendor/sokol/README.agate.md

# 3. Re-apply the patches (both idempotent) and confirm they took.
python3 tools/patch_sokol_handle_abi.py
python3 tools/patch_sokol_handle_abi.py --check
python3 tools/patch_sokol_gpu_timings.py
python3 tools/patch_sokol_gpu_timings.py --check

# 4. Record the new upstream commit and package hash at the top of
#    this file, then verify the build.
zig build test
```

## GPU timings patch (frame-level, Metal-only)

Upstream sokol-gfx has **no GPU-timestamp mechanism**: `sg_query_*`
covers features/limits/resources, and `SOKOL_TRACE_HOOKS` installs
CPU-side begin/end callbacks (a `TRACE_HOOKS_NOT_ENABLED` warning means
someone called `sg_install_trace_hooks()` without `-DSOKOL_TRACE_HOOKS`;
those hooks never yield GPU time — do not confuse the two). Per-pass
GPU attribution would require deep surgery, so v1 measures ONE timer
per frame on Metal and leaves GL4.1/per-pass for a follow-up wave.

Applied by `tools/patch_sokol_gpu_timings.py` (idempotent, `--check`
supported; 4 anchored insertions in `src/sokol/c/sokol_gfx.h`, a C
source file that is NOT machine-generated):

- public declarations after `sg_commit`:
  `sg_agate_set_gpu_timing_enabled(bool)` (default OFF) and
  `sg_agate_query_gpu_frame_ms()` (measured ms, or -1);
- file-static state (`_sg_agate_gpu_timing_enabled`,
  retained `id<MTLCommandBuffer> _sg_agate_gpu_cb`, last-completed
  cache) ahead of `_sg_mtl_commit`;
- retain hook in `_sg_mtl_commit`: while enabled, the committed
  command buffer is retained one extra frame (the previous one is
  sampled, then released, so at most one extra buffer is ever alive);
- implementations after `sg_commit`: the query samples the retained
  buffer via `_sg_agate_gpu_sample` and returns the last-completed
  cache (`status` read; on `MTLCommandBufferStatusCompleted` the cache
  refreshes from `(GPUEndTime-GPUStartTime)*1000`, 10 s sanity clamp).
  Sampling the previous buffer at replace time — a full frame after
  its commit — is what makes the value converge (a just-committed
  buffer still reports Committed, not Completed).
  Non-Metal backends (GL/D3D11/dummy/...) always return -1.

Semantics: last-completed (the value lags one CPU submit behind),
context thread only (no atomics: commit hook and query poll run on the
same thread; the Metal runtime publishes the timestamps on completion).
Disabling releases the retained buffer and resets the cache. The
`sg_shutdown` path is untouched: disable timings before shutdown to
release the held buffer (process exit reclaims it otherwise).

Agate side (`src/agate/gpu_timing.zig`, default OFF): `setEnabled` /
`AGATE_GPU_TIMINGS=1` opt-in, fail-closed `pollFrameMs()` called from
`scene/frame_render.zig` right after `sg.commit()`; the value flows
through `SceneStats.gpu_frame_ms` → `FrameRecord.gpu_frame_ms` →
session `avg/max_gpu_frame_ms` → gated HTML/MD/Chrome-trace output
(`gpu_frame_ms`, `GPU Frame (measured)`, `cat "gpu"`).

### v2: per-pass timers + GL timer-query

`sg_agate_gpu_pass_begin/end(int pass)` (pass ids 0=shadow, 1=main,
2=post; invalid ids are no-ops) bracket each engine phase from
`scene/frame_render.zig`, and `sg_agate_query_gpu_pass_ms(int)`
serves the last COMPLETED sample per pass (or -1). Agate side:
`gpu_timing.Pass` + `beginPass`/`endPass`/`pollPassMs`, fail-closed
like the frame poll; `SceneStats.gpu_shadow/main/post_ms` →
`FrameRecord` → `avg/max_gpu_*_ms` → gated HTML/MD/Chrome-trace output
(`GPU Shadow/Main/PostFX (measured)`, per-pass `cat "gpu"` events,
`hasGpuPassData` gate — Metal-only and disabled sessions render
exactly the pre-per-pass output).

- **Metal: frame-only, by construction.** sokol encodes the whole
  frame into ONE `MTLCommandBuffer` (created on the first pass in
  `_sg_mtl_begin_pass`, committed in `_sg_mtl_commit`), so
  `GPUStartTime/GPUEndTime` can only span the full frame. True
  per-encoder times would need one command buffer per pass (extra
  commits + drawable/present and in-flight-semaphore surgery) or
  `MTLCounterSampleBuffer` timestamp sampling at encoder boundaries
  (device-capability-gated, resolve + barrier overhead). Both were
  rejected for v2; per-pass on Metal is a documented follow-up. The
  pass brackets are linked no-ops there and per-pass queries are -1.
- **GL4.1 (`SOKOL_GLCORE`, non-Win32): real per-pass timers.** Each
  pass owns a `GL_TIME_ELAPSED` query pool (depth 4, generated
  lazily); a per-frame drain in `_sg_gl_commit` and every query reap
  retired queries oldest-first WITHOUT stalling (only
  `GL_QUERY_RESULT_AVAILABLE` results are read, ns → ms, 10 s sanity
  clamp). A full ring whose oldest query is still in flight drops the
  new sample instead of stalling. Disable deletes live queries and
  resets the caches. `sg_agate_query_gpu_frame_ms` on GL returns the
  sum of the last-completed per-pass values: serial GPU execution
  makes this a LOWER BOUND of the true frame span (inter-pass bubbles
  excluded), -1 until the first sample. Linux `<GL/gl.h>` stops at GL
  1.x, so the patch declares the six timer-query entry points itself
  (exported by libGL; macOS `<OpenGL/gl3.h>` already declares them).
- **Fail-closed elsewhere:** Win32-GL (the embedded loader has no
  timer entry points), GLES3 (`GL_TIME_ELAPSED_EXT` differs),
  D3D11/WGPU/Vulkan/dummy — stubs, every query -1.

Semantics (both timers): last-completed (values lag behind the CPU
submit), context thread only, never a GPU stall. CPU-submit phase
times keep their explicit `(CPU submit)` labels everywhere — a
CPU-submit value under a GPU name is a bug, not a fallback.

## Upstream

Worth reporting to both:

- **ziglang/zig** — extern struct by value is coerced to `i64` and loaded with an
  8-byte load from a 4-byte object (repro: `alloca %P` + `load i64` for
  `extern struct { id: u32 }`).
- **floooh/sokol-bindgen** — declaring handle parameters as `u32` sidesteps the
  codegen issue entirely while staying ABI-correct.
