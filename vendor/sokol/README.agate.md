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

```sh
rm -rf vendor/sokol
zig fetch --save git+https://github.com/floooh/sokol-zig.git#<commit>   # or copy the package
python3 tools/patch_sokol_handle_abi.py
```

and record the new upstream commit hash here. The script is idempotent, so
running it twice is safe.

## Upstream

Worth reporting to both:

- **ziglang/zig** — extern struct by value is coerced to `i64` and loaded with an
  8-byte load from a 4-byte object (repro: `alloca %P` + `load i64` for
  `extern struct { id: u32 }`).
- **floooh/sokol-bindgen** — declaring handle parameters as `u32` sidesteps the
  codegen issue entirely while staying ABI-correct.
