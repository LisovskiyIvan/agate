# Shared GLSL chunks (`// @include` prepass)

sokol-shdc has no `#include` support (verified: a `#include` line fails
in glslang with `'#include' : required extension not requested`, and the
vendored binary exposes no `-I` flag). Sharing is implemented as a
build-time textual prepass instead: `// @include "common/<file>"`
directives are expanded verbatim by the `expand_shader_includes` host
tool (see `../shader_material/include.zig`) before shdc runs. Wiring:
`agate/build.zig` → `createShaderWithIncludes` (engine shaders) and the
merge→expand→shdc chain in `createShaderMaterialRegistry` (hook
materials).

## Chunks and consumers

| Chunk | Lines | Consumers |
|---|---|---|
| `fullscreen_vs.glsl` | 13 | `bloom_down/up`, `glow_blur/extract`, `ssao`, `ssao_blur`, `volumetric_blur/raymarch`, `postprocess` (9) |
| `cluster.glsl` | 19 | `standard`, `pbr`, `instanced`, `instanced_pbr`, `skinned_pbr` (5) |
| `shadow_pcf.glsl` | 248 | same 5 (`hash01` → `areaLightFactor`, incl. rect area-light comment) |
| `uv_apply.glsl` | 3 | same 5 |
| `pbr_brdf.glsl` | 103 | `pbr`, `instanced_pbr`, `skinned_pbr` (3) |
| `channel_select.glsl` | 6 | same 3 |

Rules: chunk files are pure code spans with NO provenance header (any
header would break the byte-identical expansion guarantee — expansion
of a converted shader reproduces the pre-refactor source exactly, so
shdc output is unchanged). Document sharing HERE, not in the chunks.

## Error line numbers

No `#line` directives are emitted (pure substitution). A shdc error
*after* a directive reports the EXPANDED line:
`expanded = authored + Σ(include_lines − 1)` per preceding directive.
Measured: 13-line `fullscreen_vs` include shifts later lines by +12.
Includes are small versioned files; map back by inspection.

## Deliberately NOT shared (do not "fix" without review)

- `fs_params` uniform blocks (5 forward shaders): same field order by
  convention, but probe-lane comments/semantics differ and any reorder
  shifts every uniform offset. Pinned by the `fs_params probe lane
  parity` drift test in `../shader_material/include.zig`.
- Morph vertex helpers (`morphTexel`/`morphWeight`/`applyMorphDeltas`):
  arity differs (`tan_xyz` only in PBR variants). Pinned by the
  `morph arity` drift test.
- `shadow.glsl` vs `msaa_depth.glsl` (depth-only twins): must rasterize
  identically but upload different matrices (light vs camera) and
  different bias. Pinned by the `depth twin programs` drift test.
- `probe_mip.glsl` fullscreen vs: NO Y-flip (`v_uv = texcoord0`
  unconditionally) — excluded from `fullscreen_vs.glsl` on purpose.
- `particle.glsl`, `outline.glsl`, `skybox/debug/ui/ui3d_panel.glsl`,
  `particle_compute.glsl` (`@cs`), sandbox `ext_scroll.glsl`: unique
  vertex stages or standalone external contracts.
- Snippet files under `examples/shader_materials/`: hook bodies by
  design, merged — never included.
