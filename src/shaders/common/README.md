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
| `uv_apply.glsl` | 6 | same 5 (UV0/UV1 selection + transform) |
| `pbr_brdf.glsl` | 115 | `pbr`, `instanced_pbr`, `skinned_pbr` (3) |
| `channel_select.glsl` | 6 | same 3 |
| `refraction.glsl` | 25 | same 3 (screen-space refraction) |
| `hemi.glsl` | 24 | same 5 (`standard`, `pbr`, `instanced`, `instanced_pbr`, `skinned_pbr`) |
| `frame_view.glsl` | 57 | `pbr`, `instanced_pbr`, `skinned_pbr` fs view-global block (legacy slot per shader) |
| `frame_mat.glsl` | 46 | same 3 (per-material/object block, UB 5) |
| `pbr_fs_bindings.glsl` | 79 | same 3 (fs texture/sampler view bindings, merged slot-pool docs) |
| `pbr_shadow_consts.glsl` | 91 | same 3 (Poisson disk, cascade offsets, PCSS budgets) |
| `morph_helpers.glsl` | 53 | `pbr`, `skinned_pbr` vs (`morphTexel`/`morphWeight`/`applyMorphDeltas`) |
| `ssr_trace.glsl` | 33 | `postprocess`, `ssr` fs (depth-derived world pos + surface normal for the SSR raymarch) |
| `pbr_surface.glsl` | 521 | `skinned_pbr`, `instanced_pbr` (post-unlit surface/lights/IBL/emissive tail; `pbr` keeps its hook-scoped copy) |
| `wboit.glsl` | 13 | `wboit` fs (depth weighting and accumulation output) |

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

- `fs_params` uniform blocks: the three PBR variants share theirs via
  `frame_view`/`frame_mat` (pinned by the `fs_view/fs_mat parity` drift
  test in `../shader_material/include_tests.zig`); any reorder shifts
  every uniform offset, so new lanes go through the shared chunks.
- Morph vertex BLOCKS (`vs_morph` declarations: UB 2 in pbr vs UB 3 in
  skinned_pbr) and the skin variant (`vs_skin`, joints/weights
  attributes): different slots by construction, stay per-shader. Only the
  pure helpers (`morphTexel`/`morphWeight`/`applyMorphDeltas`) are shared
  (common/morph_helpers.glsl); arity is pinned by the `morph helpers live
  once in common` drift test.
- `shadow.glsl` depth-only programs (rigid/instanced/skinned entry
  points, position-only vertex input). Pinned by the `shadow depth-only
  programs` drift test.
- `probe_mip.glsl` fullscreen vs: NO Y-flip (`v_uv = texcoord0`
  unconditionally) — excluded from `fullscreen_vs.glsl` on purpose.
- `particle.glsl`, `outline.glsl`, `skybox/debug/ui/ui3d_panel.glsl`,
  `particle_compute.glsl` (`@cs`), sandbox `ext_scroll.glsl`: unique
  vertex stages or standalone external contracts.
- Snippet files under `src/shader_material/snippets/`: hook bodies by
  design, merged — never included.
