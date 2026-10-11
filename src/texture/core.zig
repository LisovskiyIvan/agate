//! Facade for the 2D texture core. `core.zig` owned the `Texture` type until
//! the by-concern split; it now only re-exports the public API so all callers
//! (`texture.zig`, `render_target.zig`, `texture/cube.zig`, the `*_tests` suites)
//! compile unchanged.
//!
//! Layout: `core_texture.zig` (the `Texture` owner: fields, lifecycle, GPU
//! upload entries, leaf re-exports), `core_sampler.zig` (Options + sampler
//! helpers), `core_raw.zig` (RawTexture + mip-chain builders),
//! `core_block.zig` (BlockSupport + format mapping), `core_decode.zig` (LDR
//! decode routing), `core_hdr.zig` (HDR decode), `core_testpng.zig` (TestPng
//! fixture). Leaves never import this facade back.

const texture_mod = @import("core_texture.zig");
const testpng_mod = @import("core_testpng.zig");

pub const Texture = texture_mod.Texture;
pub const TestPng = testpng_mod.TestPng;
