//! Facade for the texture modules. The `texture.zig` texture set was split
//! into focused leaves under `texture/` following the repo pattern (one
//! self-contained type per leaf plus a dispatcher; Zig 0.16 has no
//! usingnamespace; see `camera.zig`, `profiler.zig`,
//! `scene/render_queue.zig`, `ui.zig`):
//!
//! - `texture/color.zig` — sRGB conversion (`srgbToLinearU8`,
//!   `convertSrgbToLinearInPlace`), the particle-dot falloff, and the
//!   half-float bit helpers. Leaf: imports `std` only.
//! - `texture/mip.zig` — downsample/size helpers (`boxDownsampleU8`,
//!   `boxDownsampleF16`, `checkedFaceBytes`, `pixelFormatBytes`). Imports
//!   `color` only.
//! - `texture/core.zig` — owns the `Texture` type: 2D upload paths, the CPU
//!   decode/routing entries (`decodeMemory`/`decodeImageMemory`/HDR/DDS,
//!   incl. the KTX2/DDS/EXR readers, which stay top-level modules), the raw
//!   pixel containers, and thin forwarders to the helpers above. Imports
//!   `color` + `mip` (+ `ktx2`/`dds`/`exr`, as before the split).
//! - `texture/cube.zig` — owns `SkyboxOptions` and the `CubeTexture` type
//!   (faces, equirectangular + HDR converters, procedural skybox). Imports
//!   `core` (for `Texture.mipLevelCount` and the half-float conversions,
//!   call sites unchanged) + `mip`.
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`scene.zig`, `scene/*`, `passes/*`, `assets.zig`,
//! `material.zig`, `ktx2.zig`, `root.zig`) see the same API as when
//! everything lived in this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. Cross-leaf helpers (`color.*`, `mip.*`, `core.TestPng`) are
//! `pub` in their home module for the sibling that needs them but are
//! deliberately NOT re-exported here, so the public surface is identical to
//! the pre-split file.
const color = @import("texture/color.zig");
const mip = @import("texture/mip.zig");
const core = @import("texture/core.zig");
const cube = @import("texture/cube.zig");

// 2D texture (lives in texture/core.zig).
pub const Texture = core.Texture;
pub const TextureColorSpace = core.TextureColorSpace;
pub const TextureSlot = core.TextureSlot;

// Cube texture and its gradient options (live in texture/cube.zig).
pub const CubeTexture = cube.CubeTexture;
pub const SkyboxOptions = cube.SkyboxOptions;

// Exact per-byte sRGB -> linear conversion (lives in texture/color.zig).
pub const srgbToLinearU8 = color.srgbToLinearU8;

// Approximate bytes per pixel for a Sokol pixel format (lives in
// texture/mip.zig).
pub const pixelFormatBytes = mip.pixelFormatBytes;
