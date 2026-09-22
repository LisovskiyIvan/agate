pub const ShadowPass = @import("shadow_pass.zig").ShadowPass;
pub const SpotShadowRenderInfo = @import("shadow_pass.zig").SpotShadowRenderInfo;
pub const PointShadowRenderInfo = @import("shadow_pass.zig").PointShadowRenderInfo;
pub const POINT_SHADOW_SLOTS = @import("shadow_pass.zig").POINT_SHADOW_SLOTS;
pub const POINT_SHADOW_FACES = @import("shadow_pass.zig").POINT_SHADOW_FACES;
pub const POINT_SHADOW_RES = @import("shadow_pass.zig").POINT_SHADOW_RES;
pub const POINT_SHADOW_MAP_WIDTH = @import("shadow_pass.zig").POINT_SHADOW_MAP_WIDTH;
pub const POINT_SHADOW_MAP_HEIGHT = @import("shadow_pass.zig").POINT_SHADOW_MAP_HEIGHT;
pub const pointFaceForDir = @import("shadow_pass.zig").pointFaceForDir;
pub const pointTileOrigin = @import("shadow_pass.zig").pointTileOrigin;
pub const SkyboxPass = @import("skybox_pass.zig").SkyboxPass;
pub const ParticlePass = @import("particle_pass.zig").ParticlePass;
pub const PostProcessPass = @import("postprocess_pass.zig").PostProcessPass;
pub const BloomPass = @import("bloom_pass.zig").BloomPass;
pub const GlowPass = @import("glow_pass.zig").GlowPass;
pub const HighlightPass = @import("highlight_pass.zig").HighlightPass;
pub const HighlightDrawItem = @import("highlight_pass.zig").HighlightDrawItem;
pub const HighlightResult = @import("highlight_pass.zig").HighlightResult;
pub const makeHighlightDrawItem = @import("highlight_pass.zig").makeHighlightDrawItem;
pub const highlight_pass = @import("highlight_pass.zig");
pub const OutlinePass = @import("outline_pass.zig").OutlinePass;
pub const OutlineDrawItem = @import("outline_pass.zig").OutlineDrawItem;
// P4: makeOutlineDrawItem/OutlinePass.renderItems/OutlineDrawItem несут
// render-owned индексы вместо живых указателей — осознанное изменение
// low-level API хелперов (без врапперов совместимости); стабильные точки:
// Scene.prepareFrame/render и immediate OutlinePass.render(meshes).
pub const makeOutlineDrawItem = @import("outline_pass.zig").makeOutlineDrawItem;
pub const SSAOPass = @import("ssao_pass.zig").SSAOPass;
pub const outline_pass = @import("outline_pass.zig");
pub const DebugPass = @import("debug_pass.zig").DebugPass;
