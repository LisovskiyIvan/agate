pub const ShadowPass = @import("shadow_pass.zig").ShadowPass;
pub const SpotShadowRenderInfo = @import("shadow_pass.zig").SpotShadowRenderInfo;
pub const SkyboxPass = @import("skybox_pass.zig").SkyboxPass;
pub const ParticlePass = @import("particle_pass.zig").ParticlePass;
pub const PostProcessPass = @import("postprocess_pass.zig").PostProcessPass;
pub const BloomPass = @import("bloom_pass.zig").BloomPass;
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
