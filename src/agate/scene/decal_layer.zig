const decal_mod = @import("../mesh/decal.zig");
const DecalManager = decal_mod.DecalManager;

/// Dynamic decal hosting. The DecalManager itself lives in mesh/decal.zig
/// (forbidden-for-this-refactor territory) and needs the owning *Scene for
/// mesh creation, so the layer only holds the optional instance and Scene
/// passes itself into getOrCreate.
pub const DecalLayer = struct {
    manager: ?DecalManager = null,

    pub fn deinit(self: *DecalLayer) void {
        if (self.manager) |*dm| {
            dm.deinit();
            self.manager = null;
        }
    }

    /// Lazily creates the manager (max_decals bounds the pooled decals).
    pub fn getOrCreate(self: *DecalLayer, scene: anytype, max_decals: usize) *DecalManager {
        if (self.manager == null) {
            self.manager = decal_mod.DecalManager.init(scene, max_decals);
        }
        return &self.manager.?;
    }

    pub fn update(self: *DecalLayer, dt: f32) void {
        if (self.manager) |*dm| {
            dm.update(dt);
        }
    }
};
