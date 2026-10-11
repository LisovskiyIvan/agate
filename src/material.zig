//! Material subsystem facade.
//! Re-exports concrete materials, tagged union, PBR layer types, and draw records.

pub const types = @import("material/types.zig");
pub const pbr = @import("material/pbr.zig");
pub const shader_mat = @import("material/shader_mat.zig");
pub const union_mod = @import("material/union.zig");
pub const draw_record = @import("material/draw_record.zig");

// Types & PBR layers
pub const AlphaMode = types.AlphaMode;
pub const Channel = types.Channel;
pub const UvTransform = types.UvTransform;
pub const CoatParams = types.CoatParams;

// Concrete Materials
pub const PBRMaterial = pbr.PBRMaterial;
pub const ShaderMaterial = shader_mat.ShaderMaterial;

// Material Union
pub const Material = union_mod.Material;
pub const coatParamsFor = union_mod.coatParamsFor;

// Draw Records
pub const MaterialDrawRecord = draw_record.MaterialDrawRecord;
pub const ShaderDrawSnapshot = draw_record.ShaderDrawSnapshot;
pub const buildShaderSnapshot = draw_record.buildShaderSnapshot;
pub const buildDrawRecord = draw_record.buildDrawRecord;
