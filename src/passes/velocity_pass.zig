//! Velocity buffer pass for TAA and motion blur.
//!
//! Renders opaque rigid, skinned, and instanced meshes with current vs
//! previous transforms to generate screen-space velocity vectors
//! (uv_current - uv_previous) into an RGBA16F buffer.
//!
//! Geometry skip matrix matches itemContributesDepth: transparent, decal,
//! and hook-material meshes are skipped.
//!
//! Pipeline factory + destroy live in `velocity/pipelines.zig` (mechanical
//! leaf, `anytype` pass-through); this file owns the `VelocityPass` type,
//! targets, draws, and pure helpers with thin private forwarders.
//!
//! Fail-closed headless: all sg calls guarded, clean stats.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const velocity_shd = @import("velocity_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const mesh_mod = @import("../mesh.zig");
const Vertex = mesh_mod.Vertex;
const scene_render_queue = @import("../scene/render_queue.zig");
const RenderQueues = scene_render_queue.RenderQueues;
const stats_mod = @import("../scene/stats.zig");
const SceneStats = stats_mod.SceneStats;

/// Pure skip-matrix for depth-contributing draws (unit-testable without a
/// GPU): transparent and decal meshes write no depth in the main pass
/// either, and hook-material meshes need their custom vertex stage.
pub fn itemContributesDepth(transparent: bool, is_decal: bool, has_hook: bool) bool {
    return !transparent and !is_decal and !has_hook;
}
const material_mod = @import("../material.zig");
const MaterialDrawRecord = material_mod.MaterialDrawRecord;
const particle_pass_mod = @import("particle_pass.zig");
const ParticleDraw = particle_pass_mod.ParticlePass.ParticleDraw;
const particles_mod = @import("../particles.zig");
const taa_mod = @import("../postprocess/taa.zig");
const pipelines = @import("velocity/pipelines.zig");
const gpu_thread = @import("../gpu_thread.zig");

/// Packs the FS alpha-test uniforms from a frozen draw record (the actual
/// mapping the draws use — never hand-rolled per call): slot-0 albedo UV
/// transform, base color factor, and the cutout threshold. Opaque/blend
/// records carry cutoff 0.0, which keeps the shader's no-discard branch.
pub fn velocityAlphaUniforms(rec: MaterialDrawRecord) velocity_shd.FsAlpha {
    return .{
        .base_color_factor = rec.base_color,
        .uv_matrix = rec.uv_matrices[0],
        .uv_offset = rec.uv_offsets[0],
        .alpha_cutoff = rec.alpha_cutoff,
    };
}

/// Pure velocity-dispatch decision (no sg calls): TAA always needs the
/// buffer when it runs (taa_on already implies samples == 1); motion blur
/// needs it only at 1x. Under MSAA the velocity pass is explicitly
/// suppressed even with the depth prepass — the blur then uses camera-only
/// depth reprojection against the prepass depth, so no false per-object
/// vectors can come from a sample-count-mismatched target.
pub fn velocityNeeded(taa_on: bool, motion_blur_enabled: bool, samples: i32) bool {
    if (taa_on) return true;
    if (!motion_blur_enabled) return false;
    return samples == 1;
}

/// Cut-frame suppression (pure): a TAA reset (first frame, toggle, resize,
/// camera cut, explicit reset) or a camera cut under blur-only mode must
/// bind the zero mask and skip the blur gather for exactly that composite.
pub fn cutSuppressesVelocity(taa_reset: bool, motion_blur_enabled: bool, camera_cut: bool) bool {
    return taa_reset or (motion_blur_enabled and camera_cut);
}

/// Generation gate for frozen prev state (pure, no sg): a staged prev
/// payload is usable only when its generation equals the last actually
/// rendered generation. Anything staged-but-never-presented (latch without
/// render, superseded front, reused front replaying old motion) fails the
/// match and the draw falls back to zero motion. `maxInt` (never staged)
/// never matches: frame ids count up from 0 and never reach it.
pub fn usePresentedPrev(prev_frame: u64, last_rendered: u64) bool {
    if (prev_frame == std.math.maxInt(u64)) return false;
    return prev_frame == last_rendered;
}

/// Shared-depth pipeline state (pure plan): depth writes OFF with an EQUAL
/// compare against the borrowed main depth. EQUAL binds actual visible main
/// pixels — nearer morph-displaced surfaces and nearer opaque geometry
/// occlude farther velocity draws, and cutout holes (main depth holds the
/// farther background there) fail the match instead of writing false mask1.
/// The remaining coplanar-cutout tie is resolved in-shader: the FS replicates
/// the main pass's alpha-cutoff discard (see velocity.glsl), so discarded
/// fragments never reach the depth test at all.
pub fn velocityDepthState() sg.DepthState {
    return .{
        .compare = .EQUAL,
        .write_enabled = false,
    };
}

/// Shared-depth pass action (pure plan): own color target cleared to the
/// zero mask; borrowed main depth LOADED and STORED (never cleared, never
/// written — the post chain keeps sampling it afterwards).
pub fn velocityDepthPassAction() sg.PassAction {
    var a = sg.PassAction{};
    a.colors[0] = .{
        .load_action = .CLEAR,
        .clear_value = .{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 0.0 },
        .store_action = .STORE,
    };
    a.depth = .{
        .load_action = .LOAD,
        .store_action = .STORE,
    };
    return a;
}

/// Reactive-marker depth state (pure plan, C.3 v2): depth writes OFF like
/// the opaque velocity path, but LESS_EQUAL instead of EQUAL. Marker content
/// (particles) writes NO main depth, so its fragments sit IN FRONT of the
/// stored background depth and would fail the EQUAL match; LESS_EQUAL lets
/// exactly the visible coverage through while still rejecting occlusion by
/// nearer opaque pixels. Per-pipeline state, so opaque EQUAL draws and
/// marker draws share one pass over the same borrowed depth attachment.
pub fn reactiveMarkerDepthState() sg.DepthState {
    return .{
        .compare = .LESS_EQUAL,
        .write_enabled = false,
    };
}

/// Per-draw marker decision (pure, no sg calls): a prepared particle record
/// contributes a marker draw iff it has live instances AND a bound instance
/// buffer. The sg-side render rechecks handle validity (a zero/dead buffer
/// id fails closed to no draw); this helper is the headless-testable shape
/// of the same predicate.
pub fn particleMarkerDrawDecision(active_count: usize, has_instance_buffer: bool) bool {
    return active_count > 0 and has_instance_buffer;
}

/// Glass reactive-marker draw decision (pure, no sg calls): the inverse of
/// velocityDrawDecision for the transparent queue. Transparent items carry
/// no velocity (skipped upstream by itemContributesDepth), so they need a
/// marker iff the main pass actually blended them AND the marker vertex
/// stage can reproduce their coverage:
/// - opaque items need none (the opaque EQUAL path already wrote velocity);
/// - decals ride their receiver's velocity (the opaque path wrote the
///   receiver's vector at that pixel — stamping reactive would shed valid
///   history), so they stay on the v1 heuristic;
/// - hook materials may displace vertices in ways the marker VS cannot
///   reproduce (custom shader stage), so they stay on v1;
/// - morph-displaced draws have no morph path here (same as the opaque
///   velocity path), so they stay on v1 (mask 0 fallback).
/// Skinned vs rigid vs instanced is a pipeline selection, not a decision:
/// all three shapes draw when this returns true (skin/instance-buffer
/// validity is rechecked sg-side, fail-closed like the particle path).
pub fn glassMarkerDrawDecision(in: VelocityDrawInput) bool {
    if (in.index_count == 0) return false;
    if (!in.transparent) return false;
    if (in.is_decal) return false;
    if (in.has_hook) return false;
    if (in.morph_fallback) return false;
    return true;
}

/// Marker-draw arguments for VelocityPass.render: prepared particle records
/// (borrowed values from the consumed front slot, same records the main
/// pass drew), the camera basis for billboarding, and the master enable.
/// The same flag gates BOTH marker families (particles + transparent
/// glass): the transparent draws ride the already-passed queues object, so
/// no extra payload is needed. Disabled (or empty) renders zero marker
/// draws and leaves the velocity buffer bit-identical to the pre-marker
/// path.
pub const MarkerArgs = struct {
    enabled: bool = false,
    draws: []const ParticleDraw = &.{},
    cam_right: Vec3 = Vec3.zero,
    cam_up: Vec3 = Vec3.zero,
};

/// Velocity geometry shape for pipeline selection (opaque and glass
/// marker paths share the same three shapes).
pub const VelocityShape = enum { rigid, instanced, skinned };

// Write/read contract with the TAA resolve (taaVelocityKind in
// postprocess/taa.zig): the stamped band value must sit strictly inside the
// classify-reactive band and below the per-object gate, or marked pixels
// would misclassify on the resolve side. Fails at compile time, not as a
// silent ghosting regression. One stamped value serves both marker families
// (particles and glass share fs_marker), so this single pin covers every
// marker write site.
comptime {
    if (!(taa_mod.TAA_REACTIVE_MARKER_LO < taa_mod.TAA_REACTIVE_MARKER_ALPHA and
        taa_mod.TAA_REACTIVE_MARKER_ALPHA < taa_mod.TAA_REACTIVE_MARKER_HI)) @compileError("marker alpha outside the reactive band");
    if (!(taa_mod.TAA_REACTIVE_MARKER_HI < 0.5)) @compileError("marker band collides with the per-object gate");
}

pub const VelocityPass = struct {
    image: sg.Image = .{},
    att_view: sg.View = .{},
    tex_view: sg.View = .{},
    dummy_zero_image: sg.Image = .{},
    dummy_zero_view: sg.View = .{},
    /// Fallback albedo sampler for hand-built records with a null sampler
    /// (production records always carry the texture's own sampler, bound
    /// verbatim like the main pass). Linear clamp, color-sampling shape.
    albedo_fallback_sampler: sg.Sampler = .{},

    pip_rigid_u16: sg.Pipeline = .{},
    pip_rigid_u32: sg.Pipeline = .{},
    pip_rigid_u16_ds: sg.Pipeline = .{},
    pip_rigid_u32_ds: sg.Pipeline = .{},
    pip_inst_u16: sg.Pipeline = .{},
    pip_inst_u32: sg.Pipeline = .{},
    pip_inst_u16_ds: sg.Pipeline = .{},
    pip_inst_u32_ds: sg.Pipeline = .{},
    pip_skin_u16: sg.Pipeline = .{},
    pip_skin_u32: sg.Pipeline = .{},
    pip_skin_u16_ds: sg.Pipeline = .{},
    pip_skin_u32_ds: sg.Pipeline = .{},
    /// Reactive-marker pipelines (C.3 v2): particle billboard coverage with
    /// a constant band-alpha FS (LESS_EQUAL against the borrowed main
    /// depth). CPU layout serves .cpu AND .compute records (both ride the
    /// ParticleInstanceData stride via drawBuffer()); the GPU layout serves
    /// .gpu spawn slots. Single UINT16 index type (unit-quad ib, like the
    /// main particle pipelines).
    pip_marker_cpu: sg.Pipeline = .{},
    pip_marker_gpu: sg.Pipeline = .{},
    /// Glass reactive-marker pipelines (C.3 v2 glass half): transparent
    /// redraw coverage with the constant band-alpha FS (LESS_EQUAL against
    /// the borrowed main depth — glass writes no depth, so its fragments
    /// sit IN FRONT of the stored background depth exactly like particle
    /// billboards). Rigid/skinned share the Vertex-stride buffer 0 (the
    /// marker VS reads position only for rigid, position+joints+weights
    /// for skinned); instanced adds the single CURRENT instance buffer
    /// (no prev pairing — coverage needs no history, which also halves the
    /// per-batch binding vs the opaque velocity path). Sidedness mirrors
    /// the opaque path and the main blend twins: single-sided panes keep
    /// BACK cull (a pane seen from behind draws nothing in the main pass,
    /// so the marker must draw nothing either — cull NONE would stamp
    /// reactive on background pixels); double-sided glass uses the cull-off
    /// `_ds` twins so backfaces mark exactly the pixels the main pass
    /// shaded. Single cost note: one extra geometry pass over the
    /// transparent queue per TAA frame (vertex work only — constant FS, no
    /// texture taps, no loops).
    pip_marker_glass_rigid_u16: sg.Pipeline = .{},
    pip_marker_glass_rigid_u32: sg.Pipeline = .{},
    pip_marker_glass_rigid_u16_ds: sg.Pipeline = .{},
    pip_marker_glass_rigid_u32_ds: sg.Pipeline = .{},
    pip_marker_glass_inst_u16: sg.Pipeline = .{},
    pip_marker_glass_inst_u32: sg.Pipeline = .{},
    pip_marker_glass_inst_u16_ds: sg.Pipeline = .{},
    pip_marker_glass_inst_u32_ds: sg.Pipeline = .{},
    pip_marker_glass_skin_u16: sg.Pipeline = .{},
    pip_marker_glass_skin_u32: sg.Pipeline = .{},
    pip_marker_glass_skin_u16_ds: sg.Pipeline = .{},
    pip_marker_glass_skin_u32_ds: sg.Pipeline = .{},
    /// Marker shader (reactive_marker + reactive_marker_gpu programs).
    marker_shader: sg.Shader = .{},
    marker_gpu_shader: sg.Shader = .{},
    /// Glass marker shaders (glass_marker + glass_marker_inst +
    /// glass_marker_skin programs, all sharing fs_marker).
    glass_marker_shader: sg.Shader = .{},
    glass_marker_inst_shader: sg.Shader = .{},
    glass_marker_skin_shader: sg.Shader = .{},
    /// Unit-quad geometry for marker billboards (same shape as the particle
    /// pass quad; owned here so marker draws never borrow pass state).
    marker_quad_vb: sg.Buffer = .{},
    marker_quad_ib: sg.Buffer = .{},

    rigid_shader: sg.Shader = .{},
    inst_shader: sg.Shader = .{},
    skin_shader: sg.Shader = .{},

    width: i32 = 0,
    height: i32 = 0,
    /// Main-depth pixel format the pipelines were built for (.DEFAULT =
    /// none yet). The pass borrows the main depth attachment (never owns
    /// or destroys it); pipelines are (re)built lazily to match it.
    depth_format: sg.PixelFormat = .DEFAULT,

    pub fn init() VelocityPass {
        if (!sg.isvalid()) return .{};
        const backend = sg.queryBackend();

        // Store-first rollback throughout: every handle lands in `out`
        // before its state check, so a single `deinit` frees the failed
        // handle plus all partial handles (a nonzero FAILED id still owns
        // its pool slot and must be released, never leaked).
        var out: VelocityPass = .{};

        // 1x1 zero dummy texture (clear velocity fallback)
        const zero_pixels = [_]f16{ 0.0, 0.0, 0.0, 0.0 };
        var zero_data = sg.ImageData{};
        zero_data.mip_levels[0] = sg.asRange(&zero_pixels);
        out.dummy_zero_image = sg.makeImage(.{
            .width = 1,
            .height = 1,
            .pixel_format = .RGBA16F,
            .data = zero_data,
        });
        if (sg.queryImageState(out.dummy_zero_image) != .VALID) {
            out.deinit();
            return .{};
        }
        out.dummy_zero_view = sg.makeView(.{
            .texture = .{ .image = out.dummy_zero_image },
        });
        if (sg.queryViewState(out.dummy_zero_view) != .VALID) {
            out.deinit();
            return .{};
        }
        out.albedo_fallback_sampler = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });
        if (sg.querySamplerState(out.albedo_fallback_sampler) != .VALID) {
            out.deinit();
            return .{};
        }

        out.rigid_shader = sg.makeShader(velocity_shd.velocityShaderDesc(backend));
        if (sg.queryShaderState(out.rigid_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.inst_shader = sg.makeShader(velocity_shd.velocityInstancedShaderDesc(backend));
        if (sg.queryShaderState(out.inst_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.skin_shader = sg.makeShader(velocity_shd.velocitySkinnedShaderDesc(backend));
        if (sg.queryShaderState(out.skin_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.marker_shader = sg.makeShader(velocity_shd.reactiveMarkerShaderDesc(backend));
        if (sg.queryShaderState(out.marker_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.marker_gpu_shader = sg.makeShader(velocity_shd.reactiveMarkerGpuShaderDesc(backend));
        if (sg.queryShaderState(out.marker_gpu_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.glass_marker_shader = sg.makeShader(velocity_shd.glassMarkerShaderDesc(backend));
        if (sg.queryShaderState(out.glass_marker_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.glass_marker_inst_shader = sg.makeShader(velocity_shd.glassMarkerInstShaderDesc(backend));
        if (sg.queryShaderState(out.glass_marker_inst_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        out.glass_marker_skin_shader = sg.makeShader(velocity_shd.glassMarkerSkinShaderDesc(backend));
        if (sg.queryShaderState(out.glass_marker_skin_shader) != .VALID) {
            out.deinit();
            return .{};
        }
        // The marker quad is format-free static geometry, so it is created
        // now under the same rollback (pipelines stay unbuilt: their depth
        // format must match the borrowed main depth attachment, which is
        // only known at render time — see ensurePipelinesFor).
        const marker_quad_vertices = [_]f32{
            // x,     y,     u,   v (uv rides unconsumed in the stride)
            -0.5, -0.5, 0.0, 0.0,
            0.5,  -0.5, 1.0, 0.0,
            0.5,  0.5,  1.0, 1.0,
            -0.5, 0.5,  0.0, 1.0,
        };
        const marker_quad_indices = [_]u16{ 0, 1, 2, 0, 2, 3 };
        out.marker_quad_vb = sg.makeBuffer(.{ .data = sg.asRange(&marker_quad_vertices) });
        if (sg.queryBufferState(out.marker_quad_vb) != .VALID) {
            out.deinit();
            return .{};
        }
        out.marker_quad_ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&marker_quad_indices),
        });
        if (sg.queryBufferState(out.marker_quad_ib) != .VALID) {
            out.deinit();
            return .{};
        }
        return out;
    }

    /// Builds (or rebuilds on format change) the twelve opaque pipelines,
    /// the two particle reactive-marker pipelines, plus the twelve glass
    /// reactive-marker pipelines for the borrowed main-depth format.
    /// Body lives in `velocity/pipelines.zig` (mechanical move — same build,
    /// validate, rollback, destroy, and publish ordering); this forwarder
    /// keeps the private owner signature and all call sites unchanged.
    fn ensurePipelinesFor(self: *VelocityPass, depth_fmt: sg.PixelFormat) bool {
        return pipelines.ensurePipelinesFor(self, depth_fmt);
    }

    /// Destroys all twenty-six pipelines and resets the depth format.
    /// Body lives in `velocity/pipelines.zig`; same guard/reset ordering.
    fn destroyPipelines(self: *VelocityPass) void {
        pipelines.destroyPipelines(self);
    }

    pub fn deinit(self: *VelocityPass) void {
        self.destroyTargets();
        if (self.dummy_zero_view.id != 0) sg.destroyView(self.dummy_zero_view);
        if (self.dummy_zero_image.id != 0) sg.destroyImage(self.dummy_zero_image);
        if (self.albedo_fallback_sampler.id != 0) sg.destroySampler(self.albedo_fallback_sampler);
        self.destroyPipelines();
        if (self.rigid_shader.id != 0) sg.destroyShader(self.rigid_shader);
        if (self.inst_shader.id != 0) sg.destroyShader(self.inst_shader);
        if (self.skin_shader.id != 0) sg.destroyShader(self.skin_shader);
        if (self.marker_shader.id != 0) sg.destroyShader(self.marker_shader);
        if (self.marker_gpu_shader.id != 0) sg.destroyShader(self.marker_gpu_shader);
        if (self.glass_marker_shader.id != 0) sg.destroyShader(self.glass_marker_shader);
        if (self.glass_marker_inst_shader.id != 0) sg.destroyShader(self.glass_marker_inst_shader);
        if (self.glass_marker_skin_shader.id != 0) sg.destroyShader(self.glass_marker_skin_shader);
        self.marker_shader = .{};
        self.marker_gpu_shader = .{};
        self.glass_marker_shader = .{};
        self.glass_marker_inst_shader = .{};
        self.glass_marker_skin_shader = .{};
        if (self.marker_quad_ib.id != 0) sg.destroyBuffer(self.marker_quad_ib);
        if (self.marker_quad_vb.id != 0) sg.destroyBuffer(self.marker_quad_vb);
        self.marker_quad_ib = .{};
        self.marker_quad_vb = .{};
        self.* = .{};
    }

    pub fn destroyTargets(self: *VelocityPass) void {
        // Own color target only. The main depth attachment is borrowed per
        // render (never owned, never destroyed here).
        if (self.tex_view.id != 0) sg.destroyView(self.tex_view);
        if (self.att_view.id != 0) sg.destroyView(self.att_view);
        if (self.image.id != 0) sg.destroyImage(self.image);
        self.image = .{};
        self.att_view = .{};
        self.tex_view = .{};
        self.width = 0;
        self.height = 0;
    }

    pub fn ensure(self: *VelocityPass, width: i32, height: i32) bool {
        if (width <= 0 or height <= 0) return false;
        if (!sg.isvalid()) return false;
        if (self.width == width and self.height == height and self.image.id != 0) {
            // Match on the PUBLISHED size, but validate the whole shape,
            // not just the image: views can fail independently of it, and a
            // stale size with dead views must never read as success (the
            // caller gates the render and the composite view on this bool).
            if (sg.queryImageState(self.image) != .VALID or
                sg.queryViewState(self.att_view) != .VALID or
                sg.queryViewState(self.tex_view) != .VALID)
            {
                self.destroyTargets();
                return false;
            }
            return true;
        }

        // Build the color target into locals and validate every handle
        // (image AND views) before touching self: any failure destroys the
        // partial locals and preserves the old valid target. The depth side
        // is borrowed per render from the main target (no private depth).
        // Never publish width/height for a half-created target.
        const img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = .RGBA16F,
            .sample_count = 1,
        });
        if (sg.queryImageState(img) != .VALID) {
            if (img.id != 0) sg.destroyImage(img);
            return false;
        }
        const att_view = sg.makeView(.{
            .color_attachment = .{ .image = img },
        });
        const tex_view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        if (!gpu_thread.createdViewIsValid(att_view) or !gpu_thread.createdViewIsValid(tex_view)) {
            if (tex_view.id != 0) sg.destroyView(tex_view);
            if (att_view.id != 0) sg.destroyView(att_view);
            sg.destroyImage(img);
            return false;
        }

        self.destroyTargets();
        self.image = img;
        self.att_view = att_view;
        self.tex_view = tex_view;
        self.width = width;
        self.height = height;
        return true;
    }

    pub fn velocityTexView(self: *const VelocityPass) sg.View {
        if (self.tex_view.id != 0) return self.tex_view;
        return self.dummy_zero_view;
    }

    /// Opaque pipeline for a (shape, index type, sidedness) triple: a pure
    /// handle read from the ensure-built set, no sg calls. Single-sided
    /// (the default) returns the pre-sidedness handles; double-sided
    /// returns the cull-off `_ds` twins so backfaces emit the velocity the
    /// main pass shaded. Decals never reach selection (skipped upstream
    /// despite their forced double_sided flag). An unbuilt handle fails
    /// closed at the draw's single VALID check — never a silent fallback
    /// to the wrong culling.
    pub fn opaquePipeline(self: *const VelocityPass, shape: VelocityShape, is_u32: bool, double_sided: bool) sg.Pipeline {
        return switch (shape) {
            .rigid => if (double_sided)
                (if (is_u32) self.pip_rigid_u32_ds else self.pip_rigid_u16_ds)
            else
                (if (is_u32) self.pip_rigid_u32 else self.pip_rigid_u16),
            .instanced => if (double_sided)
                (if (is_u32) self.pip_inst_u32_ds else self.pip_inst_u16_ds)
            else
                (if (is_u32) self.pip_inst_u32 else self.pip_inst_u16),
            .skinned => if (double_sided)
                (if (is_u32) self.pip_skin_u32_ds else self.pip_skin_u16_ds)
            else
                (if (is_u32) self.pip_skin_u32 else self.pip_skin_u16),
        };
    }

    /// Glass-marker pipeline for a (shape, index type, sidedness) triple:
    /// same contract as opaquePipeline (pure handle read, single-sided
    /// default unchanged, double-sided cull-off twins, fail-closed on
    /// unbuilt).
    pub fn glassPipeline(self: *const VelocityPass, shape: VelocityShape, is_u32: bool, double_sided: bool) sg.Pipeline {
        return switch (shape) {
            .rigid => if (double_sided)
                (if (is_u32) self.pip_marker_glass_rigid_u32_ds else self.pip_marker_glass_rigid_u16_ds)
            else
                (if (is_u32) self.pip_marker_glass_rigid_u32 else self.pip_marker_glass_rigid_u16),
            .instanced => if (double_sided)
                (if (is_u32) self.pip_marker_glass_inst_u32_ds else self.pip_marker_glass_inst_u16_ds)
            else
                (if (is_u32) self.pip_marker_glass_inst_u32 else self.pip_marker_glass_inst_u16),
            .skinned => if (double_sided)
                (if (is_u32) self.pip_marker_glass_skin_u32_ds else self.pip_marker_glass_skin_u16_ds)
            else
                (if (is_u32) self.pip_marker_glass_skin_u32 else self.pip_marker_glass_skin_u16),
        };
    }

    pub fn render(
        self: *VelocityPass,
        cur_view_proj: Mat4,
        prev_view_proj: Mat4,
        queues: *const RenderQueues,
        skins: []const [scene_render_queue.MAX_BONES]Mat4,
        prev_skins: []const [scene_render_queue.MAX_BONES]Mat4,
        main_depth_att_view: sg.View,
        main_depth_format: sg.PixelFormat,
        last_rendered: u64,
        stats: *SceneStats,
        markers: MarkerArgs,
    ) void {
        if (!sg.isvalid()) return;
        // Own color target plus the BORROWED main depth attachment (never
        // owned or destroyed here): every handle revalidated, since a
        // failed ensure must never leave a stale-sized target samplable.
        if (!gpu_thread.createdViewIsValid(self.att_view) or !gpu_thread.createdViewIsValid(main_depth_att_view)) return;
        if (!self.ensurePipelinesFor(main_depth_format)) return;

        // Shared-depth plan (see velocityDepthPassAction): own color is
        // cleared to the zero mask; main depth is LOADED and STORED with
        // writes off — nearer main pixels (morph displacement, opaque
        // surfaces) occlude farther velocity draws, and cutout holes (main
        // depth holds the farther background) fail the EQUAL match instead
        // of writing false mask1. No depth clear: the post chain samples
        // this same depth afterwards.
        var pass = sg.Pass{ .action = velocityDepthPassAction() };
        pass.attachments.colors[0] = self.att_view;
        pass.attachments.depth_stencil = main_depth_att_view;
        sg.beginPass(pass);

        var draws: u32 = 0;
        for (queues.items.items) |item| {
            if (self.drawRegularOne(cur_view_proj, prev_view_proj, item, skins, prev_skins, last_rendered)) draws += 1;
        }
        for (queues.opaque_instanced.items) |batch| {
            if (self.drawInstancedOne(cur_view_proj, prev_view_proj, batch, last_rendered)) draws += 1;
        }
        // Reactive markers (C.3 v2) stamp AFTER the opaque velocity draws:
        // marker coverage overwrites the background motion underneath with
        // the reactive band value (the blended pixel's history is wrong
        // either way). Particle billboards first, then the transparent
        // glass redraw — both stamp the same constant, so overlap order is
        // irrelevant (any marked pixel is reactive). Disabled or empty
        // draws nothing — the buffer stays bit-identical to the pre-marker
        // path.
        if (markers.enabled) {
            draws += self.drawParticleMarkers(cur_view_proj, markers);
            draws += self.drawGlassMarkers(cur_view_proj, queues, skins);
        }

        sg.endPass();
        stats.main_draw_calls += draws;
        stats.draw_calls += draws;
    }

    /// Particle reactive-marker draws (C.3 v2): one quad draw per live
    /// prepared record through the LESS_EQUAL marker pipelines. Sg-free
    /// skips first (particleMarkerDrawDecision shape), then handle
    /// validation, then draws. Returns the issued draw count for stats.
    fn drawParticleMarkers(self: *VelocityPass, view_proj: Mat4, markers: MarkerArgs) u32 {
        if (markers.draws.len == 0) return 0;
        if (!gpu_thread.createdBufferIsValid(self.marker_quad_vb) or !gpu_thread.createdBufferIsValid(self.marker_quad_ib)) return 0;
        if (!gpu_thread.createdPipelineIsValid(self.pip_marker_cpu) or !gpu_thread.createdPipelineIsValid(self.pip_marker_gpu)) return 0;
        var draws: u32 = 0;
        for (markers.draws) |draw| {
            const instance_buf = draw.drawBuffer();
            if (!particleMarkerDrawDecision(draw.active_count, instance_buf.id != 0)) continue;
            if (!gpu_thread.createdBufferIsValid(instance_buf)) continue;
            const gpu = draw.simulation_mode == .gpu;
            sg.applyPipeline(if (gpu) self.pip_marker_gpu else self.pip_marker_cpu);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = self.marker_quad_vb;
            bind.vertex_buffers[1] = instance_buf;
            bind.index_buffer = self.marker_quad_ib;
            sg.applyBindings(bind);
            const vs_params = velocity_shd.VsMarkerParams{
                .view_proj = view_proj,
                .camera_right = .{ markers.cam_right.x, markers.cam_right.y, markers.cam_right.z, 0.0 },
                .camera_up = .{ markers.cam_up.x, markers.cam_up.y, markers.cam_up.z, 0.0 },
            };
            sg.applyUniforms(velocity_shd.UB_vs_marker_params, sg.asRange(&vs_params));
            if (gpu) {
                const gpu_params = velocity_shd.MarkerGpuParams{
                    .time_drag = .{ draw.clock_seconds, draw.drag, 0.0, 0.0 },
                    .gravity = .{ draw.gravity.x, draw.gravity.y, draw.gravity.z, 0.0 },
                    .sprite = .{
                        @floatFromInt(if (draw.spritesheet_columns == 0) 1 else draw.spritesheet_columns),
                        @floatFromInt(if (draw.spritesheet_rows == 0) 1 else draw.spritesheet_rows),
                        draw.spritesheet_loops,
                        0.0,
                    },
                };
                sg.applyUniforms(velocity_shd.UB_marker_gpu_params, sg.asRange(&gpu_params));
            }
            sg.draw(0, 6, @intCast(draw.active_count));
            draws += 1;
        }
        return draws;
    }

    /// Glass reactive-marker draws (C.3 v2 glass half): one redraw per
    /// transparent main-pass draw through the LESS_EQUAL marker pipelines
    /// (rigid / instanced / skinned by shape). Sg-free decision first
    /// (glassMarkerDrawDecision shape), then handle validation, then
    /// draws. Coverage-only: the CURRENT transform feeds cur_mvp (no prev
    /// pairing — the instanced path binds the single current buffer, the
    /// skinned path the current bones). Returns the issued draw count for
    /// stats. Cost: one extra vertex pass over the transparent queue per
    /// TAA frame (constant FS, no texture taps); overlapping glass layers
    /// all stamp the same value, so overdraw only costs fill, never
    /// correctness.
    fn drawGlassMarkers(
        self: *VelocityPass,
        view_proj: Mat4,
        queues: *const RenderQueues,
        skins: []const [scene_render_queue.MAX_BONES]Mat4,
    ) u32 {
        if (sg.queryPipelineState(self.pip_marker_glass_rigid_u16) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_rigid_u32) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_rigid_u16_ds) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_rigid_u32_ds) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_inst_u16) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_inst_u32) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_inst_u16_ds) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_inst_u32_ds) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_skin_u16) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_skin_u32) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_skin_u16_ds) != .VALID) return 0;
        if (sg.queryPipelineState(self.pip_marker_glass_skin_u32_ds) != .VALID) return 0;
        var draws: u32 = 0;
        for (queues.transparent.items) |item| {
            if (!glassMarkerDrawDecision(.{
                .transparent = item.transparent,
                .is_decal = item.is_decal,
                .has_hook = item.shader_index != null,
                .morph_fallback = item.velocity_depth_fallback,
                .index_count = item.index_count,
            })) continue;
            if (!gpu_thread.geometryValid(item.vertex_buffer, item.index_buffer)) continue;
            const cur_mvp = Mat4.mul(view_proj, item.model);
            if (item.is_skinned) {
                const bones = scene_render_queue.skinAt(skins, item.skin_index) orelse continue;
                const pip = self.glassPipeline(.skinned, item.is_u32, item.double_sided);
                if (sg.queryPipelineState(pip) != .VALID) continue;
                sg.applyPipeline(pip);
                var bind = sg.Bindings{};
                bind.vertex_buffers[0] = item.vertex_buffer;
                bind.index_buffer = item.index_buffer;
                sg.applyBindings(bind);
                const vs_params = velocity_shd.VsGlassParams{ .cur_mvp = cur_mvp };
                sg.applyUniforms(velocity_shd.UB_vs_glass_params, sg.asRange(&vs_params));
                const vs_skin = velocity_shd.VsGlassSkin{ .bones = bones.* };
                sg.applyUniforms(velocity_shd.UB_vs_glass_skin, sg.asRange(&vs_skin));
            } else {
                const pip = self.glassPipeline(.rigid, item.is_u32, item.double_sided);
                if (sg.queryPipelineState(pip) != .VALID) continue;
                sg.applyPipeline(pip);
                var bind = sg.Bindings{};
                bind.vertex_buffers[0] = item.vertex_buffer;
                bind.index_buffer = item.index_buffer;
                sg.applyBindings(bind);
                const vs_params = velocity_shd.VsGlassParams{ .cur_mvp = cur_mvp };
                sg.applyUniforms(velocity_shd.UB_vs_glass_params, sg.asRange(&vs_params));
            }
            sg.draw(item.base_vertex, item.index_count, 1);
            draws += 1;
        }
        for (queues.transparent_instanced.items) |batch| {
            if (!glassMarkerDrawDecision(.{
                .transparent = batch.transparent,
                .is_decal = batch.is_decal,
                .index_count = batch.index_count,
            })) continue;
            if (batch.visible_instance_count == 0 or batch.instance_buffer.id == 0) continue;
            if (!gpu_thread.instancedGeometryValid(batch.vertex_buffer, batch.index_buffer, batch.instance_buffer)) continue;
            const pip = self.glassPipeline(.instanced, batch.index_type == .UINT32, batch.double_sided);
            if (sg.queryPipelineState(pip) != .VALID) continue;
            sg.applyPipeline(pip);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = batch.vertex_buffer;
            bind.vertex_buffers[1] = batch.instance_buffer;
            bind.index_buffer = batch.index_buffer;
            sg.applyBindings(bind);
            const vs_params = velocity_shd.VsGlassInstParams{ .cur_view_proj = view_proj };
            sg.applyUniforms(velocity_shd.UB_vs_glass_inst_params, sg.asRange(&vs_params));
            sg.draw(0, batch.index_count, batch.visible_instance_count);
            draws += 1;
        }
        return draws;
    }

    fn drawRegularOne(
        self: *VelocityPass,
        cur_view_proj: Mat4,
        prev_view_proj: Mat4,
        item: scene_render_queue.RenderMeshItem,
        skins: []const [scene_render_queue.MAX_BONES]Mat4,
        prev_skins: []const [scene_render_queue.MAX_BONES]Mat4,
        last_rendered: u64,
    ) bool {
        // Sg-free skip matrix first (see velocityDrawDecision): mask0
        // fallbacks never reach handle validation, let alone draws.
        if (velocityDrawDecision(.{
            .transparent = item.transparent,
            .is_decal = item.is_decal,
            .has_hook = item.shader_index != null,
            .morph_fallback = item.velocity_depth_fallback,
            .index_count = item.index_count,
        }) != .draw) return false;
        if (!gpu_thread.geometryValid(item.vertex_buffer, item.index_buffer)) return false;

        // Generation gate: frozen prev state is usable only when it comes
        // from the last actually rendered frame. Anything staged but never
        // presented (or replayed by a reuse) collapses to zero motion here,
        // never to an adopted unrendered pose.
        const use_prev = usePresentedPrev(item.prev_frame, last_rendered);
        const prev_model = if (use_prev) item.prev_model else item.model;
        const cur_mvp = Mat4.mul(cur_view_proj, item.model);
        const prev_mvp = Mat4.mul(prev_view_proj, prev_model);

        if (item.is_skinned) {
            const bones = scene_render_queue.skinAt(skins, item.skin_index) orelse return false;
            const prev_bones = if (usePresentedPrev(item.skin_prev_frame, last_rendered))
                scene_render_queue.skinAt(prev_skins, item.skin_index) orelse bones
            else
                bones;
            const pip = self.opaquePipeline(.skinned, item.is_u32, item.double_sided);
            if (sg.queryPipelineState(pip) != .VALID) return false;
            sg.applyPipeline(pip);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = item.vertex_buffer;
            bind.index_buffer = item.index_buffer;
            bind.views[velocity_shd.VIEW_albedo_tex] = if (item.draw_record.albedo_view.id != 0)
                item.draw_record.albedo_view
            else
                self.dummy_zero_view;
            bind.samplers[velocity_shd.SMP_albedo_smp] = if (item.draw_record.albedo_sampler.id != 0)
                item.draw_record.albedo_sampler
            else
                self.albedo_fallback_sampler;
            sg.applyBindings(bind);
            const vs_params = velocity_shd.VsParams{
                .cur_mvp = cur_mvp,
                .prev_mvp = prev_mvp,
            };
            sg.applyUniforms(velocity_shd.UB_vs_params, sg.asRange(&vs_params));
            const vs_skin = velocity_shd.VsSkin{ .bones = bones.* };
            sg.applyUniforms(velocity_shd.UB_vs_skin, sg.asRange(&vs_skin));
            const vs_prev_skin = velocity_shd.VsPrevSkin{ .prev_bones = prev_bones.* };
            sg.applyUniforms(velocity_shd.UB_vs_prev_skin, sg.asRange(&vs_prev_skin));
            const alpha = velocityAlphaUniforms(item.draw_record);
            sg.applyUniforms(velocity_shd.UB_fs_alpha, sg.asRange(&alpha));
        } else {
            const pip = self.opaquePipeline(.rigid, item.is_u32, item.double_sided);
            if (sg.queryPipelineState(pip) != .VALID) return false;
            sg.applyPipeline(pip);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = item.vertex_buffer;
            bind.index_buffer = item.index_buffer;
            bind.views[velocity_shd.VIEW_albedo_tex] = if (item.draw_record.albedo_view.id != 0)
                item.draw_record.albedo_view
            else
                self.dummy_zero_view;
            bind.samplers[velocity_shd.SMP_albedo_smp] = if (item.draw_record.albedo_sampler.id != 0)
                item.draw_record.albedo_sampler
            else
                self.albedo_fallback_sampler;
            sg.applyBindings(bind);
            const vs_params = velocity_shd.VsParams{
                .cur_mvp = cur_mvp,
                .prev_mvp = prev_mvp,
            };
            sg.applyUniforms(velocity_shd.UB_vs_params, sg.asRange(&vs_params));
            const alpha = velocityAlphaUniforms(item.draw_record);
            sg.applyUniforms(velocity_shd.UB_fs_alpha, sg.asRange(&alpha));
        }
        sg.draw(item.base_vertex, item.index_count, 1);
        return true;
    }

    fn drawInstancedOne(
        self: *VelocityPass,
        cur_view_proj: Mat4,
        prev_view_proj: Mat4,
        batch: scene_render_queue.RenderInstancedBatch,
        last_rendered: u64,
    ) bool {
        if (velocityDrawDecision(.{
            .transparent = batch.transparent,
            .is_decal = batch.is_decal,
            .index_count = batch.index_count,
        }) != .draw) return false;
        if (batch.visible_instance_count == 0 or batch.instance_buffer.id == 0) return false;
        if (!gpu_thread.instancedGeometryValid(batch.vertex_buffer, batch.index_buffer, batch.instance_buffer)) return false;

        // Same generation gate as regular draws: an unmatched prev buffer
        // aliases cur (zero motion) instead of pairing unrendered history.
        const prev_buf = if (usePresentedPrev(batch.prev_frame, last_rendered) and
            gpu_thread.createdBufferIsValid(batch.prev_instance_buffer))
            batch.prev_instance_buffer
        else
            batch.instance_buffer;

        const pip = self.opaquePipeline(.instanced, batch.index_type == .UINT32, batch.double_sided);
        if (sg.queryPipelineState(pip) != .VALID) return false;
        sg.applyPipeline(pip);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = batch.vertex_buffer;
        bind.vertex_buffers[1] = batch.instance_buffer;
        bind.vertex_buffers[2] = prev_buf;
        bind.index_buffer = batch.index_buffer;
        bind.views[velocity_shd.VIEW_albedo_tex] = if (batch.draw_record.albedo_view.id != 0)
            batch.draw_record.albedo_view
        else
            self.dummy_zero_view;
        bind.samplers[velocity_shd.SMP_albedo_smp] = if (batch.draw_record.albedo_sampler.id != 0)
            batch.draw_record.albedo_sampler
        else
            self.albedo_fallback_sampler;
        sg.applyBindings(bind);
        const vs_params = velocity_shd.VsInstParams{
            .cur_view_proj = cur_view_proj,
            .prev_view_proj = prev_view_proj,
        };
        sg.applyUniforms(velocity_shd.UB_vs_inst_params, sg.asRange(&vs_params));
        const alpha = velocityAlphaUniforms(batch.draw_record);
        sg.applyUniforms(velocity_shd.UB_fs_alpha, sg.asRange(&alpha));
        sg.draw(0, batch.index_count, batch.visible_instance_count);
        return true;
    }
};

pub const VelocityDraw = enum { draw, skip_mask0 };

pub const VelocityDrawInput = struct {
    transparent: bool = false,
    is_decal: bool = false,
    has_hook: bool = false,
    morph_fallback: bool = false,
    is_cutout: bool = false,
    index_count: u32 = 0,
};

pub fn velocityDrawDecision(in: VelocityDrawInput) VelocityDraw {
    if (in.index_count == 0) return .skip_mask0;
    if (in.morph_fallback) return .skip_mask0;
    if (!itemContributesDepth(in.transparent, in.is_decal, in.has_hook)) return .skip_mask0;
    return .draw;
}
