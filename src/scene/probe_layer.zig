//! On-demand reflection probes for PBR ambient lighting.
//!
//! Design (bounded, additive, OFF by default):
//! - At most `max_probes` (4) probes per scene; `add` past the cap is a hard
//!   `error.TooManyReflectionProbes` (never a silent clamp), mirroring the
//!   directional-fill cap in `light_rig.zig`.
//! - A probe with no capture yet (`captured == false`), a disabled probe, or
//!   an object outside every probe radius uses the ambient/environment path.
//! - Capture is explicit and on-demand: `markDirty` (via
//!   `Scene.captureReflectionProbe`) flags one probe, `Scene.render` captures
//!   at most ONE dirty probe per frame on the context thread (lowest dirty +
//!   enabled index first; the rest wait for later frames), rendering the
//!   prepared draw list plus the sky into a `face_resolution` (128) cube
//!   target from the probe position. No `UploadQueue`/texture-streaming
//!   interaction on this path, so the single-upload-per-frame discipline is
//!   untouched; capture performs zero `sg.updateBuffer`/`sg.appendBuffer`
//!   traffic (only draws into probe-owned targets).
//! - After the six base faces render, GGX convolution produces specular mips
//!   and cosine-weighted convolution produces E/pi in the coarsest mip.
//!   Convolution samples only captured mip 0 into a separate scratch cube;
//!   copy passes populate the probe outputs without source feedback.
//! - Regular/skinned draws select up to two captured probes with spatial
//!   falloff weights; remaining weight belongs to the environment.
//!   Instanced draws currently use the environment only. `selectProbe`
//!   retains the explicit nearest-probe query, separate from draw blending.
//! - Uniform packing follows the light-rig pattern: `packFrame` snapshots
//!   plain data + borrowed GPU handle VALUES (view/sampler) into `FramePack`,
//!   which rides the `SceneFrameSnapshot` into the draw; the draw never reads
//!   live layer state.
//!
//! Explicit non-goals: box projection / parallax correction,
//! real-time per-frame updates (captures are
//! on-demand, at most one per frame), specular occlusion, probe-baked
//! irradiance spherical harmonics, and editor tooling.
//!
//! Threading: the layer is owned by `Scene` (context side for GPU work,
//! game side for add/remove/mark under update-vs-prepare exclusion, like the
//! sky layer). GPU targets are created lazily on the context thread and
//! retired through `GpuRetireQueue` on removal (see `remove`), never
//! destroyed inline from the game side.
//!
//! Headless behavior: every `sg.*` call is gated on `sg.isvalid()`, so all
//! bookkeeping (add/remove/dirty/selection/packing) is CPU-testable without
//! a GPU context. Without a context `ensureGpu` fails closed (returns
//! false), dirty flags are retained, and selection keeps falling back.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const blit_shd = @import("probe_mip_shader");

/// Fixed capacity: at most this many probes per scene. Bakes into
/// `FramePack.entries`, the snapshot, and the retire payload as plain
/// arrays (no per-frame allocation).
pub const max_probes = 4;
/// Cube face resolution in pixels. 128 keeps one capture at 6 small passes
/// plus a cheap blit chain; matches the procedural-skybox default scale.
pub const face_resolution: i32 = 128;
/// Full mip chain for `face_resolution` (128..1). The shader clamps
/// `roughness * max_lod` into this range.
pub const max_mips: u32 = 8;
/// Largest LOD the shader may request from a probe cube.
pub const max_lod: f32 = @floatFromInt(max_mips - 1);
/// Near clip for the six capture cameras.
pub const capture_near: f32 = 0.1;

/// Cube face order. Matches the sokol cube slice order (+X, -X, +Y, -Y, +Z,
/// -Z; see `sg.ImageData` face docs), so `slice = face` in every per-face
/// attachment view.
pub const Face = enum(u3) {
    pos_x = 0,
    neg_x = 1,
    pos_y = 2,
    neg_y = 3,
    pos_z = 4,
    neg_z = 5,
};

/// View direction of each capture face (world space, unit length).
pub fn faceDir(face: Face) Vec3 {
    return switch (face) {
        .pos_x => Vec3.new(1, 0, 0),
        .neg_x => Vec3.new(-1, 0, 0),
        .pos_y => Vec3.new(0, 1, 0),
        .neg_y => Vec3.new(0, -1, 0),
        .pos_z => Vec3.new(0, 0, 1),
        .neg_z => Vec3.new(0, 0, -1),
    };
}

/// Up vector per capture face. Chosen so no face looks parallel to its up
/// (the +/-Y faces use a Z up); in-plane orientation is arbitrary by design
/// (see the module docs: the blit chain only needs same-face content).
pub fn faceUp(face: Face) Vec3 {
    return switch (face) {
        .pos_x, .neg_x, .pos_z, .neg_z => Vec3.new(0, 1, 0),
        .pos_y => Vec3.new(0, 0, -1),
        .neg_y => Vec3.new(0, 0, 1),
    };
}

/// View matrix for one capture face from `eye`.
pub fn faceView(face: Face, eye: Vec3) Mat4 {
    return Mat4.lookAt(eye, eye.add(faceDir(face)), faceUp(face));
}

/// Combined view-projection for one capture face (90-degree cube frustum).
pub fn faceViewProj(face: Face, eye: Vec3, near: f32, far: f32) Mat4 {
    return Mat4.mul(Mat4.perspective(90.0, 1.0, near, far), faceView(face, eye));
}

/// Far clip for a capture at a probe of `radius`: the probe sees its whole
/// influence sphere plus surroundings. Floor keeps tiny radii usable.
pub fn captureFar(radius: f32) f32 {
    return @max(50.0, radius * 8.0);
}

/// Pixel width/height of mip level `mip` (`mip` in 0..`max_mips`).
pub fn mipSize(mip: u32) i32 {
    return @max(1, face_resolution >> @intCast(mip));
}

/// Convolution reads captured mip 0 only; coarse mips are its outputs.
pub fn bakeViewDesc(image: sg.Image) sg.ViewDesc {
    return .{
        .texture = .{
            .image = image,
            .mip_levels = .{ .base = 0, .count = 1 },
        },
    };
}

/// Roughness ladder for the specular bake: mip `m` in 1..`max_mips - 2`
/// maps linearly to 0..1; the coarsest mip is diffuse irradiance
/// (roughness 1.0 by convention).
pub fn roughnessForMip(mip: u32) f32 {
    if (isIrradianceMip(mip)) return 1.0;
    return @as(f32, @floatFromInt(mip)) / @as(f32, @floatFromInt(max_mips - 2));
}

/// True only for the coarsest mip, which stores cosine-weighted diffuse
/// irradiance (E/pi convention) instead of a specular lobe.
pub fn isIrradianceMip(mip: u32) bool {
    return mip == max_mips - 1;
}

/// Creation options for one probe. A new probe starts dirty (first render
/// captures it) but uncaptured, so selection falls back until the capture
/// lands and pre-capture frames stay bit-identical.
pub const ReflectionProbeOptions = struct {
    /// Influence radius in world units; objects past it ignore the probe.
    radius: f32 = 10.0,
    /// Disabled probes are skipped by selection and by capture scheduling.
    enabled: bool = true,
    /// Scalar on the probe ambient contribution in the shaders.
    intensity: f32 = 1.0,
    /// Parallax bounding box half-extent (0.0 = infinite distance / disabled).
    box_extents: f32 = 0.0,
};

/// GPU content of one probe: the mipmapped cube render target, its sample
/// view/sampler, one color-attachment view per (mip, face) for capture +
/// prefilter, and a private depth target. Plain handles, moved by value
/// into the retire queue on removal.
pub const ProbeGpu = struct {
    image: sg.Image = .{},
    tex_view: sg.View = .{},
    /// Mip-0-only texture view of `image`: the bake source for the
    /// GGX/irradiance convolution (Pass A). Capture renders mip 0 while
    /// mips 1..7 are still stale, so the convolution must never sample
    /// the full chain. Same lifetime as `tex_view`.
    bake_view: sg.View = .{},
    sampler: sg.Sampler = .{},
    mip_face_views: [max_mips][6]sg.View = [_][6]sg.View{[_]sg.View{.{}} ** 6} ** max_mips,
    depth_image: sg.Image = .{},
    depth_view: sg.View = .{},
    valid: bool = false,

    pub fn deinit(self: *ProbeGpu) void {
        // Headless guard: sokol resource destroys assert a valid context
        // (unlike the buffer path Mesh.deinit relies on), and tests retire
        // + flush empty targets without one. Production flush/deinit always
        // run with a live context, so real targets still destroy exactly
        // once. Either way the value resets.
        defer self.* = .{};
        if (!sg.isvalid()) return;
        sg.destroySampler(self.sampler);
        sg.destroyView(self.tex_view);
        sg.destroyView(self.bake_view);
        sg.destroyView(self.depth_view);
        for (&self.mip_face_views) |*mip| {
            for (mip) |*v| {
                sg.destroyView(v.*);
                v.* = .{};
            }
        }
        sg.destroyImage(self.depth_image);
        sg.destroyImage(self.image);
        self.* = .{};
    }
};

/// Estimated VRAM bytes of one fully created probe target: the linear-HDR
/// RGBA16F cube (8 B/px) with a full mip chain (all 6 faces) plus the depth
/// target. Pure (no GPU calls); feeds the profiler's render-target census.
pub fn targetBytes() usize {
    var total: usize = 0;
    var s: u32 = @intCast(face_resolution);
    while (true) {
        total += @as(usize, s) * @as(usize, s) * 8 * 6;
        if (s == 1) break;
        s = @max(1, s / 2);
    }
    total += @as(usize, @intCast(face_resolution)) * @as(usize, @intCast(face_resolution)) * 4;
    return total;
}

/// One scene probe: CPU state plus its lazily created GPU target.
pub const ReflectionProbe = struct {
    position: Vec3 = Vec3.zero,
    radius: f32 = 10.0,
    enabled: bool = true,
    intensity: f32 = 1.0,
    box_extents: f32 = 0.0,
    /// True when the GPU content is stale and needs a recapture. Set by
    /// `markDirty`, cleared by `notifyCaptured`.
    dirty: bool = true,
    /// True once at least one capture completed. Selection requires this
    /// (plus `enabled` and a live view), so pre-capture frames fall back.
    captured: bool = false,
    gpu: ProbeGpu = .{},
};

/// Snapshot entry for one probe: plain data + borrowed GPU handle VALUES
/// (never CPU refs), copied by value into `SceneFrameSnapshot` like the
/// light pack. The draw selects from these without touching live state.
pub const ProbeFrameEntry = struct {
    position: Vec3 = Vec3.zero,
    radius: f32 = 0,
    enabled: bool = false,
    captured: bool = false,
    intensity: f32 = 1.0,
    max_probe_lod: f32 = max_lod,
    box_extents: f32 = 0.0,
    view: sg.View = .{},
    sampler: sg.Sampler = .{},
};

/// Per-frame probe state published into the frame snapshot (light-rig
/// pattern: fixed array + count, no allocation).
pub const FramePack = struct {
    entries: [max_probes]ProbeFrameEntry = [_]ProbeFrameEntry{.{}} ** max_probes,
    count: usize = 0,
};

const ibl_prefilter = @import("../texture/ibl_prefilter.zig");

/// Result of per-object probe selection: the winning probe's sampling
/// state.
pub const SelectedProbe = struct {
    index: usize,
    position: Vec3 = Vec3.zero,
    intensity: f32,
    max_probe_lod: f32,
    box_extents: f32 = 0.0,
    view: sg.View,
    sampler: sg.Sampler,
};

/// One blended probe entry with its normalized spatial blend weight.
pub const ProbeBlendEntry = struct {
    probe: SelectedProbe,
    weight: f32,
};

/// Multi-probe selection result:
/// Supports up to two overlapping probes with C1 smoothstep falloff weights
/// and remaining weight allocated to the environment map.
pub const SelectedProbes = struct {
    primary: ?ProbeBlendEntry = null,
    secondary: ?ProbeBlendEntry = null,
    env_weight: f32 = 1.0,
};

/// Selects up to 2 overlapping probes at `pos` with continuous C1 smoothstep
/// falloff weights based on distance to probe centers.
/// When inside a single probe, the probe weight smoothly transitions to 0
/// at radius with zero derivative, avoiding edge pops.
/// When inside two overlapping probes, weights are normalized across both probes
/// guaranteeing smooth transition across probe boundaries without jump/pop.
pub fn selectProbes(entries: []const ProbeFrameEntry, pos: Vec3) SelectedProbes {
    var best0: ?SelectedProbe = null;
    var w0: f32 = 0.0;
    var best1: ?SelectedProbe = null;
    var w1: f32 = 0.0;

    for (entries, 0..) |*e, i| {
        if (!e.enabled or !e.captured) continue;
        if (e.view.id == 0) continue;
        const d = e.position.sub(pos).length();
        if (d >= e.radius) continue;
        const w = ibl_prefilter.smoothFalloff(d, e.radius);
        if (w <= 0.0001) continue;

        const sel = SelectedProbe{
            .index = i,
            .position = e.position,
            .intensity = e.intensity,
            .max_probe_lod = e.max_probe_lod,
            .box_extents = e.box_extents,
            .view = e.view,
            .sampler = e.sampler,
        };

        if (w > w0) {
            best1 = best0;
            w1 = w0;
            best0 = sel;
            w0 = w;
        } else if (w > w1) {
            best1 = sel;
            w1 = w;
        }
    }

    if (best0 == null) {
        return .{};
    }

    const blended = ibl_prefilter.blendProbeWeights(w0, w1);
    var res = SelectedProbes{
        .primary = .{ .probe = best0.?, .weight = blended.w0 },
        .env_weight = blended.w_env,
    };
    if (best1) |b1| {
        if (blended.w1 > 0.0001) {
            res.secondary = .{ .probe = b1, .weight = blended.w1 };
        }
    }
    return res;
}

/// Nearest enabled+captured probe whose radius contains `pos`; ties resolve
/// to the lowest probe index (strict `<` keeps the first best). Returns null
/// when no probe applies — the caller then takes today's ambient/skybox
/// path. Pure (no GPU calls).
pub fn selectProbe(entries: []const ProbeFrameEntry, pos: Vec3) ?SelectedProbe {
    var best: ?SelectedProbe = null;
    var best_d2: f32 = 0;
    for (entries, 0..) |*e, i| {
        if (!e.enabled or !e.captured) continue;
        if (e.view.id == 0) continue;
        const d2 = e.position.sub(pos).lengthSq();
        if (d2 > e.radius * e.radius) continue;
        if (best == null or d2 < best_d2) {
            best = .{
                .index = i,
                .position = e.position,
                .intensity = e.intensity,
                .max_probe_lod = e.max_probe_lod,
                .box_extents = e.box_extents,
                .view = e.view,
                .sampler = e.sampler,
            };
            best_d2 = d2;
        }
    }
    return best;
}

/// Owns the scene's probes plus the shared mip-blit resources (pipeline,
/// fullscreen quad, sampler). Fixed-size arrays: steady state allocates
/// nothing.
pub const ProbeLayer = struct {
    probes: [max_probes]ReflectionProbe = [_]ReflectionProbe{.{}} ** max_probes,
    count: usize = 0,
    blit_shader: sg.Shader = .{},
    blit_pipeline: sg.Pipeline = .{},
    blit_sampler: sg.Sampler = .{},
    blit_vb: sg.Buffer = .{},
    blit_ib: sg.Buffer = .{},
    scratch_cube: sg.Image = .{},
    scratch_tex_view: sg.View = .{},
    scratch_mip_face_views: [max_mips][6]sg.View = [_][6]sg.View{[_]sg.View{.{}} ** 6} ** max_mips,

    /// Adds a probe; returns its index. Past `max_probes` this is a hard
    /// error (never a silent clamp or replacement), mirroring
    /// `LightRig.addDirectionalLight`. New probes start dirty + uncaptured:
    /// the next render captures them (one per frame) and selection falls
    /// back until then.
    pub fn add(self: *ProbeLayer, position: Vec3, options: ReflectionProbeOptions) error{TooManyReflectionProbes}!usize {
        if (self.count >= max_probes) return error.TooManyReflectionProbes;
        const idx = self.count;
        self.probes[idx] = .{
            .position = position,
            .radius = options.radius,
            .enabled = options.enabled,
            .intensity = options.intensity,
            .box_extents = options.box_extents,
            .dirty = true,
            .captured = false,
            .gpu = .{},
        };
        self.count += 1;
        return idx;
    }

    /// Removes probe `index`, retiring its GPU target through `retire_queue`
    /// (any thread: only an epoch stamp + append happen here; the context
    /// thread destroys at the next flush). Order-preserving: higher indices
    /// shift down, so callers must not cache indices across removals.
    /// Out-of-range indices are a no-op (same contract as
    /// `Scene.removeCamera`).
    pub fn remove(self: *ProbeLayer, allocator: std.mem.Allocator, retire_queue: anytype, index: usize) void {
        if (index >= self.count) return;
        retire_queue.retireProbeTarget(allocator, self.probes[index].gpu);
        self.probes[index].gpu = .{};
        for (index..self.count - 1) |k| self.probes[k] = self.probes[k + 1];
        self.probes[self.count - 1] = .{};
        self.count -= 1;
    }

    /// Flags one probe for recapture on the next render. Out-of-range is a
    /// no-op.
    pub fn markDirty(self: *ProbeLayer, index: usize) void {
        if (index >= self.count) return;
        self.probes[index].dirty = true;
    }

    /// Flags every probe for recapture (each still captures on its own
    /// frame: at most one capture per frame, see `nextDirtyIndex`).
    pub fn markAllDirty(self: *ProbeLayer) void {
        for (self.probes[0..self.count]) |*p| p.dirty = true;
    }

    /// Lowest-index dirty + enabled probe, or null when nothing needs a
    /// capture. Disabled probes never schedule (capturing them would waste
    /// the frame's single capture slot on content selection ignores).
    pub fn nextDirtyIndex(self: *const ProbeLayer) ?usize {
        for (self.probes[0..self.count], 0..) |*p, i| {
            if (p.dirty and p.enabled) return i;
        }
        return null;
    }

    /// How many probes currently want a capture (enabled or not).
    pub fn dirtyCount(self: *const ProbeLayer) usize {
        var n: usize = 0;
        for (self.probes[0..self.count]) |*p| {
            if (p.dirty) n += 1;
        }
        return n;
    }

    /// Records a completed capture: clears `dirty`, sets `captured` so
    /// selection starts using the probe.
    pub fn notifyCaptured(self: *ProbeLayer, index: usize) void {
        if (index >= self.count) return;
        self.probes[index].dirty = false;
        self.probes[index].captured = true;
    }

    /// Snapshots probe state for the frame snapshot (light-rig pattern).
    /// Only probes with a live GPU target publish a usable view; the rest
    /// ride along as disabled-by-capture so selection skips them.
    pub fn packFrame(self: *const ProbeLayer) FramePack {
        var pack = FramePack{};
        pack.count = self.count;
        for (self.probes[0..self.count], 0..) |*p, i| {
            const usable = p.captured and p.gpu.valid and p.gpu.tex_view.id != 0;
            pack.entries[i] = .{
                .position = p.position,
                .radius = p.radius,
                .enabled = p.enabled,
                .captured = usable,
                .intensity = p.intensity,
                .max_probe_lod = max_lod,
                .box_extents = p.box_extents,
                .view = p.gpu.tex_view,
                .sampler = p.gpu.sampler,
            };
        }
        return pack;
    }

    /// Ensures the GPU target of probe `index` plus the shared blit
    /// resources exist. Context thread only (called from `Scene.render`).
    /// Fails closed (false) without a GPU context or on creation failure;
    /// the probe stays dirty and is retried on a later frame.
    pub fn ensureGpu(self: *ProbeLayer, index: usize) bool {
        if (!sg.isvalid()) return false;
        if (index >= self.count) return false;
        const probe = &self.probes[index];
        if (!probe.gpu.valid and !createTarget(&probe.gpu)) return false;
        if (self.blit_pipeline.id == 0 and !createBlit(self)) return false;
        probe.gpu.valid = true;
        return true;
    }

    fn defaultColorFormat() sg.PixelFormat {
        return .RGBA16F;
    }

    fn createTarget(gpu: *ProbeGpu) bool {
        // NOTE: plain `bool` return — `errdefer` does NOT run on `return
        // false` (only on error propagation), so every failure path below
        // destroys what it already made explicitly. Single owner: nothing
        // is published into `gpu.*` until every handle validated.
        const img = sg.makeImage(.{
            .type = .CUBE,
            .usage = .{ .color_attachment = true },
            .width = face_resolution,
            .height = face_resolution,
            .num_slices = 6,
            .num_mipmaps = @intCast(max_mips),
            .pixel_format = defaultColorFormat(),
            .sample_count = 1,
        });
        if (img.id == 0 or sg.queryImageState(img) != .VALID) {
            if (img.id != 0) sg.destroyImage(img);
            return false;
        }

        const tex_view = sg.makeView(.{ .texture = .{ .image = img } });
        if (tex_view.id == 0 or sg.queryViewState(tex_view) != .VALID) {
            if (tex_view.id != 0) sg.destroyView(tex_view);
            sg.destroyImage(img);
            return false;
        }

        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });
        if (smp.id == 0 or sg.querySamplerState(smp) != .VALID) {
            if (smp.id != 0) sg.destroySampler(smp);
            sg.destroyView(tex_view);
            sg.destroyImage(img);
            return false;
        }

        var face_views: [max_mips][6]sg.View = [_][6]sg.View{[_]sg.View{.{}} ** 6} ** max_mips;
        for (0..max_mips) |m| {
            for (0..6) |f| {
                face_views[m][f] = sg.makeView(.{
                    .color_attachment = .{
                        .image = img,
                        .mip_level = @intCast(m),
                        .slice = @intCast(f),
                    },
                });
                if (face_views[m][f].id == 0 or sg.queryViewState(face_views[m][f]) != .VALID) {
                    for (&face_views) |*mip| for (mip) |*v| {
                        if (v.id != 0) {
                            sg.destroyView(v.*);
                            v.* = .{};
                        }
                    };
                    sg.destroySampler(smp);
                    sg.destroyView(tex_view);
                    sg.destroyImage(img);
                    return false;
                }
            }
        }

        const depth_img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .width = face_resolution,
            .height = face_resolution,
            .pixel_format = .DEPTH,
            .sample_count = 1,
        });
        if (depth_img.id == 0 or sg.queryImageState(depth_img) != .VALID) {
            if (depth_img.id != 0) sg.destroyImage(depth_img);
            for (&face_views) |*mip| for (mip) |*v| sg.destroyView(v.*);
            sg.destroySampler(smp);
            sg.destroyView(tex_view);
            sg.destroyImage(img);
            return false;
        }

        const depth_view = sg.makeView(.{ .depth_stencil_attachment = .{ .image = depth_img } });
        if (depth_view.id == 0 or sg.queryViewState(depth_view) != .VALID) {
            if (depth_view.id != 0) sg.destroyView(depth_view);
            sg.destroyImage(depth_img);
            for (&face_views) |*mip| for (mip) |*v| sg.destroyView(v.*);
            sg.destroySampler(smp);
            sg.destroyView(tex_view);
            sg.destroyImage(img);
            return false;
        }

        const bake_view = sg.makeView(bakeViewDesc(img));
        if (bake_view.id == 0 or sg.queryViewState(bake_view) != .VALID) {
            if (bake_view.id != 0) sg.destroyView(bake_view);
            sg.destroyView(depth_view);
            sg.destroyImage(depth_img);
            for (&face_views) |*mip| for (mip) |*v| sg.destroyView(v.*);
            sg.destroySampler(smp);
            sg.destroyView(tex_view);
            sg.destroyImage(img);
            return false;
        }

        gpu.* = .{
            .image = img,
            .tex_view = tex_view,
            .bake_view = bake_view,
            .sampler = smp,
            .mip_face_views = face_views,
            .depth_image = depth_img,
            .depth_view = depth_view,
            .valid = true,
        };
        return true;
    }

    fn createBlit(self: *ProbeLayer) bool {
        // Same bool-fn discipline as createTarget: explicit destroy on
        // every failure path; `self.*` fields publish only on success.
        const quad_vertices = [_]f32{
            // x,     y,    u,   v
            -1.0, -1.0, 0.0, 0.0,
            1.0,  -1.0, 1.0, 0.0,
            1.0,  1.0,  1.0, 1.0,
            -1.0, 1.0,  0.0, 1.0,
        };
        const quad_indices = [_]u16{ 0, 1, 2, 0, 2, 3 };

        const vb = sg.makeBuffer(.{ .data = sg.asRange(&quad_vertices) });
        if (vb.id == 0 or sg.queryBufferState(vb) != .VALID) {
            if (vb.id != 0) sg.destroyBuffer(vb);
            return false;
        }
        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&quad_indices),
        });
        if (ib.id == 0 or sg.queryBufferState(ib) != .VALID) {
            if (ib.id != 0) sg.destroyBuffer(ib);
            sg.destroyBuffer(vb);
            return false;
        }

        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });
        if (smp.id == 0 or sg.querySamplerState(smp) != .VALID) {
            if (smp.id != 0) sg.destroySampler(smp);
            sg.destroyBuffer(ib);
            sg.destroyBuffer(vb);
            return false;
        }

        const shd = sg.makeShader(blit_shd.probeMipShaderDesc(sg.queryBackend()));
        if (shd.id == 0 or sg.queryShaderState(shd) != .VALID) {
            if (shd.id != 0) sg.destroyShader(shd);
            sg.destroySampler(smp);
            sg.destroyBuffer(ib);
            sg.destroyBuffer(vb);
            return false;
        }

        var desc = sg.PipelineDesc{
            .shader = shd,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        desc.colors[0].pixel_format = defaultColorFormat();
        desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        desc.layout.attrs[blit_shd.ATTR_probe_mip_position] = .{ .format = .FLOAT2, .offset = 0 };
        desc.layout.attrs[blit_shd.ATTR_probe_mip_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const pip = sg.makePipeline(desc);
        if (pip.id == 0 or sg.queryPipelineState(pip) != .VALID) {
            if (pip.id != 0) sg.destroyPipeline(pip);
            sg.destroyShader(shd);
            sg.destroySampler(smp);
            sg.destroyBuffer(ib);
            sg.destroyBuffer(vb);
            return false;
        }

        const scratch_cube = sg.makeImage(.{
            .type = .CUBE,
            .usage = .{ .color_attachment = true },
            .width = face_resolution,
            .height = face_resolution,
            .num_slices = 6,
            .num_mipmaps = @intCast(max_mips),
            .pixel_format = defaultColorFormat(),
            .sample_count = 1,
        });
        if (scratch_cube.id == 0 or sg.queryImageState(scratch_cube) != .VALID) {
            if (scratch_cube.id != 0) sg.destroyImage(scratch_cube);
            sg.destroyPipeline(pip);
            sg.destroyShader(shd);
            sg.destroySampler(smp);
            sg.destroyBuffer(ib);
            sg.destroyBuffer(vb);
            return false;
        }

        const scratch_tex_view = sg.makeView(.{ .texture = .{ .image = scratch_cube } });
        if (scratch_tex_view.id == 0 or sg.queryViewState(scratch_tex_view) != .VALID) {
            if (scratch_tex_view.id != 0) sg.destroyView(scratch_tex_view);
            sg.destroyImage(scratch_cube);
            sg.destroyPipeline(pip);
            sg.destroyShader(shd);
            sg.destroySampler(smp);
            sg.destroyBuffer(ib);
            sg.destroyBuffer(vb);
            return false;
        }

        var scratch_face_views: [max_mips][6]sg.View = [_][6]sg.View{[_]sg.View{.{}} ** 6} ** max_mips;
        for (0..max_mips) |m| {
            for (0..6) |f| {
                scratch_face_views[m][f] = sg.makeView(.{
                    .color_attachment = .{
                        .image = scratch_cube,
                        .mip_level = @intCast(m),
                        .slice = @intCast(f),
                    },
                });
                if (scratch_face_views[m][f].id == 0 or sg.queryViewState(scratch_face_views[m][f]) != .VALID) {
                    for (&scratch_face_views) |*mip| for (mip) |*v| {
                        if (v.id != 0) {
                            sg.destroyView(v.*);
                            v.* = .{};
                        }
                    };
                    sg.destroyView(scratch_tex_view);
                    sg.destroyImage(scratch_cube);
                    sg.destroyPipeline(pip);
                    sg.destroyShader(shd);
                    sg.destroySampler(smp);
                    sg.destroyBuffer(ib);
                    sg.destroyBuffer(vb);
                    return false;
                }
            }
        }

        self.blit_vb = vb;
        self.blit_ib = ib;
        self.blit_sampler = smp;
        self.blit_shader = shd;
        self.blit_pipeline = pip;
        self.scratch_cube = scratch_cube;
        self.scratch_tex_view = scratch_tex_view;
        self.scratch_mip_face_views = scratch_face_views;
        return true;
    }

    /// Destroys every live GPU target plus the shared blit resources.
    /// Context thread only; called from `Scene.deinit` (sg still alive).
    /// Pending retire entries for already-removed probes are drained by
    /// `GpuRetireQueue.deinit` separately — no double destroy.
    pub fn deinit(self: *ProbeLayer) void {
        for (self.probes[0..self.count]) |*p| p.gpu.deinit();
        self.count = 0;
        if (self.blit_pipeline.id != 0) sg.destroyPipeline(self.blit_pipeline);
        if (self.blit_shader.id != 0) sg.destroyShader(self.blit_shader);
        if (self.blit_sampler.id != 0) sg.destroySampler(self.blit_sampler);
        if (self.blit_vb.id != 0) sg.destroyBuffer(self.blit_vb);
        if (self.blit_ib.id != 0) sg.destroyBuffer(self.blit_ib);
        if (self.scratch_tex_view.id != 0) sg.destroyView(self.scratch_tex_view);
        for (&self.scratch_mip_face_views) |*mip| {
            for (mip) |*v| {
                if (v.id != 0) sg.destroyView(v.*);
            }
        }
        if (self.scratch_cube.id != 0) sg.destroyImage(self.scratch_cube);
        self.* = .{};
    }
};
