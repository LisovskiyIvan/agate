//! Render-to-texture target v1: a genuine runtime-renderable 2D color (+
//! optional depth) attachment pair with sampling views, owned by the caller
//! on the sokol context thread.
//!
//! What this is: the missing primitive behind refraction, mirrors, dynamic
//! thumbnails, and any effect that renders the scene (or a clear color) into
//! an offscreen target and samples it later. It is real sokol state —
//! `sg.Image` color/depth attachments plus color-attachment, texture, and
//! sampler views — never a CPU buffer pretending to be a target and never
//! bare metadata. Creation follows the probe/postfx pattern (`makeImage` +
//! `makeView` with `errdefer` rollback), destruction is immediate on the
//! context thread like every other pass-owned target (`PostProcessPass`,
//! probe cubes).
//!
//! Scope (v1, deliberately narrow):
//! - Single 2D color attachment (+ optional depth), single mip level, no
//!   array/cube/MSAA-resolve-texture subtleties beyond the standard
//!   MSAA color-resolve pair the main target already uses.
//! - Default color format is linear HDR `RGBA16F` (the renderer is always
//!   linear HDR now; `desc.color_format = .DEFAULT` resolves to it). Custom
//!   formats are allowed for clear/sample-only use; drawing scene content
//!   into a non-HDR format is refused by `renderPrimaryView` (lossless
//!   radiance contract: exactly one scene HDR format, no silent LDR clamp).
//! - No retire-queue kind: `GpuRetireQueue` lives under `scene/` (owned by
//!   the main thread) and only knows mesh/buffer/probe/ui3d payloads. This
//!   target is context-thread owned and destroys immediately in `deinit`.
//!   The candidate seam for scene-owned targets is documented in
//!   `docs/render-target.md` for the main thread.
//!
//! Threading: every GPU-touching method requires the sokol context thread
//! (`gpu_thread.assertOnContextThread`) and a live `sg` context. Without a
//! context creation fails with `error.NoContext` and the pass entries
//! (`begin`/`clear`/`resize`) return `false`; scene capture
//! (`renderPrimaryView`) reports `CaptureError` instead of a bool so every
//! unsupported capture — invalid target, missing context, unconsumed frame,
//! bad slot, missing camera, mismatched formats — is an explicit error, not
//! a silent skip. The target stays invalid until creation succeeds; dirty
//! flags are never needed because the target has no producer.
//!
//! Feedback discipline: `begin` latches `pass_open` until `end`, and every
//! sampling view reports empty handles while a pass is open — sampling a
//! target mid-capture would feed the pass its own in-flight content, so it
//! fails closed. Scene materials must additionally never bind this target's
//! views when the same scene is captured. Use a separate display pass/scene
//! or two distinct ping-pong targets; a saved previous-frame borrow of the
//! SAME attachment is still feedback and capture preflight rejects it.
//!
//! Linear HDR semantics: the target stores exactly what the bound pipeline
//! writes. The renderer is always linear HDR now: the default `RGBA16F`
//! target holds linear radiance (no display-referred encoding, no hardware
//! sRGB conversion). Explicit sRGB GPU variants (`SRGB8A8` / `SBGR8A8`, see
//! `isSrgbFormat`) perform hardware sRGB encoding on write on backends that
//! support it; v1 recommends the HDR default for any scene capture.
//!
//! Borrow rule for `asTexture`: the returned `Texture` is BORROWED — the
//! target keeps owning every handle (`owns_handles = false`); any
//! `resize`/`deinit` invalidates it. This mirrors the snapshot discipline
//! (borrowed handle VALUES, never ownership transfer).

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const math = @import("math");
const Color4 = math.Color4;
const gpu_thread = @import("gpu_thread.zig");
const msaa = @import("scene/msaa.zig");
const mip = @import("texture/mip.zig");
const Texture = @import("texture/core.zig").Texture;
const scene_draw = @import("scene/draw.zig");
const clustered_lights = @import("scene/clustered_lights.zig");

/// Absolute sanity cap for the pure dimension validation (before any live
/// `sg.queryLimits` check). Well above every real `max_image_size_2d`
/// (816... real backends report 4096..16384); the live limit is enforced
/// separately in `create` via `fitsLiveLimits`.
pub const max_target_dimension: u32 = 16384;

/// Creation descriptor. Narrow by design: size, sample count, formats, and
/// the sampling behaviour. Everything else (single level, CLAMP sampling,
/// STORE semantics) is fixed by the v1 contract below.
pub const RenderTargetDesc = struct {
    width: u32 = 256,
    height: u32 = 256,
    /// Requested MSAA sample count. Snapped through the same policy as the
    /// main target (`msaa.clampSampleCount` + the format MSAA gate): a
    /// request the backend cannot serve degrades to the closest valid count
    /// (never a creation failure). The actual count lands in `sample_count`.
    sample_count: i32 = 1,
    /// `.DEFAULT` resolves to the HDR default (`RGBA16F` — the same format
    /// as the main target, so forward pipelines match). Must be renderable
    /// (`sg.queryPixelformat(fmt).render`).
    color_format: sg.PixelFormat = .DEFAULT,
    /// `.DEFAULT` resolves to the swapchain depth format (else DEPTH).
    /// `.NONE` disables the depth attachment (same as `depth_enabled =
    /// false`). Must be a depth format when enabled.
    depth_format: sg.PixelFormat = .DEFAULT,
    /// False disables the depth attachment regardless of `depth_format`.
    depth_enabled: bool = true,
    /// Sampling filter for `sampleSampler` (LINEAR for refraction-style
    /// magnification, NEAREST for pixel-exact readback-free sampling).
    min_filter: sg.Filter = .LINEAR,
    mag_filter: sg.Filter = .LINEAR,
};

pub const CreateError = error{
    InvalidDimensions,
    ImageTooLarge,
    NoContext,
    TargetCreationFailed,
    UnsupportedColorFormat,
    UnsupportedDepthFormat,
};

/// Pure dimension validation (no `sg` calls): zero sizes are invalid, and
/// `width*height` must not overflow `u32` (the old wrap-to-small-alloc
/// hazard from the texture upload paths). The live backend limit is checked
/// separately in `create` — unit tests stay GPU-free.
pub fn validateDimensions(width: u32, height: u32) CreateError!void {
    if (width == 0 or height == 0) return error.InvalidDimensions;
    if (width > max_target_dimension or height > max_target_dimension)
        return error.ImageTooLarge;
    _ = std.math.mul(u32, width, height) catch return error.ImageTooLarge;
}

/// Snap a requested sample count through the engine MSAA policy for an
/// explicit backend. Pure (no `sg` calls) so the policy matrix stays
/// unit-testable: 1 and below means off, anything else snaps down to the
/// nearest valid count capped by the backend (real backends cap at 4x,
/// DUMMY allows 8x — see `scene/msaa.zig`).
pub fn snappedSamples(requested: i32, backend: sg.Backend) i32 {
    return msaa.clampSampleCount(backend, requested);
}

/// True when the target carries an MSAA resolve pair (the sokol resolve
/// contract: color resolves at end of pass, sampling reads the 1x image).
pub fn needsResolve(sample_count: i32) bool {
    return msaa.needsResolveAttachment(sample_count);
}

/// True for the GPU sRGB variants that perform hardware sRGB conversion on
/// sample/write (as opposed to UNORM targets, which store raw values).
/// BC1/BC2 have no sRGB variant in this sokol checkout and are UNORM-only
/// (same mapping as `Texture.sgPixelFormatForBlock`).
pub fn isSrgbFormat(fmt: sg.PixelFormat) bool {
    return switch (fmt) {
        .SRGB8A8, .SBGR8A8, .BC3_SRGBA, .BC7_SRGBA, .ETC2_SRGB8, .ETC2_SRGB8A8, .ASTC_4x4_SRGBA => true,
        else => false,
    };
}

/// True for the depth/stencil formats a depth attachment may use. Pure
/// (the live `.depth` capability bit is checked separately in `create` via
/// `sg.queryPixelformat`, which also covers backend-specific depth formats).
pub fn isDepthFormat(fmt: sg.PixelFormat) bool {
    return switch (fmt) {
        .DEPTH, .DEPTH_STENCIL => true,
        else => false,
    };
}
/// HDR color default (pure, no `sg` calls): the renderer is always linear
/// HDR, so the default capture format is `RGBA16F` unconditionally.
pub fn defaultColorFormat() sg.PixelFormat {
    return .RGBA16F;
}

/// Swapchain depth format fallback chain shared with the postfx main target:
/// environment default, else DEPTH.
pub fn defaultDepthFormat() sg.PixelFormat {
    const env_def = sg.queryDesc().environment.defaults;
    return if (env_def.depth_format != .DEFAULT and env_def.depth_format != .NONE)
        env_def.depth_format
    else
        .DEPTH;
}

/// Estimated VRAM bytes for a target of this shape: color (+ resolve twin
/// under MSAA) plus depth, via `pixelFormatBytes`. Pure; feeds capacity
/// budgets the same way `Texture.getGpuMemoryBytes` does. MSAA multiplies
/// the color/depth store by the sample count (resolve adds one 1x copy).
pub fn estimatedBytesFor(width: u32, height: u32, color_format: sg.PixelFormat, depth_format: sg.PixelFormat, sample_count: i32) usize {
    const pixels: usize = @as(usize, width) * @as(usize, height);
    const samples: usize = @as(usize, @intCast(@max(1, sample_count)));
    var total: usize = pixels * mip.pixelFormatBytes(color_format) * samples;
    if (needsResolve(sample_count)) total += pixels * mip.pixelFormatBytes(color_format);
    if (depth_format != .NONE) total += pixels * mip.pixelFormatBytes(depth_format) * samples;
    return total;
}

/// Backend capability snapshot gating target creation. Fill with
/// `queryCapabilities`; the pure predicates below it stay unit-testable.
pub const Capabilities = struct {
    backend: sg.Backend = .DUMMY,
    color_sample: bool = false,
    color_filter: bool = false,
    color_render: bool = false,
    color_blend: bool = false,
    color_msaa: bool = false,
    depth_render: bool = false,
    depth_msaa: bool = false,
    max_image_2d: i32 = 0,
    max_samples: i32 = 4,

    /// True when an MSAA count survives the backend cap (same table as the
    /// main target). Pure.
    pub fn supportsSamples(self: Capabilities, requested: i32) bool {
        if (requested <= 1) return true;
        return snappedSamples(requested, self.backend) == requested and
            self.color_msaa and self.depth_msaa;
    }

    /// True when the color format can be rendered into and the depth format
    /// (unless `.NONE`) is a depth format. Pure — the live per-format
    /// capability bits are checked separately in `create`. Blend is NOT
    /// required here: opaque/clear-only targets on other formats stay
    /// creatable; scene capture gates blend separately (see
    /// `supportsCapture`).
    pub fn supportsTarget(self: Capabilities, color_format: sg.PixelFormat, depth_format: sg.PixelFormat) bool {
        if (color_format == .NONE or color_format == .DEFAULT) return false;
        if (!self.color_render) return false;
        if (depth_format == .NONE) return true;
        if (!isDepthFormat(depth_format)) return false;
        return self.depth_render;
    }

    /// True when the pair can carry a scene capture: renderable HDR color
    /// with sample + filter + blend (every scene material family blends
    /// alpha) plus a renderable depth format. Pure.
    pub fn supportsCapture(self: Capabilities, color_format: sg.PixelFormat, depth_format: sg.PixelFormat) bool {
        if (color_format != .RGBA16F) return false;
        if (!self.color_render or !self.color_sample or !self.color_filter or !self.color_blend) return false;
        if (depth_format == .NONE or !isDepthFormat(depth_format)) return false;
        return self.depth_render;
    }
};

/// Live backend capabilities for a format pair. Requires a valid `sg`
/// context (`error.NoContext` headless — the pure `Capabilities` predicates
/// above cover the context-free policy tests).
pub fn queryCapabilities(color_format: sg.PixelFormat, depth_format: sg.PixelFormat) CreateError!Capabilities {
    if (!sg.isvalid()) return error.NoContext;
    const backend = sg.queryBackend();
    const ci = sg.queryPixelformat(color_format);
    const di = if (depth_format != .NONE) sg.queryPixelformat(depth_format) else sg.PixelformatInfo{};
    return .{
        .backend = backend,
        .color_sample = ci.sample,
        .color_filter = ci.filter,
        .color_render = ci.render,
        .color_blend = ci.blend,
        .color_msaa = ci.msaa,
        .depth_render = di.depth,
        .depth_msaa = depth_format == .NONE or di.msaa,
        .max_image_2d = sg.queryLimits().max_image_size_2d,
        .max_samples = msaa.maxSamplesForBackend(backend),
    };
}

/// Scene-capture failures. Every unsupported capture is an error here, not
/// prose: callers (and the smoke in `examples/render_target_basic.zig`)
/// must handle each case instead of silently capturing garbage.
pub const CaptureError = error{
    FeedbackLoop,
    /// Target never created, destroyed, or only partially created.
    InvalidTarget,
    /// No live `sg` context (headless, worker thread, pre-setup).
    NoContext,
    /// A capture pass is already open on this target (re-entrant `begin`).
    /// Close it with `end` first; sampling views stay empty until then.
    PassAlreadyOpen,
    /// `scene.hasConsumableFrame()` is false: no prepare published a frame
    /// yet (or nothing was ever prepared). Prepare first.
    NoConsumableFrame,
    /// Camera index out of the clustered slot table.
    InvalidViewSlot,
    /// The prepared snapshot carries no camera (scene has none).
    NoCamera,
    /// Target color format is not the scene HDR format (`RGBA16F` — the
    /// only scene-capture format, so radiance survives losslessly). Any
    /// other format (explicit RGBA8/BGRA8, sRGB variants, RGBA32F) is
    /// refused. Clear/sample-only use of such targets is unaffected.
    IncompatibleColorFormat,
    /// Target has no depth attachment. Scene capture needs depth: the
    /// forward pipelines are depth-tested with writes on, and a pass
    /// without a depth attachment leaves their depth state undefined.
    DepthRequired,
    /// Target depth format differs from the main-target depth format
    /// (`defaultDepthFormat()`). Same silent-corruption rationale as color.
    IncompatibleDepthFormat,
    /// Defensive: all gates above passed yet `begin` still refused. Cannot
    /// happen on one thread (nothing changes between the gates and the
    /// begin); present so a future lock-free path cannot silently skip.
    PassBeginFailed,
};

pub const SceneRenderOptions = struct {
    clear_color: Color4 = Color4.new(0, 0, 0, 1),
    depth_clear: f32 = 1.0,
    /// Clustered-light view slot this capture uploads (see `view_render`:
    /// every view uploads only its own slot — the sokol one-update rule
    /// forbids two uploads of one slot per frame). Primary main views use
    /// slots 0..7; automatic refraction uses its own slot 8. The default
    /// RTT slot 9 is disjoint from both. Multiple captures of one scene in
    /// one frame require distinct unused slots. With no clustered lights
    /// in the snapshot no upload happens and the slot is irrelevant.
    view_slot: usize = clustered_lights.RTT_VIEW_SLOT,
};

/// A genuine runtime-renderable 2D target: color attachment (+ MSAA resolve
/// pair when `sample_count > 1`), optional depth attachment, a texture view
/// over the sampled image, and a sampler. Context-thread owned; `deinit`
/// destroys immediately (no retire queue — see the module docs).
pub const RenderTarget = struct {
    width: u32 = 0,
    height: u32 = 0,
    sample_count: i32 = 1,
    color_format: sg.PixelFormat = .DEFAULT,
    depth_format: sg.PixelFormat = .NONE,
    min_filter: sg.Filter = .LINEAR,
    mag_filter: sg.Filter = .LINEAR,
    color_image: sg.Image = .{},
    color_att_view: sg.View = .{},
    color_tex_view: sg.View = .{},
    resolve_image: sg.Image = .{},
    resolve_att_view: sg.View = .{},
    depth_image: sg.Image = .{},
    depth_att_view: sg.View = .{},
    depth_tex_view: sg.View = .{},
    sampler: sg.Sampler = .{},
    valid: bool = false,
    /// True between a successful `begin` and its `end` (context thread
    /// only). While open, every sampling view (`sampleView`,
    /// `depthSampleView`, `sampleSampler`, `asTexture`) reports empty
    /// handles: sampling a target mid-capture would feed the pass its own
    /// in-flight content (self-sampling feedback), so sampling fails closed
    /// until the pass closes. Scene materials must additionally never bind
    /// this target's views when the same scene is captured — the smoke
    /// proves the discipline by construction (separate capture/display
    /// scenes with plain materials on the captured side).
    pass_open: bool = false,

    /// Creates the GPU target with `errdefer` rollback: any partial failure
    /// destroys the handles already made and returns an error — never a
    /// half-alive target. Fails closed without a context (`error.NoContext`)
    /// and keeps no allocation behind (all GPU state; no CPU allocs).
    pub fn create(desc: RenderTargetDesc) CreateError!RenderTarget {
        try validateDimensions(desc.width, desc.height);
        if (!sg.isvalid()) return error.NoContext;
        gpu_thread.assertOnContextThread();

        const color_fmt = if (desc.color_format != .DEFAULT) desc.color_format else defaultColorFormat();
        const depth_fmt: sg.PixelFormat = if (!desc.depth_enabled) .NONE else if (desc.depth_format != .DEFAULT) desc.depth_format else defaultDepthFormat();

        const ci = sg.queryPixelformat(color_fmt);
        if (!ci.render or !ci.sample or ((desc.min_filter == .LINEAR or desc.mag_filter == .LINEAR) and !ci.filter))
            return error.UnsupportedColorFormat;
        if (depth_fmt != .NONE) {
            const di = sg.queryPixelformat(depth_fmt);
            if (!di.depth) return error.UnsupportedDepthFormat;
        }

        // Live size gate: the backend's own 2D limit (0 = unknown, skip).
        const live_max = sg.queryLimits().max_image_size_2d;
        if (live_max > 0 and (desc.width > live_max or desc.height > live_max))
            return error.ImageTooLarge;

        // Same effective-sample policy as the main target: unsupported MSAA
        // degrades to 1x rather than failing creation.
        const backend = sg.queryBackend();
        const formats_msaa_capable = ci.msaa and (depth_fmt == .NONE or sg.queryPixelformat(depth_fmt).msaa);
        const samples = msaa.effectiveSampleCount(desc.sample_count, .{
            .formats_msaa_capable = formats_msaa_capable,
            .backend = backend,
        });

        var self = RenderTarget{
            .width = desc.width,
            .height = desc.height,
            .sample_count = samples,
            .color_format = color_fmt,
            .depth_format = depth_fmt,
            .min_filter = desc.min_filter,
            .mag_filter = desc.mag_filter,
        };
        // Rollback: every handle below is stored into `self` first and each
        // step is validated with `query*State != .VALID` (never id-only), so
        // a later failure destroys all earlier FAILED-but-nonzero objects —
        // views before images — and never publishes a half-alive target.
        errdefer self.deinit();

        const w: i32 = @intCast(desc.width);
        const h: i32 = @intCast(desc.height);

        self.color_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = w,
            .height = h,
            .pixel_format = color_fmt,
            .sample_count = samples,
        });
        if (sg.queryImageState(self.color_image) != .VALID) return error.TargetCreationFailed;

        self.color_att_view = sg.makeView(.{ .color_attachment = .{ .image = self.color_image } });
        if (sg.queryViewState(self.color_att_view) != .VALID) return error.TargetCreationFailed;

        if (needsResolve(samples)) {
            self.resolve_image = sg.makeImage(.{
                .usage = .{ .resolve_attachment = true },
                .width = w,
                .height = h,
                .pixel_format = color_fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(self.resolve_image) != .VALID) return error.TargetCreationFailed;

            self.resolve_att_view = sg.makeView(.{ .resolve_attachment = .{ .image = self.resolve_image } });
            if (sg.queryViewState(self.resolve_att_view) != .VALID) return error.TargetCreationFailed;

            self.color_tex_view = sg.makeView(.{ .texture = .{ .image = self.resolve_image } });
            if (sg.queryViewState(self.color_tex_view) != .VALID) return error.TargetCreationFailed;
        } else {
            self.color_tex_view = sg.makeView(.{ .texture = .{ .image = self.color_image } });
            if (sg.queryViewState(self.color_tex_view) != .VALID) return error.TargetCreationFailed;
        }

        if (depth_fmt != .NONE) {
            self.depth_image = sg.makeImage(.{
                .usage = .{ .depth_stencil_attachment = true },
                .width = w,
                .height = h,
                .pixel_format = depth_fmt,
                .sample_count = samples,
            });
            if (sg.queryImageState(self.depth_image) != .VALID) return error.TargetCreationFailed;

            self.depth_att_view = sg.makeView(.{ .depth_stencil_attachment = .{ .image = self.depth_image } });
            if (sg.queryViewState(self.depth_att_view) != .VALID) return error.TargetCreationFailed;

            // No depth resolve in this sokol version (see scene/msaa.zig):
            // only the 1x depth gets a texture view; MSAA depth is
            // write-only for the capturing pass by design.
            if (samples == 1) {
                self.depth_tex_view = sg.makeView(.{ .texture = .{ .image = self.depth_image } });
                if (sg.queryViewState(self.depth_tex_view) != .VALID) return error.TargetCreationFailed;
            }
        }

        self.sampler = sg.makeSampler(.{
            .min_filter = desc.min_filter,
            .mag_filter = desc.mag_filter,
            .mipmap_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });
        if (sg.querySamplerState(self.sampler) != .VALID) return error.TargetCreationFailed;

        self.valid = true;
        return self;
    }

    /// Destroys every handle immediately (context thread; the caller owns
    /// the frame ordering — same discipline as `PostProcessPass.deinit`).
    /// Headless-safe: without a context there is nothing to destroy, the
    /// value still resets so no dangling ids survive. Idempotent.
    pub fn deinit(self: *RenderTarget) void {
        defer self.* = .{};
        if (!sg.isvalid()) return;
        gpu_thread.assertOnContextThread();
        // Destroying mid-pass would leave the open pass dangling: a
        // leak-free misuse tripwire (same discipline as the pin guards —
        // asserts in Debug/ReleaseSafe, compiled out of ReleaseFast/Small
        // with the assert helper).
        std.debug.assert(!self.pass_open);
        if (self.sampler.id != 0) sg.destroySampler(self.sampler);
        if (self.color_tex_view.id != 0) sg.destroyView(self.color_tex_view);
        if (self.depth_tex_view.id != 0) sg.destroyView(self.depth_tex_view);
        if (self.color_att_view.id != 0) sg.destroyView(self.color_att_view);
        if (self.resolve_att_view.id != 0) sg.destroyView(self.resolve_att_view);
        if (self.depth_att_view.id != 0) sg.destroyView(self.depth_att_view);
        if (self.resolve_image.id != 0) sg.destroyImage(self.resolve_image);
        if (self.depth_image.id != 0) sg.destroyImage(self.depth_image);
        if (self.color_image.id != 0) sg.destroyImage(self.color_image);
    }

    pub fn isValid(self: *const RenderTarget) bool {
        return self.valid and self.color_image.id != 0 and self.color_att_view.id != 0 and self.color_tex_view.id != 0;
    }

    pub fn hasDepth(self: *const RenderTarget) bool {
        return self.depth_format != .NONE and self.depth_image.id != 0;
    }

    /// True between a successful `begin` and its `end`. Sampling views read
    /// empty while this holds (see the `pass_open` field docs).
    pub fn isCapturing(self: *const RenderTarget) bool {
        return self.pass_open;
    }

    /// Estimated VRAM bytes for this target's current shape. Pure.
    pub fn estimatedBytes(self: *const RenderTarget) usize {
        if (!self.isValid()) return 0;
        return estimatedBytesFor(self.width, self.height, self.color_format, self.depth_format, self.sample_count);
    }

    /// Recreates the target at a new size, preserving formats, sample
    /// count, and sampler behaviour. Error-safe with rollback: the old
    /// target (and its content) survives when recreation fails. Same-size
    /// calls are a no-op `true`. Returns false without a context or on
    /// creation failure (fail-closed: the old target stays valid).
    pub fn resize(self: *RenderTarget, width: u32, height: u32) bool {
        // Never rebuild under an open pass: the old handles are bound
        // right now. Fail closed, old target (and content) intact.
        if (self.pass_open) return false;
        if (width == self.width and height == self.height) return self.isValid();
        const desc = RenderTargetDesc{
            .width = width,
            .height = height,
            .sample_count = self.sample_count,
            // Resolved formats ride through verbatim (no re-defaulting:
            // the target keeps the exact formats it was created with).
            .color_format = if (self.color_format != .DEFAULT) self.color_format else .DEFAULT,
            .depth_format = self.depth_format,
            .depth_enabled = self.depth_format != .NONE,
            .min_filter = self.min_filter,
            .mag_filter = self.mag_filter,
        };
        const rebuilt = RenderTarget.create(desc) catch return false;
        self.deinit();
        self.* = rebuilt;
        return true;
    }

    /// Begins a render pass into the target, clearing color (+ depth when
    /// present). Returns false fail-closed (invalid, headless, or a pass
    /// already open) — the caller must then skip both draws AND `end`.
    /// Viewport/scissor are set to the target rect; the caller restores them
    /// after `end` (or uses `renderPrimaryView`, which restores
    /// automatically).
    pub fn begin(self: *RenderTarget, clear_color: Color4, depth_clear: f32) bool {
        if (!self.isValid()) return false;
        if (!sg.isvalid()) return false;
        gpu_thread.assertOnContextThread();
        if (self.pass_open) return false;
        var action = sg.PassAction{};
        // MSAA color/depth stores are DONTCARE (the resolve carries the
        // color out; MSAA depth is write-only by design — same rule as the
        // postfx main target). 1x stores both for later sampling.
        const msaa_active = needsResolve(self.sample_count);
        action.colors[0] = .{
            .load_action = .CLEAR,
            .store_action = if (msaa_active) .DONTCARE else .STORE,
            .clear_value = .{ .r = clear_color.r, .g = clear_color.g, .b = clear_color.b, .a = clear_color.a },
        };
        if (self.hasDepth()) {
            action.depth = .{
                .load_action = .CLEAR,
                .store_action = if (msaa_active) .DONTCARE else .STORE,
                .clear_value = depth_clear,
            };
        }
        var pass = sg.Pass{ .action = action };
        pass.attachments.colors[0] = self.color_att_view;
        if (needsResolve(self.sample_count)) pass.attachments.resolves[0] = self.resolve_att_view;
        if (self.hasDepth()) pass.attachments.depth_stencil = self.depth_att_view;
        sg.beginPass(pass);
        const w: i32 = @intCast(self.width);
        const h: i32 = @intCast(self.height);
        sg.applyViewport(0, 0, w, h, true);
        sg.applyScissorRect(0, 0, w, h, true);
        self.pass_open = true;
        return true;
    }

    /// Ends the pass opened by `begin`. Call exactly once per successful
    /// `begin`, on the same thread. Fail-safe no-op without an open pass
    /// (never issues a bare `sg.endPass`, which would be a validation
    /// error).
    pub fn end(self: *RenderTarget) void {
        gpu_thread.assertOnContextThread();
        if (!self.pass_open) return;
        sg.endPass();
        self.pass_open = false;
    }

    /// Clear-only pass (no draws): genuine GPU clear of the attachments.
    /// Returns false fail-closed (invalid, headless, or pass already open).
    pub fn clear(self: *RenderTarget, clear_color: Color4, depth_clear: f32) bool {
        if (!self.begin(clear_color, depth_clear)) return false;
        self.end();
        return true;
    }

    /// Texture view for sampling the captured color in later passes (the
    /// 1x resolve image under MSAA, the color image otherwise). Empty when
    /// invalid — and empty while a capture pass is open (self-sampling
    /// feedback fails closed): bind sites must fall back while it is.
    pub fn sampleView(self: *const RenderTarget) sg.View {
        if (self.pass_open) return .{};
        return self.color_tex_view;
    }

    pub fn sampleSampler(self: *const RenderTarget) sg.Sampler {
        if (self.pass_open) return .{};
        return self.sampler;
    }

    /// Depth texture view for sampling captured depth (soft-particle style
    /// fades, refraction absorption). Empty without depth, under MSAA
    /// (write-only MSAA depth — same rule as the main target), or while a
    /// capture pass is open.
    pub fn depthSampleView(self: *const RenderTarget) sg.View {
        if (self.pass_open) return .{};
        return self.depth_tex_view;
    }

    pub fn colorAttachmentView(self: *const RenderTarget) sg.View {
        return self.color_att_view;
    }

    pub fn depthAttachmentView(self: *const RenderTarget) sg.View {
        return self.depth_att_view;
    }

    /// Borrowed `Texture` over the captured color for slots that speak
    /// `Texture` (material uniforms, debug views). BORROWED: the target
    /// keeps owning every handle (`Texture.deinit` on the borrow is a no-op), and
    /// any `resize`/`deinit` invalidates it. The sampler/filter state is
    /// the target's own (single level, `num_mipmaps = 1`). While a capture
    /// pass is open the handles report empty (dims/format metadata intact)
    /// — same self-sampling rule as `sampleView`.
    pub fn asTexture(self: *const RenderTarget) Texture {
        const hdr = self.color_format == .RGBA16F or self.color_format == .RGBA32F;
        if (self.pass_open) {
            return .{
                .image = .{},
                .view = .{},
                .sampler = .{},
                .width = self.width,
                .height = self.height,
                .num_mipmaps = 1,
                .format = self.color_format,
                .is_hdr = hdr,
                .owns_handles = false,
            };
        }
        return .{
            .image = if (needsResolve(self.sample_count)) self.resolve_image else self.color_image,
            .view = self.color_tex_view,
            .sampler = self.sampler,
            .width = self.width,
            .height = self.height,
            .num_mipmaps = 1,
            .format = self.color_format,
            .is_hdr = hdr,
            .owns_handles = false,
        };
    }

    /// Renders the prepared PRIMARY view into the target through the
    /// existing scene seam (`Scene.renderSceneView`): the
    /// same opaque + transparent queues, sky, particles, and debug lines the
    /// main pass would draw, from the staged snapshot the last prepare
    /// published.
    ///
    /// Hardened front-slot discipline (the v1 borrow is gone): the call
    /// requires `scene.hasConsumableFrame()` and holds a consumer pin
    /// (`pinFront`/`unpin`, same as `Scene.render`) across the whole read,
    /// so a concurrent producer cannot reclaim the slot mid-capture and an
    /// invalid/never-prepared front is refused up front instead of drawn.
    /// Scene-capture format contract is enforced, not documented-and-hoped:
    /// the color format must be the scene HDR format (`RGBA16F`) with live
    /// sample + filter + render + blend caps (every scene material family
    /// blends alpha), and a depth attachment in `defaultDepthFormat()` is
    /// required (see `CaptureError`; clear/sample-only targets with exotic
    /// formats keep working — only scene capture is gated).
    ///
    /// Frame-ordering contract (for the pipeline owner): call between
    /// a staged begin/finish and `Scene.render`, while
    /// the prepared draws are consumable. Shadow maps and probe contents
    /// are the previous frame's when the capture runs before this frame's
    /// shadow/probe passes — the same documented one-frame lag as probe
    /// captures. Clustered slot selection follows `SceneRenderOptions`;
    /// out-of-range slots are an error (the draw would otherwise silently
    /// clamp onto another view's slot).
    pub fn renderPrimaryView(self: *RenderTarget, scene: anytype, opts: SceneRenderOptions) CaptureError!void {
        if (!self.isValid()) return error.InvalidTarget;
        if (self.pass_open) return error.PassAlreadyOpen;
        if (opts.view_slot >= clustered_lights.MAX_VIEW_SLOTS) return error.InvalidViewSlot;
        if (!sg.isvalid()) return error.NoContext;
        gpu_thread.assertOnContextThread();
        if (!scene.hasConsumableFrame()) return error.NoConsumableFrame;

        // HDR scene contract before touching any slot: scene capture keeps
        // radiance lossless in RGBA16F through the matching forward set.
        // Float32 is deliberately NOT accepted (one contract, no silent
        // precision/format drift). Blend is required: scene materials blend.
        if (self.color_format != .RGBA16F) return error.IncompatibleColorFormat;
        if (!self.hasDepth()) return error.DepthRequired;
        if (self.depth_format != defaultDepthFormat()) return error.IncompatibleDepthFormat;
        {
            const caps = sg.queryPixelformat(self.color_format);
            if (!caps.render or !caps.sample or !caps.blend) return error.IncompatibleColorFormat;
            if ((self.min_filter == .LINEAR or self.mag_filter == .LINEAR) and !caps.filter)
                return error.IncompatibleColorFormat;
        }

        // Resolve the forward set BEFORE opening the pass: forwardFor may
        // recreate the MSAA/format twin (destroying the previous one), which
        // must never happen mid-pass. The pointer borrows Scene state and
        // stays valid across the capture (no producer mutation mid-capture:
        // the pin below holds the prepared slot).
        const fwd_pipelines = scene.forwardFor(self.sample_count, self.color_format);

        // Consumer pin across the whole capture (render's own discipline):
        // the presenting capture holds the lease, so a concurrent producer
        // build cannot reclaim this slot mid-draw. Unpin is mandatory —
        // the defer covers every return below, including the NoCamera and
        // begin paths (no partial pass is ever left open: begin runs only
        // after every gate passed).
        const pinned_idx = scene.draws.pinFront();
        defer scene.draws.unpin(pinned_idx) catch {};
        const draws = scene.preparedDraws();
        const snap = &draws.snapshot;
        if (!snap.has_camera) return error.NoCamera;
        // A saved Texture borrow bypasses sampleView's pass-open gate.
        // Refuse feedback before opening a pass, including alternate views
        // into the same image. Inspect staged records, never live materials.
        if (self.queuesSampleSelf(&draws.primary, true)) return error.FeedbackLoop;
        for (scene.particles.frame.items) |record| if (self.recordSamplesSelf(record)) return error.FeedbackLoop;
        if (!self.begin(opts.clear_color, opts.depth_clear)) return error.PassBeginFailed;

        // Render-owned draw environment: snapshot copies only (same shape
        // as frame_render's main pass, minus multi-camera and TAA jitter —
        // the capture is always the primary view, unjittered). The forward
        // set was resolved above (never mid-pass).
        const env = scene_draw.Environment{
            .pipelines = fwd_pipelines,
            .stats = &scene.stats,
            .default_white = snap.default_white,
            .default_normal = snap.default_normal,
            .default_cube = snap.default_cube,
            .sky_texture = snap.sky_texture,
            .ibl_intensity = snap.ibl_intensity,
            .probes = snap.probe_pack.entries[0..snap.probe_pack.count],
            .shadow_pass = &scene.shadows.pass,
            .shadow_uniforms = snap.shadow_uniforms,
            .clustered = &scene.clustered,
        };
        var capture_snap = snap.*;
        capture_snap.screen_w = @intCast(self.width);
        capture_snap.screen_h = @intCast(self.height);
        var capture_cam = snap.primary_cam;
        capture_cam.viewport = .{};
        scene.renderSceneView(
            capture_cam,
            &draws.primary,
            draws.outline_items.items,
            draws.outline_skins.items,
            self.sample_count,
            &capture_snap,
            env,
            opts.view_slot,
        );
        self.end();

        // Viewport/scissor restore: the main pass sets its own per-view
        // rects, but leave no stale target-sized state behind (same habit
        // as the probe capture path).
        const rw = if (snap.screen_w > 0) snap.screen_w else sapp.width();
        const rh = if (snap.screen_h > 0) snap.screen_h else sapp.height();
        sg.applyViewport(0, 0, rw, rh, true);
        sg.applyScissorRect(0, 0, rw, rh, true);
    }

    /// Capture preflight over frozen geometry/material handles. Refraction
    /// excludes transparent geometry, so its preflight does the same.
    pub fn queuesSampleSelf(self: *const RenderTarget, queues: anytype, include_transparent: bool) bool {
        for (queues.items.items) |item| if (self.itemSamplesSelf(item, queues)) return true;
        for (queues.opaque_instanced.items) |batch| if (self.recordSamplesSelf(batch.draw_record)) return true;
        if (include_transparent) {
            for (queues.transparent.items) |item| if (self.itemSamplesSelf(item, queues)) return true;
            for (queues.transparent_instanced.items) |batch| if (self.recordSamplesSelf(batch.draw_record)) return true;
        }
        return false;
    }

    fn itemSamplesSelf(self: *const RenderTarget, item: anytype, queues: anytype) bool {
        if (self.recordSamplesSelf(item.draw_record) or self.viewSamplesSelf(item.morph_view)) return true;
        if (item.shader_index) |i| {
            if (i < queues.shader_storage.items.len and self.recordSamplesSelf(queues.shader_storage.items[i])) return true;
        }
        return false;
    }

    pub fn recordSamplesSelf(self: *const RenderTarget, record: anytype) bool {
        inline for (@typeInfo(@TypeOf(record)).@"struct".fields) |field| {
            if (field.type == sg.View) {
                if (self.viewSamplesSelf(@field(record, field.name))) return true;
            } else if (field.type == ?sg.View) {
                if (@field(record, field.name)) |view| if (self.viewSamplesSelf(view)) return true;
            }
        }
        return false;
    }

    pub fn viewSamplesSelf(self: *const RenderTarget, view: sg.View) bool {
        if (view.id == 0) return false;
        if (view.id == self.color_tex_view.id or view.id == self.depth_tex_view.id) return true;
        if (!sg.isvalid() or sg.queryViewState(view) != .VALID) return false;
        const image = sg.queryViewDesc(view).texture.image;
        return image.id != 0 and (image.id == self.color_image.id or
            image.id == self.resolve_image.id or image.id == self.depth_image.id);
    }
};

test {
    _ = @import("render_target_tests.zig");
}
