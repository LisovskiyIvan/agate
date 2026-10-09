//! Two-level application facade over sokol_app window lifecycle and GPU context.
//!
//! Level 1 (High-Level API):
//! - `agate.App` owns the sokol_app window lifecycle (`init`, `frame`, `cleanup`, `event`).
//! - Manages resize, DPI scaling, delta-timing, frame counting, and input dispatch.
//! - Integrated with `agate.Scene` and `agate.Runtime` staged-frame execution.
//! - Configuration options for backend floor (GL 4.3+, Metal, WebGPU, D3D11), MSAA,
//!   sRGB backbuffer, and pool sizes.
//!
//! Level 2 (Low-Level Escape):
//! - Direct, raw access to `sokol.gfx` (`sg.*`) and `sokol.app` (`sapp.*`) is fully
//!   preserved and supported alongside `App`.
//! - Context thread registration (`gpu_thread.markContextThread()`) is performed on init,
//!   allowing immediate raw GPU resource creation and commands without wrapping.
//!
//! Headless Mode:
//! - `App.runHeadless(steps)` executes the complete lifecycle on CPU without opening a
//!   window or requiring a GPU, enabling headless testing and CI validation.
//!
//! Dimension units: `window_width`/`window_height` (and `width()`/`height()`)
//! are FRAMEBUFFER pixels, matching `sapp.width()`/`sapp.height()`. The
//! `RESIZED` event carries both units; `App` stores the framebuffer pair.
//! Logical (screen-coordinate) size is `logicalWidth()`/`logicalHeight()`
//! (framebuffer divided by `dpi_scale`).
//!
//! Event contract: when an `event` callback is installed it owns ALL input
//! routing (including forwarding to `App.scene`, e.g. after UI layers
//! consume). `App` forwards events to the scene only when no callback is
//! installed, so cameras never observe UI-consumed input.
//!
//! Ownership: `initScene`/`initRuntime` create App-owned systems (repeated
//! calls return the existing instance, never dropping live state).
//! `onCleanup` quiesces the owned worker BEFORE user cleanup runs, then
//! deinits the owned runtime, the owned scene, and the sg context.
//! Borrowed (`scene`/`runtime` pointing at caller memory) systems are never
//! touched on cleanup.

const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const sg = sokol.gfx;
const sglue = sokol.glue;
const stime = sokol.time;

const Scene = @import("scene.zig").Scene;
const runtime_mod = @import("runtime.zig");
const Runtime = runtime_mod.Runtime;
const FrameResult = runtime_mod.FrameResult;
const gpu_thread = @import("gpu_thread.zig");

/// Explicit window and graphics backend configuration.
pub const AppConfig = struct {
    /// Window title shown in window decorations.
    title: [:0]const u8 = "Agate Application",
    /// Initial logical window width in screen coordinates.
    width: i32 = 1280,
    /// Initial logical window height in screen coordinates.
    height: i32 = 720,
    /// Multi-sample anti-aliasing sample count (1 = disabled, 2, 4).
    sample_count: i32 = 1,
    /// Support high-DPI (Retina) display resolutions.
    high_dpi: bool = true,
    /// Disable vertical sync for unconstrained benchmarking/profiling.
    disable_vsync: bool = true,
    /// Request hardware sRGB format for the default swapchain backbuffer.
    srgb_backbuffer: bool = false,
    /// Desktop OpenGL major floor (4.3 required for clustered compute/storage).
    gl_major: i32 = 4,
    /// Desktop OpenGL minor floor.
    gl_minor: i32 = 3,
    /// Request WebGPU device timing extensions on startup.
    wgpu_gpu_timing_enabled: bool = true,
    /// Disable display synchronization on Metal macOS backends.
    metal_disable_display_sync: bool = true,
    /// Custom sokol log callback.
    logger: ?*const fn ([*c]const u8, u32, u32, [*c]const u8, u32, [*c]const u8, ?*anyopaque) callconv(.c) void = sokol.log.func,

    // Resource pool sizing for sokol-gfx setup:
    buffer_pool_size: i32 = 4096,
    image_pool_size: i32 = 256,
    sampler_pool_size: i32 = 128,
    view_pool_size: i32 = 256,
    pipeline_pool_size: i32 = 256,
    shader_pool_size: i32 = 64,
    uniform_buffer_size: i32 = 16 * 1024 * 1024,

    /// When true, runs in CPU-only headless mode without calling `sapp.run`.
    headless: bool = false,
};

/// High-level user callbacks (event-routing contract: see module docs).
pub const AppCallbacks = struct {
    /// Invoked once after GPU context and subsystems are initialized.
    init: ?*const fn (*App) void = null,
    /// Invoked each frame to update simulation and/or issue rendering.
    frame: ?*const fn (*App) void = null,
    /// Invoked before context teardown to release user resources.
    cleanup: ?*const fn (*App) void = null,
    /// Invoked for window, keyboard, and mouse input events.
    event: ?*const fn (*App, [*c]const sapp.Event) void = null,
};

/// Singleton active app pointer for sokol_app C-ABI callbacks.
var active_app: ?*App = null;

export fn appInitCb() callconv(.c) void {
    if (active_app) |app| app.onInit();
}

export fn appFrameCb() callconv(.c) void {
    if (active_app) |app| app.onFrame();
}

export fn appCleanupCb() callconv(.c) void {
    if (active_app) |app| app.onCleanup();
}

export fn appEventCb(ev: [*c]const sapp.Event) callconv(.c) void {
    if (active_app) |app| {
        app.onEvent(ev);
    }
}

/// The Agate Application coordinator.
pub const App = struct {
    allocator: std.mem.Allocator,
    config: AppConfig,
    callbacks: AppCallbacks,
    user_data: ?*anyopaque = null,

    // Optional engine systems owned directly by App
    scene: ?*Scene = null,
    runtime: ?*Runtime = null,
    own_scene: ?Scene = null,
    own_runtime: ?Runtime = null,

    // Timing and window metrics
    frame_count: u64 = 0,
    time_seconds: f64 = 0.0,
    delta_time: f32 = 0.0,
    last_time: u64 = 0,
    /// Framebuffer width in physical pixels (see module docs).
    window_width: i32 = 1280,
    /// Framebuffer height in physical pixels (see module docs).
    window_height: i32 = 720,
    dpi_scale: f32 = 1.0,

    running: bool = false,
    is_headless: bool = false,

    /// Initializes an application descriptor with the specified configuration and callbacks.
    pub fn init(allocator: std.mem.Allocator, config: AppConfig, callbacks: AppCallbacks) App {
        return .{
            .allocator = allocator,
            .config = config,
            .callbacks = callbacks,
            .window_width = config.width,
            .window_height = config.height,
            .is_headless = config.headless,
        };
    }

    /// Convenience helper to allocate and initialize an internal Scene owned by App.
    /// Guarded: repeated calls return the existing owned scene (never drops it).
    /// Requires a live GPU context (Scene builds GPU resources); call from the
    /// init callback or later, never headless.
    pub fn initScene(self: *App) *Scene {
        if (self.own_scene) |*owned| {
            self.scene = owned;
            return self.scene.?;
        }
        self.own_scene = Scene.init(self.allocator);
        self.scene = &self.own_scene.?;
        return self.scene.?;
    }

    /// Convenience helper to initialize an internal Runtime owned by App.
    /// Guarded: repeated calls return the existing owned runtime (a live
    /// worker is never dropped).
    pub fn initRuntime(self: *App) *Runtime {
        if (self.own_runtime) |*owned| {
            self.runtime = owned;
            return self.runtime.?;
        }
        self.own_runtime = Runtime.init();
        self.runtime = &self.own_runtime.?;
        return self.runtime.?;
    }

    /// sokol-app startup descriptor built from config.
    pub fn sappDesc(self: *const App) sapp.Desc {
        return .{
            .init_cb = appInitCb,
            .frame_cb = appFrameCb,
            .cleanup_cb = appCleanupCb,
            .event_cb = appEventCb,
            .window_title = self.config.title.ptr,
            .width = self.config.width,
            .height = self.config.height,
            .sample_count = self.config.sample_count,
            .srgb = self.config.srgb_backbuffer,
            .gl = .{ .major_version = self.config.gl_major, .minor_version = self.config.gl_minor },
            .high_dpi = self.config.high_dpi,
            .disable_vsync = self.config.disable_vsync,
            .wgpu_gpu_timing_enabled = self.config.wgpu_gpu_timing_enabled,
            .metal = .{
                .disable_display_sync = self.config.metal_disable_display_sync,
            },
            .logger = if (self.config.logger) |l| .{ .func = l } else .{},
        };
    }

    /// sokol-gfx descriptor built from config (`env` supplied by the caller).
    pub fn sgDesc(self: *const App, env: sg.Environment) sg.Desc {
        return .{
            .environment = env,
            .logger = if (self.config.logger) |l| .{ .func = l } else .{},
            .buffer_pool_size = self.config.buffer_pool_size,
            .image_pool_size = self.config.image_pool_size,
            .sampler_pool_size = self.config.sampler_pool_size,
            .view_pool_size = self.config.view_pool_size,
            .pipeline_pool_size = self.config.pipeline_pool_size,
            .shader_pool_size = self.config.shader_pool_size,
            .uniform_buffer_size = self.config.uniform_buffer_size,
        };
    }

    /// Starts the application lifecycle. Calls `sapp.run` in windowed mode,
    /// or runs a single tick in headless mode.
    pub fn run(self: *App) void {
        if (self.config.headless) {
            self.runHeadless(1);
            return;
        }

        std.debug.assert(active_app == null);
        active_app = self;
        sapp.run(self.sappDesc());
    }

    /// Executes `steps` frames headlessly without window creation or GPU requirement.
    pub fn runHeadless(self: *App, steps: usize) void {
        std.debug.assert(active_app == null);
        active_app = self;
        self.is_headless = true;
        self.onInit();
        var i: usize = 0;
        while (i < steps) : (i += 1) {
            if (!self.running) break;
            self.onFrame();
        }
        self.onCleanup();
    }

    /// Requests window closure and termination of the application loop.
    pub fn requestQuit(self: *App) void {
        self.running = false;
        if (!self.is_headless) {
            sapp.quit();
        }
    }

    /// Lifecycle hook: context initialization.
    pub fn onInit(self: *App) void {
        gpu_thread.markContextThread();
        if (!self.is_headless) {
            stime.setup();
            self.last_time = stime.now();
            if (!sg.isvalid()) {
                sg.setup(self.sgDesc(sglue.environment()));
            }
            self.window_width = sapp.width();
            self.window_height = sapp.height();
            self.dpi_scale = sapp.dpiScale();
        }
        self.running = true;

        if (self.callbacks.init) |cb| {
            cb(self);
        }
    }

    /// Lifecycle hook: frame update and rendering.
    pub fn onFrame(self: *App) void {
        if (!self.is_headless) {
            self.window_width = sapp.width();
            self.window_height = sapp.height();
            self.dpi_scale = sapp.dpiScale();
            const now = stime.now();
            if (self.last_time != 0) {
                self.delta_time = @floatCast(stime.sec(stime.since(self.last_time)));
                self.time_seconds += self.delta_time;
            }
            self.last_time = now;
        } else {
            self.delta_time = 1.0 / 60.0;
            self.time_seconds += self.delta_time;
        }

        if (self.callbacks.frame) |cb| {
            cb(self);
        } else {
            // Default frame behaviour: advance staged frame if runtime/scene are available.
            _ = self.renderStagedFrame();
        }

        self.frame_count += 1;
    }

    /// Lifecycle hook: event handling.
    pub fn onEvent(self: *App, ev: [*c]const sapp.Event) void {
        if (ev != null and ev.*.type == .RESIZED) {
            // Framebuffer units (matches sapp.width()/height()); fall back
            // to window units when the backend reports no framebuffer size.
            self.window_width = if (ev.*.framebuffer_width > 0) ev.*.framebuffer_width else ev.*.window_width;
            self.window_height = if (ev.*.framebuffer_height > 0) ev.*.framebuffer_height else ev.*.window_height;
        }

        if (self.callbacks.event) |cb| {
            cb(self, ev);
        } else if (self.scene) |sc| {
            if (ev != null) {
                sc.handleEvent(ev);
            }
        }
    }

    /// Lifecycle hook: cleanup and context teardown.
    pub fn onCleanup(self: *App) void {
        self.running = false;
        if (self.own_runtime) |*rt| {
            rt.quiesce();
        }

        if (self.callbacks.cleanup) |cb| {
            cb(self);
        }

        if (self.own_runtime) |*rt| {
            rt.deinit();
            self.own_runtime = null;
            self.runtime = null;
        }

        if (self.own_scene) |*sc| {
            sc.deinit();
            self.own_scene = null;
            self.scene = null;
        }

        if (!self.is_headless) {
            if (sg.isvalid()) {
                sg.shutdown();
            }
        }

        if (active_app == self) {
            active_app = null;
        }
    }

    /// Advances the staged-frame pipeline using `runtime` and `scene`.
    pub fn renderStagedFrame(self: *App) FrameResult {
        const rt = self.runtime orelse return .skipped;
        const sc = self.scene orelse return .skipped;
        return rt.renderFrame(sc);
    }

    // Accessors (framebuffer-pixel units unless noted).

    /// Framebuffer width in physical pixels (matches `sapp.width()`).
    pub inline fn width(self: *const App) i32 {
        return self.window_width;
    }

    /// Framebuffer height in physical pixels (matches `sapp.height()`).
    pub inline fn height(self: *const App) i32 {
        return self.window_height;
    }

    pub inline fn dpiScale(self: *const App) f32 {
        return self.dpi_scale;
    }

    /// Logical width in screen coordinates (framebuffer divided by DPI scale).
    pub inline fn logicalWidth(self: *const App) i32 {
        return @intFromFloat(@as(f32, @floatFromInt(self.window_width)) / self.dpi_scale);
    }

    /// Logical height in screen coordinates (framebuffer divided by DPI scale).
    pub inline fn logicalHeight(self: *const App) i32 {
        return @intFromFloat(@as(f32, @floatFromInt(self.window_height)) / self.dpi_scale);
    }

    /// Framebuffer width in physical pixels (same as `width()`; for GPU-target sizing).
    pub inline fn framebufferWidth(self: *const App) i32 {
        return self.window_width;
    }

    /// Framebuffer height in physical pixels (same as `height()`; for GPU-target sizing).
    pub inline fn framebufferHeight(self: *const App) i32 {
        return self.window_height;
    }

    pub inline fn deltaTime(self: *const App) f32 {
        return self.delta_time;
    }

    pub inline fn time(self: *const App) f64 {
        return self.time_seconds;
    }

    pub inline fn frameCount(self: *const App) u64 {
        return self.frame_count;
    }

    pub fn getUserData(self: *App, comptime T: type) *T {
        return @ptrCast(@alignCast(self.user_data.?));
    }
};
