//! Full test-suite aggregation. Imported from root.zig's `test` block: a plain
//! `@import` in non-test code never collects `test` decls, only the
//! `test { _ = @import(...); }` chain rooted at the test binary does.
//!
//! math is a separate build module (see build.zig), so it must be imported by
//! module name here; math.zig aggregates the tests of math/*.zig itself.

test {
    _ = @import("math");

    _ = @import("ai.zig"); // pulls ai/tests.zig
    _ = @import("animation/animation.zig"); // pulls animation/tests.zig
    _ = @import("animation/easing.zig");
    _ = @import("animation/skeleton.zig");
    _ = @import("audio.zig"); // pulls audio/tests.zig
    _ = @import("camera.zig");
    _ = @import("export/obj.zig");
    _ = @import("export/stl.zig");
    _ = @import("lights.zig");
    _ = @import("loader/gltf_util.zig");
    _ = @import("loader/lights.zig");
    _ = @import("loader/materials.zig");
    _ = @import("loader/meshopt.zig");
    _ = @import("loader/obj.zig");
    _ = @import("loader/ply.zig");
    _ = @import("loader/stl.zig");
    _ = @import("material.zig");
    _ = @import("mesh.zig"); // pulls mesh/tests.zig and mesh/csg_tests.zig
    _ = @import("particles.zig");
    _ = @import("passes/debug_pass.zig");
    _ = @import("passes/outline_pass.zig");
    _ = @import("physics.zig"); // pulls physics/tests.zig
    _ = @import("physics_mesh.zig");
    _ = @import("postprocess.zig");
    _ = @import("ragdoll.zig");
    _ = @import("scene.zig");
    _ = @import("scene/animation_runtime.zig");
    _ = @import("scene/content.zig");
    _ = @import("scene/light_selection.zig");
    _ = @import("scene/light_rig.zig");
    _ = @import("scene/pipelines.zig");
    _ = @import("scene/forward_pipelines.zig");
    _ = @import("scene/project_cache.zig");
    _ = @import("scene/render_queue.zig");
    _ = @import("scene/shadow_pcss.zig");
    _ = @import("scene/shadow_system.zig");
    _ = @import("scene/sky_layer.zig");
    _ = @import("scene/postfx_stack.zig");
    _ = @import("scene/uniforms.zig");
    _ = @import("serialization.zig");
    _ = @import("texture.zig");
    _ = @import("ui.zig");
    _ = @import("ui/types.zig");
    _ = @import("ui/theme.zig");
    _ = @import("ui/transition.zig");
    _ = @import("ui/css_parser.zig");
    _ = @import("vehicle.zig");
    _ = @import("visibility/mod.zig"); // pulls visibility/tests.zig
}
