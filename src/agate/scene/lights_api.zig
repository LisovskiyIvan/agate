//! Scene light API: hemispheric/point/spot/directional/area/clustered
//! create/add/remove/get/count plus the update-phase `updateLights`
//! top-k pack. Split out of `scene.zig` (facade).
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const math = @import("math");
const Vec3 = math.Vec3;
const lights = @import("../lights.zig");
const HemisphericLight = lights.HemisphericLight;
const DirectionalLight = lights.DirectionalLight;
const DirectionalLightOptions = lights.DirectionalLightOptions;
const PointLight = lights.PointLight;
const PointLightOptions = lights.PointLightOptions;
const SpotLight = lights.SpotLight;
const SpotLightOptions = lights.SpotLightOptions;
const AreaLight = lights.AreaLight;
const AreaLightOptions = lights.AreaLightOptions;
const ClusteredPointLight = lights.ClusteredPointLight;
const ClusteredPointLightOptions = lights.ClusteredPointLightOptions;

// ---- Lights. ----

pub fn createHemisphericLight(self: anytype, name: []const u8, options: lights.HemisphericLightOptions) HemisphericLight {
    return self.lights.setHemispheric(name, options);
}

pub fn createPointLight(self: anytype, name: []const u8, options: PointLightOptions) !*PointLight {
    return self.lights.createPointLight(self.allocator, name, options);
}

pub fn createSpotLight(self: anytype, name: []const u8, options: SpotLightOptions) !*SpotLight {
    return self.lights.createSpotLight(self.allocator, name, options);
}

pub fn createDirectionalLight(self: anytype, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
    return self.lights.createDirectionalLight(self.allocator, name, options);
}

/// Appends a shadowless directional fill (uniform slots 1..3, creation
/// order). Hard-errors past 4 suns total (see
/// LightRig.addDirectionalLight); fills are session-local.
pub fn addDirectionalLight(self: anytype, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
    return self.lights.addDirectionalLight(self.allocator, name, options);
}

/// Sets the sun direction from azimuth and elevation angles (radians).
pub fn setSunAngles(self: anytype, azimuth_rad: f32, elevation_rad: f32) void {
    self.lights.setSunAngles(azimuth_rad, elevation_rad);
}

/// Sets the sun color from correlated color temperature in Kelvin (1000K to 12000K).
pub fn setSunColorTemperature(self: anytype, kelvin: f32) void {
    self.lights.setSunColorTemperature(kelvin);
}

// ---- Rect area lights (wave 26, v1). ----
//
// A bounded, additive, OFF-by-default feature: with zero area lights
// every appended uniform lane is zeroed and all five forward shaders
// skip the area loop, rendering bit-identically to today.
//
// Orientation is two half-extent vectors (`right` = local +X axis scaled
// by half-width, `up` = local +Y axis scaled by half-height; the
// emitting normal is cross(right, up)). Degenerate rects (zero area)
// emit nothing.
//
// Explicit non-goals in v1: area-light shadows (unshadowed, document
// accordingly), LTC integration (closest-point approximation instead —
// see the shader header), glTF import (KHR_lights_punctual has no rect
// type; area lights are API-only), persistence (session-local like
// directional fills: save/load never writes them, load never clears
// live ones).
//
// Capacity: at most `lights.max_area_lights` (2); past the cap is a
// hard `error.TooManyAreaLights`.

/// Appends a rect area light; returns the live pointer (creation order
/// == uniform slot order). Hard-errors past the cap.
pub fn addAreaLight(self: anytype, name: []const u8, options: AreaLightOptions) !*AreaLight {
    return self.lights.addAreaLight(self.allocator, name, options);
}

/// Removes area light `index`, destroying it. Order-preserving: higher
/// indices shift down. Out-of-range indices are a no-op.
pub fn removeAreaLight(self: anytype, index: usize) void {
    self.lights.removeAreaLight(self.allocator, index);
}

/// Live area-light state (center/right/up/color/intensity/enabled are
/// freely mutable game-side under update-vs-prepare exclusion). Null
/// when out of range.
pub fn getAreaLight(self: anytype, index: usize) ?*AreaLight {
    return self.lights.getAreaLight(index);
}

/// Number of owned area lights (at most lights.max_area_lights).
pub fn areaLightCount(self: anytype) usize {
    return self.lights.areaLightCount();
}

// ---- Clustered forward point lights (wave 30, v1). ----
//
// A bounded, additive, OFF-by-default pool of EXTRA point lights beyond
// the legacy 4-slot top-k lanes: with zero clustered lights every
// appended pack lane is zeroed, the tile build writes empty headers,
// and all five forward shaders skip the clustered loop, rendering
// bit-identically to today.
//
// Staging (1-frame lag, probe-pack pattern): add/remove/moves mutate
// game-side values under update-vs-prepare exclusion; `updateLights`
// stages them into the light handoff, `packFrameSnapshot` freezes the
// staged copy, and the context thread rebuilds the 2D screen tiles
// from that frozen copy during render — so a move is visible on the
// NEXT presented frame, never the current one. Removing a light also
// retires the live tile storage buffers through the epoch retire queue
// (safe under update||render overlap: the context thread destroys at
// the next flush); the next context rebuild recreates exact-fit
// buffers.
//
// Explicit non-goals in v1: shadows for clustered lights (unshadowed,
// document accordingly), depth-aware tiles (2D full-depth columns with
// documented over-inclusion), hysteresis/fades, glTF import, and
// persistence (session-local like directional fills and area lights:
// save/load never writes the pool, load never clears live lights).
//
// Capacity: at most `lights.max_clustered_lights` (64); past the cap
// is a hard `error.TooManyClusteredLights`.

/// Appends a clustered forward point light at `position`; returns its
/// index (pack order follows creation order). Hard-errors past the cap.
/// Also retires any live tile storage buffers (uniform discipline, so
/// stale GPU can never serve the next frame).
pub fn addClusteredPointLight(self: anytype, position: Vec3, options: ClusteredPointLightOptions) error{TooManyClusteredLights}!usize {
    const idx = try self.lights.addClusteredPointLight(position, options);
    self.clustered.retireBuffers(self.allocator, &self.gpu_retire);
    return idx;
}

/// Removes clustered light `index`, retiring its tile storage buffers
/// through the epoch retire queue (safe under update||render overlap).
/// Order-preserving: higher indices shift down. Out-of-range indices
/// are a no-op (and never retire).
pub fn removeClusteredPointLight(self: anytype, index: usize) void {
    if (index >= self.lights.clusteredPointLightCount()) return;
    self.lights.removeClusteredPointLight(index);
    self.clustered.retireBuffers(self.allocator, &self.gpu_retire);
}

/// Live clustered-light state (position/color/intensity/radius/enabled
/// are freely mutable game-side under update-vs-prepare exclusion;
/// edits stage through updateLights and appear next frame). Null when
/// out of range.
pub fn getClusteredPointLight(self: anytype, index: usize) ?*ClusteredPointLight {
    return self.lights.getClusteredPointLight(index);
}

/// Number of owned clustered lights (at most
/// lights.max_clustered_lights).
pub fn clusteredPointLightCount(self: anytype) usize {
    return self.lights.clusteredPointLightCount();
}

/// Stage 3, slice 2: update-phase light packing. Selects the top-k
/// point/spot lights for the camera (advancing the incumbency-
/// hysteresis fades with `dt`) and stores the uniform-ready pack.
/// Called once per frame BEFORE render(); render consumes
/// `snapshot.light_pack` (via the staged slot snapshot), never
/// `self.light_pack` and never live light state — so the direct
/// `self.light_pack` fallback below stays game/prepare-side ownership
/// (update vs prepare excluded under phase_mutex) and needs no atomic
/// mailbox. No new mailbox: the render reads the snapshot copy taken by
/// packFrameSnapshot.
pub fn updateLights(self: anytype, dt: f32) void {
    // Zero-eye fallback keeps the pack defined for camera-less scenes
    // (render early-returns without a camera anyway).
    const eye = if (self.active_camera) |cam|
        cam.getPosition()
    else if (self.cameras.items.len > 0)
        self.cameras.items[0].camera.getPosition()
    else
        Vec3.zero;
    const pack = self.lights.packFrame(eye, self.shadows.enabled, dt);
    // Publish through the mailbox. Same-thread today (publish is
    // visible to this frame's render takeLatest); after the split the
    // same call sequence crosses the thread boundary unchanged.
    if (self.light_handoff.claim()) |i| {
        self.light_handoff.slot(i).* = pack;
        self.light_handoff.publish(i);
    } else {
        // Same saturation shape as publishFrameSnapshot: drain stale
        // published packs (consumer excluded by phase ownership) so the
        // newest pack wins; direct fallback only on a still-failed claim.
        self.light_handoff.releasePublished();
        if (self.light_handoff.claim()) |i| {
            self.light_handoff.slot(i).* = pack;
            self.light_handoff.publish(i);
        } else {
            // Both slots still published (consumer lagging): fall back to
            // the consumed copy as the carrier.
            self.light_pack = pack;
        }
    }
}
