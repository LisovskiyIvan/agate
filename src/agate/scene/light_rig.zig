const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;

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
const light_selection = @import("light_selection.zig");
const passes = @import("../passes/mod.zig");

/// Owns every light of the scene: the legacy hemispheric sun, the primary
/// directional sun plus up to 3 shadowless directional fills (4 suns total,
/// see lights.max_directional_lights), and the point/spot collections. Also
/// packs the shader-visible subset per frame (top-k selection + uniform
/// arrays), so light data has exactly one home and Scene never touches
/// light pointers.
pub const LightRig = struct {
    // Legacy hemispheric sun: always present, also drives the ambient term.
    hemi: HemisphericLight = .{},
    // Primary scene sun (uniform slot 0, the only shadow caster). Null keeps
    // the legacy hemispheric sun exactly.
    directional: ?*DirectionalLight = null,
    // Shadowless directional fills (uniform slots 1..3, creation order).
    // Capped at lights.max_fill_directionals; extras are session-local
    // (save/load persists only the primary sun).
    extra_directionals: std.ArrayListUnmanaged(*DirectionalLight) = .empty,
    point_lights: std.ArrayListUnmanaged(*PointLight) = .empty,
    spot_lights: std.ArrayListUnmanaged(*SpotLight) = .empty,
    /// Rect area lights (wave 26, v1, creation order, at most
    /// lights.max_area_lights). Session-local like directional fills:
    /// save/load never persists them. No shadows, no hysteresis, no
    /// significance selection — every owned light packs verbatim.
    area_lights: std.ArrayListUnmanaged(*AreaLight) = .empty,
    /// Clustered forward point lights (wave 30, v1): value array, at most
    /// lights.max_clustered_lights, session-local like area lights. No
    /// shadows, no hysteresis, no significance selection — every ENABLED
    /// light packs verbatim into FramePack and the 2D tile build culls
    /// per tile (see scene/clustered_lights.zig).
    clustered: [lights.max_clustered_lights]ClusteredPointLight = [_]ClusteredPointLight{.{}} ** lights.max_clustered_lights,
    clustered_count: usize = 0,

    // Slot hysteresis (see light_selection.Hysteresis): without it two lights
    // trading the 4th/2nd slot pop every frame. `hysteresis_enabled = false`
    // restores the legacy instant-swap behavior bit-for-bit.
    hysteresis_enabled: bool = true,
    // A challenger must beat an incumbent's raw score by this factor to
    // displace it — the dampener that makes slot swaps rare and decisive.
    incumbency_bonus: f32 = 0.5,
    // Seconds for a slot hand-off (old light fades out, then the new one in).
    fade_time: f32 = 0.25,
    point_hysteresis: light_selection.Hysteresis(PointLight, point_slots, light_selection.scorePoint) = .{},
    spot_hysteresis: light_selection.Hysteresis(SpotLight, spot_slots, light_selection.scoreSpot) = .{},

    pub fn init(name: []const u8, options: lights.HemisphericLightOptions) LightRig {
        return .{ .hemi = HemisphericLight.init(name, options) };
    }

    pub fn setHemispheric(self: *LightRig, name: []const u8, options: lights.HemisphericLightOptions) HemisphericLight {
        self.hemi = HemisphericLight.init(name, options);
        return self.hemi;
    }

    pub fn createPointLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: PointLightOptions) !*PointLight {
        const pl = try allocator.create(PointLight);
        pl.* = PointLight.init(name, options);
        try self.point_lights.append(allocator, pl);
        return pl;
    }

    pub fn createSpotLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: SpotLightOptions) !*SpotLight {
        const sl = try allocator.create(SpotLight);
        sl.* = SpotLight.init(name, options);
        try self.spot_lights.append(allocator, sl);
        return sl;
    }

    /// Creates (or replaces) the primary scene sun (uniform slot 0, the
    /// only shadow caster). Replacing destroys the previous primary; fills
    /// added via addDirectionalLight are untouched.
    pub fn createDirectionalLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        if (self.directional) |old| {
            if (old.owns_name) allocator.free(old.name);
            allocator.destroy(old);
            self.directional = null;
        }
        const dl = try allocator.create(DirectionalLight);
        dl.* = DirectionalLight.init(name, options);
        self.directional = dl;
        return dl;
    }

    /// Appends a shadowless directional fill (uniform slots 1..3, creation
    /// order). Returns error.TooManyDirectionalLights once the rig holds 4
    /// suns total (1 primary + 3 fills) — hard error, never silent clamp or
    /// replacement, so callers notice the cap. Fills are session-local:
    /// save/load persists only the primary sun.
    pub fn addDirectionalLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        if (self.extra_directionals.items.len >= lights.max_fill_directionals) return error.TooManyDirectionalLights;
        const dl = try allocator.create(DirectionalLight);
        dl.* = DirectionalLight.init(name, options);
        try self.extra_directionals.append(allocator, dl);
        return dl;
    }

    /// Total directional lights owned (primary + fills), at most
    /// lights.max_directional_lights.
    pub fn directionalCount(self: *const LightRig) usize {
        var n: usize = self.extra_directionals.items.len;
        if (self.directional != null) n += 1;
        return n;
    }

    /// Uniform-slot view: index 0 is the primary sun (null when absent),
    /// indices 1..3 are the fills in creation order. Null past the end.
    pub fn directionalAt(self: *const LightRig, index: usize) ?*DirectionalLight {
        if (index == 0) return self.directional;
        const fill = index - 1;
        if (fill < self.extra_directionals.items.len) return self.extra_directionals.items[fill];
        return null;
    }

    /// Appends a rect area light (uniform slots follow creation order).
    /// Returns error.TooManyAreaLights past lights.max_area_lights — hard
    /// error, never silent clamp or replacement, so callers notice the cap.
    /// Area lights are session-local (save/load never persists them),
    /// API-only in v1 (no KHR_lights_punctual mapping — glTF has no rect
    /// light type), and shadowless (no area-light shadows in v1).
    pub fn addAreaLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: AreaLightOptions) !*AreaLight {
        if (self.area_lights.items.len >= lights.max_area_lights) return error.TooManyAreaLights;
        const al = try allocator.create(AreaLight);
        al.* = AreaLight.init(name, options);
        try self.area_lights.append(allocator, al);
        return al;
    }

    /// Removes area light `index`, destroying it. Order-preserving: higher
    /// indices shift down, so callers must not cache indices across
    /// removals. Out-of-range indices are a no-op (same contract as
    /// Scene.removeCamera).
    pub fn removeAreaLight(self: *LightRig, allocator: std.mem.Allocator, index: usize) void {
        if (index >= self.area_lights.items.len) return;
        const al = self.area_lights.items[index];
        if (al.owns_name) allocator.free(al.name);
        allocator.destroy(al);
        for (index..self.area_lights.items.len - 1) |k| self.area_lights.items[k] = self.area_lights.items[k + 1];
        self.area_lights.items.len -= 1;
    }

    /// Live area-light state (center/right/up/color/intensity/enabled are
    /// freely mutable game-side under update-vs-prepare exclusion). Null
    /// when out of range.
    pub fn getAreaLight(self: *const LightRig, index: usize) ?*AreaLight {
        if (index >= self.area_lights.items.len) return null;
        return self.area_lights.items[index];
    }

    /// Number of owned area lights (at most lights.max_area_lights).
    pub fn areaLightCount(self: *const LightRig) usize {
        return self.area_lights.items.len;
    }

    /// Appends a clustered forward point light (EXTRA pool beyond the
    /// legacy 4-slot top-k lanes); returns its index. Pack order follows
    /// creation order. Returns error.TooManyClusteredLights past
    /// lights.max_clustered_lights — hard error, never silent clamp or
    /// replacement, so callers notice the cap. No shadows in v1
    /// (unshadowed by design). Session-local: save/load never persists
    /// the pool.
    pub fn addClusteredPointLight(self: *LightRig, position: Vec3, options: ClusteredPointLightOptions) error{TooManyClusteredLights}!usize {
        if (self.clustered_count >= lights.max_clustered_lights) return error.TooManyClusteredLights;
        const idx = self.clustered_count;
        self.clustered[idx] = ClusteredPointLight.init(position, options);
        self.clustered_count += 1;
        return idx;
    }

    /// Removes clustered light `index`, order-preserving (higher indices
    /// shift down, so callers must not cache indices across removals).
    /// Out-of-range indices are a no-op (same contract as
    /// Scene.removeCamera). Value array: no heap to free; GPU buffers
    /// retire via the Scene-level cache (see Scene.removeClusteredPointLight).
    pub fn removeClusteredPointLight(self: *LightRig, index: usize) void {
        if (index >= self.clustered_count) return;
        for (index..self.clustered_count - 1) |k| self.clustered[k] = self.clustered[k + 1];
        self.clustered[self.clustered_count - 1] = .{};
        self.clustered_count -= 1;
    }

    /// Live clustered-light state (position/color/intensity/radius/enabled
    /// are freely mutable game-side under update-vs-prepare exclusion;
    /// moves take effect in the next staged snapshot, see
    /// Scene.removeClusteredPointLight docs). Null when out of range.
    pub fn getClusteredPointLight(self: *LightRig, index: usize) ?*ClusteredPointLight {
        if (index >= self.clustered_count) return null;
        return &self.clustered[index];
    }

    /// Number of owned clustered lights (at most
    /// lights.max_clustered_lights).
    pub fn clusteredPointLightCount(self: *const LightRig) usize {
        return self.clustered_count;
    }

    // Sun resolution helpers (directional override, hemispheric fallback).
    pub fn sunDirection(self: *const LightRig) Vec3 {
        return lights.resolveSunDirection(self.directional, self.hemi);
    }

    pub fn sunColor(self: *const LightRig) Color3 {
        return lights.resolveSunColor(self.directional, self.hemi);
    }

    pub fn sunIntensity(self: *const LightRig) f32 {
        return lights.resolveSunIntensity(self.directional, self.hemi);
    }

    /// Sets the sun direction from azimuth and elevation angles (in radians).
    /// Updates the primary directional light (if created) and the hemispheric
    /// light direction fallback.
    pub fn setSunAngles(self: *LightRig, azimuth_rad: f32, elevation_rad: f32) void {
        const dir = lights.sunDirectionFromAngles(azimuth_rad, elevation_rad);
        if (self.directional) |dl| {
            dl.direction = dir;
        }
        self.hemi.direction = dir;
    }

    /// Sets the sun color from correlated color temperature in Kelvin.
    /// Updates the primary directional light diffuse color and the hemispheric
    /// diffuse color fallback.
    pub fn setSunColorTemperature(self: *LightRig, kelvin: f32) void {
        const col = lights.colorTemperatureToRgb(kelvin);
        if (self.directional) |dl| {
            dl.diffuse = col;
        }
        self.hemi.diffuse = col;
    }

    /// Frees every owned light. Leaves the collections empty so a deinit
    /// path stays idempotent under testing allocators. Slot hysteresis state
    /// is cleared too — it holds raw light pointers.
    pub fn deinit(self: *LightRig, allocator: std.mem.Allocator) void {
        self.point_hysteresis.reset();
        self.spot_hysteresis.reset();

        for (self.point_lights.items) |pl| {
            if (pl.owns_name) allocator.free(pl.name);
            allocator.destroy(pl);
        }
        self.point_lights.deinit(allocator);

        for (self.spot_lights.items) |sl| {
            if (sl.owns_name) allocator.free(sl.name);
            allocator.destroy(sl);
        }
        self.spot_lights.deinit(allocator);

        for (self.extra_directionals.items) |dl| {
            if (dl.owns_name) allocator.free(dl.name);
            allocator.destroy(dl);
        }
        self.extra_directionals.deinit(allocator);

        for (self.area_lights.items) |al| {
            if (al.owns_name) allocator.free(al.name);
            allocator.destroy(al);
        }
        self.area_lights.deinit(allocator);

        if (self.directional) |dl| {
            if (dl.owns_name) allocator.free(dl.name);
            allocator.destroy(dl);
            self.directional = null;
        }
    }

    // Shader uniform arrays for point/spot lights. Sizes mirror the shader
    // declarations (4 point + 2 spot slots); selection picks the best subset.
    pub const point_slots = 4;
    pub const spot_slots = 2;
    // Point-light shadow slots: at most this many packed point lights carry
    // a 6-face tile row in the point shadow atlas (matches the shader
    // point_view_proj[12] layout and ShadowPass.POINT_SHADOW_SLOTS).
    pub const point_shadow_slots = 2;

    /// Everything the renderer needs from the light rig for one frame:
    /// packed uniform arrays plus the spot shadow list for the depth pass.
    /// `shadows_enabled` gates spot shadow info generation exactly like the
    /// legacy inline render() code.
    pub const FramePack = struct {
        counts: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
        // Up to 4 directional suns (see lights.max_directional_lights):
        // slot 0 mirrors the primary sun (the only shadow caster; zeroed
        // when absent or disabled), slots 1..3 hold the fills in creation
        // order (disabled/unused slots zeroed, so the shader skip on
        // intensity <= 0 costs nothing). Directions are normalized here so
        // the fragment shaders can use them raw.
        directional_dir: [4][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
        directional_color_int: [4][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
        point_pos_range: [4][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
        point_color_int: [4][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
        spot_pos_range: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        spot_dir_inner: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        spot_color_outer: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        spot_intensity: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        spot_view_proj: [2]Mat4 = [_]Mat4{Mat4.identity} ** 2,
        spot_shadow_params: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        spot_shadows: [2]passes.SpotShadowRenderInfo = [_]passes.SpotShadowRenderInfo{.{}} ** 2,
        num_spot_shadows: usize = 0,
        // Point-light shadows, keyed by PACKED point slot (the shader light
        // loop index): x carries shadow_slot + 1 (0 = no shadow), y/z the
        // light's bias/normal-bias knobs. The 6 face matrices of shadow slot
        // s live in point_view_proj[s * 6 .. s * 6 + 6]; the depth pass tiles
        // come from point_shadows (6 entries per slot). All zeroed when no
        // point light casts shadows, so the shaders early-out with zero cost.
        point_view_proj: [12]Mat4 = [_]Mat4{Mat4.identity} ** 12,
        point_shadow_params: [4][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
        point_shadows: [12]passes.PointShadowRenderInfo = [_]passes.PointShadowRenderInfo{.{}} ** 12,
        num_point_shadows: usize = 0,
        // APPENDED LAST (rect area lights, wave 26): up to
        // lights.max_area_lights rects in creation order. xyz + intensity
        // in area_center_int (w = 0 when disabled/unused, so the shader
        // skip on intensity <= 0 renders bit-identically with zero lights),
        // right/up half-extent vectors and emitted rgb alongside. Appended
        // last so every existing lane stays bit-identical.
        area_center_int: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        area_right: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        area_up: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        area_color: [2][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
        // APPENDED LAST (clustered forward point lights, wave 30): up to
        // lights.max_clustered_lights EXTRA lights in creation order,
        // enabled only (disabled/degenerate lanes stay zeroed, so the tile
        // build skips them and the shader range-gates them exactly like a
        // zero-range legacy lane). xyz + radius in clustered_pos_range,
        // rgb + intensity in clustered_color_int, staged count in
        // clustered_count (usize: snapshot-side plain data, not a shader
        // lane). Appended last so every existing lane stays bit-identical;
        // empty pool zeroes everything (exact legacy path).
        clustered_pos_range: [lights.max_clustered_lights][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** lights.max_clustered_lights,
        clustered_color_int: [lights.max_clustered_lights][4]f32 = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** lights.max_clustered_lights,
        clustered_count: usize = 0,
    };

    /// Packs the directional suns for the frame uniforms: slot 0 mirrors
    /// the primary sun (zeroed when absent or disabled), slots 1..3 hold
    /// the fills in creation order (disabled fills stay zeroed, as do
    /// unused slots). No hysteresis or significance selection: fills are
    /// few and order-stable by construction, and the primary slot must
    /// never swap (it feeds the CSM shadow term).
    fn packDirectionals(self: *const LightRig, pack: *FramePack) void {
        if (self.directional) |dl| {
            if (dl.is_enabled) {
                const dir = if (dl.direction.lengthSq() > 1e-12) dl.direction.normalize() else Vec3.zero;
                pack.directional_dir[0] = .{ dir.x, dir.y, dir.z, 0.0 };
                pack.directional_color_int[0] = .{ dl.diffuse.r, dl.diffuse.g, dl.diffuse.b, dl.intensity };
            }
        }
        for (self.extra_directionals.items, 0..) |dl, k| {
            if (k >= lights.max_fill_directionals) break; // defensive: insert path caps
            if (!dl.is_enabled) continue; // stays zeroed
            const slot = k + 1;
            const dir = if (dl.direction.lengthSq() > 1e-12) dl.direction.normalize() else Vec3.zero;
            pack.directional_dir[slot] = .{ dir.x, dir.y, dir.z, 0.0 };
            pack.directional_color_int[slot] = .{ dl.diffuse.r, dl.diffuse.g, dl.diffuse.b, dl.intensity };
        }
    }

    /// Packs the rect area lights for the frame uniforms: creation order,
    /// disabled/unused lanes zeroed (the shader skips on intensity <= 0).
    /// No hysteresis, no significance selection, nothing frame-dependent in
    /// v1 (dt is ignored): area lights are few and order-stable, and an
    /// intensity fade would need a second look-dev pass before it earns a
    /// slot protocol.
    fn packAreaLights(self: *const LightRig, pack: *FramePack) void {
        for (self.area_lights.items, 0..) |al, i| {
            if (i >= lights.max_area_lights) break; // defensive: insert path caps
            if (!al.is_enabled) continue; // stays zeroed
            pack.area_center_int[i] = .{ al.center.x, al.center.y, al.center.z, al.intensity };
            pack.area_right[i] = .{ al.right.x, al.right.y, al.right.z, 0.0 };
            pack.area_up[i] = .{ al.up.x, al.up.y, al.up.z, 0.0 };
            pack.area_color[i] = .{ al.color.r, al.color.g, al.color.b, 0.0 };
        }
    }

    /// Stages the clustered forward pool for the tile build: creation-order
    /// lanes (position + radius, color + intensity), disabled lanes zeroed
    /// (the tile build and the shader range-gate skip them, so the packed
    /// index always equals the creation index). No hysteresis, no
    /// selection, nothing frame-dependent (dt ignored): tiling culls per
    /// tile, and fades would need a look-dev pass. `pack.clustered_count`
    /// stages the owned count (<= lights.max_clustered_lights); lanes past
    /// it stay zeroed (exact legacy path when empty).
    fn packClusteredLights(self: *const LightRig, pack: *FramePack) void {
        for (self.clustered[0..self.clustered_count], 0..) |*cl, i| {
            if (i >= lights.max_clustered_lights) break; // defensive
            if (!cl.is_enabled) continue; // stays zeroed
            pack.clustered_pos_range[i] = .{ cl.position.x, cl.position.y, cl.position.z, cl.radius };
            pack.clustered_color_int[i] = .{ cl.color.r, cl.color.g, cl.color.b, cl.intensity };
        }
        pack.clustered_count = self.clustered_count;
    }

    /// Picks up to point_shadow_slots shadow casters among the packed point
    /// lights (pack order; ties resolve to the earlier pack index) by raw
    /// significance, and fills the point shadow uniforms + depth-pass tiles.
    /// Non-casters keep zeroed params, so their shader cost is one early-out.
    fn packPointShadows(pack: *FramePack, in_lights: []const *PointLight, eye: Vec3, shadows_enabled: bool) void {
        var best_idx: [point_shadow_slots]usize = undefined;
        var best_score: [point_shadow_slots]f32 = undefined;
        var n: usize = 0;
        for (in_lights, 0..) |pl, i| {
            if (!(pl.cast_shadows and shadows_enabled and pl.is_enabled)) continue;
            const s = light_selection.scorePoint(pl, eye);
            var pos: usize = n;
            for (0..n) |j| {
                if (s > best_score[j]) {
                    pos = j;
                    break;
                }
            }
            if (pos < point_shadow_slots) {
                const end = if (n < point_shadow_slots) n else point_shadow_slots - 1;
                var k = end;
                while (k > pos) : (k -= 1) {
                    best_idx[k] = best_idx[k - 1];
                    best_score[k] = best_score[k - 1];
                }
                best_idx[pos] = i;
                best_score[pos] = s;
                if (n < point_shadow_slots) n += 1;
            }
        }
        for (best_idx[0..n], 0..) |pack_i, slot| {
            const pl = in_lights[pack_i];
            pack.point_shadow_params[pack_i] = .{ @floatFromInt(slot + 1), pl.shadow_bias, pl.shadow_normal_bias, 0.0 };
            for (0..passes.POINT_SHADOW_FACES) |f| {
                const vp = pl.getShadowFaceViewProj(f);
                pack.point_view_proj[slot * passes.POINT_SHADOW_FACES + f] = vp;
                const o = passes.pointTileOrigin(slot, f);
                pack.point_shadows[pack.num_point_shadows] = .{
                    .tile_x = o.x,
                    .tile_y = o.y,
                    .view_proj = vp,
                };
                pack.num_point_shadows += 1;
            }
        }
    }

    /// Packs point & spot lights for the camera. Directions are normalized
    /// once here so the fragment shaders can use them raw (no per-pixel
    /// normalize()). The directional light overrides the legacy hemispheric
    /// sun; ambient stays hemispheric ground_color. `dt` advances the slot
    /// fade state (pass 0 to freeze transitions).
    pub fn packFrame(self: *LightRig, eye: Vec3, shadows_enabled: bool, dt: f32) FramePack {
        var pack = FramePack{
            .counts = .{ 0.0, 0.0, 0.0, 0.0 },
            .directional_dir = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
            .directional_color_int = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
            .point_pos_range = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
            .point_color_int = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
            .spot_pos_range = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_dir_inner = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_color_outer = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_intensity = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_view_proj = [_]Mat4{Mat4.identity} ** 2,
            .spot_shadow_params = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_shadows = undefined,
            .num_spot_shadows = 0,
            .point_view_proj = [_]Mat4{Mat4.identity} ** 12,
            .point_shadow_params = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
            .point_shadows = undefined,
            .num_point_shadows = 0,
        };
        // Directional suns ride both pack paths verbatim (no hysteresis):
        // slot 0 is the primary, slots 1..3 the fills. Area lights ride
        // both paths verbatim too (no selection, nothing frame-dependent).
        // Clustered lights ride both paths verbatim as well (creation-order
        // lanes; the tile build culls per tile, see packClusteredLights).
        self.packDirectionals(&pack);
        self.packAreaLights(&pack);
        self.packClusteredLights(&pack);

        // Pick the most relevant lights for the camera before packing; the
        // shader uniform arrays only hold 4 point + 2 spot slots. With
        // hysteresis the packed subset may differ from the raw top-k: a
        // displaced light keeps its slot fading out before the challenger
        // fades in, and intensities are scaled by the fade factor.
        if (self.hysteresis_enabled) {
            var point_packed: [point_slots]light_selection.Hysteresis(PointLight, point_slots, light_selection.scorePoint).Packed = undefined;
            const num_point = self.point_hysteresis.update(
                self.point_lights.items,
                eye,
                dt,
                self.incumbency_bonus,
                self.fade_time,
                &point_packed,
            );
            for (point_packed[0..num_point], 0..) |p, i| {
                const pl = p.light;
                pack.point_pos_range[i] = .{ pl.position.x, pl.position.y, pl.position.z, pl.range };
                pack.point_color_int[i] = .{ pl.color.r, pl.color.g, pl.color.b, pl.intensity * p.factor };
            }
            pack.counts[0] = @floatFromInt(num_point);

            var point_buf_h: [point_slots]*PointLight = undefined;
            for (point_packed[0..num_point], 0..) |p, i| point_buf_h[i] = p.light;
            packPointShadows(&pack, point_buf_h[0..num_point], eye, shadows_enabled);

            var spot_packed: [spot_slots]light_selection.Hysteresis(SpotLight, spot_slots, light_selection.scoreSpot).Packed = undefined;
            const num_spot = self.spot_hysteresis.update(
                self.spot_lights.items,
                eye,
                dt,
                self.incumbency_bonus,
                self.fade_time,
                &spot_packed,
            );
            for (spot_packed[0..num_spot], 0..) |p, i| {
                const sl = p.light;
                const dir = sl.direction.normalize();
                const cos_inner = @cos(sl.inner_angle_deg * (std.math.pi / 180.0));
                const cos_outer = @cos(sl.outer_angle_deg * (std.math.pi / 180.0));
                const fade_intensity = sl.intensity * p.factor;
                pack.spot_pos_range[i] = .{ sl.position.x, sl.position.y, sl.position.z, sl.range };
                pack.spot_dir_inner[i] = .{ dir.x, dir.y, dir.z, cos_inner };
                pack.spot_color_outer[i] = .{ sl.color.r, sl.color.g, sl.color.b, cos_outer };
                pack.spot_intensity[i] = .{ fade_intensity, 0.0, 0.0, 0.0 };

                // A fading-out shadow caster keeps its shadow: dropping the
                // shadow earlier than the light itself would pop twice.
                if (sl.cast_shadows and shadows_enabled and sl.is_enabled) {
                    const svp = sl.getShadowViewProj();
                    pack.spot_view_proj[i] = svp;
                    pack.spot_shadow_params[i] = .{ 1.0, sl.shadow_bias, sl.shadow_normal_bias, 0.0 };
                    pack.spot_shadows[pack.num_spot_shadows] = .{
                        .spot_index = i,
                        .view_proj = svp,
                    };
                    pack.num_spot_shadows += 1;
                } else {
                    pack.spot_shadow_params[i] = .{ 0.0, 0.0, 0.0, 0.0 };
                }
            }
            pack.counts[1] = @floatFromInt(num_spot);
            return pack;
        }

        // Legacy path: instant top-k, no fade state (bit-identical to the
        // pre-hysteresis packing).
        var point_buf: [point_slots]*PointLight = undefined;
        const num_point = light_selection.selectPoint(self.point_lights.items, eye, &point_buf);
        for (point_buf[0..num_point], 0..) |pl, i| {
            pack.point_pos_range[i] = .{ pl.position.x, pl.position.y, pl.position.z, pl.range };
            pack.point_color_int[i] = .{ pl.color.r, pl.color.g, pl.color.b, pl.intensity };
        }
        pack.counts[0] = @floatFromInt(num_point);
        packPointShadows(&pack, point_buf[0..num_point], eye, shadows_enabled);

        var spot_buf: [spot_slots]*SpotLight = undefined;
        const num_spot = light_selection.selectSpot(self.spot_lights.items, eye, &spot_buf);

        for (spot_buf[0..num_spot], 0..) |sl, i| {
            const dir = sl.direction.normalize();
            const cos_inner = @cos(sl.inner_angle_deg * (std.math.pi / 180.0));
            const cos_outer = @cos(sl.outer_angle_deg * (std.math.pi / 180.0));
            pack.spot_pos_range[i] = .{ sl.position.x, sl.position.y, sl.position.z, sl.range };
            pack.spot_dir_inner[i] = .{ dir.x, dir.y, dir.z, cos_inner };
            pack.spot_color_outer[i] = .{ sl.color.r, sl.color.g, sl.color.b, cos_outer };
            pack.spot_intensity[i] = .{ sl.intensity, 0.0, 0.0, 0.0 };

            if (sl.cast_shadows and shadows_enabled and sl.is_enabled) {
                const svp = sl.getShadowViewProj();
                pack.spot_view_proj[i] = svp;
                pack.spot_shadow_params[i] = .{ 1.0, sl.shadow_bias, sl.shadow_normal_bias, 0.0 };
                pack.spot_shadows[pack.num_spot_shadows] = .{
                    .spot_index = i,
                    .view_proj = svp,
                };
                pack.num_spot_shadows += 1;
            } else {
                pack.spot_shadow_params[i] = .{ 0.0, 0.0, 0.0, 0.0 };
            }
        }
        pack.counts[1] = @floatFromInt(num_spot);

        return pack;
    }
};

test "createDirectionalLight replaces the primary sun" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    const first = try rig.createDirectionalLight(allocator, "sun1", .{});
    try std.testing.expect(rig.directional == first);

    const second = try rig.createDirectionalLight(allocator, "sun2", .{ .intensity = 2.0 });
    try std.testing.expect(rig.directional == second);
    try std.testing.expectEqualStrings("sun2", rig.directional.?.name);
    try std.testing.expectEqual(@as(f32, 2.0), rig.directional.?.intensity);

    // Sun resolution prefers the directional once one is set.
    const sun_dir = rig.sunDirection();
    const want_dir = second.direction.normalize();
    try std.testing.expectEqual(want_dir.x, sun_dir.x);
    try std.testing.expectEqual(want_dir.y, sun_dir.y);
    try std.testing.expectEqual(want_dir.z, sun_dir.z);
}

test "single directional packs slot 0 with the rest zeroed (legacy-neutral)" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    // No suns at all: all slots zeroed, sun falls back to hemispheric.
    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    for (0..4) |i| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[i]);
    }
    try std.testing.expectEqual(@as(usize, 0), rig.directionalCount());
    try std.testing.expect(rig.directionalAt(0) == null);

    // Exactly one light (the default): slot 0 mirrors the primary with a
    // normalized direction, slots 1..3 stay zeroed — the shader fill loop
    // (1..3, skip on intensity <= 0) then adds nothing bit-identically.
    const sun = try rig.createDirectionalLight(allocator, "sun", .{
        .direction = Vec3.new(0.0, -2.0, 0.0),
        .diffuse = Color3.new(1.0, 0.5, 0.25),
        .intensity = 2.0,
    });
    _ = sun;
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ 0.0, -1.0, 0.0, 0.0 }, pack.directional_dir[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 0.25, 2.0 }, pack.directional_color_int[0]);
    for (1..4) |i| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[i]);
    }
    try std.testing.expectEqual(@as(usize, 1), rig.directionalCount());
}

test "fills pack in creation order, disabled entries zeroed" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    _ = try rig.createDirectionalLight(allocator, "sun", .{
        .direction = Vec3.new(0.0, -1.0, 0.0),
        .intensity = 1.0,
    });
    const fill_a = try rig.addDirectionalLight(allocator, "fill-a", .{
        .direction = Vec3.new(2.0, 0.0, 0.0),
        .diffuse = Color3.new(0.0, 1.0, 0.0),
        .intensity = 0.5,
    });
    const fill_b = try rig.addDirectionalLight(allocator, "fill-b", .{
        .direction = Vec3.new(0.0, 0.0, 3.0),
        .diffuse = Color3.new(0.0, 0.0, 1.0),
        .intensity = 0.25,
    });
    fill_b.is_enabled = false;
    try std.testing.expectEqual(@as(usize, 3), rig.directionalCount());
    try std.testing.expect(rig.directionalAt(1) == fill_a);
    try std.testing.expect(rig.directionalAt(2) == fill_b);
    try std.testing.expect(rig.directionalAt(3) == null);

    const pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    // Slot 0 is the primary, slot 1 the enabled fill (normalized).
    try std.testing.expectEqual([4]f32{ 0.0, -1.0, 0.0, 0.0 }, pack.directional_dir[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 1.0, 0.0, 0.5 }, pack.directional_color_int[1]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, pack.directional_dir[1]);
    // Disabled fill zeroed, unused slot zeroed.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[3]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[3]);
}

test "addDirectionalLight caps at three fills with a hard error" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.createDirectionalLight(allocator, "sun", .{});
    _ = try rig.addDirectionalLight(allocator, "f1", .{});
    _ = try rig.addDirectionalLight(allocator, "f2", .{});
    _ = try rig.addDirectionalLight(allocator, "f3", .{});
    try std.testing.expectEqual(@as(usize, 4), rig.directionalCount());
    // Beyond 4 total: hard error, never silent clamp or replacement.
    try std.testing.expectError(error.TooManyDirectionalLights, rig.addDirectionalLight(allocator, "f4", .{}));
    try std.testing.expectEqual(@as(usize, 4), rig.directionalCount());
}

test "primary replacement keeps fills and sun resolution ignores fills" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.createDirectionalLight(allocator, "sun1", .{
        .direction = Vec3.new(0.0, -1.0, 0.0),
        .diffuse = Color3.new(1.0, 0.0, 0.0),
        .intensity = 1.0,
    });
    const fill = try rig.addDirectionalLight(allocator, "fill", .{
        .direction = Vec3.new(1.0, 0.0, 0.0),
        .diffuse = Color3.new(0.0, 1.0, 0.0),
        .intensity = 5.0,
    });
    const sun2 = try rig.createDirectionalLight(allocator, "sun2", .{
        .direction = Vec3.new(0.0, 0.0, -1.0),
        .diffuse = Color3.new(0.0, 0.0, 1.0),
        .intensity = 2.0,
    });
    // Replacement swaps only slot 0; the fill pointer survives.
    try std.testing.expect(rig.directional == sun2);
    try std.testing.expect(rig.directionalAt(1) == fill);
    try std.testing.expectEqual(@as(usize, 2), rig.directionalCount());
    // Significance/ordering for the primary is unchanged: fills never win
    // the sun, no matter how intense.
    const dir = rig.sunDirection();
    try std.testing.expectApproxEqAbs(dir.z, -1.0, 1e-6);
    const col = rig.sunColor();
    try std.testing.expectApproxEqAbs(col.b, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(rig.sunIntensity(), 2.0, 1e-6);
}

test "disabled primary zeroes slot 0 and falls back to hemispheric sun" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{ .direction = Vec3.new(0.0, 1.0, 0.0) });
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    const sun = try rig.createDirectionalLight(allocator, "sun", .{
        .direction = Vec3.new(0.0, -1.0, 0.0),
        .intensity = 3.0,
    });
    sun.is_enabled = false;
    const pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[0]);
    const dir = rig.sunDirection();
    try std.testing.expectApproxEqAbs(dir.y, 1.0, 1e-6);
}

test "packFrame packs selected point and spot lights into uniform arrays" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    // This test asserts the raw packing layout, not the fade transitions —
    // the legacy instant-swap path keeps it independent of fade timing.
    rig.hysteresis_enabled = false;

    const eye = Vec3.zero;
    // Empty rig: zero counts, zeroed shadow params.
    var pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.counts);
    try std.testing.expectEqual(@as(usize, 0), pack.num_spot_shadows);

    const pl = try rig.createPointLight(allocator, "lamp", .{
        .position = Vec3.new(1, 2, 3),
        .range = 10.0,
        .intensity = 2.0,
    });
    const sl = try rig.createSpotLight(allocator, "cone", .{
        .position = Vec3.new(0, 5, 0),
        .direction = Vec3.new(0, -1, 0),
        .range = 20.0,
        .intensity = 4.0,
    });
    sl.cast_shadows = true;
    sl.shadow_bias = 0.25;
    sl.shadow_normal_bias = 0.5;

    pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[0]);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[1]);
    try std.testing.expectEqual([4]f32{ pl.position.x, pl.position.y, pl.position.z, pl.range }, pack.point_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 4.0, 0.0, 0.0, 0.0 }, pack.spot_intensity[0]);
    // Shadow-casting spot contributes one entry for the depth pass.
    try std.testing.expectEqual(@as(usize, 1), pack.num_spot_shadows);
    try std.testing.expectEqual([4]f32{ 1.0, 0.25, 0.5, 0.0 }, pack.spot_shadow_params[0]);

    // Disabled shadows clear the params but keep the light packing.
    pack = rig.packFrame(eye, false, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.num_spot_shadows);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.spot_shadow_params[0]);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[1]);
}

test "packFrame scales intensity by the slot fade factor" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.createPointLight(allocator, "lamp", .{
        .position = Vec3.new(1, 0, 0),
        .range = 10.0,
        .intensity = 2.0,
    });
    const eye = Vec3.zero;
    const dt: f32 = 0.25 / 2.0; // two frames per half of the fade

    // First frame: the light just claimed its slot at factor 0 — color
    // channels pass through, the intensity component is scaled to zero.
    var pack = rig.packFrame(eye, false, dt);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[0]);
    try std.testing.expectEqual(@as(f32, 0.0), pack.point_color_int[0][3]);

    // Halfway through the fade-in the packed intensity is scaled.
    pack = rig.packFrame(eye, false, dt);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pack.point_color_int[0][3], 1e-6);

    // Fade complete: full intensity, steady from here on.
    pack = rig.packFrame(eye, false, dt);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), pack.point_color_int[0][3], 1e-6);
    pack = rig.packFrame(eye, false, dt);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), pack.point_color_int[0][3], 1e-6);
}

test "packFrame leaves point shadow uniforms zeroed when nothing casts" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    // Empty rig: neutrality by construction.
    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.num_point_shadows);
    for (pack.point_shadow_params) |p| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, p);
    }

    // A non-casting point light keeps the shadow lanes zeroed (off by
    // default): lighting still packs, the shader shadow path early-outs.
    _ = try rig.createPointLight(allocator, "lamp", .{
        .position = Vec3.new(1, 2, 3),
        .range = 10.0,
        .intensity = 2.0,
    });
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[0]);
    try std.testing.expectEqual(@as(usize, 0), pack.num_point_shadows);
    for (pack.point_shadow_params) |p| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, p);
    }
    for (pack.point_view_proj) |m| try std.testing.expectEqual(Mat4.identity, m);
}

test "packFrame caps point shadow casters at two by significance" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;
    const eye = Vec3.zero;

    // Scores at the eye (intensity * range / (1 + dist^2)):
    // c0 at (1,0,0): 1*10/2 = 5; c1 at origin: 3*10/1 = 30;
    // c2 at (3,0,0): 4*10/10 = 4. Pack order (instant): c1, c0, c2.
    _ = try rig.createPointLight(allocator, "c0", .{ .position = Vec3.new(1, 0, 0), .intensity = 1.0, .range = 10.0, .cast_shadows = true });
    const c1 = try rig.createPointLight(allocator, "c1", .{ .position = Vec3.zero, .intensity = 3.0, .range = 10.0, .cast_shadows = true });
    _ = try rig.createPointLight(allocator, "c2", .{ .position = Vec3.new(3, 0, 0), .intensity = 4.0, .range = 10.0, .cast_shadows = true });
    c1.shadow_bias = 0.25;
    c1.shadow_normal_bias = 0.5;

    const pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 3.0), pack.counts[0]);
    // Two shadow slots taken (6 face tiles each), the weakest caster drops.
    try std.testing.expectEqual(@as(usize, 12), pack.num_point_shadows);
    // Pack indices: 0 = c1 (strongest), 1 = c0, 2 = c2 (dropped).
    try std.testing.expectEqual([4]f32{ 1.0, 0.25, 0.5, 0.0 }, pack.point_shadow_params[0]);
    try std.testing.expectEqual(@as(f32, 2.0), pack.point_shadow_params[1][0]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.point_shadow_params[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.point_shadow_params[3]);
    // Tiles cover both slot rows: faces 0..5 at y 0 and y 256.
    for (pack.point_shadows[0..6], 0..) |t, f| {
        try std.testing.expectEqual(@as(i32, @intCast(f)) * 256, t.tile_x);
        try std.testing.expectEqual(@as(i32, 0), t.tile_y);
    }
    for (pack.point_shadows[6..12], 0..) |t, f| {
        try std.testing.expectEqual(@as(i32, @intCast(f)) * 256, t.tile_x);
        try std.testing.expectEqual(@as(i32, 256), t.tile_y);
    }
}

test "addAreaLight caps at two with a hard error" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.addAreaLight(allocator, "a0", .{});
    _ = try rig.addAreaLight(allocator, "a1", .{});
    try std.testing.expectEqual(@as(usize, 2), rig.areaLightCount());
    // Beyond the cap: hard error, never silent clamp or replacement.
    try std.testing.expectError(error.TooManyAreaLights, rig.addAreaLight(allocator, "a2", .{}));
    try std.testing.expectEqual(@as(usize, 2), rig.areaLightCount());
}

test "area lights pack creation-order lanes, disabled entries zeroed" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    // Empty rig: all area lanes zeroed (zero lights render bit-identically).
    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    for (0..lights.max_area_lights) |i| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_center_int[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_right[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_up[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_color[i]);
    }

    const a0 = try rig.addAreaLight(allocator, "key", .{
        .center = Vec3.new(1.0, 2.0, 3.0),
        .right = Vec3.new(2.0, 0.0, 0.0),
        .up = Vec3.new(0.0, 0.5, 0.0),
        .color = Color3.new(1.0, 0.5, 0.25),
        .intensity = 3.0,
    });
    const a1 = try rig.addAreaLight(allocator, "fill", .{
        .center = Vec3.new(-1.0, 0.0, 0.0),
        .intensity = 0.5,
    });
    a1.is_enabled = false;
    try std.testing.expect(rig.getAreaLight(0) == a0);
    try std.testing.expect(rig.getAreaLight(1) == a1);
    try std.testing.expect(rig.getAreaLight(2) == null);

    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 3.0 }, pack.area_center_int[0]);
    try std.testing.expectEqual([4]f32{ 2.0, 0.0, 0.0, 0.0 }, pack.area_right[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.5, 0.0, 0.0 }, pack.area_up[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 0.25, 0.0 }, pack.area_color[0]);
    // Disabled light: zeroed lane, unused w stays 0.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_center_int[1]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_color[1]);

    // Re-enable: second lane packs verbatim (defaults: unit axes).
    a1.is_enabled = true;
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ -1.0, 0.0, 0.0, 0.5 }, pack.area_center_int[1]);
    try std.testing.expectEqual([4]f32{ 0.5, 0.0, 0.0, 0.0 }, pack.area_right[1]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.5, 0.0, 0.0 }, pack.area_up[1]);
}

test "removeAreaLight destroys order-preserving, out-of-range is a no-op" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    const a0 = try rig.addAreaLight(allocator, "a0", .{});
    const a1 = try rig.addAreaLight(allocator, "a1", .{});
    rig.removeAreaLight(allocator, 7); // no-op
    try std.testing.expectEqual(@as(usize, 2), rig.areaLightCount());
    rig.removeAreaLight(allocator, 0);
    try std.testing.expectEqual(@as(usize, 1), rig.areaLightCount());
    try std.testing.expect(rig.getAreaLight(0) == a1);
    _ = a0;
    rig.removeAreaLight(allocator, 0);
    try std.testing.expectEqual(@as(usize, 0), rig.areaLightCount());
    try std.testing.expect(rig.getAreaLight(0) == null);
}

test "area packing is frame-independent (no fade, no selection)" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    // Hysteresis ON here on purpose: area lanes must still pack verbatim
    // on the very first frame (no enter-fade) and identically for any dt.
    _ = try rig.addAreaLight(allocator, "key", .{
        .center = Vec3.new(0.0, 3.0, 0.0),
        .intensity = 2.0,
    });
    const eye = Vec3.new(10.0, 0.0, 0.0);
    const p0 = rig.packFrame(eye, true, 0.016);
    const p1 = rig.packFrame(eye, true, 0.5);
    try std.testing.expectEqual([4]f32{ 0.0, 3.0, 0.0, 2.0 }, p0.area_center_int[0]);
    try std.testing.expectEqual(p0.area_center_int, p1.area_center_int);
    try std.testing.expectEqual(p0.area_right, p1.area_right);
    try std.testing.expectEqual(p0.area_up, p1.area_up);
    try std.testing.expectEqual(p0.area_color, p1.area_color);
}

test "packFrame gates point shadows on enabled lights and global shadows" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;
    const eye = Vec3.zero;

    const caster = try rig.createPointLight(allocator, "caster", .{
        .position = Vec3.zero,
        .intensity = 5.0,
        .range = 10.0,
        .cast_shadows = true,
    });

    // Global shadows off: no tiles, zeroed params, light still packs.
    var pack = rig.packFrame(eye, false, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[0]);
    try std.testing.expectEqual(@as(usize, 0), pack.num_point_shadows);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.point_shadow_params[0]);

    // Disabled light: skipped even with shadows on.
    caster.is_enabled = false;
    pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 0.0), pack.counts[0]);
    try std.testing.expectEqual(@as(usize, 0), pack.num_point_shadows);

    // Re-enabled: one slot, six face tiles.
    caster.is_enabled = true;
    pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 6), pack.num_point_shadows);
    try std.testing.expectEqual(@as(f32, 1.0), pack.point_shadow_params[0][0]);
}

test "addClusteredPointLight caps at 64 with a hard error" {
    var rig = LightRig.init("hemi", .{});
    // Value array: no allocator traffic, but deinit stays idempotent.
    const allocator = std.testing.allocator;
    defer rig.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), rig.clusteredPointLightCount());
    try std.testing.expect(rig.getClusteredPointLight(0) == null);

    var i: usize = 0;
    while (i < lights.max_clustered_lights) : (i += 1) {
        const idx = try rig.addClusteredPointLight(Vec3.new(@floatFromInt(i), 0, 0), .{});
        try std.testing.expectEqual(i, idx);
    }
    try std.testing.expectEqual(lights.max_clustered_lights, rig.clusteredPointLightCount());
    // Beyond the cap: hard error, never silent clamp or replacement.
    try std.testing.expectError(error.TooManyClusteredLights, rig.addClusteredPointLight(Vec3.zero, .{}));
    try std.testing.expectEqual(lights.max_clustered_lights, rig.clusteredPointLightCount());
}

test "clustered lights pack creation-order lanes, disabled entries zeroed" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    // Empty pool: zero count, all lanes zeroed (the tile build writes empty
    // headers and the shader takes the exact legacy path).
    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.clustered_count);
    for (0..lights.max_clustered_lights) |k| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_pos_range[k]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_color_int[k]);
    }

    const idx0 = try rig.addClusteredPointLight(Vec3.new(1.0, 2.0, 3.0), .{
        .color = Color3.new(1.0, 0.5, 0.25),
        .intensity = 3.0,
        .radius = 7.0,
    });
    const idx1 = try rig.addClusteredPointLight(Vec3.new(-1.0, 0.0, 0.0), .{ .intensity = 0.5 });
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 1), idx1);
    try std.testing.expect(rig.getClusteredPointLight(0).?.position.x == 1.0);
    try std.testing.expect(rig.getClusteredPointLight(2) == null);

    // Packed index == creation index (stable identity for tile lists).
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 2), pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 7.0 }, pack.clustered_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 0.25, 3.0 }, pack.clustered_color_int[0]);
    try std.testing.expectEqual([4]f32{ -1.0, 0.0, 0.0, 10.0 }, pack.clustered_pos_range[1]);

    // Disable: lane zeroes out (tile build + shader skip it), count still
    // stages the owned lanes.
    rig.getClusteredPointLight(0).?.is_enabled = false;
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 2), pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_color_int[0]);
    try std.testing.expectEqual([4]f32{ -1.0, 0.0, 0.0, 10.0 }, pack.clustered_pos_range[1]);
    rig.getClusteredPointLight(0).?.is_enabled = true;

    // Legacy lanes are untouched by the clustered pool (append-last: the
    // legacy top-k selection never sees these lights).
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.counts);
}

test "removeClusteredPointLight destroys order-preserving, out-of-range is a no-op" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.addClusteredPointLight(Vec3.new(1, 0, 0), .{});
    _ = try rig.addClusteredPointLight(Vec3.new(2, 0, 0), .{});
    _ = try rig.addClusteredPointLight(Vec3.new(3, 0, 0), .{});
    rig.removeClusteredPointLight(9); // no-op
    try std.testing.expectEqual(@as(usize, 3), rig.clusteredPointLightCount());
    rig.removeClusteredPointLight(0);
    try std.testing.expectEqual(@as(usize, 2), rig.clusteredPointLightCount());
    // Order-preserving: the tail shifted down (index 0 now holds x=2).
    try std.testing.expectEqual(@as(f32, 2.0), rig.getClusteredPointLight(0).?.position.x);
    try std.testing.expectEqual(@as(f32, 3.0), rig.getClusteredPointLight(1).?.position.x);
    rig.removeClusteredPointLight(1);
    rig.removeClusteredPointLight(0);
    try std.testing.expectEqual(@as(usize, 0), rig.clusteredPointLightCount());
    try std.testing.expect(rig.getClusteredPointLight(0) == null);
    // Empty again: pack returns to the zeroed legacy-neutral state.
    const pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_pos_range[0]);
}

test "clustered packing is frame-independent (no fade, no selection)" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    // Hysteresis ON here on purpose: clustered lanes must still pack
    // verbatim on the very first frame (no enter-fade) and identically for
    // any dt — tiling culls per tile instead of fading slots.
    _ = try rig.addClusteredPointLight(Vec3.new(0.0, 3.0, 0.0), .{ .intensity = 2.0 });
    const eye = Vec3.new(10.0, 0.0, 0.0);
    const p0 = rig.packFrame(eye, true, 0.016);
    const p1 = rig.packFrame(eye, true, 0.5);
    try std.testing.expectEqual(@as(usize, 1), p0.clustered_count);
    try std.testing.expectEqual([4]f32{ 0.0, 3.0, 0.0, 10.0 }, p0.clustered_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 1.0, 2.0 }, p0.clustered_color_int[0]);
    try std.testing.expectEqual(p0.clustered_pos_range, p1.clustered_pos_range);
    try std.testing.expectEqual(p0.clustered_color_int, p1.clustered_color_int);
}
