//! Facade for the light modules. The `lights.zig` light set was split into
//! focused leaves under `lights/` following the repo pattern (one
//! self-contained type per leaf plus helpers; Zig 0.16 has no
//! usingnamespace; see `camera.zig`, `profiler.zig`,
//! `scene/render_queue.zig`, `texture.zig`):
//!
//! - `lights/hemispheric.zig` — `HemisphericLightOptions` + `HemisphericLight`.
//!   Leaf: imports `math` only.
//! - `lights/directional.zig` — `DirectionalLightOptions` + `DirectionalLight`
//!   (+ `max_directional_lights`/`max_fill_directionals`). Imports `math` only.
//! - `lights/point.zig` — `PointLightOptions` + `PointLight` (incl. the cube-face
//!   shadow view-projection). Imports `math` only.
//! - `lights/clustered.zig` — `ClusteredPointLightOptions` +
//!   `ClusteredPointLight` (+ `max_clustered_lights`). Imports `math` only.
//! - `lights/area.zig` — `AreaLightOptions` + `AreaLight` (+ `max_area_lights`,
//!   incl. the rect normal/area helpers). Imports `math` only.
//! - `lights/spot.zig` — `SpotLightOptions` + `SpotLight` (incl. the shadow
//!   view-projection). Imports `std` + `math`.
//! - `lights/sun.zig` — active-sun resolvers (`resolveSunDirection`,
//!   `resolveSunColor`, `resolveSunIntensity`), `sunDirectionFromAngles`,
//!   `colorTemperatureToRgb`. Imports `std` + `math` + the `directional` +
//!   `hemispheric` siblings.
//! - `lights/tests.zig` — all light tests, unchanged apart from the header.
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`root.zig`, `scene.zig`, `scene/*`,
//! `serialization/*`, `loader/*`) see the same API as when everything lived
//! in this file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. Cross-leaf type uses (`sun.zig` reaching `directional` +
//! `hemispheric`) are direct sibling imports. Moved tests reach the API
//! through this facade through a block-scoped import that exists only in
//! test builds.
const hemispheric = @import("lights/hemispheric.zig");
pub const HemisphericLightOptions = hemispheric.HemisphericLightOptions;
pub const HemisphericLight = hemispheric.HemisphericLight;

const directional = @import("lights/directional.zig");
pub const DirectionalLightOptions = directional.DirectionalLightOptions;
pub const max_directional_lights = directional.max_directional_lights;
pub const max_fill_directionals = directional.max_fill_directionals;
pub const DirectionalLight = directional.DirectionalLight;

const point = @import("lights/point.zig");
pub const PointLightOptions = point.PointLightOptions;
pub const PointLight = point.PointLight;

const clustered = @import("lights/clustered.zig");
pub const max_clustered_lights = clustered.max_clustered_lights;
pub const max_clustered_spots = clustered.max_clustered_spots;
pub const ClusteredPointLightOptions = clustered.ClusteredPointLightOptions;
pub const ClusteredPointLight = clustered.ClusteredPointLight;
pub const ClusteredSpotLightOptions = clustered.ClusteredSpotLightOptions;
pub const ClusteredSpotLight = clustered.ClusteredSpotLight;

const area = @import("lights/area.zig");
pub const AreaLightOptions = area.AreaLightOptions;
pub const max_area_lights = area.max_area_lights;
pub const AreaLight = area.AreaLight;

const spot = @import("lights/spot.zig");
pub const SpotLightOptions = spot.SpotLightOptions;
pub const SpotLight = spot.SpotLight;

const sun = @import("lights/sun.zig");
pub const resolveSunDirection = sun.resolveSunDirection;
pub const resolveSunColor = sun.resolveSunColor;
pub const resolveSunIntensity = sun.resolveSunIntensity;
pub const sunDirectionFromAngles = sun.sunDirectionFromAngles;
pub const colorTemperatureToRgb = sun.colorTemperatureToRgb;

test {
    _ = @import("lights/tests.zig");
}
