const std = @import("std");
const agate = @import("root.zig");

test "root re-exports cover recent Scene APIs" {
    _ = agate.ClusteredPointLight;
    _ = agate.AreaLight;
    _ = agate.sunDirectionFromAngles;
    _ = agate.colorTemperatureToRgb;
    _ = agate.ReflectionProbe;
    _ = agate.Ui3dPanel;
    _ = agate.HighlightLayer;
    _ = agate.HighlightEntry;
    _ = agate.HighlightOptions;
    _ = agate.CollisionMode;
    _ = agate.writeGlbAlloc;
    _ = agate.AssetManager;
    _ = agate.AssetTask;
    _ = agate.AssetCache;
    _ = agate.Crowd;
    _ = agate.CameraRig;
    _ = agate.Handoff;
    _ = agate.RenderTarget;
    _ = agate.RenderTargetDesc;
}

test "root re-exports cover the runtime frame facade" {
    _ = agate.Runtime;
    _ = agate.runtime.BeginResult;
    _ = agate.runtime.FrameResult;
}
