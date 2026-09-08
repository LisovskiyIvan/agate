const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;

pub const HemisphericLightOptions = struct {
    direction: Vec3 = Vec3.up,
    diffuse: Color3 = Color3.white,
    ground_color: Color3 = Color3.new(0.2, 0.2, 0.2),
    intensity: f32 = 1.0,
};

pub const HemisphericLight = struct {
    name: []const u8 = "HemisphericLight",
    direction: Vec3 = Vec3.up,
    diffuse: Color3 = Color3.white,
    ground_color: Color3 = Color3.new(0.2, 0.2, 0.2),
    intensity: f32 = 1.0,

    pub fn init(name: []const u8, options: HemisphericLightOptions) HemisphericLight {
        return .{
            .name = name,
            .direction = options.direction.normalize(),
            .diffuse = options.diffuse,
            .ground_color = options.ground_color,
            .intensity = options.intensity,
        };
    }
};

pub const DirectionalLightOptions = struct {
    direction: Vec3 = Vec3.new(0.5, 1.0, 0.5),
    diffuse: Color3 = Color3.white,
    intensity: f32 = 1.0,
};

pub const DirectionalLight = struct {
    name: []const u8 = "DirectionalLight",
    direction: Vec3 = Vec3.new(0.5, 1.0, 0.5),
    diffuse: Color3 = Color3.white,
    intensity: f32 = 1.0,

    pub fn init(name: []const u8, options: DirectionalLightOptions) DirectionalLight {
        return .{
            .name = name,
            .direction = options.direction.normalize(),
            .diffuse = options.diffuse,
            .intensity = options.intensity,
        };
    }
};
