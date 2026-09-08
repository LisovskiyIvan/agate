pub const Color3 = extern struct {
    r: f32 = 1.0,
    g: f32 = 1.0,
    b: f32 = 1.0,

    pub const white = Color3{ .r = 1, .g = 1, .b = 1 };
    pub const black = Color3{ .r = 0, .g = 0, .b = 0 };
    pub const red = Color3{ .r = 1, .g = 0, .b = 0 };
    pub const green = Color3{ .r = 0, .g = 1, .b = 0 };
    pub const blue = Color3{ .r = 0, .g = 0, .b = 1 };
    pub const yellow = Color3{ .r = 1, .g = 1, .b = 0 };
    pub const gray = Color3{ .r = 0.5, .g = 0.5, .b = 0.5 };

    pub fn new(r: f32, g: f32, b: f32) Color3 {
        return .{ .r = r, .g = g, .b = b };
    }

    pub fn toColor4(self: Color3, a: f32) Color4 {
        return Color4.new(self.r, self.g, self.b, a);
    }
};

pub const Color4 = extern struct {
    r: f32 = 1.0,
    g: f32 = 1.0,
    b: f32 = 1.0,
    a: f32 = 1.0,

    pub const white = Color4{ .r = 1, .g = 1, .b = 1, .a = 1 };
    pub const black = Color4{ .r = 0, .g = 0, .b = 0, .a = 1 };
    pub const transparent = Color4{ .r = 0, .g = 0, .b = 0, .a = 0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) Color4 {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    pub fn toArray(self: Color4) [4]f32 {
        return .{ self.r, self.g, self.b, self.a };
    }
};
