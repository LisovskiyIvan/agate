//! NodeMaterial v1: a typed material graph that compiles to a hook snippet.
//!
//! A graph of small typed nodes (const/color, UV, time, texture sample,
//! arithmetic, mix, output) compiles — purely on CPU, deterministically —
//! into a `// @hook(albedo)` snippet plus `// @param` declarations that feed
//! the EXISTING ShaderMaterial path (build.zig `user_shader_materials` hook
//! merger, runtime packing via shader_material.zig). No engine sources are
//! touched to add a new material: bake the snippet through the hook table
//! once, then drive every graph parameter at runtime with the ordinary
//! `ShaderMaterial.setUniform` (that uniform update is the "no engine
//! rebuild" story — parameter tweaks never recompile anything).
//!
//! Pipeline:
//!
//! ```zig
//! var g = try agate.node_material.Graph.init(allocator);
//! defer g.deinit();
//! const red = try g.addNode(.const_color, .{ .color = .{ 0.8, 0.1, 0.1 } });
//! const out = try g.addNode(.output, .{});
//! try g.connect(out, 0, red);
//! var compiled = try g.compile(allocator, "my_red");
//! defer compiled.deinit(allocator);
//! // compiled.snippet -> register in build.zig user_shader_materials,
//! // compiled.params  -> declarative uniform table (merge.Param-compatible).
//! ```
//!
//! Codegen contract (stable, golden-tested):
//!   - one temp per reachable node, `float|vec2|vec3 _n<id>` in topological
//!     order (inputs before users, ties broken by node id — compiling twice
//!     yields byte-identical output);
//!   - unreachable nodes are dropped (no dead code in the snippet);
//!   - `// @param` decls: auto `u_time` first (only when a time node is
//!     used), then user params in node-id order; float params pack one word,
//!     color params pack a 4-aligned vec4 (rgb + 1.0 pad) mirroring
//!     merge.parseParamDecl, so `compiled.params` plugs straight into
//!     `shader_material.setUniform` / `defaultUniformStorage`;
//!   - the albedo body ends with `base.rgb = <color>;` and, only when the
//!     alpha port is wired, `base.a *= <alpha>;` (multiply preserves the
//!     sampled texture alpha; unwired alpha leaves `base` untouched).
//!
//! Time source: hook shaders expose no clock, so a `time` node becomes an
//! ordinary `u_time` float param (default 0.0) the app bumps per frame with
//! `setUniform("u_time", .{ .scalar = t })` — same rule as every graph param.
//!
//! Texture sampling v1: slot 0 only, sampled as
//! `texture(sampler2D(diffuse_tex, smp), <uv>).rgb` — the identifiers the
//! standard template guarantees in hook scope. Other slots are an explicit
//! `error.UnsupportedTextureSlot` (engine-template materials have a fixed
//! texture contract; a second sampler is a runtime-source feature).
//!
//! Out of v1 scope (documented non-goals): full PBR node set
//! (coat/sheen/etc. stay scalar ShaderMaterial/preset features), KHR/glTF
//! material import, hot-reload, serialization of graphs.

const std = @import("std");
const merge = @import("shader_material/merge.zig");

/// Port value type. Strict: connecting mismatched types is a hard
/// `error.TypeMismatch`, never an implicit conversion.
pub const Type = enum {
    float,
    vec2,
    vec3,
};

fn typeKeyword(t: Type) []const u8 {
    return switch (t) {
        .float => "float",
        .vec2 => "vec2",
        .vec3 => "vec3",
    };
}

/// Node operation. Ports are positional (`connect(dst, port, src)`):
/// const/uv/time expose no inputs; texture_sample takes uv on port 0;
/// add/multiply take (a, b) on ports 0..1 (same type); mix takes
/// (a, b, t) on ports 0..2 with a/b the same type and t a float;
/// sin takes x on port 0; output takes color (vec3, required) on port 0
/// and alpha (float, optional — defaults to untouched `base.a`) on port 1.
pub const Kind = enum {
    const_float,
    const_color,
    uv,
    time,
    texture_sample,
    add,
    multiply,
    mix,
    sin,
    output,
};

/// Max user-param name length (fixed buffer per node, no allocation).
pub const max_param_len: usize = 48;
/// Auto param backing every `time` node (single shared declaration).
pub const time_param_name = "u_time";

pub const Error = error{
    MissingOutput,
    MultipleOutputs,
    DanglingInput,
    TypeMismatch,
    Cycle,
    UnknownNode,
    DuplicateParam,
    BadParamName,
    UnsupportedTextureSlot,
    OutOfMemory,
};

/// Options for `Graph.addNode`. Only the fields matching `kind` are read.
pub const NodeOptions = struct {
    float: f32 = 0.0,
    color: [3]f32 = .{ 1, 1, 1 },
    /// Texture slot for `texture_sample` (v1: must be 0).
    tex_slot: u8 = 0,
    /// Exposes a const_float/const_color node as a `// @param` uniform.
    /// Empty = baked literal. Must be a C-like identifier, must not collide
    /// with merge.reserved_param_names / `u_time`, must be unique per graph.
    param: []const u8 = "",
};

const Node = struct {
    kind: Kind,
    float_val: f32 = 0.0,
    color_val: [3]f32 = .{ 1, 1, 1 },
    tex_slot: u8 = 0,
    param_buf: [max_param_len]u8 = [_]u8{0} ** max_param_len,
    param_len: u8 = 0,
    inputs: [3]?u32 = .{ null, null, null },

    fn paramName(self: *const Node) []const u8 {
        return self.param_buf[0..self.param_len];
    }

    fn hasParam(self: *const Node) bool {
        return self.param_len > 0;
    }
};

/// Material graph: append-only node store with positional edges.
/// Not thread-safe (build on one thread like every other scene-side value).
pub const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(Node) = .empty,

    pub fn init(allocator: std.mem.Allocator) Graph {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Graph) void {
        self.nodes.deinit(self.allocator);
    }

    pub fn nodeCount(self: *const Graph) usize {
        return self.nodes.items.len;
    }

    /// Appends a node, returns its id. Param names longer than
    /// `max_param_len` are a hard `error.BadParamName` (never truncated).
    pub fn addNode(self: *Graph, kind: Kind, opts: NodeOptions) Error!u32 {
        if (opts.param.len > max_param_len) return error.BadParamName;
        const id: u32 = @intCast(self.nodes.items.len);
        var node = Node{
            .kind = kind,
            .float_val = opts.float,
            .color_val = opts.color,
            .tex_slot = opts.tex_slot,
        };
        @memcpy(node.param_buf[0..opts.param.len], opts.param);
        node.param_len = @intCast(opts.param.len);
        self.nodes.append(self.allocator, node) catch return error.OutOfMemory;
        return id;
    }

    /// Wires `src` into positional `port` of `dst`. Bounds are checked at
    /// compile time (`error.UnknownNode`); port range is asserted here.
    pub fn connect(self: *Graph, dst: u32, port: u8, src: u32) void {
        std.debug.assert(dst < self.nodes.items.len);
        std.debug.assert(src < self.nodes.items.len);
        std.debug.assert(port < 3);
        self.nodes.items[dst].inputs[port] = src;
    }

    /// Runs validation only (same checks as `compile`, no output).
    pub fn validate(self: *const Graph, allocator: std.mem.Allocator) Error!void {
        var ctx = try ResolveContext.init(allocator, self);
        defer ctx.deinit();
        _ = try ctx.resolveOutput();
    }

    /// Validates and emits the hook snippet + declarative param table.
    /// Caller owns the result (see `Compiled.deinit`).
    pub fn compile(self: *const Graph, allocator: std.mem.Allocator, name: []const u8) Error!Compiled {
        var ctx = try ResolveContext.init(allocator, self);
        defer ctx.deinit();
        const out_id = try ctx.resolveOutput();
        return ctx.emit(allocator, name, out_id);
    }
};

/// Compiled graph: hook snippet text plus the merge-compatible param table.
/// `params` plugs into `shader_material.defaultUniformStorage` /
/// `shader_material.setUniform` unchanged (offsets follow the same float /
/// 4-aligned-vec4 packing as merge.parseParamDecl).
pub const Compiled = struct {
    snippet: []u8,
    params: []merge.Param,

    pub fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        allocator.free(self.snippet);
        allocator.free(self.params);
    }
};

// ---------------------------------------------------------------------------
// Resolution + codegen.
// ---------------------------------------------------------------------------

const Mark = enum { visiting, done };

const ResolveContext = struct {
    graph: *const Graph,
    types: []?Type, // resolved output type per node
    marks: []?Mark, // DFS cycle-guard marks
    order: std.ArrayListUnmanaged(u32) = .empty, // post-order (inputs first)

    fn init(allocator: std.mem.Allocator, graph: *const Graph) Error!ResolveContext {
        const n = graph.nodes.items.len;
        const types = allocator.alloc(?Type, n) catch return error.OutOfMemory;
        errdefer allocator.free(types);
        const marks = allocator.alloc(?Mark, n) catch {
            allocator.free(types);
            return error.OutOfMemory;
        };
        @memset(types, null);
        @memset(marks, null);
        return .{ .graph = graph, .types = types, .marks = marks };
    }

    fn deinit(self: *ResolveContext) void {
        // Allocator is not retained; free via the graph's allocator.
        self.graph.allocator.free(self.types);
        self.graph.allocator.free(self.marks);
        self.order.deinit(self.graph.allocator);
    }

    fn input(self: *ResolveContext, id: u32, port: u8) Error!u32 {
        const node = &self.graph.nodes.items[id];
        return node.inputs[port] orelse return error.DanglingInput;
    }

    fn checkNode(self: *ResolveContext, id: u32) Error!void {
        if (id >= self.graph.nodes.items.len) return error.UnknownNode;
    }

    fn resolve(self: *ResolveContext, id: u32) Error!Type {
        try self.checkNode(id);
        if (self.types[id]) |t| return t;
        if (self.marks[id]) |m| {
            if (m == .visiting) return error.Cycle;
            unreachable; // .done always pairs with a resolved type
        }
        self.marks[id] = .visiting;
        const node = &self.graph.nodes.items[id];
        const t: Type = switch (node.kind) {
            .const_float, .time, .sin => t: {
                if (node.kind == .sin) {
                    const x = try self.input(id, 0);
                    const xt = try self.resolve(x);
                    if (xt != .float) return error.TypeMismatch;
                }
                break :t .float;
            },
            .const_color, .texture_sample => t: {
                if (node.kind == .texture_sample) {
                    if (node.tex_slot != 0) return error.UnsupportedTextureSlot;
                    const uv = try self.input(id, 0);
                    const uvt = try self.resolve(uv);
                    if (uvt != .vec2) return error.TypeMismatch;
                }
                break :t .vec3;
            },
            .uv => .vec2,
            .add, .multiply => t: {
                const a = try self.input(id, 0);
                const b = try self.input(id, 1);
                const at = try self.resolve(a);
                const bt = try self.resolve(b);
                if (at != bt) return error.TypeMismatch;
                break :t at;
            },
            .mix => t: {
                const a = try self.input(id, 0);
                const b = try self.input(id, 1);
                const k = try self.input(id, 2);
                const at = try self.resolve(a);
                const bt = try self.resolve(b);
                const kt = try self.resolve(k);
                if (at != bt or kt != .float) return error.TypeMismatch;
                break :t at;
            },
            .output => t: {
                const c = try self.input(id, 0);
                const ct = try self.resolve(c);
                if (ct != .vec3) return error.TypeMismatch;
                if (node.inputs[1]) |alpha_id| {
                    const at = try self.resolve(alpha_id);
                    if (at != .float) return error.TypeMismatch;
                }
                // The sink has no value type of its own; mark vec3 so a
                // second resolution pass short-circuits consistently.
                break :t .vec3;
            },
        };
        // Named consts validate their param names up front (fail-closed:
        // a bad name never reaches the snippet).
        if ((node.kind == .const_float or node.kind == .const_color) and node.hasParam()) {
            try checkParamName(node.paramName());
        }
        self.types[id] = t;
        self.marks[id] = .done;
        self.order.append(self.graph.allocator, id) catch return error.OutOfMemory;
        return t;
    }

    fn resolveOutput(self: *ResolveContext) Error!u32 {
        var out_id: ?u32 = null;
        var count: usize = 0;
        for (self.graph.nodes.items, 0..) |n, i| {
            if (n.kind == .output) {
                count += 1;
                out_id = @intCast(i);
            }
        }
        if (count == 0) return error.MissingOutput;
        if (count > 1) return error.MultipleOutputs;
        const id = out_id.?;
        _ = try self.resolve(id);
        // Duplicate user-param detection across the whole graph (reachable
        // or not — a duplicated name is a typo even in dead code).
        var seen: std.ArrayListUnmanaged([]const u8) = .empty;
        defer seen.deinit(self.graph.allocator);
        var uses_time = false;
        for (self.graph.nodes.items) |*n| {
            if (n.kind == .time) uses_time = true;
            if ((n.kind == .const_float or n.kind == .const_color) and n.hasParam()) {
                for (seen.items) |s| {
                    if (std.mem.eql(u8, s, n.paramName())) return error.DuplicateParam;
                }
                seen.append(self.graph.allocator, n.paramName()) catch return error.OutOfMemory;
            }
        }
        if (uses_time) {
            for (seen.items) |s| {
                if (std.mem.eql(u8, s, time_param_name)) return error.DuplicateParam;
            }
        }
        return id;
    }

    fn emit(self: *ResolveContext, allocator: std.mem.Allocator, name: []const u8, out_id: u32) Error!Compiled {
        // Reachable set = post-order list (output last). Param decl order:
        // auto u_time first, then user params in node-id order.
        var uses_time = false;
        for (self.order.items) |id| {
            if (self.graph.nodes.items[id].kind == .time) {
                uses_time = true;
                break;
            }
        }
        var params: std.ArrayListUnmanaged(merge.Param) = .empty;
        errdefer params.deinit(allocator);
        var next_word: u8 = 0;
        if (uses_time) {
            try params.append(allocator, .{ .name = time_param_name, .offset = 0, .comps = 1 });
            next_word = 1;
        }
        // Node ids ascend in `order` except the output lands last; collect
        // user params in id order explicitly for determinism.
        const ids = self.order.items;
        std.mem.sort(u32, ids, {}, struct {
            fn less(_: void, a: u32, b: u32) bool {
                return a < b;
            }
        }.less);
        for (ids) |id| {
            const node = &self.graph.nodes.items[id];
            if ((node.kind == .const_float or node.kind == .const_color) and node.hasParam()) {
                if (node.kind == .const_float) {
                    try params.append(allocator, .{
                        .name = node.paramName(),
                        .offset = next_word,
                        .comps = 1,
                        .default = .{ node.float_val, 0, 0, 0 },
                    });
                    next_word += 1;
                } else {
                    const aligned: u8 = @intCast(std.mem.alignForward(u32, next_word, 4));
                    try params.append(allocator, .{
                        .name = node.paramName(),
                        .offset = aligned,
                        .comps = 4,
                        .default = .{ node.color_val[0], node.color_val[1], node.color_val[2], 1.0 },
                    });
                    next_word = aligned + 4;
                }
                if (next_word > merge.user_word_count) return error.OutOfMemory; // param space overflow
            }
        }

        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(allocator);
        try out.print(allocator, "// agate node material: {s} (generated — do not edit)\n// base: standard\n", .{name});
        if (uses_time) {
            try out.print(allocator, "// @param {s} float = 0.0\n", .{time_param_name});
        }
        for (params.items) |p| {
            if (std.mem.eql(u8, p.name, time_param_name)) continue;
            if (p.comps == 1) {
                try out.print(allocator, "// @param {s} float = {d}\n", .{ p.name, p.default[0] });
            } else {
                try out.print(allocator, "// @param {s} vec4 = {d} {d} {d} {d}\n", .{ p.name, p.default[0], p.default[1], p.default[2], p.default[3] });
            }
        }
        try out.print(allocator, "// @hook(albedo)\n", .{});
        // Body in post-order (inputs before users). The DFS from the single
        // output visits ports in order, so post-order is deterministic.
        for (self.order.items) |id| {
            const node = &self.graph.nodes.items[id];
            if (node.kind == .output) continue;
            try self.emitNode(allocator, &out, id);
        }
        const out_node = &self.graph.nodes.items[out_id];
        const color_src = out_node.inputs[0].?;
        try out.print(allocator, "base.rgb = _n{d};\n", .{color_src});
        if (out_node.inputs[1]) |alpha_src| {
            try out.print(allocator, "base.a *= _n{d};\n", .{alpha_src});
        }

        return .{
            .snippet = try out.toOwnedSlice(allocator),
            .params = try params.toOwnedSlice(allocator),
        };
    }

    fn emitNode(self: *ResolveContext, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), id: u32) Error!void {
        const node = &self.graph.nodes.items[id];
        const t = self.types[id].?;
        const printEmit = struct {
            fn emit(list: *std.ArrayListUnmanaged(u8), alloc: std.mem.Allocator, comptime format: []const u8, args: anytype) Error!void {
                list.print(alloc, format, args) catch return error.OutOfMemory;
            }
        }.emit;
        switch (node.kind) {
            .const_float => {
                if (node.hasParam()) {
                    try printEmit(out, allocator, "float _n{d} = {s};\n", .{ id, node.paramName() });
                } else {
                    try printEmit(out, allocator, "float _n{d} = {d};\n", .{ id, node.float_val });
                }
            },
            .const_color => {
                if (node.hasParam()) {
                    try printEmit(out, allocator, "vec3 _n{d} = {s}.rgb;\n", .{ id, node.paramName() });
                } else {
                    try printEmit(out, allocator, "vec3 _n{d} = vec3({d}, {d}, {d});\n", .{ id, node.color_val[0], node.color_val[1], node.color_val[2] });
                }
            },
            .uv => try printEmit(out, allocator, "vec2 _n{d} = v_uv;\n", .{id}),
            .time => try printEmit(out, allocator, "float _n{d} = {s};\n", .{ id, time_param_name }),
            .texture_sample => {
                const uv = node.inputs[0].?;
                try printEmit(out, allocator, "vec3 _n{d} = texture(sampler2D(diffuse_tex, smp), _n{d}).rgb;\n", .{ id, uv });
            },
            .add => {
                const a = node.inputs[0].?;
                const b = node.inputs[1].?;
                try printEmit(out, allocator, "{s} _n{d} = (_n{d} + _n{d});\n", .{ typeKeyword(t), id, a, b });
            },
            .multiply => {
                const a = node.inputs[0].?;
                const b = node.inputs[1].?;
                try printEmit(out, allocator, "{s} _n{d} = (_n{d} * _n{d});\n", .{ typeKeyword(t), id, a, b });
            },
            .mix => {
                const a = node.inputs[0].?;
                const b = node.inputs[1].?;
                const k = node.inputs[2].?;
                try printEmit(out, allocator, "{s} _n{d} = mix(_n{d}, _n{d}, _n{d});\n", .{ typeKeyword(t), id, a, b, k });
            },
            .sin => {
                const x = node.inputs[0].?;
                try printEmit(out, allocator, "float _n{d} = sin(_n{d});\n", .{ id, x });
            },
            .output => unreachable,
        }
    }
};

fn isIdentChar(c: u8, first: bool) bool {
    if (c == '_') return true;
    if (c >= 'a' and c <= 'z') return true;
    if (c >= 'A' and c <= 'Z') return true;
    if (!first and c >= '0' and c <= '9') return true;
    return false;
}

fn checkParamName(name: []const u8) Error!void {
    if (name.len == 0 or name.len > max_param_len) return error.BadParamName;
    if (!isIdentChar(name[0], true)) return error.BadParamName;
    for (name[1..]) |c| {
        if (!isIdentChar(c, false)) return error.BadParamName;
    }
    if (std.mem.eql(u8, name, time_param_name)) return error.DuplicateParam;
    for (merge.reserved_param_names) |res| {
        if (std.mem.eql(u8, name, res)) return error.BadParamName;
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

test "minimal graph compiles to the golden snippet" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const red = try g.addNode(.const_color, .{ .color = .{ 0.8, 0.1, 0.1 } });
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, red);

    var c = try g.compile(alloc, "flat_red");
    defer c.deinit(alloc);

    const expected =
        \\// agate node material: flat_red (generated — do not edit)
        \\// base: standard
        \\// @hook(albedo)
        \\vec3 _n0 = vec3(0.8, 0.1, 0.1);
        \\base.rgb = _n0;
        \\
    ;
    try std.testing.expectEqualStrings(expected, c.snippet);
    try std.testing.expectEqual(@as(usize, 0), c.params.len);
}

test "compile is deterministic across runs" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const uv = try g.addNode(.uv, .{});
    const tex = try g.addNode(.texture_sample, .{});
    g.connect(tex, 0, uv);
    const tint = try g.addNode(.const_color, .{ .color = .{ 1, 0.5, 0.25 }, .param = "u_tint" });
    const mul = try g.addNode(.multiply, .{});
    g.connect(mul, 0, tex);
    g.connect(mul, 1, tint);
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, mul);

    var a = try g.compile(alloc, "textured");
    defer a.deinit(alloc);
    var b = try g.compile(alloc, "textured");
    defer b.deinit(alloc);
    try std.testing.expectEqualStrings(a.snippet, b.snippet);
    // Param packing: vec4 4-aligned at word 0 with rgb + 1.0 pad.
    try std.testing.expectEqual(@as(usize, 1), a.params.len);
    try std.testing.expectEqualStrings("u_tint", a.params[0].name);
    try std.testing.expectEqual(@as(u8, 0), a.params[0].offset);
    try std.testing.expectEqual(@as(u8, 4), a.params[0].comps);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0.5, 0.25, 1.0 }, &a.params[0].default);
    // Golden body spot-checks.
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "// @param u_tint vec4 = 1 0.5 0.25 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "vec2 _n0 = v_uv;") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "texture(sampler2D(diffuse_tex, smp), _n0).rgb;") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "vec3 _n2 = u_tint.rgb;") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "vec3 _n3 = (_n1 * _n2);") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "base.rgb = _n3;") != null);
    // Unwired alpha leaves base.a alone.
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "base.a") == null);
}

test "time/mix/alpha graph wires uniforms and the alpha port" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const t = try g.addNode(.time, .{});
    const speed = try g.addNode(.const_float, .{ .float = 2.0, .param = "u_speed" });
    const rate = try g.addNode(.multiply, .{});
    g.connect(rate, 0, t);
    g.connect(rate, 1, speed);
    const wave = try g.addNode(.sin, .{});
    g.connect(wave, 0, rate);
    const lo = try g.addNode(.const_color, .{});
    const hi = try g.addNode(.const_color, .{ .color = .{ 0, 0, 1 } });
    const m = try g.addNode(.mix, .{});
    g.connect(m, 0, lo);
    g.connect(m, 1, hi);
    // Reuse the wave as the mix factor AND the output alpha (fan-out).
    g.connect(m, 2, wave);
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, m);
    g.connect(out, 1, wave);

    var c = try g.compile(alloc, "pulse");
    defer c.deinit(alloc);
    // Auto u_time first, then u_speed in node-id order.
    try std.testing.expectEqual(@as(usize, 2), c.params.len);
    try std.testing.expectEqualStrings("u_time", c.params[0].name);
    try std.testing.expectEqualStrings("u_speed", c.params[1].name);
    try std.testing.expectEqual(@as(u8, 1), c.params[1].offset);
    try std.testing.expect(std.mem.indexOf(u8, c.snippet, "float _n0 = u_time;") != null);
    try std.testing.expect(std.mem.indexOf(u8, c.snippet, "float _n3 = sin(_n2);") != null);
    try std.testing.expect(std.mem.indexOf(u8, c.snippet, "base.a *= _n3;") != null);
}

test "compiled params drive shader_material packing (runtime updates)" {
    const alloc = std.testing.allocator;
    const shader_material = @import("shader_material.zig");
    var g = Graph.init(alloc);
    defer g.deinit();
    const speed = try g.addNode(.const_float, .{ .float = 4.0, .param = "u_speed" });
    const white = try g.addNode(.const_color, .{});
    const mx = try g.addNode(.mix, .{});
    g.connect(mx, 0, white);
    g.connect(mx, 1, white);
    g.connect(mx, 2, speed);
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, mx);
    var c = try g.compile(alloc, "params_only");
    defer c.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), c.params.len);

    // Compiled params are merge-level declarations; convert to the runtime
    // table view (identical layout, asserted below) for uniform packing.
    comptime {
        const M = @import("shader_material/merge.zig").Param;
        const S = shader_material.Param;
        std.debug.assert(@sizeOf(M) == @sizeOf(S));
        std.debug.assert(@offsetOf(M, "name") == @offsetOf(S, "name"));
        std.debug.assert(@offsetOf(M, "offset") == @offsetOf(S, "offset"));
        std.debug.assert(@offsetOf(M, "comps") == @offsetOf(S, "comps"));
        std.debug.assert(@offsetOf(M, "default") == @offsetOf(S, "default"));
    }
    const sm_params = try alloc.alloc(shader_material.Param, c.params.len);
    defer alloc.free(sm_params);
    for (c.params, 0..) |p, i| {
        sm_params[i] = .{ .name = p.name, .offset = p.offset, .comps = p.comps, .default = p.default };
    }

    var storage = shader_material.defaultUniformStorage(sm_params);
    try std.testing.expectEqual(@as(f32, 4.0), storage[0][0]);
    // Runtime update without any rebuild — the v1 "no engine rebuild" path.
    try shader_material.setUniform(&storage, sm_params, "u_speed", .{ .scalar = 9.0 });
    try std.testing.expectEqual(@as(f32, 9.0), storage[0][0]);
    try std.testing.expectError(error.UnknownParam, shader_material.setUniform(&storage, sm_params, "u_nope", .{ .scalar = 1 }));
    try std.testing.expectEqual(@as(usize, 128), shader_material.uniformBytes(&storage).len);
}

test "generated snippet merges through the hook merger" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const k = try g.addNode(.const_float, .{ .float = 0.5, .param = "u_k" });
    const c0 = try g.addNode(.const_color, .{});
    // Scale via mix against black (float * vec3 would be a type error).
    const black = try g.addNode(.const_color, .{ .color = .{ 0, 0, 0 } });
    const mx = try g.addNode(.mix, .{});
    g.connect(mx, 0, black);
    g.connect(mx, 1, c0);
    g.connect(mx, 2, k);
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, mx);
    var c = try g.compile(alloc, "merge_probe");
    defer c.deinit(alloc);

    const tmpl =
        \\// @hook(decls)
        \\// @endhook
        \\void main() {
        \\    vec3 base = vec3(1.0);
        \\    // @hook(albedo)
        \\    // @endhook
        \\}
        \\
    ;
    const res = try merge.merge(alloc, .{ .template = tmpl, .snippet = c.snippet, .base = "standard", .material_name = "merge_probe" });
    defer alloc.free(res.glsl);
    defer alloc.free(res.params);
    try std.testing.expectEqual(@as(usize, 1), res.params.len);
    try std.testing.expectEqualStrings("u_k", res.params[0].name);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "base.rgb = ") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "#define u_k (sm_user_0.x)") != null);
}

test "validation rejects cycles, mismatches and dangling inputs" {
    const alloc = std.testing.allocator;
    // Cycle: a feeds b feeds a (via output root).
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.add, .{});
        const b = try g.addNode(.add, .{});
        const f = try g.addNode(.const_float, .{});
        g.connect(a, 0, b);
        g.connect(a, 1, f);
        g.connect(b, 0, a);
        g.connect(b, 1, f);
        const out = try g.addNode(.output, .{});
        const w = try g.addNode(.const_color, .{});
        g.connect(out, 0, w);
        // Output is fine but a/b are unreachable — unreachable cycles are
        // dropped, so this compiles. Wire the cycle to the output instead.
        _ = try g.addNode(.output, .{});
        try std.testing.expectError(error.MultipleOutputs, g.compile(alloc, "x"));
    }
    // True cycle through the output.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.add, .{});
        const f = try g.addNode(.const_float, .{});
        g.connect(a, 0, a); // self-loop
        g.connect(a, 1, f);
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, a);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        try std.testing.expectError(error.Cycle, g.compile(alloc, "x"));
    }
    // Type mismatch: float into a vec3 color port.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const f = try g.addNode(.const_float, .{});
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, f);
        try std.testing.expectError(error.TypeMismatch, g.compile(alloc, "x"));
    }
    // Dangling: add with only one input wired.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.add, .{});
        const f = try g.addNode(.const_float, .{});
        g.connect(a, 0, f);
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        const k = try g.addNode(.const_float, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, k);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        // `a` is unreachable: compiles fine (dead code dropped).
        var c = try g.compile(alloc, "dead_ok");
        defer c.deinit(alloc);
        try std.testing.expect(std.mem.indexOf(u8, c.snippet, "_n0") == null);
    }
    // Dangling on the REACHABLE path is an error.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.add, .{});
        const f = try g.addNode(.const_float, .{});
        g.connect(a, 0, f);
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, a);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        try std.testing.expectError(error.DanglingInput, g.compile(alloc, "x"));
    }
    // Missing output / unknown hookups.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        _ = try g.addNode(.const_float, .{});
        try std.testing.expectError(error.MissingOutput, g.compile(alloc, "x"));
    }
}

test "param names are guarded (reserved, duplicate, malformed, slots)" {
    const alloc = std.testing.allocator;
    // Reserved GLSL-ish name from merge.reserved_param_names, left
    // unreachable: dead code (and its names) never reach the snippet.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        _ = try g.addNode(.const_float, .{ .param = "roughness" });
        const w = try g.addNode(.const_color, .{});
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, w);
        _ = try g.addNode(.const_float, .{});
        _ = try g.addNode(.mix, .{});
        var c = try g.compile(alloc, "ok");
        defer c.deinit(alloc);
    }
    // Reserved name ON the reachable path is a hard error.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const k = try g.addNode(.const_float, .{ .param = "albedo" });
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, k);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        try std.testing.expectError(error.BadParamName, g.compile(alloc, "x"));
    }
    // Duplicate user params.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.const_float, .{ .param = "u_dup" });
        _ = try g.addNode(.const_float, .{ .param = "u_dup" });
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, a);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        try std.testing.expectError(error.DuplicateParam, g.compile(alloc, "x"));
    }
    // Malformed identifiers.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        try std.testing.expectError(error.BadParamName, g.addNode(.const_float, .{ .param = "this name has spaces and is way too long for the buffer limit ok" }));
        const w = try g.addNode(.const_color, .{});
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, w);
        const bad = try g.addNode(.const_float, .{ .param = "9lives" });
        // Reachable bad names fail: wire it in as the mix factor.
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, bad);
        g.connect(out, 0, mx);
        try std.testing.expectError(error.BadParamName, g.compile(alloc, "x"));
    }
    // Non-zero texture slot.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const uv = try g.addNode(.uv, .{});
        const tx = try g.addNode(.texture_sample, .{ .tex_slot = 1 });
        g.connect(tx, 0, uv);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, tx);
        try std.testing.expectError(error.UnsupportedTextureSlot, g.compile(alloc, "x"));
    }
    // texture_sample fed a float is a mismatch.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const f = try g.addNode(.const_float, .{});
        const tx = try g.addNode(.texture_sample, .{});
        g.connect(tx, 0, f);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, tx);
        try std.testing.expectError(error.TypeMismatch, g.compile(alloc, "x"));
    }
}
