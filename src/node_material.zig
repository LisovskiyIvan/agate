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
    cos,
    clamp,
    step,
    smoothstep,
    pow,
    dot,
    length,
    normalize,
    fract,
    fresnel,
    panner,
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
    InvalidJson,
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

    /// Serializes the graph description to JSON text.
    pub fn serializeJson(self: *const Graph, allocator: std.mem.Allocator) ![]u8 {
        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(allocator);
        try out.print(allocator, "{{\n  \"nodes\": [\n", .{});
        for (self.nodes.items, 0..) |n, i| {
            try out.print(allocator, "    {{\"id\": {d}, \"kind\": \"{s}\"", .{ i, @tagName(n.kind) });
            if (n.kind == .const_float) {
                try out.print(allocator, ", \"float\": {d}", .{n.float_val});
            } else if (n.kind == .const_color) {
                try out.print(allocator, ", \"color\": [{d}, {d}, {d}]", .{ n.color_val[0], n.color_val[1], n.color_val[2] });
            } else if (n.kind == .texture_sample) {
                try out.print(allocator, ", \"tex_slot\": {d}", .{n.tex_slot});
            }
            if (n.hasParam()) {
                try out.print(allocator, ", \"param\": \"{s}\"", .{n.paramName()});
            }
            try out.print(allocator, ", \"inputs\": [", .{});
            var first = true;
            for (n.inputs) |inp| {
                if (inp) |src| {
                    if (!first) try out.print(allocator, ", ", .{});
                    try out.print(allocator, "{d}", .{src});
                    first = false;
                }
            }
            try out.print(allocator, "]}}", .{});
            if (i + 1 < self.nodes.items.len) {
                try out.print(allocator, ",\n", .{});
            } else {
                try out.print(allocator, "\n", .{});
            }
        }
        try out.print(allocator, "  ]\n}}\n", .{});
        return out.toOwnedSlice(allocator);
    }

    /// Deserializes a graph from JSON text.
    pub fn deserializeJson(allocator: std.mem.Allocator, json_text: []const u8) !Graph {
        const NodeEntry = struct {
            id: ?u32 = null,
            kind: []const u8,
            float: ?f32 = null,
            color: ?[3]f32 = null,
            tex_slot: ?u8 = null,
            param: ?[]const u8 = null,
            inputs: ?[]const u32 = null,
        };
        const Schema = struct {
            nodes: []const NodeEntry,
        };
        const parsed = std.json.parseFromSlice(Schema, allocator, json_text, .{ .ignore_unknown_fields = true }) catch return error.InvalidJson;
        defer parsed.deinit();

        var g = Graph.init(allocator);
        errdefer g.deinit();

        for (parsed.value.nodes) |ne| {
            const kind = std.meta.stringToEnum(Kind, ne.kind) orelse return error.InvalidJson;
            _ = try g.addNode(kind, .{
                .float = ne.float orelse 0.0,
                .color = ne.color orelse .{ 1, 1, 1 },
                .tex_slot = ne.tex_slot orelse 0,
                .param = ne.param orelse "",
            });
        }
        for (parsed.value.nodes, 0..) |ne, dst| {
            if (ne.inputs) |inps| {
                for (inps, 0..) |src, port| {
                    if (port < 3) {
                        g.connect(@intCast(dst), @intCast(port), src);
                    }
                }
            }
        }
        return g;
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
            .const_float, .time, .sin, .cos => t: {
                if (node.kind == .sin or node.kind == .cos) {
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
            .pow => t: {
                const a = try self.input(id, 0);
                const b = try self.input(id, 1);
                const at = try self.resolve(a);
                const bt = try self.resolve(b);
                if (at != .float or bt != .float) return error.TypeMismatch;
                break :t .float;
            },
            .dot => t: {
                const a = try self.input(id, 0);
                const b = try self.input(id, 1);
                const at = try self.resolve(a);
                const bt = try self.resolve(b);
                if (at != bt or at == .float) return error.TypeMismatch;
                break :t .float;
            },
            .length => t: {
                const a = try self.input(id, 0);
                const at = try self.resolve(a);
                if (at == .float) return error.TypeMismatch;
                break :t .float;
            },
            .normalize => t: {
                const a = try self.input(id, 0);
                const at = try self.resolve(a);
                if (at == .float) return error.TypeMismatch;
                break :t at;
            },
            .fract => t: {
                const a = try self.input(id, 0);
                const at = try self.resolve(a);
                break :t at;
            },
            .step => t: {
                const edge = try self.input(id, 0);
                const x = try self.input(id, 1);
                const et = try self.resolve(edge);
                const xt = try self.resolve(x);
                if (et != xt or et != .float) return error.TypeMismatch;
                break :t .float;
            },
            .smoothstep => t: {
                const e0 = try self.input(id, 0);
                const e1 = try self.input(id, 1);
                const x = try self.input(id, 2);
                const e0t = try self.resolve(e0);
                const e1t = try self.resolve(e1);
                const xt = try self.resolve(x);
                if (e0t != .float or e1t != .float or xt != .float) return error.TypeMismatch;
                break :t .float;
            },
            .clamp => t: {
                const x = try self.input(id, 0);
                const min_v = try self.input(id, 1);
                const max_v = try self.input(id, 2);
                const xt = try self.resolve(x);
                const mint = try self.resolve(min_v);
                const maxt = try self.resolve(max_v);
                if (xt != mint or xt != maxt) return error.TypeMismatch;
                break :t xt;
            },
            .fresnel => t: {
                const p = try self.input(id, 0);
                const pt = try self.resolve(p);
                if (pt != .float) return error.TypeMismatch;
                break :t .float;
            },
            .panner => t: {
                const uv = try self.input(id, 0);
                const speed = try self.input(id, 1);
                const time = try self.input(id, 2);
                const uvt = try self.resolve(uv);
                const speedt = try self.resolve(speed);
                const timet = try self.resolve(time);
                if (uvt != .vec2 or speedt != .vec2 or timet != .float) return error.TypeMismatch;
                break :t .vec2;
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
            .cos => {
                const x = node.inputs[0].?;
                try printEmit(out, allocator, "float _n{d} = cos(_n{d});\n", .{ id, x });
            },
            .pow => {
                const a = node.inputs[0].?;
                const b = node.inputs[1].?;
                try printEmit(out, allocator, "float _n{d} = pow(_n{d}, _n{d});\n", .{ id, a, b });
            },
            .dot => {
                const a = node.inputs[0].?;
                const b = node.inputs[1].?;
                try printEmit(out, allocator, "float _n{d} = dot(_n{d}, _n{d});\n", .{ id, a, b });
            },
            .length => {
                const a = node.inputs[0].?;
                try printEmit(out, allocator, "float _n{d} = length(_n{d});\n", .{ id, a });
            },
            .normalize => {
                const a = node.inputs[0].?;
                try printEmit(out, allocator, "{s} _n{d} = normalize(_n{d});\n", .{ typeKeyword(t), id, a });
            },
            .fract => {
                const a = node.inputs[0].?;
                try printEmit(out, allocator, "{s} _n{d} = fract(_n{d});\n", .{ typeKeyword(t), id, a });
            },
            .step => {
                const edge = node.inputs[0].?;
                const x = node.inputs[1].?;
                try printEmit(out, allocator, "float _n{d} = step(_n{d}, _n{d});\n", .{ id, edge, x });
            },
            .smoothstep => {
                const e0 = node.inputs[0].?;
                const e1 = node.inputs[1].?;
                const x = node.inputs[2].?;
                try printEmit(out, allocator, "float _n{d} = smoothstep(_n{d}, _n{d}, _n{d});\n", .{ id, e0, e1, x });
            },
            .clamp => {
                const x = node.inputs[0].?;
                const mn = node.inputs[1].?;
                const mx = node.inputs[2].?;
                try printEmit(out, allocator, "{s} _n{d} = clamp(_n{d}, _n{d}, _n{d});\n", .{ typeKeyword(t), id, x, mn, mx });
            },
            .fresnel => {
                const p = node.inputs[0].?;
                try printEmit(out, allocator, "float _n{d} = pow(1.0 - clamp(dot(N, normalize(eye_pos - v_world_pos)), 0.0, 1.0), _n{d});\n", .{ id, p });
            },
            .panner => {
                const uv = node.inputs[0].?;
                const speed = node.inputs[1].?;
                const time = node.inputs[2].?;
                try printEmit(out, allocator, "vec2 _n{d} = fract(_n{d} + _n{d} * _n{d});\n", .{ id, uv, speed, time });
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
