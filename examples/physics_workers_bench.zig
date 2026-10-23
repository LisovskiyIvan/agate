//! Headless benchmark for Box3D's internal worker count.
//!
//! Builds the same deterministic box scene in a `PhysicsWorld` created with
//! `workers = 1, 2, 4, ...` and reports median wall time per step plus a
//! position checksum. The checksum answers the determinism question: for a
//! fixed scene and dt stream, does the worker count change the result?
//!
//! Run from examples/:
//!   zig build run-physics-workers -Doptimize=ReleaseFast -- \
//!       --bodies 768 --steps 120 --reps 5 --workers 1,2,4
//!
//! The benchmark is CPU-only: no window, no GPU context, no sokol calls.

const std = @import("std");
const z = @import("agate");

const Cfg = struct {
    bodies: usize = 768,
    steps: usize = 120,
    reps: usize = 5,
    workers: [8]u8 = .{ 1, 2, 4, 0, 0, 0, 0, 0 },
    workers_len: usize = 3,
};

fn usage() void {
    std.debug.print(
        \\usage: physics-workers-bench [options]
        \\  --bodies N       dynamic boxes (default 768)
        \\  --steps N        fixed 1/60 steps per rep (default 120)
        \\  --reps N         repetitions per worker count (default 5)
        \\  --workers LIST   comma-separated worker counts (default 1,2,4)
        \\
    , .{});
}

fn parseArgs(args: std.process.Args, alloc: std.mem.Allocator) !Cfg {
    var cfg = Cfg{};
    var it = try std.process.Args.Iterator.initAllocator(args, alloc);
    defer it.deinit();
    _ = it.next(); // program name
    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            usage();
            return error.Help;
        } else if (std.mem.eql(u8, a, "--bodies")) {
            const v = it.next() orelse return error.BadArg;
            cfg.bodies = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--steps")) {
            const v = it.next() orelse return error.BadArg;
            cfg.steps = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--reps")) {
            const v = it.next() orelse return error.BadArg;
            cfg.reps = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--workers")) {
            const v = it.next() orelse return error.BadArg;
            cfg.workers_len = 0;
            var wit = std.mem.splitScalar(u8, v, ',');
            while (wit.next()) |part| {
                if (cfg.workers_len == cfg.workers.len) return error.BadArg;
                cfg.workers[cfg.workers_len] = try std.fmt.parseInt(u8, part, 10);
                cfg.workers_len += 1;
            }
            if (cfg.workers_len == 0) return error.BadArg;
        } else {
            std.debug.print("unknown argument: {s}\n", .{a});
            usage();
            return error.BadArg;
        }
    }
    return cfg;
}

fn makeMesh(pos: z.Vec3) z.Mesh {
    var m = z.Mesh{
        .name = "bench_box",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = pos,
    };
    m.local_bounding_box = z.BoundingBox.init(
        z.Vec3.new(-0.45, -0.45, -0.45),
        z.Vec3.new(0.45, 0.45, 0.45),
    );
    return m;
}

fn layoutMesh(i: usize) z.Mesh {
    const cols: usize = 8;
    const layer = i / (cols * cols);
    const in_layer = i % (cols * cols);
    const cx = @as(f32, @floatFromInt(in_layer % cols)) - @as(f32, @floatFromInt(cols)) / 2.0;
    const cz = @as(f32, @floatFromInt(in_layer / cols)) - @as(f32, @floatFromInt(cols)) / 2.0;
    // Deterministic asymmetry so the stack is not perfectly symmetric.
    const jitter = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 0.06;
    return makeMesh(z.Vec3.new(cx + jitter, 1.2 + @as(f32, @floatFromInt(layer)) * 1.05, cz));
}

fn runOne(
    alloc: std.mem.Allocator,
    workers: u8,
    meshes: []z.Mesh,
    steps: usize,
    reps: usize,
) !struct { median_ns_per_step: f64, checksum: u64, repeatable: bool, actual_workers: u8 } {
    var samples = try alloc.alloc(u64, reps);
    defer alloc.free(samples);
    var checksums = try alloc.alloc(u64, reps);
    defer alloc.free(checksums);

    var actual_workers: u8 = 0;

    for (0..reps) |rep| {
        var world = z.PhysicsWorld.initWithWorkerCount(alloc, workers);
        defer world.deinit();
        actual_workers = world.getWorkerCount();

        // Static floor.
        var floor = z.Mesh{
            .name = "floor",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 0,
            .position = z.Vec3.new(0, -0.5, 0),
        };
        floor.local_bounding_box = z.BoundingBox.init(
            z.Vec3.new(-20, -0.5, -20),
            z.Vec3.new(20, 0.5, 20),
        );
        _ = try world.createBody(&floor, .box, 0.0);

        // Every rep starts from the same initial pose.
        for (meshes, 0..) |*m, i| m.* = layoutMesh(i);

        for (meshes) |*m| {
            _ = try world.createBody(m, .box, 1.0);
        }

        const t0 = z.jobs.monoNs();
        for (0..steps) |_| world.step(1.0 / 60.0);
        samples[rep] = z.jobs.monoNs() -% t0;

        var hasher = std.hash.Wyhash.init(0x9e3779b97f4a7c15);
        for (meshes) |*m| {
            hasher.update(std.mem.asBytes(&m.position));
            hasher.update(std.mem.asBytes(&m.rotation));
            if (!std.math.isFinite(m.position.y)) return error.NonFinite;
        }
        checksums[rep] = hasher.final();
    }

    var repeatable = true;
    for (checksums[1..]) |c| {
        if (c != checksums[0]) repeatable = false;
    }

    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const median = samples[samples.len / 2];
    const per_step = @as(f64, @floatFromInt(median)) / @as(f64, @floatFromInt(steps));
    return .{
        .median_ns_per_step = per_step,
        .checksum = checksums[checksums.len - 1],
        .repeatable = repeatable,
        .actual_workers = actual_workers,
    };
}

pub fn main(init: std.process.Init) !void {
    const alloc = std.heap.smp_allocator;
    const cfg = parseArgs(init.minimal.args, alloc) catch |err| switch (err) {
        error.Help => return,
        else => return err,
    };

    const meshes = try alloc.alloc(z.Mesh, cfg.bodies);
    defer alloc.free(meshes);
    for (meshes, 0..) |*m, i| m.* = layoutMesh(i);

    std.debug.print("physics workers bench: bodies={d} steps={d} reps={d}\n", .{ cfg.bodies, cfg.steps, cfg.reps });

    for (cfg.workers[0..cfg.workers_len]) |workers| {
        const res = try runOne(alloc, workers, meshes, cfg.steps, cfg.reps);
        std.debug.print(
            "workers={d} (actual {d}): {d:.3} ms/step  checksum=0x{x:0>16}  repeatable={s}\n",
            .{ workers, res.actual_workers, res.median_ns_per_step / 1_000_000.0, res.checksum, if (res.repeatable) "yes" else "NO" },
        );
    }
}
