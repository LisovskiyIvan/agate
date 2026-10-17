//! Headless ReleaseFast benchmark for the fork-join pool.
//!
//! Measures the real `jobs.Pool.forkJoin` path against an inline run of the
//! same disjoint-index kernel, for worker counts 1/2/4/8 and three job
//! shapes:
//!   - large:          len=65536, CPU-heavy per element (throughput path)
//!   - medium:         len=4096, more rounds per element (wake amortization)
//!   - fixed_overhead: len=4, direct `forkJoin` (not `parallelFor`, which
//!                     would inline below `min_len_for_workers`): isolates
//!                     publish + wake-all + spin-wait cost.
//!
//! Correctness is checked before timing: inline and pool runs start from the
//! same seed and must produce byte-identical data. The benchmark prints the
//! median ns per pass and ns per element; no GPU, no window.
//!
//! Run (from examples/):
//!   zig build run-jobs-bench -Doptimize=ReleaseFast -- --reps 7
//!
//! Result interpretation: a large speedup on `large` means the pool is worth
//! wiring (the engine currently leaves `jobs.global` to the host); a large
//! `fixed_overhead` relative to inline is the wake/spin cost the audit
//! flagged for a future per-worker wake strategy.

const std = @import("std");
const z = @import("agate");
const sokol = @import("sokol");

const jobs = z.jobs;
const alloc_global = std.heap.smp_allocator;

const Cfg = struct {
    reps: usize = 7,
    warmup: usize = 1,
};

const Shape = struct {
    name: []const u8,
    len: usize,
    rounds: u32,
    iters: usize,
};

const shapes = [_]Shape{
    .{ .name = "large_len65536", .len = 65_536, .rounds = 64, .iters = 32 },
    .{ .name = "medium_len4096", .len = 4_096, .rounds = 256, .iters = 160 },
    .{ .name = "fixed_overhead_len4", .len = 4, .rounds = 4_096, .iters = 2_000 },
};

const Work = struct {
    data: []u64,
    rounds: u32,

    fn run(self: *Work, start: usize, end: usize) void {
        var i = start;
        while (i < end) : (i += 1) {
            var x = self.data[i] ^ @as(u64, @intCast(i));
            var r: u32 = 0;
            while (r < self.rounds) : (r += 1) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
            }
            self.data[i] = x;
        }
    }
};

const Result = struct {
    checksum: u64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    total_ms: f64,
};

fn lessF64(_: void, a: f64, b: f64) bool {
    return a < b;
}

fn median(times: []f64) f64 {
    std.mem.sort(f64, times, {}, lessF64);
    return times[times.len / 2];
}

fn fill(data: []u64) void {
    for (data, 0..) |*v, i| {
        v.* = 0x9e3779b97f4a7c15 *% (@as(u64, @intCast(i)) +% 1);
    }
}

fn checksum(data: []const u64) u64 {
    var h = std.hash.Wyhash.init(0);
    h.update(std.mem.sliceAsBytes(data));
    return h.final();
}

fn runPass(w: *Work, pool: ?*jobs.Pool) void {
    if (pool) |p| {
        p.forkJoin(Work, w, Work.run, w.data.len);
    } else {
        w.run(0, w.data.len);
    }
}

fn measure(pool: ?*jobs.Pool, shape: Shape, cfg: Cfg, seed_data: []const u64, scratch: []u64) !Result {
    var w = Work{ .data = scratch, .rounds = shape.rounds };
    // Warmup on a copy; each measured sample starts from the same seed so
    // every run performs the same work (xorshift is not idempotent, but the
    // amount of work is what we time).
    for (0..cfg.warmup) |_| {
        @memcpy(scratch, seed_data);
        for (0..shape.iters) |_| runPass(&w, pool);
    }
    const times = try alloc_global.alloc(f64, cfg.reps);
    defer alloc_global.free(times);
    var total: f64 = 0;
    for (0..cfg.reps) |r| {
        @memcpy(scratch, seed_data);
        const t0 = sokol.time.now();
        for (0..shape.iters) |_| runPass(&w, pool);
        const ms: f64 = @floatCast(sokol.time.ms(sokol.time.now() -% t0));
        times[r] = ms;
        total += ms;
    }
    var mn = times[0];
    var mx = times[0];
    for (times) |t| {
        mn = @min(mn, t);
        mx = @max(mx, t);
    }
    return .{
        .checksum = checksum(scratch),
        .median_ms = median(times),
        .min_ms = mn,
        .max_ms = mx,
        .total_ms = total,
    };
}

fn report(shape: Shape, workers: usize, r: Result, cfg: Cfg) void {
    const passes = @as(f64, @floatFromInt(shape.iters));
    const elems = @as(f64, @floatFromInt(shape.iters * shape.len));
    const ns_per_pass = r.median_ms * 1e6 / passes;
    const ns_per_elem = r.median_ms * 1e6 / elems;
    std.debug.print(
        "case={s} workers={d} reps={d} iters={d} checksum={x} median={d:.3}ms min={d:.3}ms max={d:.3}ms total={d:.1}ms ns_per_pass={d:.1} ns_per_elem={d:.3}\n",
        .{ shape.name, workers, cfg.reps, shape.iters, r.checksum, r.median_ms, r.min_ms, r.max_ms, r.total_ms, ns_per_pass, ns_per_elem },
    );
    if (r.total_ms < 100.0) {
        std.debug.print("case={s} workers={d} INCONCLUSIVE: total {d:.1}ms < 100ms\n", .{ shape.name, workers, r.total_ms });
        std.process.exit(2);
    }
}

fn parseReps(args: []const []const u8) usize {
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--reps") and i + 1 < args.len) {
            const v = std.fmt.parseInt(usize, args[i + 1], 10) catch 7;
            return std.math.clamp(v, 3, 15);
        }
    }
    return 7;
}

pub fn main(init: std.process.Init) !void {
    sokol.time.setup();
    var args_buf: [16][]const u8 = undefined;
    var n_args: usize = 0;
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc_global);
    defer it.deinit();
    _ = it.next();
    while (it.next()) |a| {
        if (n_args < args_buf.len) {
            args_buf[n_args] = a;
            n_args += 1;
        }
    }
    const cfg = Cfg{ .reps = parseReps(args_buf[0..n_args]) };
    std.debug.print("jobs_bench: reps={d} warmup={d} allocator=smp clock=sokol.time units=ms\n", .{ cfg.reps, cfg.warmup });

    const max_len = 65_536;
    const seed = try alloc_global.alloc(u64, max_len);
    defer alloc_global.free(seed);
    fill(seed);

    // Worker configurations. `0` is the inline reference (no pool).
    const worker_counts = [_]usize{ 0, 1, 2, 4, 8 };

    for (shapes) |shape| {
        // Correctness: inline and each pool must agree on the final bytes.
        const ref = try alloc_global.alloc(u64, shape.len);
        defer alloc_global.free(ref);
        @memcpy(ref, seed[0..shape.len]);
        var w_ref = Work{ .data = ref, .rounds = shape.rounds };
        w_ref.run(0, shape.len);
        const ref_checksum = checksum(ref);

        for (worker_counts) |wc| {
            var pool: ?*jobs.Pool = null;
            if (wc > 0) {
                pool = try jobs.Pool.init(alloc_global, wc);
            }
            defer if (pool) |p| p.deinit();

            const scratch = try alloc_global.alloc(u64, shape.len);
            defer alloc_global.free(scratch);
            @memcpy(scratch, seed[0..shape.len]);
            var w = Work{ .data = scratch, .rounds = shape.rounds };
            runPass(&w, pool);
            if (checksum(scratch) != ref_checksum) {
                std.debug.panic("jobs_bench: checksum mismatch shape={s} workers={d}", .{ shape.name, wc });
            }

            const r = try measure(pool, shape, cfg, seed[0..shape.len], scratch);
            report(shape, wc, r, cfg);
        }
    }
    std.debug.print("jobs_bench: done\n", .{});
}
