//! User-configurable simulation threading (phase 1: physics stepping).
//!
//! `ThreadingOptions` is the configuration surface for how the simulation
//! phases inside `Scene.update` are placed on threads. Today the only axis
//! is `physics`: step the `PhysicsWorld` inline on the calling (game)
//! thread (the default, bit-identical to all prior behavior, no thread
//! spawned) or on a Scene-owned dedicated worker overlapped with the
//! physics-independent phases (fork-join inside `Scene.update`).
//!
//! See `docs/threading.md` for the thread model, the determinism argument,
//! the parallel-window contract, fallback rules and counters.

pub const ThreadingOptions = struct {
    /// Physics world stepping placement.
    physics: PhysicsMode = .@"inline",

    pub const PhysicsMode = enum {
        /// Step inside `Scene.update` on the calling (game) thread.
        /// Default; bit-identical to all prior behavior; no thread spawned.
        ///
        /// NOTE: `inline` is a Zig keyword, so the tag is quoted
        /// (`ThreadingOptions.PhysicsMode.@"inline"`, `.@"inline"` at use
        /// sites) while keeping the brief-specified name.
        @"inline",
        /// Step on a Scene-owned dedicated worker thread, fork-joined
        /// inside `Scene.update`. Physics overlaps the
        /// physics-independent phases (animations, soft bodies, particles,
        /// decals). The worker spawns lazily on the first needed update;
        /// switching back to `.inline` parks it (no work posted, thread
        /// kept until `Scene.deinit`). When the worker cannot spawn, the
        /// scene degrades permanently to inline and bumps the worker's
        /// fallback counter.
        dedicated,
    };
};

test "threading defaults to inline physics" {
    const t = @import("std").testing;
    const opts = ThreadingOptions{};
    try t.expectEqual(ThreadingOptions.PhysicsMode.@"inline", opts.physics);
}
