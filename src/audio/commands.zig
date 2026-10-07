//! Voice trigger queue: lock-free SPSC command ring + slot stealing.
//! Split out of `audio.zig` (facade).
//!
//! `Command` is the trigger payload (`play`/`playClip` in `playback.zig`
//! enqueue one; the mixer drains them in `renderFrames`). `pushCommand`
//! runs on any thread, `drainCommands`/`applyCommand`/`acquireSlot` run on
//! the audio thread inside `renderFrames` — same 0-allocation, lock-free
//! discipline as before the split; only the file changed, no
//! synchronization or memory behavior.
//!
//! Anti-cycle rule (same as `profiler/*`, `particles/*`): every function
//! takes the engine as `anytype` (a `*AudioEngine` from `engine.zig` in
//! practice) and this module never imports `engine.zig` or the `audio.zig`
//! facade back. Field access through `anytype` is structural, so no
//! owner edge exists at all. `playback.zig` and `mixer.zig` reach these
//! helpers through direct sibling imports (documented in the facade).
//! None of this is re-exported by the facade: `Command` was private before
//! the split and stays reachable only through the sibling that needs it.

const clip_mod = @import("clip.zig");
const types = @import("types.zig");

const BusId = types.BusId;
const Voice = types.Voice;
const VoiceKind = types.VoiceKind;
const AudioClip = clip_mod.AudioClip;

pub const Command = union(enum) {
    voice: struct {
        kind: VoiceKind,
        bus: ?BusId = null,
        volume: f32,
        pan: f32,
        duration: f32,
        freq: f32,
        freq_end: f32,
        cutoff: f32,
        cutoff_end: f32,
        seed: u32,
    },
    clip: struct {
        clip: *const AudioClip,
        bus: ?BusId = null,
        volume: f32,
        pan: f32,
        duration: f32,
        sample_step: f64,
        loop: bool,
        seed: u32,
        cutoff: f32 = 20000.0,
    },
};

/// Pushes an audio trigger command to the lock-free SPSC queue.
pub fn pushCommand(self: anytype, cmd: Command) bool {
    const head = self.cmd_head.load(.monotonic);
    const tail = self.cmd_tail.load(.acquire);
    if (head -% tail >= types.max_commands) {
        return false;
    }
    self.cmd_ring[head % types.max_commands] = cmd;
    self.cmd_head.store(head +% 1, .release);
    return true;
}

/// Drains pending commands from the lock-free SPSC queue into voices.
pub fn drainCommands(self: anytype) void {
    const head = self.cmd_head.load(.acquire);
    var tail = self.cmd_tail.load(.monotonic);
    while (tail != head) {
        const cmd = self.cmd_ring[tail % types.max_commands];
        applyCommand(self, cmd);
        tail +%= 1;
    }
    self.cmd_tail.store(tail, .release);
}

fn applyCommand(self: anytype, cmd: Command) void {
    const v = acquireSlot(self) orelse return;
    switch (cmd) {
        .voice => |p| {
            v.* = .{
                .active = true,
                .kind = p.kind,
                .bus = p.bus,
                .volume = p.volume,
                .pan = p.pan,
                .duration = p.duration,
                .freq = p.freq,
                .freq_end = p.freq_end,
                .cutoff = p.cutoff,
                .cutoff_end = p.cutoff_end,
                .seed = p.seed,
            };
        },
        .clip => |c| {
            v.* = .{
                .active = true,
                .kind = .sample,
                .bus = c.bus,
                .volume = c.volume,
                .pan = c.pan,
                .duration = c.duration,
                .cutoff = c.cutoff,
                .cutoff_end = c.cutoff,
                .lp = 0.0,
                .lp_r = 0.0,
                .clip = c.clip,
                .sample_pos = 0.0,
                .sample_step = c.sample_step,
                .loop = c.loop,
                .seed = c.seed,
            };
        },
    }
}

/// Picks a free voice, or steals the one closest to finishing.
fn acquireSlot(self: anytype) ?*Voice {
    for (&self.voices) |*v| {
        if (!v.active) return v;
    }
    // Steal the voice closest to finishing.
    var best: ?*Voice = null;
    var best_k: f32 = -1.0;
    for (&self.voices) |*v| {
        const k = v.t / @max(v.duration, 1e-6);
        if (k > best_k) {
            best_k = k;
            best = v;
        }
    }
    return best;
}
