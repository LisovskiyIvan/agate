//! Facade for the audio modules. The `audio.zig` engine was split into
//! focused leaves under `audio/` following the repo pattern (free functions
//! + thin forwarders; Zig 0.16 has no usingnamespace; see `camera.zig`,
//! `particles.zig`, `profiler.zig`, `scene/render_queue/build.zig`):
//!
//! - `audio/types.zig` — shared vocabulary: `BusId` (+ `AudioBus`,
//!   capacity constants, `invalid_bus`), `AudioConfig`/`AudioEngineConfig`,
//!   attenuation vocabulary (`AttenuationModel`, `BusAttenuation`,
//!   `BusConfig`), the `VoiceKind`/`Voice` slot layout, and the engine
//!   sizing limits (`max_voices`, …, `max_streams`) re-exported by the
//!   engine as nested constants. Imports `dsp` + `occlusion` + `clip` only.
//! - `audio/engine.zig` — owns the `AudioEngine` type: fields, the trivial
//!   lifecycle (`init`/`configure`/capacity, `start`/`shutdown`, listener,
//!   master/mute, `stopAll`) and thin forwarders into the siblings below,
//!   so every call site keeps working unchanged. Imports every leaf.
//! - `audio/commands.zig` — voice trigger queue (`Command`, lock-free SPSC
//!   ring, slot stealing). Imports `types` + `clip` only.
//! - `audio/playback.zig` — `play`/`playClip`, one-shot helpers
//!   (impact/explosion/blip), `PlayOptions`/`ClipPlayOptions`,
//!   `SpatializeResult`/`spatializeWith`. Imports `types` + `clip` +
//!   `occlusion` + the `commands` + `buses` siblings.
//! - `audio/buses.zig` — bus registry and routing DAG (lifecycle, identity,
//!   hierarchy, attenuation/doppler, volume/mute cascade, `stopBus`).
//!   Imports `types` + the `effects` sibling.
//! - `audio/effects.zig` — bus DSP filters, pooled reverb units + presets,
//!   occlusion state and raycast wiring. Imports `types` + `dsp` +
//!   `occlusion`.
//! - `audio/mixer.zig` — `renderFrames` + the synth kernels (`renderNoise`,
//!   `renderTone`, `renderSample`) and the private `ActiveVoice` state.
//!   The real-time callback path stays 0-allocation and lock-free.
//!   Imports `types` + `dsp` + the `commands` sibling.
//! - `audio/streams.zig` — stream slots, the sound/music playback API,
//!   `MusicPlayOptions`, `playMusicClip`. Imports `stream` + `clip` +
//!   `types` + the `buses` + `playback` siblings.
//! - Pre-existing leaves, untouched apart from the import repoint noted
//!   below: `audio/clip.zig` (`AudioClip` + WAV decode), `audio/decode.zig`
//!   (MP3/Ogg decoders), `audio/dsp.zig` (biquad + reverb),
//!   `audio/occlusion.zig` (raycast occlusion), `audio/stream.zig`
//!   (`AudioStream`), `audio/tests.zig` (all audio tests, unchanged).
//!
//! Everything that was public before the split is re-exported here
//! unchanged; consumers (`root.zig`, `scene.zig`, `physics/world.zig`,
//! `audio/tests.zig`) see the same API as when everything lived in this
//! file.
//!
//! Documented anti-cycle rule: leaves must never import this facade —
//! importing it back would make the re-exports depend on their own
//! consumers. Method bodies take the engine as `anytype` (same discipline
//! as `profiler/*` taking a generic profiler), so library code has no
//! leaf-to-owner edge at all; leaf-to-leaf calls use direct sibling
//! imports (same discipline as `particles/cpu.zig` reaching `subemitters`).
//! Cross-leaf helpers (`commands.pushCommand`/`drainCommands`,
//! `buses.isBusSpatial`/`getBusAttenuation`/…, `effects.setBusFilter`/…,
//! `playback.playClip`, …) are `pub` in their home module for the sibling
//! that needs them but are deliberately NOT re-exported here, so the public
//! surface is identical to the pre-split file.
//!
//! Honest structural notes (see commit message):
//! - The pre-existing facade back-imports (`stream.zig` and `occlusion.zig`
//!   both did `@import("../audio.zig").BusId`) are repointed at
//!   `audio/types.zig`, so the module DAG is now genuinely acyclic.
//! - Private owner methods cannot be called through `anytype` (verified:
//!   `'x' is not marked 'pub'`), so the moved bodies call the owning
//!   sibling directly (`commands.pushCommand(self, …)` instead of
//!   `self.pushCommand(…)`); behavior is identical, only the call path
//!   changed.
//! - No `Phase(@TypeOf(self))` namespace was needed: unlike
//!   `particles/cpu.zig`, no generic payload struct or range function
//!   crosses the leaf boundary here.
//! - The dead private `spatialize` helper (no callers before the split) is
//!   preserved verbatim as a private function in `playback.zig`.

const types = @import("audio/types.zig");
const engine = @import("audio/engine.zig");

const clip_mod = @import("audio/clip.zig");
pub const AudioClip = clip_mod.AudioClip;

const dsp = @import("audio/dsp.zig");
pub const BiquadFilterType = dsp.BiquadFilterType;
pub const BusFilterConfig = dsp.BusFilterConfig;
pub const BusReverbConfig = dsp.BusReverbConfig;
pub const BiquadFilter = dsp.BiquadFilter;
pub const ReverbProcessor = dsp.ReverbProcessor;

const occlusion_mod = @import("audio/occlusion.zig");
pub const AudioOcclusionConfig = occlusion_mod.AudioOcclusionConfig;
pub const RaycastFn = occlusion_mod.RaycastFn;
pub const evaluateRaycastOcclusion = occlusion_mod.evaluateRaycastOcclusion;
pub const AudioOcclusionTracker = occlusion_mod.AudioOcclusionTracker;
pub const AudioEmitter = occlusion_mod.AudioEmitter;

const stream_mod = @import("audio/stream.zig");
pub const StreamFormat = stream_mod.StreamFormat;
pub const StreamState = stream_mod.StreamState;
pub const StreamError = stream_mod.StreamError;
pub const StreamOptions = stream_mod.StreamOptions;
pub const PlaySoundOptions = stream_mod.PlaySoundOptions;
pub const AudioStream = stream_mod.AudioStream;

// Shared vocabulary (lives in audio/types.zig).
pub const BusId = types.BusId;
pub const AudioBus = types.AudioBus;
pub const default_max_buses = types.default_max_buses;
pub const max_bus_capacity = types.max_bus_capacity;
pub const max_buses = types.max_buses;
pub const invalid_bus = types.invalid_bus;
pub const AudioConfig = types.AudioConfig;
pub const AudioEngineConfig = types.AudioEngineConfig;
pub const AttenuationModel = types.AttenuationModel;
pub const BusAttenuation = types.BusAttenuation;
pub const BusConfig = types.BusConfig;
pub const VoiceKind = types.VoiceKind;
pub const Voice = types.Voice;

// Audio engine owner (lives in audio/engine.zig).
pub const AudioEngine = engine.AudioEngine;

test {
    _ = @import("audio/tests.zig");
}
