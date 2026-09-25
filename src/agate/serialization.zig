//! Compact binary snapshot of a scene (".babylon"-style, but binary).
//!
//! Facade: the implementation lives in serialization/ (free functions + thin
//! forwarders; Zig 0.16 has no usingnamespace). Public entry points keep
//! identical signatures/behavior:
//!   format.zig - header/version constants, caps, binary primitives (Writer,
//!     Reader, options-field codec, postprocess persistence contract).
//!   props.zig  - snapshot types (SceneState and entries), custom game
//!     properties, entity IDs / hierarchy + material-kind helpers.
//!   writer.zig - scene -> bytes (capture, serializeAlloc, saveFile, async save).
//!   reader.zig - bytes -> scene (deserializeAlloc, restore, loadFile, async load).
//!
//! Only what is listed in format/writer is captured; everything else is
//! intentionally left out (see NOT SERIALIZED). All strings/slices in
//! SceneState are owned (allocator.dupe) and released by SceneState.deinit.
//!
//! NOT SERIALIZED (by design): geometry (vertices/indices), textures and
//! cube maps (skybox/IBL contents), skeletons/animations, morph targets,
//! physics bodies, particles, UI, instanced meshes, bone
//! attachments, follow-camera target link (target_position is kept),
//! material sharing topology (values are per-mesh), SSAO config, shadow
//! tuning beyond softness, pipeline/GPU handles. Meshes are matched by id
//! or name on restore; geometry is referenced, never stored.

const std = @import("std");
const jobs = @import("jobs.zig");
const math = @import("math");
const Vec3 = math.Vec3;
const format_mod = @import("serialization/format.zig");
const props_mod = @import("serialization/props.zig");
const writer_mod = @import("serialization/writer.zig");
const reader_mod = @import("serialization/reader.zig");
const PostProcessOptions = @import("postprocess.zig").PostProcessOptions;
const CameraModule = @import("camera.zig");
const Camera = CameraModule.Camera;
const TargetCamera = CameraModule.TargetCamera;
const FlyCamera = CameraModule.FlyCamera;
const MaterialModule = @import("material.zig");
const StandardMaterial = MaterialModule.StandardMaterial;

// Format re-exports (header/version constants, caps, decode errors).
pub const MAGIC = format_mod.MAGIC;
pub const VERSION = format_mod.VERSION;
pub const MAX_ENTRIES = format_mod.MAX_ENTRIES;
pub const MAX_STRING_BYTES = format_mod.MAX_STRING_BYTES;
pub const MAX_FILE_BYTES = format_mod.MAX_FILE_BYTES;
pub const DecodeError = format_mod.DecodeError;

// Snapshot-type re-exports (props.zig owns them).
pub const StandardEntry = props_mod.StandardEntry;
pub const PbrEntry = props_mod.PbrEntry;
pub const MaterialEntry = props_mod.MaterialEntry;
pub const MeshEntry = props_mod.MeshEntry;
pub const HemiEntry = props_mod.HemiEntry;
pub const DirectionalEntry = props_mod.DirectionalEntry;
pub const PointEntry = props_mod.PointEntry;
pub const SpotEntry = props_mod.SpotEntry;
pub const ArcRotateEntry = props_mod.ArcRotateEntry;
pub const FreeEntry = props_mod.FreeEntry;
pub const FollowEntry = props_mod.FollowEntry;
pub const TargetEntry = props_mod.TargetEntry;
pub const FlyEntry = props_mod.FlyEntry;
pub const CameraEntry = props_mod.CameraEntry;
pub const RenderEntry = props_mod.RenderEntry;
pub const GameProperty = props_mod.GameProperty;
pub const SceneState = props_mod.SceneState;

// Scene -> bytes (writer.zig).
pub const capture = writer_mod.capture;
pub const serializeAlloc = writer_mod.serializeAlloc;
pub const saveFile = writer_mod.saveFile;
pub const AsyncSaveTask = writer_mod.AsyncSaveTask;
pub const saveFileAsync = writer_mod.saveFileAsync;

// Bytes -> scene (reader.zig).
pub const restore = reader_mod.restore;
pub const deserializeAlloc = reader_mod.deserializeAlloc;
pub const loadFile = reader_mod.loadFile;
pub const AsyncLoadTask = reader_mod.AsyncLoadTask;
pub const loadFileAsync = reader_mod.loadFileAsync;

// Private test-only aliases: Writer/writePostProcess lived in this file
// before the split; tests below keep their original bodies via these.
const Writer = format_mod.Writer;
const writePostProcess = format_mod.writePostProcess;

// ---------------------------------------------------------------------------
// GPU-free tests
// ---------------------------------------------------------------------------

// CPU-only Scene/Mesh fixtures live in testing.zig (shared with
// loader/materials.zig); the GPU-backed fields left `undefined` there are
// never dereferenced by capture()/restore().
const testScene = @import("testing.zig").testScene;
const testMesh = @import("testing.zig").testMesh;
