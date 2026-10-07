//! Facade for the physics modules. The `PhysicsWorld` core (type, fields,
//! methods) lives in `physics/world.zig` so the `physics/*` submodules can
//! import the concrete world type without importing this facade — importing
//! it back would make the type depend on its own consumers. Everything that
//! was public before the split is re-exported here unchanged; consumers
//! (root.zig, sandbox) see the same API as when the world lived in this file.
const types = @import("physics/types.zig");
const body_mod = @import("physics/body.zig");
const world_mod = @import("physics/world.zig");

// Re-exports of the extracted leaf modules (public API unchanged).
pub const PickingInfo = types.PickingInfo;
pub const ColliderType = types.ColliderType;
pub const HeightFieldOptions = types.HeightFieldOptions;
pub const PhysicsRayHit = body_mod.PhysicsRayHit;
pub const CollisionFilter = types.CollisionFilter;
pub const BodyOptions = types.BodyOptions;
pub const SensorEvent = body_mod.SensorEvent;
pub const ContactEvent = body_mod.ContactEvent;
pub const ContactHitEvent = body_mod.ContactHitEvent;
pub const ChildShape = body_mod.ChildShape;
pub const ChildShapeOptions = types.ChildShapeOptions;
pub const JointId = types.JointId;
pub const DistanceJointOptions = types.DistanceJointOptions;
pub const SphericalJointOptions = types.SphericalJointOptions;
pub const RevoluteJointOptions = types.RevoluteJointOptions;
pub const WheelJointOptions = types.WheelJointOptions;
pub const MotorJointOptions = types.MotorJointOptions;
pub const WeldJointOptions = types.WeldJointOptions;
pub const ParallelJointOptions = types.ParallelJointOptions;
pub const PrismaticJointOptions = types.PrismaticJointOptions;
pub const RigidBody = body_mod.RigidBody;
pub const DebugLine = types.DebugLine;

// The world type itself, moved verbatim to `physics/world.zig`.
pub const PhysicsWorld = world_mod.PhysicsWorld;
pub const PhysicsProfile = world_mod.PhysicsProfile;
pub const PhysicsCounters = world_mod.PhysicsCounters;

const character_mod = @import("physics/character.zig");
pub const CharacterController = character_mod.CharacterController;

const rope_mod = @import("physics/rope.zig");
pub const Rope = rope_mod.Rope;
pub const RopeOptions = rope_mod.RopeOptions;

test {
    _ = @import("physics/tests.zig");
}
