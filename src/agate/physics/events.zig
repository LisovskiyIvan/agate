//! Sensor/contact event collection as free functions.
//! Extracted from `PhysicsWorld` methods in `physics.zig`; behavior unchanged.
//! Each function takes `world: anytype` (concretely `*PhysicsWorld`) so this
//! module never imports `physics.zig` (no import cycle).
const c = @import("../c.zig").c;
const convert = @import("convert.zig");
const queries = @import("queries.zig");
const fromB3Pos = convert.fromB3Pos;
const fromB3Vec = convert.fromB3Vec;

/// Events are collected inside step() and cleared at the start of each
/// call, so they describe the most recent frame only.
pub fn clearEvents(world: anytype) void {
    world.sensor_events.clearRetainingCapacity();
    world.contact_events.clearRetainingCapacity();
    world.contact_hit_events.clearRetainingCapacity();
}

pub fn drainEvents(world: anytype) void {
    const sensor = c.b3World_GetSensorEvents(world.world_id);
    var i: i32 = 0;
    while (i < sensor.beginCount) : (i += 1) {
        const ev = sensor.beginEvents[@intCast(i)];
        world.sensor_events.append(world.allocator, .{
            .sensor = queries.findBodyByShape(world, ev.sensorShapeId),
            .visitor = queries.findBodyByShape(world, ev.visitorShapeId),
            .began = true,
        }) catch {};
    }
    i = 0;
    while (i < sensor.endCount) : (i += 1) {
        const ev = sensor.endEvents[@intCast(i)];
        world.sensor_events.append(world.allocator, .{
            .sensor = queries.findBodyByShape(world, ev.sensorShapeId),
            .visitor = queries.findBodyByShape(world, ev.visitorShapeId),
            .began = false,
        }) catch {};
    }

    const contacts = c.b3World_GetContactEvents(world.world_id);
    i = 0;
    while (i < contacts.beginCount) : (i += 1) {
        const ev = contacts.beginEvents[@intCast(i)];
        world.contact_events.append(world.allocator, .{
            .a = queries.findBodyByShape(world, ev.shapeIdA),
            .b = queries.findBodyByShape(world, ev.shapeIdB),
            .began = true,
        }) catch {};
    }
    i = 0;
    while (i < contacts.endCount) : (i += 1) {
        const ev = contacts.endEvents[@intCast(i)];
        world.contact_events.append(world.allocator, .{
            .a = queries.findBodyByShape(world, ev.shapeIdA),
            .b = queries.findBodyByShape(world, ev.shapeIdB),
            .began = false,
        }) catch {};
    }
    i = 0;
    while (i < contacts.hitCount) : (i += 1) {
        const ev = contacts.hitEvents[@intCast(i)];
        world.contact_hit_events.append(world.allocator, .{
            .a = queries.findBodyByShape(world, ev.shapeIdA),
            .b = queries.findBodyByShape(world, ev.shapeIdB),
            .point = fromB3Pos(ev.point),
            .normal = fromB3Vec(ev.normal),
            .approach_speed = ev.approachSpeed,
        }) catch {};
    }
}
