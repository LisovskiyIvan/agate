pub const navmesh = @import("ai/navmesh.zig");
pub const NavNode = navmesh.NavNode;
pub const NavMesh = navmesh.NavMesh;

pub const funnel = @import("ai/funnel.zig");
pub const Portal = funnel.Portal;
pub const triArea2D = funnel.triArea2D;
pub const stringPull = funnel.stringPull;

pub const pathfinding = @import("ai/pathfinding.zig");
pub const Pathfinding = pathfinding.Pathfinding;

pub const agent = @import("ai/agent.zig");
pub const NavAgent = agent.NavAgent;

pub const crowd = @import("ai/crowd.zig");
pub const Crowd = crowd.Crowd;
pub const CrowdAgent = crowd.CrowdAgent;
pub const CrowdAgentParams = crowd.CrowdAgentParams;

test {
    _ = @import("ai/tests.zig");
}
