"""Read-only, bounded kinematic checks; these are not physical action trials."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import DOWN, body_name, colliding, joint_name, solve_ik


def collision_details(world, data, pair):
    """Describe the actual collision selected by the controller's safety rules."""
    model = world.model
    for contact in data.contact:
        names = tuple(
            model.geom(int(g)).name or model.body(int(model.geom_bodyid[g])).name
            for g in contact.geom
        )
        if names == tuple(pair) and contact.dist <= -0.0008:
            return {
                "geometry_pair": list(pair),
                "body_pair": [
                    model.body(int(model.geom_bodyid[g])).name for g in contact.geom
                ],
                "position": contact.pos.tolist(),
                "penetration": max(0.0, -float(contact.dist)),
            }
    return {"geometry_pair": list(pair)}


def path_obstruction(world, qs, held_id=None):
    """First blocked joint sample, using the controller's existing tolerances.

    Held-object transforms are only used internally for a controller's already
    declared grasp. Public preflights cannot invent a grasp or suppress contacts.
    """
    scratch = mujoco.MjData(world.model)
    mujoco.mj_copyData(scratch, world.model, world.data)
    hand = world.model.site("grasp").id
    rel_pos = rel_rot = None
    if held_id:
        hand_rot = world.data.site_xmat[hand].reshape(3, 3)
        body = world.data.body(body_name(world, held_id))
        rel_pos = hand_rot.T @ (body.xpos - world.data.site_xpos[hand])
        rel_rot = hand_rot.T @ body.xmat.reshape(3, 3)
    start = world.arm_q
    for index, target in enumerate(qs):
        samples = max(2, int(np.max(np.abs(target - start)) / 0.012) + 1)
        for fraction in np.linspace(0, 1, samples):
            scratch.qpos[:7] = start + fraction * (target - start)
            mujoco.mj_fwdPosition(world.model, scratch)
            if held_id:
                rot = scratch.site_xmat[hand].reshape(3, 3)
                joint = scratch.joint(joint_name(world, held_id))
                joint.qpos[:3] = scratch.site_xpos[hand] + rot @ rel_pos
                quat = Rotation.from_matrix(rot @ rel_rot).as_quat()
                joint.qpos[3:] = quat[[3, 0, 1, 2]]
                mujoco.mj_fwdPosition(world.model, scratch)
            pair = colliding(world, scratch, held_id)
            if pair:
                return {
                    "failed_waypoint_index": index,
                    "segment_fraction": float(fraction),
                    "collision": collision_details(world, scratch, pair),
                }
        start = target
    return None


def check_approaches(world, arguments):
    from .action_lab import PhysicsSnapshot
    from .tools import CheckApproachesArgs

    arguments = CheckApproachesArgs.model_validate(
        arguments.model_dump() if isinstance(arguments, CheckApproachesArgs) else arguments
    )
    if getattr(getattr(world, "spec", None), "robot", None) != "panda":
        return {
            "ok": False,
            "scene_revision": world.revision,
            "error_code": "unsupported_robot",
            "detail": "Approach checks require a general world with a Panda.",
        }
    snapshot = PhysicsSnapshot.capture(world)
    candidates = []
    for candidate in arguments.candidates:
        copied = snapshot.clone()
        q = copied.arm_q
        qs = []
        result = {
            "name": candidate.name,
            "reachable": True,
            "clear": False,
            "checked_waypoints": 0,
        }
        for index, waypoint in enumerate(candidate.waypoints):
            rotation = (
                DOWN if waypoint.rotation is None
                else Rotation.from_euler("XYZ", waypoint.rotation).as_matrix()
            )
            q = solve_ik(copied, waypoint.position, q, rotation)
            if q is None:
                result.update(
                    reachable=False,
                    error_code="unreachable",
                    detail=(
                        "No IK solution found with the bounded solver; "
                        "this does not prove geometric impossibility."
                    ),
                    failed_waypoint_index=index,
                    requested_pose=waypoint.model_dump(mode="json"),
                )
                break
            qs.append(q)
        else:
            obstruction = path_obstruction(copied, qs)
            if obstruction:
                index = obstruction["failed_waypoint_index"]
                result.update(
                    obstruction,
                    error_code="no_path",
                    checked_waypoints=index,
                    detail="The sampled joint interpolation is obstructed.",
                    requested_pose=candidate.waypoints[index].model_dump(mode="json"),
                )
            else:
                result.update(clear=True, checked_waypoints=len(qs))
        candidates.append(result)
    return {
        "ok": True,
        "scene_revision": world.revision,
        "payload": {
            "coordinate_frame": "world",
            "physical_trial": False,
            "scope": (
                "Bounded IK and sampled joint interpolation at the current gripper "
                "configuration, with static objects. Omitted rotation is downward. "
                "No grasp, contact stroke, dynamics, settling, or full-action "
                "feasibility is verified."
            ),
            "candidates": candidates,
        },
    }
