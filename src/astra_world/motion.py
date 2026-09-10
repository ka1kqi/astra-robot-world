"""Kinematics on scratch data; actuator-driven motion on the live world."""

from threading import Event
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .scene import HOME

DOWN = np.diag([1.0, -1.0, -1.0])


def body_name(world, entity_id):
    return world.body_name(entity_id) if hasattr(world, "body_name") else entity_id


def joint_name(world, entity_id):
    return (
        world.joint_name(entity_id)
        if hasattr(world, "joint_name")
        else entity_id + "_joint"
    )


class MotionError(RuntimeError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def solve_ik(world, position, seed_q, rotation=DOWN):
    model = world.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = world.data.qpos
    target = np.asarray(position)
    site = model.site("grasp").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    limits = model.jnt_range[:7]

    def solve_from(seed):
        q = np.asarray(seed).copy()
        for _ in range(160):
            scratch.qpos[:7] = q
            mujoco.mj_fwdPosition(model, scratch)
            delta = target - scratch.site_xpos[site]
            orient = Rotation.from_matrix(
                rotation @ scratch.site_xmat[site].reshape(3, 3).T
            ).as_rotvec()
            if np.linalg.norm(delta) < 0.0004 and np.linalg.norm(orient) < 0.004:
                return q
            mujoco.mj_jacSite(model, scratch, jacp, jacr, site)
            jac = np.vstack([jacp[:, :7], jacr[:, :7]])
            error = np.r_[delta, orient]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.0001 * np.eye(6), error)
            q = np.clip(
                q + dq * min(1.0, 0.15 / max(np.max(np.abs(dq)), 0.001)),
                limits[:, 0] + 0.001,
                limits[:, 1] - 0.001,
            )
        return None

    # Keep the original solution branch for ordinary incremental motion.
    result = solve_from(seed_q)
    if result is not None:
        return result
    # A stalled local search does not establish geometric unreachability. Try a
    # small deterministic set of shoulder/elbow postures, still on scratch data.
    perturbation = np.array([1.2, 0, 1.0, 0, 0, 0, 0])
    for seed in (limits.mean(axis=1), HOME, HOME + perturbation, HOME - perturbation):
        seed = np.clip(seed, limits[:, 0] + 0.001, limits[:, 1] - 0.001)
        if np.array_equal(seed, seed_q):
            continue
        result = solve_from(seed)
        if result is not None:
            return result
    return None


class Controller:
    def __init__(self, world, cancel=None, tick=None, status=None):
        self.world = world
        self.held_id = None
        self.contact_ids = ()
        self.cancel = cancel or Event()
        self.tick = tick or (lambda: None)
        self.status = status or (lambda text: None)

    def step(self, seconds):
        w = self.world
        for _ in range(max(1, int(seconds / w.model.opt.timestep))):
            if self.cancel.is_set():
                w.data.ctrl[:7] = w.arm_q
                raise MotionError(
                    "cancelled", "Action stopped; holding the current arm position."
                )
            # Gravity compensation supplements the Panda position servos.
            w.data.qfrc_applied[:7] = w.data.qfrc_bias[:7]
            mujoco.mj_step(w.model, w.data)
            if not np.isfinite(w.data.qpos).all():
                raise MotionError("unstable_physics", "The simulation became unstable.")
            contact = colliding(w, w.data, self.held_id, self.contact_ids)
            if contact:
                w.data.ctrl[:7] = w.arm_q
                raise MotionError(
                    "unexpected_contact",
                    f"Unexpected contact between {contact[0]} and {contact[1]}.",
                )
            self.tick()

    def move_q(self, target, seconds=1.2):
        w = self.world
        start = w.arm_q
        duration = max(seconds, np.max(np.abs(target - start)) / 0.65)
        n = max(1, int(duration / 0.01))
        for i in range(1, n + 1):
            t = i / n
            w.data.ctrl[:7] = start + (target - start) * (t * t * (3 - 2 * t))
            self.step(0.01)
        self.step(0.18)
        if np.max(np.abs(w.arm_q - target)) > 0.07:
            raise MotionError(
                "tracking_error", "Arm could not reach the commanded pose."
            )

    def move(self, pos, seconds=1.2, rotation=DOWN):
        q = solve_ik(self.world, pos, self.world.arm_q, rotation)
        if q is None:
            raise MotionError(
                "unreachable",
                f"Target {np.round(pos, 3).tolist()} is outside the usable workspace.",
            )
        if not path_is_clear(self.world, [q], self.held_id):
            raise MotionError(
                "no_path", "The approach or placement path is obstructed."
            )
        self.move_q(q, seconds)

    def gripper(self, opened):
        self.world.data.ctrl[7] = 255 if opened else 0
        self.step(0.8)


    def release(self):
        """Opening releases the load; allow only its residual finger contacts.

        Once opening starts, ordinary object/support impacts belong to the free
        dynamics. The palm, arm, and unrelated finger contacts remain guarded.
        """
        released = self.held_id
        self.held_id = None
        self.contact_ids = (released,) if released is not None else ()
        try:
            self.gripper(True)
        finally:
            self.contact_ids = ()


def colliding(world, scratch, held_id=None, contact_ids=()):
    """Reject arm/self/environment and held-object contacts, except finger grasp."""
    m = world.model
    robot_bodies = {
        m.body(n).id
        for n in [
            "link0",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
            "left_finger",
            "right_finger",
        ]
    }
    fingers = {m.body("left_finger").id, m.body("right_finger").id}
    held_body = m.body(body_name(world, held_id)).id if held_id else -1
    contact_bodies = {m.body(body_name(world, name)).id for name in contact_ids}
    base = m.body("link0").id
    for contact in scratch.contact:
        if contact.dist > -0.0008:
            continue
        a, b = (int(m.geom_bodyid[g]) for g in contact.geom)
        if a not in robot_bodies and b not in robot_bodies and held_body not in (a, b):
            continue
        if (a == held_body and b in fingers) or (b == held_body and a in fingers):
            continue
        if (a in contact_bodies and b in fingers) or (b in contact_bodies and a in fingers):
            continue
        if base in (a, b) and 0 in (a, b):
            continue
        # The held block may still touch its support during initial lift/lowering.
        if held_body in (a, b) and not (a in robot_bodies or b in robot_bodies):
            other_geom = int(contact.geom[1] if a == held_body else contact.geom[0])
            name = m.geom(other_geom).name or ""
            if (
                hasattr(world, "observe")
                and contact.dist > -0.003
                and abs(contact.frame[2]) > 0.8
                and contact.pos[2] < scratch.body(held_body).xpos[2] - 0.005
            ):
                continue
            if name == "table" or name.endswith("_floor"):
                continue
        return (
            m.geom(int(contact.geom[0])).name or m.body(a).name,
            m.geom(int(contact.geom[1])).name or m.body(b).name,
        )
    return None


def path_is_clear(world, qs, held_id=None):
    scratch = mujoco.MjData(world.model)
    scratch.qpos[:] = world.data.qpos
    hand = world.model.site("grasp").id
    rel_pos = rel_rot = None
    if held_id:
        hand_rot = world.data.site_xmat[hand].reshape(3, 3)
        rel_pos = hand_rot.T @ (
            world.data.body(body_name(world, held_id)).xpos - world.data.site_xpos[hand]
        )
        rel_rot = hand_rot.T @ world.data.body(body_name(world, held_id)).xmat.reshape(
            3, 3
        )
    start = world.arm_q
    for target in qs:
        samples = max(2, int(np.max(np.abs(target - start)) / 0.012) + 1)
        for t in np.linspace(0, 1, samples):
            scratch.qpos[:7] = start + t * (target - start)
            mujoco.mj_fwdPosition(world.model, scratch)
            if held_id:
                rot = scratch.site_xmat[hand].reshape(3, 3)
                joint = scratch.joint(joint_name(world, held_id))
                joint.qpos[:3] = scratch.site_xpos[hand] + rot @ rel_pos
                quat_xyzw = Rotation.from_matrix(rot @ rel_rot).as_quat()
                joint.qpos[3:] = quat_xyzw[[3, 0, 1, 2]]
                mujoco.mj_fwdPosition(world.model, scratch)
            if colliding(world, scratch, held_id):
                return False
        start = target
    return True


def plan_transport(world, target, held_id=None, rotation=DOWN):
    """Try bounded Cartesian corridors; every joint segment is collision checked."""
    start = world.data.site("grasp").xpos.copy()
    target = np.asarray(target)
    candidates = [("direct", [target])]
    if world.obstacles or hasattr(world, "observe"):
        # Side corridors stay below the barrier top and visibly go around it.
        for x, label in [(0.66, "outer detour"), (0.23, "inner detour")]:
            z = max(start[2], target[2], 0.24)
            candidates.append(
                (
                    label,
                    [np.array([x, start[1], z]), np.array([x, target[1], z]), target],
                )
            )
    for label, positions in candidates:
        q = world.arm_q
        qs = []
        for pos in positions:
            q = solve_ik(world, pos, q, rotation)
            if q is None:
                break
            qs.append(q)
        else:
            if path_is_clear(world, qs, held_id):
                return {
                    "scene_revision": world.revision,
                    "label": label,
                    "qs": qs,
                    "points": [start.tolist()] + [p.tolist() for p in positions],
                }
    return None


def execute_transport(ctl, target, held_id=None, rotation=DOWN):
    w = ctl.world
    path = plan_transport(w, target, held_id, rotation)
    if path is None:
        raise MotionError(
            "no_path", "No collision-free route was found in the supported corridors."
        )
    if path["scene_revision"] != w.revision:
        raise MotionError(
            "stale_path", "The scene changed; the trajectory must be replanned."
        )
    ctl.status("Taking the " + path["label"] + " route.")
    w.paths.append({k: v for k, v in path.items() if k != "qs"})
    w.paths[:] = w.paths[-12:]
    for q in path["qs"]:
        ctl.move_q(q, 1.6)
        if (
            held_id
            and np.linalg.norm(
                w.data.body(body_name(w, held_id)).xpos - w.data.site("grasp").xpos
            )
            > 0.065
        ):
            raise MotionError("grasp_lost", "The block slipped from the gripper.")
    return path["label"]
