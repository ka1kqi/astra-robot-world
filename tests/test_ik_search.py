"""Bounded IK retries recover reachable poses after local convergence failures."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from astra_world.motion import colliding, solve_ik
from astra_world.world_builder import build_world
from astra_world.world_spec import WorldSpec

# Collision-free Panda posture whose FK pose defeats the original HOME-only solve.
REACHABLE_Q = np.array([
    -1.1235443883478355, -.7371783623423563, 2.2815331819204743,
    -1.6110275940349907, -1.3458355458021705, 2.1217290493602188,
    -1.7028467035641488,
])


def world():
    return build_world(WorldSpec(name='ik_search', robot='panda'))


def fk(w, q):
    scratch = mujoco.MjData(w.model)
    scratch.qpos[:] = w.data.qpos
    scratch.qpos[:7] = q
    mujoco.mj_fwdPosition(w.model, scratch)
    return scratch


def state(w):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(w.model, kind))
    mujoco.mj_getState(w.model, w.data, result, kind)
    return result


def test_retry_recovers_known_reachable_pose_and_preserves_entire_world_state():
    w = world()
    target = fk(w, REACHABLE_Q)
    assert colliding(w, target) is None
    position = target.site('grasp').xpos.copy()
    rotation = target.site('grasp').xmat.reshape(3, 3).copy()
    # Include solver, actuator, external-force, and timing state, not just qpos.
    w.data.time = 9.25
    w.data.qvel[:] = .01
    w.data.qacc_warmstart[:] = .02
    w.data.qfrc_applied[:] = .03
    w.data.xfrc_applied[:] = .04
    original = state(w)
    seed = w.arm_q.copy()
    solution = solve_ik(w, position, seed, rotation)
    assert solution is not None, 'Known collision-free FK target must be recoverable'
    observed = fk(w, solution).site('grasp')
    assert np.linalg.norm(observed.xpos - position) < .0004
    residual = Rotation.from_matrix(rotation @ observed.xmat.reshape(3, 3).T)
    assert residual.magnitude() < .004
    limits = w.model.jnt_range[:7]
    assert np.all(solution >= limits[:, 0] + .001)
    assert np.all(solution <= limits[:, 1] - .001)
    np.testing.assert_array_equal(seed, w.arm_q)
    np.testing.assert_array_equal(state(w), original)
    np.testing.assert_array_equal(solve_ik(w, position, seed, rotation), solution)


def test_successful_original_seed_is_returned_without_switching_posture():
    w = world()
    target = fk(w, REACHABLE_Q)
    seed = REACHABLE_Q.copy()
    result = solve_ik(w, target.site('grasp').xpos, seed,
                      target.site('grasp').xmat.reshape(3, 3))
    np.testing.assert_array_equal(result, seed)
    np.testing.assert_array_equal(seed, REACHABLE_Q)


def test_out_of_reach_remains_failure_with_bounded_work_and_no_mutation(monkeypatch):
    w = world()
    before = state(w)
    calls = 0
    real_forward = mujoco.mj_fwdPosition
    def forward(model, data):
        nonlocal calls
        calls += 1
        return real_forward(model, data)
    monkeypatch.setattr(mujoco, 'mj_fwdPosition', forward)
    assert solve_ik(w, (3, 3, 3), w.arm_q) is None
    assert calls <= 800
    np.testing.assert_array_equal(state(w), before)
