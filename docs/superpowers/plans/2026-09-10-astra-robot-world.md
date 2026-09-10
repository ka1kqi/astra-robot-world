# Astra Robot World Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. This plan has been implemented with the bounded decisions documented in the implementation record below. The original task details are retained for context; final verification status is recorded at the end.

**Goal:** Run the user's four-prompt sorting-station demo on their Mac, including physical manipulation and obstacle-aware retry.

**Architecture:** A Python process owns MuJoCo and its native viewer; a local chat service queues typed commands from Astra. A curated scene builder and deterministic robot skills provide validated actions, with simulator state returned after execution.

**Tech Stack:** Native MuJoCo, Menagerie Panda, Python, NumPy/SciPy, Pydantic, pytest; FastAPI/Uvicorn and a plain HTML/CSS/JavaScript chat page at the interface milestone. Add the model provider's SDK only after verifying the available Astra endpoint and model identifier.

**Spec:** `docs/superpowers/specs/2026-09-10-astra-robot-world-design.md`

## Global constraints

- Target: MacBook Pro, Apple M4 Pro, 24 GB unified memory, 14-core CPU.
- Local simulation/rendering; model inference may be remote.
- One Panda arm; six default blocks, two bins, one obstacle; maximum 12 blocks.
- Meters, radians, Z-up; stable object and joint IDs; fixed camera defines left/right labels.
- Physical grasping and actuator-driven motion; no teleporting or attaching blocks to fake successful execution.
- Retry explicitly restores pre-sort movable state while preserving the new obstacle.
- One simulation owner and one mutating action at a time.
- Final verification uses live Astra calls; offline fixtures only validate development plumbing.

## File map

Environment setup is complete in `.python-version`, `pyproject.toml`, `uv.lock`, `scripts/fetch_assets.py`, and `scripts/check_mujoco.py`. Do not replace successful installation work. The existing project is dependency-only: when adding `src/astra_world`, configure a package build backend and verify editable installation so `uv run mjpython -m astra_world` resolves the package.

Verified locally on 2026-09-10: Python 3.12.13 arm64, MuJoCo 3.13.0, NumPy 2.5.3, SciPy 1.18.1, Pydantic 2.13.5, pytest 9.1.1. Panda assets live at `assets/menagerie/franka_emika_panda`, pinned to Menagerie commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`. The robot loaded with 9 position coordinates and 8 actuators; 1,000 physics steps and 640×480 offscreen rendering passed. The installation subagent also verified the native viewer. `uv pip check` passed. Reproduce with the commands in `README.md`.

| File | Responsibility |
|---|---|
| `src/astra_world/contracts.py` | Typed scene requests, observations, goals, results, snapshots |
| `src/astra_world/assets.py` | Curated asset IDs and model paths |
| `src/astra_world/scene.py` | Compose/validate scenes and preserve state on recompilation |
| `src/astra_world/simulation.py` | Physics stepping, native viewer, queue, cancellation |
| `src/astra_world/motion.py` | IK, candidate routes, swept collision validation |
| `src/astra_world/manipulation.py` | Grasp/place/sort state machine and success checks |
| `src/astra_world/session.py` | Last goal, pre-trial snapshot, retry semantics |
| `src/astra_world/tools.py` | Typed command dispatch and tool results |
| `src/astra_world/astra.py` | Provider adapter and bounded tool-calling loop |
| `src/astra_world/server.py` | Local HTTP interface and status events |
| `src/astra_world/__main__.py` | Launch worker/viewer and local service |
| `web/index.html`, `web/app.js`, `web/styles.css` | Chat/status interface |
| `tests/` | Geometry, state, physical integration, and orchestration checks |

## Task 1: Load and construct a valid sorting station

**Files:** `contracts.py`, `assets.py`, `scene.py`, `simulation.py`, `__main__.py`, `tests/test_scene.py`.

**Interfaces:** `SceneSpec(seed: int, block_colors: list[str])`; `build_scene(spec: SceneSpec) -> World`; `World` owns `model`, `data`, `revision`, `object_ids`, `bin_ids`, and name-to-joint mappings. `observe(world: World) -> WorldObservation` returns stable IDs and poses.

- [ ] Read installation notes; rerun its Panda smoke check using the locked environment.
- [ ] Define strict request models with extra fields forbidden, allowed colors red/blue/green, and a block-count limit of 12. Define bin labels independently of world-axis signs.
- [ ] Write scene validation tests before implementing composition:

```python
def test_default_station_has_expected_objects():
    world = build_scene(SceneSpec(seed=7, block_colors=["red", "blue", "green"] * 2))
    assert len(world.object_ids) == 6
    assert set(world.bin_ids) == {"left_bin", "right_bin"}
    assert all_objects_have_support(world)
    assert not unintended_contacts(world)
```

`all_objects_have_support` and `unintended_contacts` are test helpers in `tests/test_scene.py`: inspect settled object heights against table height and enumerate MuJoCo contacts, allowing floor/table support and robot-base support only.

- [ ] Compose the Panda, tabletop, six free-joint blocks, and two five-piece bins with `MjSpec` or generated MJCF. Keep source model paths absolute or explicitly resolve mesh assets. Use namespaced stable IDs.
- [ ] Validate broad-phase bounds before compiling, then settle and inspect contacts. Pin seed 7 as the first demo fixture; add seed 11 after the first is usable.
- [ ] Start the passive viewer through `mjpython -m astra_world`; keep viewer ownership and physics stepping in the worker.
- [ ] Run `pytest tests/test_scene.py -v`; visually confirm bins, colors, camera labels, and a stationary stable Panda.

**Deliverable:** a visible, valid sorting station built from a structured request.

## Task 2: Prove physical pick-and-place before model integration

**Files:** `motion.py`, `manipulation.py`, `tests/test_manipulation.py`.

**Interfaces:** `solve_ik(world, target_pose, seed_q) -> ndarray | None`; `plan_transport(world, target_pose, held_object_id) -> Trajectory | None`; `sort_blocks(world, goal: SortGoal, cancel: Event) -> ActionResult`. `SortGoal` has `color` and `destination_id`. `Trajectory` contains `scene_revision`, sampled joint targets, and the route label. `ActionResult` contains `ok`, `scene_revision`, `payload`, `error_code`, and `detail`.

- [ ] Add contract types above to `contracts.py`. Establish Panda arm/gripper actuator mappings from actual model names and control ranges; do not assume every actuator is a position actuator.
- [ ] Write a physical integration test for one red block and a failure test for an unreachable target. The passing condition measures bin containment and velocity over 0.5 simulated seconds, not a command return value.
- [ ] Implement damped least-squares IK using `mj_jacSite`, joint limits, bounded iterations, and fixed downward hand orientation. Return failure for excessive final position/orientation residual.
- [ ] Implement approach, descend, close, lift, transport, lower, release, retreat as explicit states with simulation-time deadlines. Track contact and relative block/hand pose to verify retention after lifting.
- [ ] Use smooth actuator target interpolation and physical stepping throughout. Reserve direct state changes for scene initialization and the explicitly announced retry reset.
- [ ] Execute both red blocks sequentially into distinct interior bin slots; return partial progress if the second grasp fails.
- [ ] Run `pytest tests/test_manipulation.py -v`, then inspect a full pick/place in the native viewer. Save a seed and grasp configuration that succeed three consecutive times.

**Deliverable:** one reliable physical sorting task callable directly from Python.

## Task 3: Add the obstacle and make retry adapt

**Files:** `scene.py`, `motion.py`, `session.py`, `tests/test_retry.py`, `tests/test_motion.py`.

**Interfaces:** `add_obstacle(world, spec: ObstacleSpec) -> World`; `capture_trial(world, goal) -> TrialSnapshot`; `retry_last_task(session, world, cancel) -> ActionResult`. `TrialSnapshot` stores the goal plus robot and movable-object qpos/qvel by joint name. `ObstacleSpec` contains ID, position, and half-extents. `Session` retains the last snapshot/goal.

- [ ] Write a regression test that performs a sort, adds an obstacle, restores the trial, and verifies that the obstacle remains while block positions match the pre-sort snapshot.

```python
def test_retry_keeps_new_obstacle(station, completed_trial):
    world = add_obstacle(station, completed_trial.corridor_obstacle)
    restored = completed_trial.session.restore_trial(world)
    assert "obstacle_1" in restored.object_ids
    assert restored.revision > completed_trial.original_revision
    assert movable_poses(restored) == completed_trial.initial_poses
```

Define `station` and `completed_trial` fixtures in `tests/conftest.py` using Task 1/2 functions; `movable_poses` returns named qpos values rounded to six decimals for comparison before physics resumes. `Session.restore_trial(world) -> World` validates restored placements, preserves static additions, and resets actuator targets to restored joint positions.

- [ ] Implement transactional scene recompilation and named state transfer. Increment scene revision on scene changes and invalidate cached paths. Keep the old scene untouched on invalid obstacle placement.
- [ ] Record the first successful transport corridor. Choose a box placement that intersects it but leaves at least one side detour feasible and does not intersect snapshot poses. Return a placement failure when no such position is found.
- [ ] Generate direct/raised/left/right waypoint candidates; solve sequential IK and sample joint-space segments. Check all robot collision geoms and the held block on scratch data, using grasp-relative pose for the latter.
- [ ] Write tests showing the old route collides, a new side route passes, and an intentionally enclosed target returns `no_path`. Verify trajectory revision mismatches are rejected before motion.
- [ ] Implement visible retry status, pre-sort restoration, preserved semantic goal, and re-execution. If no previous task exists, return `no_previous_task` without changing the world.
- [ ] Run `pytest tests/test_retry.py tests/test_motion.py -v`; manually complete the four-step sequence through direct tool calls and compare rendered old/new paths.

**Deliverable:** the full physical demo works before involving an LLM.

## Task 4: Connect typed tools to Astra

**Files:** `tools.py`, `astra.py`, `contracts.py`, `tests/test_tools.py`, `tests/test_astra.py`.

**Interfaces:** `dispatch(command: ToolCommand) -> ActionResult`; `AstraAdapter.respond(messages, tool_schemas) -> ModelTurn`; `ModelTurn` carries assistant text and validated tool requests. Define these types in `contracts.py`. The adapter is the only provider-specific component.

- [ ] Establish the actual Astra service/endpoint and model ID through available project configuration or user input. Keep credentials in environment variables; never commit them. Read provider documentation before adding the SDK or implementing its tool-call protocol.
- [ ] Register `list_assets`, `build_sorting_station`, `observe_world`, `sort_blocks`, `add_obstacle`, `retry_last_task`, and `stop` with strict schemas. Restrict IDs to the active catalog/world.
- [ ] Write dispatcher tests for unknown IDs, unsupported colors, extra fields, busy simulation, `no_path`, and retry with no history. Assert invalid requests leave state unchanged.
- [ ] Implement one simulation command queue with an immediate cancellation event. Return busy status rather than concurrently executing mutations. Observe/status uses worker-produced snapshots.
- [ ] Implement a bounded model loop: maximum 12 tool calls per user turn, provider timeout, cancellation, and actual tool-result messages supplied back to the model. On failure, report the specific action outcome; do not fabricate success or silently retry scene resets.
- [ ] Use an offline model fixture to test the exact four-prompt conversation, including resolving “that bin” to `left_bin` and “try again” to the retained task. Mark fixture runs explicitly as offline.
- [ ] Run `pytest tests/test_tools.py tests/test_astra.py -v`; execute one real Astra turn and confirm it generates a valid tool request using the configured model.

**Deliverable:** live natural-language scene/action requests drive the previously verified simulation.

## Task 5: Add the demo interface and rehearse

**Files:** `server.py`, `__main__.py`, `web/index.html`, `web/app.js`, `web/styles.css`, `tests/test_server.py`, `README.md`.

**Interfaces:** `POST /chat` accepts `{message: string}` and returns a turn ID; `GET /events` streams turn/action/status events; `POST /stop` signals cancellation. Bind to loopback by default. Browser refresh retrieves current scene/action status without restarting simulation.

- [ ] Add and lock FastAPI/Uvicorn only at this milestone. Build a small chat panel with transcript, action status, Stop button, and the four sample prompts. Use the frontend-design skill at implementation time.
- [ ] Write API tests for message validation, busy state, error propagation, and cancellation. Ensure HTML output escapes user/model text and API failures leave the simulation usable.
- [ ] Run the native viewer and local HTTP service from one documented `mjpython -m astra_world` command. Keep the native viewer on its required thread and networking/model work outside the stepping loop.
- [ ] Make retry announce restoration and obstacle preservation. Display different route labels and optionally previous/current path overlays in the viewer.
- [ ] Run the complete test suite once after integration. Run the exact four prompts with live Astra three consecutive times on seed 7 and a scene-construction check on seed 11.
- [ ] Save a concise demo log: package versions, seed, tool calls/results, task completion, collision/planning failures, and actual timing. Verify Stop interrupts within the stated target under normal simulation operation.
- [ ] Update README with setup, model credential names, launch commands, demo prompts, retry semantics, and known limits. Record the demo only after the live checks pass.

**Deliverable:** a reproducible hackathon presentation with honest action feedback.

## Completion checklist

- [x] Environment and Panda verified on the actual Mac.
- [x] Scene generation uses supplied parameters and curated assets.
- [x] Physical sorting works reliably.
- [x] New obstacle invalidates the old route.
- [x] Retry preserves the obstacle and repeats the intended task with a new route.
- [x] Live Astra drives all four user turns.
- [x] Impossible actions and Stop behave visibly and correctly.
- [x] Installation instructions and demo evidence match the final code.

## Implementation record — 2026-09-10

The five milestones are implemented. The headless live-model demo passed three consecutive times. Native launch, rebuild, physical sort, obstacle insertion, retry, and another rebuild also passed; the browser's live build prompt completed against the real native runtime.

- Scene construction: real Menagerie Panda, six default blocks, configurable colors/count/seed, two bins, validated transactional obstacle insertion.
- Manipulation: numerical IK, actuator-driven pinch grasps, four bin slots, collision-checked approaches and transport, unexpected-contact stops, continuous 0.5-second settled-state checks.
- Retry: named joint snapshots preserve the obstacle, restore the trial, and rerun the retained goal. The validated default barrier produces outer detours on both red-block transports.
- Astra: `gpt-6-astra` verified available on the configured account. Uses Responses API, low reasoning, stateless encrypted reasoning continuity, and typed function tools. Keys remain in ignored local configuration; `.env.example` contains no key.
- Interface: browser chat/status/manual controls on port 8765 plus native viewer with bin labels and route overlays. Stop cancels provider turns and physical actions.
- Rehearsal: `uv run python scripts/rehearse_live.py --repeats 3` completed all 12 user turns with successful actual tool results, correct final block positions, and obstacle detours. Local evidence: `artifacts/live-rehearsal.json`.

Implementation decisions relative to the original plan:

1. Use generated MJCF for the prepared station and transactional recompilation; no need for a general asset editor in this demo.
2. Use polling `GET /state` instead of an event stream. The small local interface receives cached world observations without sharing live MuJoCo state across threads.
3. Bins have 0.25 m interiors and are spaced to clear the open Panda gripper. A strict contact review caught and eliminated bin-wall brushing in the original geometry.
4. The bounded planner implements direct and inner/outer side corridors. It reports `no_path` instead of adding an unvalidated raised/global-planning fallback.
5. Automatic obstacle placement is tuned to the accepted left-bin demo. A right-bin arrangement may reject insertion because it overlaps restored source blocks; the rejection is explicit and leaves the scene intact.
6. No synthetic model is used for the final demo. Manual controls and offline rehearsals are labeled separately from live Astra.

Final validation: 38 automated tests passed; JavaScript syntax and dependency checks passed. Two upstream test-client deprecation warnings remain. The macOS asynchronous viewer-close race is covered by regression tests and an actual native sequence.
