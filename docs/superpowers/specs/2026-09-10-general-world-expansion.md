# General-world expansion

Status: initial expansion implemented. User selected many objects/tasks with the Panda: stacking, packing, sorting, obstacles. The architecture sections below retain the longer-term design; they are not a claim that every proposed interface is implemented.

## Implemented and verified

- 28 procedural/composite catalog assets; OBJ/STL importer with centered visuals, convex collision hulls, and supplied attribution. No external object dataset bundled.
- Validated WorldSpec with up to 64 entities, Panda or no robot, physical parameters, and geometry-aware scaling restrictions.
- Search/describe/create/observe/add_entity/simulate/save/load/reset tools. Insertion preserves current named joint state and rejects detected penetrations transactionally.
- General pick_place physically tested with small_box, block, and short_cylinder in upright, ground-level setups. One skill supports stacking and tray packing; physical obstacle regression forces an outer route.
- General scenes pause between actions; snapshot persistence includes current physics state. Original sorting preset and retry remain separate, preserving the four-prompt demo.
- Asset browsing and task prompts are available in the browser. Actual rendering remains native MuJoCo.
- Live Astra trials passed: custom stacking, two-shape tray packing, state-preserving wall insertion, and robot-free ball drop. Original live four-prompt demo passed again after expansion.
- Native Mac smoke passed scene swaps, simulate/reset, physical stacking, and wall insertion. Full suite: 82 passed, two upstream deprecation warnings.

Not yet implemented: multiple robots, general RobotAdapter interface, push skill, arbitrary scene removal/movement patches, declarative goal evaluator, arbitrary grasp generation, or global motion planning. Initial scene layout overlap is not comprehensively prevalidated.

## Intended change

Replace task-specific world creation with an asset catalog, a declarative scene description, and robot-specific capabilities. Astra chooses objects, assembles scenes, defines measurable goals, and composes available actions. A scene may contain zero, one, or multiple robots; simulation does not imply that every imported robot has a working controller.

The shared foundation below is useful whether the next priority is many Panda tasks or rooms and different robots. Begin with that foundation and choose the controller/asset packs according to the user's priority.

## Three approaches

1. **Recommended: catalog + scene schema + capability tools.** Reproducible, inspectable scenes and explicit support boundaries. New assets can be added without adding a new scenario function.
2. **Generate arbitrary simulator Python.** Flexible, but changes are harder to validate, replay, and diagnose; it conflates scene construction with application modification.
3. **Adopt a complete environment framework.** RoboCasa/robosuite supply valuable assets and tasks, but full adoption adds framework and runtime constraints. Start with compatible asset packs and integrate a framework only where its controllers/tasks save substantial work.

## Shared architecture

### Asset registry

Each `AssetDefinition` has a stable ID, semantic tags, source URL, pinned source revision/hash, license, local model path, format, bounds, units, collision representation, mass/friction defaults, grasp candidates, support surfaces, and optional joint/robot metadata.

Separate validation levels:

- imported: visual/model file loads;
- physics-ready: contacts and inertia validated for the intended scale;
- manipulation-ready: an available robot skill has tested grasp/release behavior;
- controller-ready: a robot has a supported controller and named actuator/end-effector mappings.

A model must never be advertised as graspable or navigable merely because its mesh imported. Download packs on demand and cache locally; do not fetch thousands of assets at startup. Keep license attribution beside each pack.

Sources to investigate per pack:

- MuJoCo Menagerie for robot descriptions: https://github.com/google-deepmind/mujoco_menagerie
- robosuite for primitive, composite, and prepared object models: https://robosuite.ai/docs/_sources/modules/objects.html
- RoboCasa for household assets and environments: https://robocasa.ai/docs/build/html/assets/objects.html

RoboCasa documents over 3,200 objects across over 150 categories. These are potential source assets, not a claim that this app has imported or validated that collection.

### Scene representation

A `WorldSpec` contains entities, explicit transforms, physical parameters, cameras, a seed, and optional goals. Each entity references an asset ID and declares static, free-body, or articulated behavior. Physical parameters include gravity and supported material overrides.

A layout resolver can translate relations such as “on the table,” “inside the tray,” and “between these objects” into concrete placements. Compilation receives only concrete coordinates and validated asset references. Use stable namespacing for composed MJCF assets and preserve their internal mesh/material references.

A `ScenePatch` supports adding/removing/moving entities while paused. Validate a candidate scene before replacing the current world. Snapshot records must include the scene specification and named dynamic state; changing layouts cannot leave snapshots referencing deleted entities. Every accepted patch increments a revision and invalidates planned trajectories.

Containers require hollow/composite collision shapes. A convex hull of a bowl or bin closes its interior and is not an acceptable substitute.

### Generic tools

- `search_assets(query, tags, capability)`
- `describe_asset(asset_id)`
- `create_world(spec)`
- `patch_world(patch)`
- `observe_world()`
- `get_robot_capabilities(robot_id)`
- `execute_skill(robot_id, skill, arguments)`
- `simulate(duration, measurements)`
- `evaluate_goal(goal)`
- `save_scenario(name)` / `load_scenario(name)`
- `reset_trial()` / `stop()`

The current `build_sorting_station` becomes a preset that emits a `WorldSpec`. The current `sort_blocks` becomes a composition of observations and individual pick/place skills, rather than the core action interface.

### Controllers and skills

Move Panda-specific home angles, actuator mapping, end-effector site, gripper commands, and contact allowances into a `RobotAdapter`. General world/snapshot code must not assume that the first seven coordinates belong to a Panda arm.

Initial manipulation skills: reach, open/close gripper, pick, place, push. Each declares preconditions and supported object properties. Physics remains in MuJoCo, trajectory generation in the controller, and task sequencing in Astra. Training-free scripts can control suitable arms; stable walking or general dexterous handling needs a compatible existing policy or separate training.

### Task goals and observations

Typed predicates include `inside_region`, `on_support`, `near`, `joint_at`, and `settled`. They evaluate actual geometry/state over a required duration. Goals are data, never dynamically evaluated Python strings. Report partial outcomes and failures to Astra with concrete object IDs.

Examples:

- sort several object types into trays;
- pack a set of graspable objects into a box;
- arrange blocks into a pattern or simple stack;
- compare how shapes slide down ramps with different friction;
- vary obstacle positions and measure whether the available arm planner succeeds;
- later, navigate a room using a robot adapter with a navigation controller.

## Implementation stages

1. Introduce `AssetDefinition`, `WorldSpec`, and an extensible primitive/composite catalog; compile scenes with no robot or a Panda. Preserve the existing sorting preset and all current tests.
2. Add a versioned asset-pack importer and a small, verified external object pack. Include import smoke tests, source/license records, dimensional checks, and representative physical drop/contact tests.
3. Extract `RobotAdapter` and individual Panda pick/place skills from the sorting state machine. Replace hard-coded block geometry and bin IDs with entity metadata and target regions.
4. Expose scene composition, capability discovery, skill execution, and goal tools to Astra. Add scene preset/save/load UI and visible object/capability browsing.
5. Verify at least three distinct prompted scenarios using different object types, plus one physics-only scene. Keep unsupported operations explicit.
6. Add another robot/controller pack or richer room assets according to the user's selected priority.

## Verification boundary

The current M4 Pro/24 GB machine should be benchmarked with representative packs. Avoid promises about thousands of simultaneous objects or photorealistic rooms before measurement. Asset-library size, simultaneous scene size, and number of available robot skills are independent dimensions of generality.
