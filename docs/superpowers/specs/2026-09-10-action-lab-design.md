# Action Lab: generated Panda motions

Status: implemented after user approval. Added user-requested gripper circle goals, embedded real MuJoCo rendering, and a fixed-port launcher. Multi-robot controllers remain deferred; Panda is the supported hackathon platform.

## Implementation record

- Typed pose/gripper/contact-stroke/wait/pick-place programs, bounded contact/tracking/time checks.
- Immutable copies of compiled models plus mjSTATE_INTEGRATION for trial isolation; fixed physical success predicates and three-run verification.
- Circle goals measure actual gripper winding and radial/plane error, excluding the initial approach.
- Create action UI freezes goal/target/budget server-side; its tool scope permits same-draft testing/saving and reads only.
- Trial progress/measurements appear in the browser; embedded frames show the live world, which remains unchanged during trials. Run live displays physical replay. No trial video preview in this version.
- One launcher, ./run.sh, uses8765; --native opens an optional desktop viewer in the same process, --restart gracefully restarts only this project's recognized app.
- Live Astra created and saved tower and circle programs after three physical verification runs each; both replayed successfully. Full automated suite:118 tests passed at initial integration.
- Reviewed and fixed early failed-circle handling, goal preservation exemptions, compiled-model snapshot fidelity, settled-time floating-point boundary, and stale UI trial cards.


## Problem and scope

The user requested knocking over a red/green tower. The current pick_place skill approaches downward and the controller rejects contact with the upper block. A toppling task needs intentional contact, which is outside this skill's contract. The live scene has red centered near (0.400,-0.120,0.020) and green near (0.401,-0.120,0.060).

Add a Create action workflow where Astra generates a reusable motion program, evaluates it in physical trials, revises failed attempts, and saves a successful version. Initial scope is the existing Panda in general scenes, beginning with push/topple. Existing pick_place and sorting behavior remain regression requirements.

## Approaches

1. Recommended: typed motion programs over reusable controller primitives. Astra controls sequencing, relative poses, contact target, stroke direction/distance, speed, and timing. Programs are inspectable and trials reproducible.
2. Generate unrestricted Python functions: expressive but requires a separate code execution environment, resource controls, and protection against changing the measured world or success check. Defer this execution backend.
3. Add only a fixed push tool: quickest immediate fix, but does not deliver user-created reusable actions. Implement contact-aware pushing as a foundation for approach 1.

## Motion contract

Provide approach/move-to-pose, gripper open/close, Cartesian contact stroke, retreat, wait, and existing pick_place composition. Object-relative poses resolve from the trial's initial observation. IK supports explicit orientation. Cartesian strokes use sufficiently close waypoints instead of assuming interpolation between endpoint joint angles follows a straight Cartesian path.

Separate collision-free travel from intentional contact strokes. A contact stroke declares which object and which gripper surfaces may touch; contacts involving other robot links, ground, or unrelated objects remain checked. Enforce bounded speed, stroke length, time, joint tracking, and measured contact loads. Tune numeric limits with physical regression tests before exposing the primitive. Do not globally disable collision checking or inject object displacement/forces to fake a robot action.

A generated action is a versioned typed program, not arbitrary Python. Validate step count, numeric finiteness, references, workspace limits, contact policy, and duration before executing. Saved actions declare parameters and tested preconditions; success on a small box does not establish support for every asset.

## Trial lifecycle

Create action captures an immutable initial scene and physics snapshot. Each trial uses a separate MuJoCo data/model context rebuilt from that snapshot; include controls, velocities, simulation time, applied forces and solver state needed for repeatability. Failed attempts never become the next trial's starting state. Preserve the user's live scene during experiments.

Astra submits candidate programs and receives structured outcomes: reachability, prohibited contact, tracking error, target poses/orientations, displacement, settledness, and goal score. It can adjust approach, contact height, direction, stroke length, speed, and sequence. Default limits: five trials, 30 simulated seconds per trial, five minutes overall; Stop cancels execution and further API iterations. These limits stop failed exploration rather than guaranteeing a successful action.

Successful candidates receive two fresh-snapshot verification runs before being marked tested for the scene. Verification consumes the overall budget. If budget is exhausted, show the best failed attempt and diagnostics without saving it as successful. Scene changes invalidate a pending live replay; revalidate before Run.

## Success contract

Define measurable criteria before optimization and keep them fixed through revisions. For the reported tower: upper block loses support from the lower block, reaches ground-level support, and remains settled for at least 0.5 simulated seconds. Report rotation separately so sliding an upper block off is not falsely described as rotating the whole tower. Preserve unrelated objects within a declared displacement tolerance. Evaluate physical state/contact geometry; model prose alone cannot mark success. Exclude impossible requests such as pushing a fixed body before trial execution.

Initial goal types: object displaced into a requested region; upper block no longer supported by lower block and settled. Broader free-text goals require selecting a supported measurable predicate, not arbitrary generated evaluator code.

## UI and tools

Create action opens a goal input, target selection, trial budget, and visible outcome definition. During execution show candidate steps, trial number, measured failures, and current best result. Distinguish trial preview from the live world. Trial replay can be shown in the existing native viewer with an explicit trial label; never silently replace the live session.

Add draft_action, test_action, save_action, list_actions, and run_action tools. The server owns session budgets, immutable snapshots, validation, and measurements. Astra owns proposals and revisions. Store versioned programs and verification reports under ignored local actions/. Saved actions appear in an action library; Astra invokes them through run_action(name, arguments), so dynamically expanding the provider's tool schema is unnecessary.

User flow: Create action -> 'Knock over this tower' -> see bounded trials and measured results -> inspect saved successful action -> Run in current scene. Trial execution is automatic once requested. Running the saved action is an explicit UI operation; ordinary subsequent chat requests can also authorize run_action.

## Verification

Reproduce the user's two-block tower as the first physical fixture. Verify intentional fingertip contact can topple/dislodge the upper block, unrelated robot collisions still abort, and no direct object state edits occur during action execution. Test failed and cancelled trials preserve live state, every trial starts from the same snapshot, finite budgets terminate, unreachable/fixed targets reject, and goal scores cannot be set by generated steps. Repeat successful action from fresh state and report the tested geometry. Run the complete existing suite and live Astra Create action flow, then native preview/replay smoke checks.

Live discovery regression also passed: Astra tested an insufficient 5 mm stroke, observed failure, revised its program, passed three fresh-state trials, and saved the action. Trace: artifacts/action-discovery.json (ignored).
