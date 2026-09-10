# Action Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Complete all tasks without further permission checkpoints; the user approved the design and implementation.

**Goal:** Let Astra create, physically test, save and run reusable Panda contact actions through the UI.
**Architecture:** Typed motion steps execute through Panda controllers; an ActionLab owns immutable snapshots, bounded trials and fixed goal measurements. HTTP/UI and Astra tools expose progress and saved actions.
**Tech Stack:** Python, Pydantic, MuJoCo, FastAPI, existing vanilla JS UI.
**Spec:** docs/superpowers/specs/2026-09-10-action-lab-design.md

## Global Constraints

- Preserve current live world during trials, including failures and cancellation.
- Default limits: five trials, 30 simulated seconds per trial, five minutes overall. Successful candidates require two additional fresh-snapshot verification runs within budget.
- Astra cannot directly set physical object state or the outcome measurement. Existing picking/sorting collision behavior stays covered.
- Generated programs are typed motion data; no Python evaluation. Local files only; no publishing.
- General scenes with Panda only for initial version. UI labels experimentation separately from live execution.

## Interfaces and tasks

- [x] Task 1 (parent): Create action_contracts.py with strict GoalSpec, MotionStep, ActionProgram and tool argument models. MotionStep operations: move_to_pose, gripper, contact_stroke, wait, pick_place. Positions are absolute or offsets from reference_id's initial position. Contact strokes use direction/distance/speed and explicit contact_ids. Reject malformed/nonfinite/out-of-bounds values. Write contract regression tests before integration.
- [x] Task 2 (motion agent): action_motion.py exports execute_program(world, program, cancel=None, tick=None, status=None, max_seconds=30) -> {ok,error_code?,detail?,payload}. Use scratch planning + actuator motion, Cartesian contact strokes, intentional finger-only contacts, finite force/speed/time limits. Tests physically topple/dislodge upper block, reject unrelated arm contact, hold/cancel, preserve original motion regressions. Agent may edit motion.py only for minimal backwards-compatible hooks.
- [x] Task 3 (trial agent): action_lab.py exports ActionLab(world), draft(name, goal, trial_budget=5), test(draft_id, program, cancel=None, status=None), save(draft_id), list_actions(), run(world,name,bindings=None,cancel=None,tick=None,status=None). Results use existing {ok,scene_revision,payload} convention. Optional snapshot() exposes browser progress. Clone via mjSTATE_INTEGRATION and WorldSpec; goal fixed before trial. Test budgets, fresh state each run, immutability, saved-program validation, measured success and two confirmation runs. No server/runtime edits.
- [x] Task 4 (UI agent): add Create action form with description, target/support selection, budget, visible trial status/measurements/program and saved action Run buttons. POST /actions/create accepts {message,target_id,support_id,trial_budget}; GET /actions returns library via runtime; GET /state world.action_lab carries draft/trial status. Handle missing/failed/empty states. Existing chat/stop/preset controls remain usable.
- [x] Task 5 (parent): integrate tool schemas, dispatch, runtime progress, server bounded Action Lab conversation and cancellation. New tools draft_action/test_action/save_action/list_actions/run_action. Create action uses isolated chat history, fixed target/support context, overall asyncio deadline. Failure within experiment is a trial result, not necessarily failed creation. Scene edits during active turn remain serialized. Add adapter/server/runtime tests with mocked provider and real physics.
- [x] Task 6 (parent + review): run existing suite and live Astra tower experiment; check saved Run on original scene, native viewer and browser. Add reproducible script and docs; independent review controller+trials+integration and fix demonstrated defects.

## Acceptance examples

```python
# Successful candidate is tested three times and leaves live physics untouched.
before = live.data.qpos.copy()
draft = lab.draft('topple_tower', GoalSpec(kind='topple', object_id='green', support_id='red'), 5)
result = lab.test(draft['payload']['draft_id'], program)
assert result['payload']['verified']
np.testing.assert_array_equal(live.data.qpos, before)
assert lab.save(draft['payload']['draft_id'])['ok']
```

```python
# Failed experiments cannot become successful saved actions.
assert not lab.save(failed_draft_id)['ok']
# Stop is delivered without waiting for the provider or physics completion.
assert client.post('/stop').status_code == 200
```

## Execution record

- Approved by user: “sounds good to me, use high effort to solve this”.
- Ruling: use typed motion programs first, as approved; raw generated Python is deferred.
- Ruling: trial preview initially uses structured positions/progress in browser; native viewer retains live world. This preserves live context and avoids replacing the user's world during experiments. Saved Run animates native physics.

- Added circle goals, streamed MuJoCo frames, and run.sh fixed-port launcher per subsequent user requests. Live tower/circle create-save-replay passed. Full suite118passed. Independent reviews completed and concrete findings fixed.
