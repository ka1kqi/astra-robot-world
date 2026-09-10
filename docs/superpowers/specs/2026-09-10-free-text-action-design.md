# Free-text action requests

Status: first increment implemented after user approval, including visible trial previews and replay. Free text maps to the existing topple/displace/circle checks; unsupported/ambiguous requests are explicit. Broader predicate families below remain proposed.

Implemented preview extension: browser Live world / Experiment preview switch, real scratch-world MuJoCo frames, current hypothesis and measured results, auto-follow, and bounded-session failed/successful replay. The desktop viewer remains attached to the unchanged live world. Replay videos are in-memory; the notebook remains durable.

Implemented proposal guard: interpretation has only one structured-output tool and cannot execute simulation actions. Start uses the server-stored proposal and an authoritative owner-thread physics-state check. Immediate Stop also cancels proposals that have not started their coroutine yet.

## Product behavior

Replace the goal dropdown with a request: “What should the robot do?” Keep tower and circle as example chips. Astra reads the current scene and relevant notebook entries, then proposes an action and a measurable outcome. The user reviews one compact card before starting trials; experimentation then proceeds automatically within the displayed budget.

Example request: “Push the blue block gently into the tray.”

Proposed card:
- Move blue_block into tray_1 and let it settle for 0.5 seconds.
- Keep contact below 5 N and leave the other objects in place.
- Five trials, maximum five minutes. Live world stays unchanged.
- Buttons: Start experiments / Edit outcome.

The force threshold is a visible proposed interpretation of “gently”, not something the model silently equates with the word. Ask a clarification only when it materially changes the task, such as two possible blue objects or “spin” meaning a circle versus joint rotation. Highlight target IDs in the scene listing; do not require users to type IDs.

## Recommended architecture

Separate request interpretation, motion generation, and success evaluation. Add a proposal endpoint returning a versioned typed specification: request, human-readable interpretation, entity bindings, success predicates, preservation/contact/time constraints, relevant notebook references, and capability gaps. Freeze this specification when Start experiments is clicked. Scene revision and initial physics state must still match; otherwise regenerate or revalidate the proposal before starting.

Astra composes programs from the existing motion primitives and revises them using measured failures. The simulator evaluates predicates independently of model prose. A candidate cannot change its own evaluation criteria or exempt unrelated objects. Changing the intended outcome creates a new experiment linked to the earlier one.

Keep run_action(name, bindings) as the reusable tool interface. Saved actions gain declared parameters and readable descriptions; a new API function is unnecessary for each learned motion.

## Approaches considered

1. **Recommended: compose measurable predicates.** Expand beyond named tower/circle tasks using a reusable goal vocabulary. This provides broader action creation with inspectable evidence.
2. **Generate arbitrary Python controllers and evaluators.** More expressive, but requires a separate execution backend and independent evaluator validation. A generated evaluator can accidentally reward a trivial or wrong outcome; do not introduce this as a shortcut to generality.
3. **Let Astra judge rendered video alone.** Useful as supplemental interpretation for expressive tasks, but insufficient as the sole verification signal. Visual confidence and physical measurements should remain separately labeled.

## Goal vocabulary and staged scope

First expose the existing displaced-object, topple, and circle checks through free text. Then add composable predicates:
- Object inside a region or container, accounting for its extent.
- Object stably supported by another object/surface.
- Gripper reaches an explicit pose or follows a sampled reference path.
- A path includes repeated directional changes, enabling a geometrically specified wave.
- Object remains below speed/contact-force limits and settles for a duration.
- All-of and ordered-sequence combinations, with explicit timing.

Example “wave hello” proposal: move the gripper above the table and sweep left/right three times over a stated width. This verifies the specified gesture geometry; it does not establish that every viewer will recognize a greeting. Unlimited joint spins remain impossible where hardware joint limits forbid them.

If a request needs an unavailable primitive or evaluator (such as compliant twisting of a threaded cap), show the missing capability and offer an explicit narrower interpretation. Do not relabel a pose sequence as having learned the unsupported manipulation.

## Persistent memory

Use the implemented notebook as evidence and experiment memory. Retrieve experiments by goal and physical context (robot, asset geometry/scale, supports), then read their notes and programs as starting hypotheses. Persist each candidate and measured trial automatically. Astra writes concise hypothesis/revision/lesson notes tied to trial numbers; notes cannot change results.

Each new experiment starts from current-scene validation, even when borrowing a previously successful program. Failed and cancelled experiments remain searchable. A server restart preserves their records, but does not implicitly resume an unfinished physical experiment; a new snapshot starts a linked attempt.

## UX during and after trials

Show the interpreted goal, candidate steps, trial counter, measured outcome, and notebook notes. Distinguish live scene from experiments. Successful actions require two confirmation runs from the same snapshot and show the exact tested conditions. Run live revalidates the current scene before moving the robot. Exhaustion shows the best attempt and why it failed, without granting verified status.

## Verification and delivery

Deliver in two increments:
1. Free-text interpretation into currently supported goals, proposal/review UI, notebook retrieval, unknown/ambiguous request handling. No broader motion claims.
2. Predicate composition and additional path/region/support checks, with physical regression cases and held-out placements for each new goal family.

Acceptance cases: free-text tower, circle, and object displacement; ambiguous object selection; impossible joint spin; wave request translated to explicit gesture geometry only when its evaluator exists; attempted criterion mutation rejected; prior-note retrieval cannot override the user or evaluator; cancelled trials preserve live state; new bindings require revalidation. Evaluate task interpretation separately from physical success so a robot cannot “succeed” at the wrong task.
