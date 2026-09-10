# Astra Robot World

Astra composes MuJoCo scenes from an extensible asset catalog and directs a Franka Panda arm to stack, pack, and sort small objects, with collision-checked motion. Physics and the native viewer run locally on macOS; live language requests use your configured API.

Explore the [interactive architecture canvas](web/architecture.html) or [architecture reference](docs/architecture.md) for Astra’s role, all 25 tools, the motion controller, experiments, and storage. In the running app, click **Architecture**.

## Run on your Mac

Prerequisites: Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync --locked
uv run python scripts/fetch_assets.py
./run.sh
```

Open **http://127.0.0.1:8765**. The browser embeds actual MuJoCo rendering alongside chat, Action Lab, Stop, and manual physics controls. The default launcher uses one simulation process and a fixed port; invoking it again reuses the existing app.

Conversation tool calls expand to show their exact submitted parameters. Expand a trial or live execution result to inspect its ordered motion steps, then expand each step's parameters. Pick/place results also record their internal gripper, move, and transport commands. Submitted steps and completed steps are labeled separately; older calls may lack recorded arguments. Expanded sections remain open while the conversation updates. Refresh an already-open browser tab after frontend updates; HTML, JavaScript, and CSS responses disable caching.

```sh
./run.sh                   # Embedded simulation and chat
./run.sh --restart         # Restart after code/config changes; save your scene first
./run.sh --restart --native # Same app/world plus a connected desktop MuJoCo window
```

The native and browser views share physics state; their camera angles may differ. No second port is silently selected. A server without this project's health endpoint is not killed by the launcher. Close an old manually started instance if port 8765 is occupied. Closing the native window stops its shared app. Saved scenes can be restored after restart through chat, e.g. “Load scenario before_action_lab.”

The tested machine is an Apple M4 Pro MacBook Pro with 24 GB unified memory. Python is pinned in `.python-version`; `uv.lock` pins the complete environment. Packages live in `.venv`.

## Connect Astra

Copy `.env.example` to `.env` if you have not configured it already, and put your own key in the blank `ASTRA_API_KEY` field. `.env` is ignored by Git. Existing process environment variables override the file. Restart after changing configuration.

The default configuration uses `gpt-6-astra` at `https://api.openai.com/v1`. The Astra adapter uses the Responses API for function calling with reasoning; other explicitly configured models use an OpenAI-compatible Chat Completions endpoint. The key stays on the local server and is never returned to the browser. API calls incur usage on the configured account.

Without API configuration, manual physics controls still work and the app clearly reports that live Astra is unconfigured.

## The demo

Send these prompts one at a time, waiting for each action to finish:

1. “Build a sorting station with colored blocks and two bins.”
2. “Put the red blocks in the left bin.”
3. “Now add an obstacle between the robot and that bin.”
4. “Try again.”

The first sort uses a direct transport route. Retry explicitly restores the blocks and robot to the pre-sort snapshot while retaining the obstacle; the new transport route goes around the barrier. Grasping uses physical finger contact and actuator control. Blocks are not teleported or attached to the gripper during execution.

Stop cancels the active turn and motion and holds the current arm position. Scene changes run one at a time, and rejected placements leave the previous world intact.

## Create new actions

Open **Create action** and describe the desired action in your own words. Astra proposes an interpretation, targets, and measurable outcome. Review the card, then select **Start experiments**. Ambiguous requests ask for clarification; unsupported requests name the missing capability. Proposing never executes robot tools. Editing a request creates a new proposal; starting rejects a proposal if the live physics state has changed.

Free-text requests support toppling, object displacement, gripper circles, and extraction from underneath another object. Extraction names the lower object, upper object, and landing surface: the lower must move and separate horizontally by at least 4 cm by default, the upper must lose its lower support and land on the named surface, and both must settle for 0.5 seconds. Unrelated objects, including the landing surface, remain preserved. Optional destination coordinates further constrain the lower object's final position. It does not yet verify arbitrary gestures, exact container containment, custom force limits, or general ordered compound goals.

Astra can use `check_approaches` to compare up to eight candidate routes of up to eight pose waypoints each before spending a trial. It checks bounded inverse kinematics and sampled joint-path collisions on a physics copy with the current gripper configuration; it does not prove a grasp or contact action will succeed. Failed motions include the step, requested pose, and available collision diagnostics. Extraction reports track both blocks.

Each conversation turn permits up to 36 experiment/read-only calls and 12 live calls, with a final tool-free summary when either allowance is exhausted. Notebook queries and sandbox tests no longer consume the live-call allowance. Per-draft trial and time limits still apply. Instructions about experiment strategy—such as trying a short stroke before revising—are passed to the experiment agent separately from the measured goal.

Astra generates typed motion steps, tests them on copies of the current physics state, and revises failed candidates. Switch between **Live world** and **Experiment preview** to watch actual trial physics while the original world remains unchanged. Preview shows trial number, hypothesis, confirmation status, and measured outcome. Auto-follow tracks new trials; select an earlier failed or successful trial to play, pause, or scrub its replay.

**Replay last action** plays the most recent recorded live robot command without moving the robot again. Play, pause, and seek are separate from experiment playback. Recordings are held in memory, capped at 300 frames and 64 MiB; truncated recordings are labeled, and a restart clears them. A rejected command that never moves the robot preserves the previous recording.

Trial clips are approximately 10 fps, retained in memory with limits of10 trials,300 frames per clip, and64MiB total. Older frames/clips may be evicted; restarting clears video replays. Notebook records and saved actions persist. The optional desktop window continues to show the live world.

The default budget is five trials and five minutes, with at most 30 simulated seconds per trial. A candidate needs three successful fresh-state trials before saving. Click **Run live** to validate the saved action against the current scene and then execute it physically. Stop cancels further trials and motion.

Available motion steps are `move_to_pose`, `gripper`, `contact_stroke`, `wait`, and `pick_place`. Astra supplies poses, orientations and stroke parameters; Python computes actuator commands. Contact strokes explicitly allow fingertip contact with selected targets while continuing to check other robot collisions and contact loads. Generated programs are data, not arbitrary Python.

A tower success means the upper block has lost the lower support and settled on the ground; the lower block may also move. Circle success uses measured gripper positions to verify a complete turn with bounded radial and plane error. Saved actions record tested geometry and preconditions; they are not a claim of arbitrary-robot or arbitrary-object support.

New tools: `draft_action`, `test_action`, `save_action`, `list_actions`, `run_action`. Local records live under ignored `actions/`. For live discovery verification, including a failed short stroke followed by revision:

```sh
uv run python scripts/rehearse_action_lab.py
```

This uses API credits and a separate headless test world, without starting another HTTP server or desktop window.

## General scenes and tasks

Browse the 28 built-in assets in the browser. Astra can search and inspect their sizes, compose up to 64 entities with position, rotation, scale, color, mass, and friction, and insert another asset without resetting current object positions. Scenes support Panda or no robot. General scenes pause between actions; ask to simulate a duration to run an experiment.

Try these prompts in sequence:

1. “Create a Panda stacking scene with a red small_box at [0.4,-0.12,0.021] and a blue small_box at [0.4,0.12,0.021].”
2. “Stack the red block on the blue block.”
3. “Save this scenario as stacked_blocks.”

For packing:

1. “Create a Panda packing scene: a red block at [0.4,-0.12,0.021], a blue short_cylinder at [0.55,-0.12,0.021], and a tray at [0.48,0.35,0.025].”
2. “Pack both objects into the tray, placing their centers at [0.43,0.35,0.03] and [0.53,0.35,0.03].”

Positions are object centers in meters; rotations are Euler radians. Astra translates a task into reusable `pick_place` actions; the simulator computes IK, checks candidate routes, drives physical finger contact, and checks that the object settles at the target. The same action supports stacking, packing, and sorting by sequencing placements.

`save_scenario` stores the scene plus current joint positions, velocities, controls, and simulation time in ignored `scenarios/`. `load_scenario` restores that snapshot. `reset_world` restores the current scene's initial poses. These tools apply to general scenes; the original sorting demo retains its separate pre-sort retry behavior.

### Import your own assets

```sh
uv run python scripts/import_asset.py /path/to/object.obj --id custom_object --units m --license 'Your asset license'
```

OBJ and STL import preserve source attribution, normalize the origin to the bounding-box center, and register separate visual geometry and a convex collision hull under ignored `assets/custom/`. Imported meshes become available to scene composition, but do not automatically acquire a robot grasp skill. Concave openings are filled by the convex collision approximation.

## Verification

```sh
uv run pytest -q
uv run python -m astra_world --headless --demo
uv run python scripts/check_mujoco.py
uv run mjpython scripts/check_mujoco.py --viewer
```

The offline demo writes `artifacts/demo.json` and calls the real physics tools directly. It does **not** use an LLM. The smoke check loads Panda, advances physics 1,000 steps, and renders `artifacts/panda-smoke.ppm`.

To explicitly run the four prompts through the configured live Astra API:

```sh
uv run python scripts/rehearse_live.py --repeats 3
uv run python scripts/rehearse_general.py
```

This makes real API calls, runs physical simulation without a viewer, checks tool outcomes and object positions, and writes `artifacts/live-rehearsal.json`.

Use `./run.sh` for everyday operation. The lower-level CLI also supports `--headless` and `--port` for isolated tests; manually starting multiple ports creates separate worlds.

## Scope and limits

- General scenes offer 28 procedural/composite props and local mesh import, up to 64 entities, with one fixed Panda or no robot. The original sorting preset has 1–12 blocks, two bins, and one obstacle.
- Each bin has four placement slots. Insufficient capacity returns an error before motion.
- Physical grasp tests cover `small_box` cubes resting flat on any face, and upright `block` and `short_cylinder` at default scale in reachable ground-level arrangements. Other assets can be simulated; arbitrary grasps, walking, and additional robot controllers are not implemented. Scaling, mass, clutter, or different placements can cause manipulation to fail.
- Custom motion programs can close the gripper with `{"op":"gripper","opened":false,"object_id":"green_block"}`. Both fingers must contact the named free object; subsequent moves carry its collision allowance until `{"op":"gripper","opened":true}` releases it. This does not attach the object: lifting and placement still depend on physics, with collision checks and the 40 N action force limit active.
- Initial scene poses are not automatically settled or guaranteed free of overlap. Later entity insertion rejects detected penetrations. Arrange objects on valid support surfaces; some rotated/nonuniform primitive scales are rejected.
- Motion planning searches a small set of direct and side corridors with sampled whole-arm/held-object collision checks. Unreachable or obstructed motions can return `no_path`; this is not a general global planner.
- Automatic obstacle placement is tuned to the accepted left-bin demo. Other arrangements can reject the obstacle if it overlaps current or restored objects. Explicit obstacle dimensions and positions are available through the tool API.
- The robot uses simulator state, not image perception. The embedded browser viewport renders actual simulator data; the optional desktop window shares the same world. The browser image is a streamed camera view, not a draggable 3D canvas.
- Rendering on a logged-out/headless macOS host may require a display context. Use `mjpython` for the passive native viewer.

## Assets and licenses

`scripts/fetch_assets.py` downloads a sparse checkout of [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), pinned to commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`, selecting `franka_emika_panda` and repository metadata. It preserves existing local asset changes. Override the model location with `ASTRA_PANDA_PATH` if needed.

The Panda's [Apache-2.0 license](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/LICENSE) and attribution remain in `assets/menagerie/franka_emika_panda/`. Keep those notices when redistributing assets. The built-in prop catalog is project-authored procedural geometry, including composite hollow containers; no external prop dataset is bundled. Imported files retain their supplied license/source metadata.

## Design

See the [design](docs/superpowers/specs/2026-09-10-astra-robot-world-design.md) and [implementation plan](docs/superpowers/plans/2026-09-10-astra-robot-world.md). The code separates scene compilation, motion, manipulation, retry memory, single-owner simulation, typed tools, model integration, and the browser interface.

## Persistent experiment notebook

Action Lab now records drafts, proposed programs, every completed physical trial (including failures), final experiment state, and links to newly saved actions in `actions/notebook.sqlite3`. SQLite transactions persist each entry as it happens. Existing action JSON files remain compatible; historical failed trials from before this feature cannot be reconstructed automatically.

Astra can call `search_action_notes`, `read_action_notes`, and `write_action_note`. Notes are labeled model interpretations and may reference a real trial number. They never change measured outcomes or verified status. Create action retrieves related experiments by goal type, and Astra is instructed to record hypotheses and lessons. The notebook is searchable under Action Lab in the browser.

Notebook records survive app restarts and scene changes. An unfinished experiment's log survives, but its in-memory execution session is not resumed automatically. Saved executable actions remain separate JSON files. The first increment of the [free-text action design](docs/superpowers/specs/2026-09-10-free-text-action-design.md) is implemented; broader predicate composition remains proposed.


To rehearse free-text proposal, a deliberately failed short push, revision, three successful trials, and replay-frame checks against the existing app:

```sh
uv run python scripts/rehearse_visible_learning.py
```

This uses API credits and requires a red/green tower in the current general Panda scene. Trials preserve the live world. Optional `--scenario NAME` explicitly loads a saved scene first. The script connects to8765 and does not launch another app instance.
