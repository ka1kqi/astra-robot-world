# Astra Robot World

Astra builds a MuJoCo sorting station from prepared assets, directs a Franka Panda arm to sort blocks, and replans when you add an obstacle. Physics and the native viewer run locally on macOS; live language requests use your configured API.

## Run on your Mac

Prerequisites: Git and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync --locked
uv run python scripts/fetch_assets.py
uv run mjpython -m astra_world
```

Open **http://127.0.0.1:8765** alongside the MuJoCo window. The native viewer shows actual motion, labeled bins, and cyan/grey route overlays. The browser has chat, action feedback, Stop, and explicit manual physics controls.

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
```

This makes real API calls, runs physical simulation without a viewer, checks tool outcomes and object positions, and writes `artifacts/live-rehearsal.json`.

For an HTTP service without a native window, use `uv run python -m astra_world --headless`. Change the port with `--port 8767`.

## Scope and limits

- One fixed Panda, a table, two bins, 1–12 red/blue/green blocks, and one box obstacle. The default is six blocks with seed 7; seed 11 is also checked.
- Each bin has four placement slots. Insufficient capacity returns an error before motion.
- Scene generation composes the supported asset catalog. Arbitrary robot types, external furniture, mesh generation, walking, and vision-only control are not implemented.
- Motion planning searches a small set of direct and side corridors with sampled whole-arm/held-object collision checks. Unreachable or obstructed motions can return `no_path`; this is not a general global planner.
- Automatic obstacle placement is tuned to the accepted left-bin demo. Other arrangements can reject the obstacle if it overlaps current or restored objects. Explicit obstacle dimensions and positions are available through the tool API.
- The robot uses simulator state, not image perception. The small browser diagram is an illustration; actual rendering appears in the native MuJoCo window.
- Rendering on a logged-out/headless macOS host may require a display context. Use `mjpython` for the passive native viewer.

## Assets and licenses

`scripts/fetch_assets.py` downloads a sparse checkout of [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), pinned to commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`, selecting `franka_emika_panda` and repository metadata. It preserves existing local asset changes. Override the model location with `ASTRA_PANDA_PATH` if needed.

The Panda's [Apache-2.0 license](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda/LICENSE) and attribution remain in `assets/menagerie/franka_emika_panda/`. Keep those notices when redistributing assets. Blocks, bins, and the obstacle are procedural primitives; no additional prop library is needed.

## Design

See the [design](docs/superpowers/specs/2026-09-10-astra-robot-world-design.md) and [implementation plan](docs/superpowers/plans/2026-09-10-astra-robot-world.md). The code separates scene compilation, motion, manipulation, retry memory, single-owner simulation, typed tools, model integration, and the browser interface.
