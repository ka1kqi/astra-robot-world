"""Single-owner MuJoCo runtime. HTTP threads see only copied observations."""

from concurrent.futures import Future
from copy import deepcopy
import hashlib
import json
import logging
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread, main_thread
import time
from uuid import uuid4

import mujoco
import numpy as np
from pydantic import ValidationError

from .assets import CATALOG
from .experiment_preview import ExperimentClips
from .scene import BINS, build_scene, observe
from .tools import ARGUMENTS, dispatch, error

log = logging.getLogger(__name__)


class SimulationRuntime:
    def __init__(self, *, headless=False, render_frames=False):
        self.headless = headless
        self.render_frames = render_frames
        self._renderer = None
        self._preview_renderer = None
        self._experiments = ExperimentClips()
        self._action_clips = ExperimentClips(max_trials=1)
        self._action_recording = None
        self._preview_failed = False
        self._preview_wall_origin = self._preview_sim_origin = 0.0
        self._preview_last_time = -1.0
        self._frame = None
        self._render_error = None
        self._last_render = 0.0
        self._render_failed_model = None
        self._lock = Lock()
        self._queue = Queue(maxsize=1)
        self._cancel = Event()
        self._shutdown = Event()
        self.ready = Event()
        self._busy = False
        self._snapshot = {
            "scene_revision": 0,
            "status": "Starting simulation",
            "busy": False,
        }
        self._thread = None
        self._viewer = None
        self._viewer_model = None
        self._session = None
        self._startup_error = None
        self._last_publish = self._last_sync = 0.0
        self._wall_origin = self._sim_origin = 0.0

    def start(self):
        """Start a background owner for headless embedding and tests."""
        if not self.headless:
            raise RuntimeError(
                "Run the native viewer with run() on the main thread under mjpython."
            )
        if self._thread is not None:
            raise RuntimeError("Runtime already started.")
        self._thread = Thread(target=self.run, name="mujoco-owner", daemon=True)
        self._thread.start()
        if not self.ready.wait(30):
            raise RuntimeError("Simulation startup timed out.")
        if self._startup_error:
            raise RuntimeError("Simulation startup failed.") from self._startup_error
        return self

    def snapshot(self):
        with self._lock:
            return deepcopy(self._snapshot)

    def frame(self):
        with self._lock:
            return self._frame

    def experiment_metadata(self):
        return self._experiments.metadata()

    def experiment_frame(self, trial_id=None, frame_index=None):
        return self._experiments.frame(trial_id, frame_index)

    def action_replay_metadata(self):
        metadata = self._action_clips.metadata()
        return {
            "clip": metadata["trials"][-1] if metadata["trials"] else None,
            "available": metadata["available"],
            "error": metadata["error"],
        }

    def action_replay_frame(self, clip_id=None, frame_index=None):
        return self._action_clips.frame(clip_id, frame_index)

    def _arm_action_recording(self, name, arguments):
        self._action_recording = None
        if name not in {"pick_place", "run_action", "sort_blocks", "retry_last_task"}:
            return
        if not self.render_frames:
            return
        self._render_frame()
        args = arguments.model_dump() if hasattr(arguments, "model_dump") else arguments
        self._action_recording = {
            "id": uuid4().hex,
            "tool": name,
            "name": args.get("name") or args.get("object_id") or name,
            "origin": float(self._session.world.data.time),
            "initial_frame": self.frame(),
            "started": False,
            "last_sample": -1.0,
        }

    def _record_action_frame(self, final=False):
        recording = self._action_recording
        if recording is None:
            return
        elapsed = float(self._session.world.data.time) - recording["origin"]
        if not recording["started"]:
            # Revalidation previews never tick the live world. Rejected commands
            # therefore preserve the previous recording, including its clip id.
            if abs(elapsed) < 1e-9:
                return
            if elapsed < 0:  # A retry can restore an earlier simulation clock.
                recording["origin"] = float(self._session.world.data.time)
                elapsed = 0.0
            self._action_clips.begin({
                "id": recording["id"], "tool": recording["tool"],
                "name": recording["name"], "scope": "recorded_live_action",
            })
            recording["started"] = True
            if recording["initial_frame"] is not None:
                self._action_clips.append(recording["id"], recording["initial_frame"], 0.0)
            recording["initial_frame"] = None
        if final or elapsed - recording["last_sample"] >= .1 - 1e-9:
            self._render_frame()
            frame = self.frame()
            if frame is not None:
                self._action_clips.append(recording["id"], frame, elapsed)
            else:
                self._action_clips.rendering_error(recording["id"], self._render_error or "No live frame available.")
            recording["last_sample"] = elapsed

    def _finish_action_recording(self, result):
        if self._action_recording is None:
            return
        try:
            self._record_action_frame(final=True)
            recording = self._action_recording
            if recording["started"]:
                self._action_clips.end(recording["id"], {
                    "ok": bool(result.get("ok")),
                    "goal_success": bool(result.get("ok")),
                    "error_code": result.get("error_code"),
                    "detail": result.get("detail"),
                    "sim_seconds": max(0.0, float(self._session.world.data.time) - recording["origin"]),
                })
        finally:
            self._action_recording = None

    def _preview(self, event, world, metadata):
        """Called only by the physics owner, with the isolated trial's MjData."""
        now = time.monotonic()
        if event == "begin":
            self._experiments.begin(metadata)
            self._preview_failed = False
            self._preview_wall_origin = now
            self._preview_sim_origin = float(world.data.time)
            self._preview_last_time = -1.0
        elapsed = float(world.data.time) - self._preview_sim_origin
        if self.render_frames and not self._preview_failed:
            # Follow simulation time at 1x; Event.wait makes Stop interruptible.
            while event != "begin" and not self._cancel.is_set():
                delay = self._preview_wall_origin + elapsed - time.monotonic()
                if delay <= 0:
                    break
                self._cancel.wait(min(delay, .02))
                if not self.headless and time.monotonic() - self._last_sync >= 1 / 30:
                    self._sync_viewer()
            if event in ("begin", "end") or elapsed - self._preview_last_time >= .1 - 1e-9:
                try:
                    if self._preview_renderer is None:
                        from .rendering import WorldRenderer

                        self._preview_renderer = WorldRenderer()
                    frame = self._preview_renderer.render(world)
                    self._experiments.append(metadata["id"], frame, elapsed)
                    self._preview_last_time = elapsed
                except Exception as exc:
                    log.exception("Trial rendering failed")
                    self._preview_failed = True
                    self._experiments.rendering_error(metadata["id"], str(exc))
        if event == "end":
            self._experiments.end(metadata["id"], metadata)
        if not self.headless and time.monotonic() - self._last_sync >= 1 / 30:
            self._sync_viewer()
        if time.monotonic() - self._last_publish >= .1:
            self._publish()

    def _state_token(self):
        """Hash owner-thread physics without forward(), which changes solver state."""
        world = self._session.world
        digest = hashlib.sha256()
        digest.update(world.spec.model_dump_json().encode())
        digest.update(str(world.revision).encode())
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(world.model, kind))
        mujoco.mj_getState(world.model, world.data, state, kind)
        digest.update(state.tobytes())
        # Compiled asset mesh/texture buffers are immutable for this model instance;
        # spec captures their identities. Include all mutable physical parameters.
        digest.update(str(id(world.model)).encode())
        for name in dir(world.model):
            if name.startswith(("_", "mesh_", "tex_", "skin_", "bvh_", "name_", "names", "paths")):
                continue
            value = getattr(world.model, name)
            if isinstance(value, np.ndarray):
                digest.update(name.encode())
                digest.update(value.tobytes())
        for name in dir(world.model.opt):
            if name.startswith("_"):
                continue
            value = getattr(world.model.opt, name)
            if isinstance(value, np.ndarray):
                digest.update(name.encode())
                digest.update(value.tobytes())
            elif isinstance(value, (int, float)):
                digest.update(json.dumps([name, value]).encode())
        return digest.hexdigest()

    def _render_frame(self):
        world = self._session.world
        if not self.render_frames or self._render_failed_model is world.model:
            return
        try:
            if self._renderer is None:
                from .rendering import WorldRenderer

                self._renderer = WorldRenderer()
            frame = self._renderer.render(world)
            with self._lock:
                self._frame = frame
                self._render_error = None
        except Exception as exc:
            log.exception("Browser rendering failed")
            with self._lock:
                self._frame = None
                self._render_error = str(exc)
            self._render_failed_model = world.model
        self._last_render = time.monotonic()

    def _completed(self, result):
        future = Future()
        future.set_result(result)
        return future

    def submit(self, name, args, *, expected_state_token=None):
        revision = self.snapshot()["scene_revision"]
        if name not in ARGUMENTS:
            return self._completed(
                error("unknown_tool", "Unknown tool name.", revision)
            )
        try:
            arguments = ARGUMENTS[name].model_validate(args)
        except (ValidationError, TypeError) as exc:
            return self._completed(error("invalid_arguments", str(exc), revision))
        if name in ("search_assets", "describe_asset"):
            from .catalog import search_assets, get_asset

            try:
                if name == "search_assets":
                    matches = search_assets(arguments.query, arguments.limit)
                    payload = {"assets": matches, "count": len(matches)}
                else:
                    payload = {"asset": get_asset(arguments.asset_id)}
                return self._completed(
                    {"ok": True, "scene_revision": revision, "payload": payload}
                )
            except (ValueError, KeyError) as exc:
                return self._completed(error("unknown_asset", str(exc), revision))
        if name == "stop":
            return self._completed(self.stop())
        with self._lock:
            revision = self._snapshot["scene_revision"]
            if self._shutdown.is_set():
                return self._completed(
                    error("shutdown", "Simulation is closed.", revision)
                )
            if name == "observe_world":
                return self._completed(
                    {
                        "ok": True,
                        "scene_revision": revision,
                        "payload": deepcopy(self._snapshot),
                    }
                )
            if name == "list_actions":
                return self._completed(
                    {
                        "ok": True,
                        "scene_revision": revision,
                        "payload": {
                            "actions": deepcopy(self._snapshot.get("actions", []))
                        },
                    }
                )
            if name == "list_assets":
                return self._completed(
                    {
                        "ok": True,
                        "scene_revision": revision,
                        "payload": {"assets": deepcopy(CATALOG)},
                    }
                )
            if not self.ready.is_set():
                return self._completed(
                    error("not_ready", "Simulation is starting.", revision)
                )
            if self._busy:
                return self._completed(
                    error("busy", "An action is running. Wait or press Stop.", revision)
                )
            self._busy = True
            self._snapshot["busy"] = True
            self._cancel.clear()
            future = Future()
            self._queue.put_nowait((name, arguments, not args, future, expected_state_token))
            return future

    def stop(self):
        # Event delivery never waits for the physics thread or a provider response.
        with self._lock:
            self._cancel.set()
            return {
                "ok": True,
                "scene_revision": self._snapshot["scene_revision"],
                "payload": {"status": "Stop requested."},
            }

    def close(self):
        self._shutdown.set()
        self.stop()
        if self._thread and current_thread() is not self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Simulation did not shut down within five seconds.")

    def _publish(self, **updates):
        world = self._session.world
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(world.model, kind))
        mujoco.mj_getState(world.model, world.data, state, kind)
        try:
            observation = observe(world)
        finally:
            # Observation may run forward dynamics, including control callbacks.
            mujoco.mj_setState(world.model, world.data, state, kind)
        observation["state_token"] = self._state_token()
        lab = getattr(self._session, "action_lab", None)
        if lab is not None:
            observation["action_lab"] = lab.snapshot()
            observation["actions"] = (
                lab.list_actions().get("payload", {}).get("actions", [])
            )
        observation["rendering"] = {
            "available": self._frame is not None,
            "error": self._render_error,
        }
        with self._lock:
            for key in ("status", "last_result"):
                if key in self._snapshot:
                    observation[key] = self._snapshot[key]
            observation.update(updates)
            observation["busy"] = self._busy
            self._snapshot = deepcopy(observation)
        self._last_publish = time.monotonic()

    def _status(self, text):
        self._publish(status=text)

    def _open_viewer(self, world):
        import mujoco.viewer

        if self._viewer:
            self._viewer.close()
        self._viewer = None
        self._viewer_model = None
        # close() only requests exit. is_running() also becomes false before
        # mjpython releases its single UI slot, and there is no public join API.
        deadline = time.monotonic() + 5
        while not self._shutdown.is_set():
            if self._busy and self._cancel.is_set():
                return False  # The idle loop will reopen the updated world.
            try:
                self._viewer = mujoco.viewer.launch_passive(world.model, world.data)
                return True
            except RuntimeError as exc:
                if "another mujoco viewer is already open" not in str(exc).lower():
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "The previous MuJoCo viewer did not close within five seconds."
                    ) from exc
                wake = self._cancel if self._busy else self._shutdown
                wake.wait(0.02)
        return False

    def _sync_viewer(self):
        if self.headless:
            return
        world = self._session.world
        if self._viewer_model is not world.model:
            if not self._open_viewer(world):
                return
            general = hasattr(world, "observe")
            self._viewer.cam.lookat[:] = (
                world.model.stat.center if general else [0.35, 0, 0.3]
            )
            self._viewer.cam.distance = (
                max(1.5, world.model.stat.extent * 1.8) if general else 2.1
            )
            self._viewer.cam.azimuth = 180
            self._viewer.cam.elevation = -35
            self._viewer_model = world.model
            self._wall_origin, self._sim_origin = time.monotonic(), world.data.time
        if not self._viewer.is_running():
            self._shutdown.set()
            self._cancel.set()
        else:
            with self._viewer.lock():
                self._draw_overlays(self._viewer.user_scn)
            self._viewer.sync()
        self._last_sync = time.monotonic()

    def _draw_overlays(self, scene):
        """Viewer-only geometry; labels and routes never enter physics contacts."""
        scene.ngeom = 0
        identity = np.eye(3).ravel()
        bins = {} if hasattr(self._session.world, "observe") else BINS
        for name, center in bins.items():
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                center + [0, 0, 0.15],
                identity,
                np.ones(4),
            )
            geom.label = name.replace("_", " ").upper()
            scene.ngeom += 1

        current = self._session.world.paths
        previous = self._session.last_paths
        paths = current if previous == current else previous + current
        paths = paths[-4:]
        # Limit overlay work even if a future planner emits much denser routes.
        limit = min(scene.maxgeom, 130)
        for index, path in enumerate(paths):
            color = (
                [0.1, 0.9, 1.0, 0.9]
                if index == len(paths) - 1 and current
                else [0.55, 0.6, 0.66, 0.5]
            )
            points = path.get("points", [])
            for start, end in zip(points, points[1:]):
                if scene.ngeom >= limit:
                    return
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    identity,
                    np.array(color),
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    3,
                    np.asarray(start),
                    np.asarray(end),
                )
                scene.ngeom += 1

    def _tick(self):
        self._record_action_frame()
        now = time.monotonic()
        if self.render_frames and now - self._last_render >= 0.12:
            self._render_frame()
        if now - self._last_publish >= 0.1:
            self._publish()
        if not self.headless:
            if now - self._last_sync >= 1 / 30:
                self._sync_viewer()
        if not self.headless or self.render_frames:
            # Viewer replacement may reset the clock origin while synchronizing.
            delay = (
                self._wall_origin
                + self._session.world.data.time
                - self._sim_origin
                - time.monotonic()
            )
            if delay > 0:
                wake = self._cancel if self._busy else self._shutdown
                wake.wait(min(delay, 0.02))

    def run(self):
        """Own all simulation state here; native operation requires the main thread."""
        if not self.headless and current_thread() is not main_thread():
            raise RuntimeError("The native viewer must run on the main thread.")
        try:
            from .session import Session

            self._session = Session(build_scene())
            from .action_lab import ActionLab

            self._session.action_lab = ActionLab(self._session.world)
            self._render_frame()
            self._publish(status="Ready")
            self._sync_viewer()
            self.ready.set()
            while not self._shutdown.is_set():
                try:
                    command = self._queue.get(timeout=0.01)
                except Empty:
                    if not self.headless or self.render_frames:
                        world = self._session.world
                        if not hasattr(world, "observe"):
                            world.data.qfrc_applied[:7] = world.data.qfrc_bias[:7]
                            for _ in range(5):
                                mujoco.mj_step(world.model, world.data)
                        self._tick()
                    continue
                name, arguments, automatic_obstacle, future, expected_state_token = command
                result = error(
                    "cancelled",
                    "Action cancelled before execution.",
                    self._session.world.revision,
                )
                running = future.set_running_or_notify_cancel()
                try:
                    if running and expected_state_token is not None and expected_state_token != self._state_token():
                        result = error(
                            "stale_scene", "The live physics changed since this proposal was prepared.",
                            self._session.world.revision,
                        )
                    elif running and not self._cancel.is_set():
                        self._wall_origin = time.monotonic()
                        self._sim_origin = self._session.world.data.time
                        self._status(name.replace("_", " ").capitalize())
                        self._arm_action_recording(name, arguments)
                        self._session, result = dispatch(
                            self._session,
                            name,
                            arguments,
                            automatic_obstacle=automatic_obstacle,
                            cancel=self._cancel,
                            tick=self._tick,
                            status=self._status,
                            preview=self._preview,
                        )
                        if not hasattr(self._session, "action_lab"):
                            from .action_lab import ActionLab

                            self._session.action_lab = ActionLab(self._session.world)
                        self._sync_viewer()
                        self._render_frame()
                except ValueError as exc:
                    result = error(
                        "invalid_scene", str(exc), self._session.world.revision
                    )
                except Exception as exc:
                    log.exception("Simulation action failed")
                    result = error(
                        "execution_failed", str(exc), self._session.world.revision
                    )
                finally:
                    self._finish_action_recording(result)
                    with self._lock:
                        self._busy = False
                    self._publish(
                        last_result=result,
                        status="Ready"
                        if result["ok"]
                        else result.get("detail", "Action failed"),
                    )
                    if running:
                        future.set_result(result)
        except Exception as exc:
            self._startup_error = exc
            log.exception("Simulation runtime failed")
        finally:
            self._shutdown.set()
            self.ready.set()
            if self._renderer:
                self._renderer.close()
            if self._preview_renderer:
                self._preview_renderer.close()
            if self._viewer:
                self._viewer.close()
            while not self._queue.empty():
                _, _, _, future, _ = self._queue.get_nowait()
                if not future.done():
                    future.set_result(
                        error(
                            "shutdown",
                            "Simulation is closed.",
                            self.snapshot()["scene_revision"],
                        )
                    )
