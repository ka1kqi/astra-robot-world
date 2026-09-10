"""Single-owner MuJoCo runtime. HTTP threads see only copied observations."""
from concurrent.futures import Future
from copy import deepcopy
import logging
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread, main_thread
import time

import mujoco
import numpy as np
from pydantic import ValidationError

from .assets import CATALOG
from .scene import BINS, build_scene, observe
from .tools import ARGUMENTS, dispatch, error

log = logging.getLogger(__name__)


class SimulationRuntime:
    def __init__(self, *, headless=False):
        self.headless = headless
        self._lock = Lock()
        self._queue = Queue(maxsize=1)
        self._cancel = Event()
        self._shutdown = Event()
        self.ready = Event()
        self._busy = False
        self._snapshot = {'scene_revision': 0, 'status': 'Starting simulation', 'busy': False}
        self._thread = None
        self._viewer = None
        self._viewer_model = None
        self._session = None
        self._startup_error = None
        self._last_publish = self._last_sync = 0.
        self._wall_origin = self._sim_origin = 0.

    def start(self):
        """Start a background owner for headless embedding and tests."""
        if not self.headless:
            raise RuntimeError('Run the native viewer with run() on the main thread under mjpython.')
        if self._thread is not None:
            raise RuntimeError('Runtime already started.')
        self._thread = Thread(target=self.run, name='mujoco-owner', daemon=True)
        self._thread.start()
        if not self.ready.wait(30):
            raise RuntimeError('Simulation startup timed out.')
        if self._startup_error:
            raise RuntimeError('Simulation startup failed.') from self._startup_error
        return self

    def snapshot(self):
        with self._lock:
            return deepcopy(self._snapshot)

    def _completed(self, result):
        future = Future()
        future.set_result(result)
        return future

    def submit(self, name, args):
        revision = self.snapshot()['scene_revision']
        if name not in ARGUMENTS:
            return self._completed(error('unknown_tool', 'Unknown tool name.', revision))
        try:
            arguments = ARGUMENTS[name].model_validate(args)
        except (ValidationError, TypeError) as exc:
            return self._completed(error('invalid_arguments', str(exc), revision))
        if name == 'stop':
            return self._completed(self.stop())
        with self._lock:
            revision = self._snapshot['scene_revision']
            if self._shutdown.is_set():
                return self._completed(error('shutdown', 'Simulation is closed.', revision))
            if name == 'observe_world':
                return self._completed({'ok': True, 'scene_revision': revision, 'payload': deepcopy(self._snapshot)})
            if name == 'list_assets':
                return self._completed({'ok': True, 'scene_revision': revision, 'payload': {'assets': deepcopy(CATALOG)}})
            if not self.ready.is_set():
                return self._completed(error('not_ready', 'Simulation is starting.', revision))
            if self._busy:
                return self._completed(error('busy', 'An action is running. Wait or press Stop.', revision))
            self._busy = True
            self._snapshot['busy'] = True
            self._cancel.clear()
            future = Future()
            self._queue.put_nowait((name, arguments, not args, future))
            return future

    def stop(self):
        # Event delivery never waits for the physics thread or a provider response.
        with self._lock:
            self._cancel.set()
            return {'ok': True, 'scene_revision': self._snapshot['scene_revision'],
                    'payload': {'status': 'Stop requested.'}}

    def close(self):
        self._shutdown.set()
        self.stop()
        if self._thread and current_thread() is not self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError('Simulation did not shut down within five seconds.')

    def _publish(self, **updates):
        observation = observe(self._session.world)
        with self._lock:
            for key in ('status', 'last_result'):
                if key in self._snapshot:
                    observation[key] = self._snapshot[key]
            observation.update(updates)
            observation['busy'] = self._busy
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
                if 'another mujoco viewer is already open' not in str(exc).lower():
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError('The previous MuJoCo viewer did not close within five seconds.') from exc
                wake = self._cancel if self._busy else self._shutdown
                wake.wait(.02)
        return False

    def _sync_viewer(self):
        if self.headless:
            return
        world = self._session.world
        if self._viewer_model is not world.model:
            if not self._open_viewer(world):
                return
            self._viewer.cam.lookat[:] = [.35, 0, .3]
            self._viewer.cam.distance = 2.1
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
        for name, center in BINS.items():
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                               center + [0, 0, .15], identity, np.ones(4))
            geom.label = name.replace('_', ' ').upper()
            scene.ngeom += 1

        current = self._session.world.paths
        previous = self._session.last_paths
        paths = current if previous == current else previous + current
        paths = paths[-4:]
        # Limit overlay work even if a future planner emits much denser routes.
        limit = min(scene.maxgeom, 130)
        for index, path in enumerate(paths):
            color = [.1, .9, 1., .9] if index == len(paths)-1 and current else [.55, .6, .66, .5]
            points = path.get('points', [])
            for start, end in zip(points, points[1:]):
                if scene.ngeom >= limit:
                    return
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                                   np.zeros(3), np.zeros(3), identity, np.array(color))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3,
                                    np.asarray(start), np.asarray(end))
                scene.ngeom += 1

    def _tick(self):
        now = time.monotonic()
        if now - self._last_publish >= .1:
            self._publish()
        if not self.headless:
            if now - self._last_sync >= 1/30:
                self._sync_viewer()
            # Viewer replacement may reset the clock origin while synchronizing.
            delay = self._wall_origin + self._session.world.data.time - self._sim_origin - time.monotonic()
            if delay > 0:
                wake = self._cancel if self._busy else self._shutdown
                wake.wait(min(delay, .02))

    def run(self):
        """Own all simulation state here; native operation requires the main thread."""
        if not self.headless and current_thread() is not main_thread():
            raise RuntimeError('The native viewer must run on the main thread.')
        try:
            from .session import Session
            self._session = Session(build_scene())
            self._publish(status='Ready')
            self._sync_viewer()
            self.ready.set()
            while not self._shutdown.is_set():
                try:
                    command = self._queue.get(timeout=.01)
                except Empty:
                    if not self.headless:
                        world = self._session.world
                        world.data.qfrc_applied[:7] = world.data.qfrc_bias[:7]
                        for _ in range(5):
                            mujoco.mj_step(world.model, world.data)
                        self._tick()
                    continue
                name, arguments, automatic_obstacle, future = command
                result = error('cancelled', 'Action cancelled before execution.', self._session.world.revision)
                running = future.set_running_or_notify_cancel()
                try:
                    if running and not self._cancel.is_set():
                        self._wall_origin = time.monotonic()
                        self._sim_origin = self._session.world.data.time
                        self._status(name.replace('_', ' ').capitalize())
                        self._session, result = dispatch(self._session, name, arguments,
                            automatic_obstacle=automatic_obstacle, cancel=self._cancel,
                            tick=self._tick, status=self._status)
                        self._sync_viewer()
                except ValueError as exc:
                    result = error('invalid_scene', str(exc), self._session.world.revision)
                except Exception as exc:
                    log.exception('Simulation action failed')
                    result = error('execution_failed', str(exc), self._session.world.revision)
                finally:
                    with self._lock:
                        self._busy = False
                    self._publish(last_result=result, status='Ready' if result['ok'] else result.get('detail', 'Action failed'))
                    if running:
                        future.set_result(result)
        except Exception as exc:
            self._startup_error = exc
            log.exception('Simulation runtime failed')
        finally:
            self._shutdown.set()
            self.ready.set()
            if self._viewer:
                self._viewer.close()
            while not self._queue.empty():
                *_, future = self._queue.get_nowait()
                if not future.done():
                    future.set_result(error('shutdown', 'Simulation is closed.', self.snapshot()['scene_revision']))
