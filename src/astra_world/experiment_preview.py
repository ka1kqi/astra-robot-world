"""Bounded, thread-safe JPEG replay storage; never owns MuJoCo state."""

from collections import OrderedDict, deque
from copy import deepcopy
from threading import Lock


class ExperimentClips:
    def __init__(self, max_bytes=64 * 1024 * 1024, max_trials=10, max_frames=300):
        self.max_bytes = max_bytes
        self.max_trials = max_trials
        self.max_frames = max_frames
        self._clips = OrderedDict()
        self._bytes = 0
        self._lock = Lock()
        self._error = None

    def _evict(self, keep=None):
        trial_id = next(identifier for identifier in self._clips if identifier != keep)
        clip = self._clips.pop(trial_id)
        self._bytes -= sum(len(frame) for frame in clip["frames"])

    def begin(self, metadata):
        with self._lock:
            while len(self._clips) >= self.max_trials:
                self._evict()
            self._clips[metadata["id"]] = {
                "metadata": {
                    **deepcopy(metadata),
                    "state": "testing",
                    "duration": 0.0,
                    "goal_success": None,
                    "error_code": None,
                    "fps": 10,
                    "dropped_frames": 0,
                    "truncated": False,
                },
                "frames": deque(),
                "times": deque(),
            }
            self._error = None

    def append(self, trial_id, frame, elapsed):
        frame = bytes(frame)
        with self._lock:
            clip = self._clips.get(trial_id)
            if clip is None or len(frame) > self.max_bytes:
                return
            while len(clip["frames"]) >= self.max_frames:
                self._bytes -= len(clip["frames"].popleft())
                clip["times"].popleft()
                clip["metadata"]["dropped_frames"] += 1
                clip["metadata"]["truncated"] = True
            while self._bytes + len(frame) > self.max_bytes and len(self._clips) > 1:
                self._evict(keep=trial_id)
            while self._bytes + len(frame) > self.max_bytes:
                self._bytes -= len(clip["frames"].popleft())
                clip["times"].popleft()
                clip["metadata"]["dropped_frames"] += 1
                clip["metadata"]["truncated"] = True
            clip["frames"].append(frame)
            clip["times"].append(float(elapsed))
            clip["metadata"]["duration"] = float(elapsed)
            self._bytes += len(frame)

    def end(self, trial_id, report):
        with self._lock:
            clip = self._clips.get(trial_id)
            if clip is None:
                return
            metadata = clip["metadata"]
            metadata.update(deepcopy(report))
            metadata["duration"] = report.get("sim_seconds", metadata["duration"])
            if report.get("error_code") == "cancelled":
                metadata["state"] = "cancelled"
            elif report.get("goal_success"):
                metadata["state"] = "succeeded"
            else:
                metadata["state"] = "failed"

    def rendering_error(self, trial_id, message):
        with self._lock:
            self._error = message
            if trial_id in self._clips:
                self._clips[trial_id]["metadata"]["render_error"] = message

    def metadata(self):
        with self._lock:
            trials = [
                {
                    **deepcopy(clip["metadata"]),
                    "frame_count": len(clip["frames"]),
                    "frame_times": list(clip["times"]),
                }
                for clip in self._clips.values()
            ]
            return {
                "trials": trials,
                "latest_trial_id": next(reversed(self._clips), None),
                "available": self._bytes > 0,
                "error": self._error,
            }

    def frame(self, trial_id=None, frame_index=None):
        with self._lock:
            if trial_id is None:
                trial_id = next(reversed(self._clips), None)
            clip = self._clips.get(trial_id)
            if clip is None or not clip["frames"]:
                return None
            if frame_index is None:
                return clip["frames"][-1]
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or not 0 <= frame_index < len(clip["frames"])
            ):
                return None
            return clip["frames"][frame_index]
