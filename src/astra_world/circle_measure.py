"""Measure a complete end-effector circle from physics samples, not commands."""

import numpy as np


def evaluate_circle(trace, goal):
    points = np.asarray(trace, dtype=float)
    failed = {
        "success": False,
        "swept_radians": 0.0,
        "required_radians": float(2 * np.pi - 0.20),
        "radial_and_plane_tolerance": min(0.012, goal.radius * 0.18),
        "detail": "No complete measured circle within tolerance.",
    }
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 12
        or not np.isfinite(points).all()
    ):
        return failed
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[goal.plane]
    relative = points - np.asarray(goal.target_position)
    radial_error = np.abs(np.linalg.norm(relative[:, axes[:2]], axis=1) - goal.radius)
    plane_error = np.abs(relative[:, axes[2]])
    tolerance = min(0.012, goal.radius * 0.18)
    valid = (radial_error <= tolerance) & (plane_error <= tolerance)
    best = 0.0
    first = None
    for index, good in enumerate(np.r_[valid, False]):
        if good and first is None:
            first = index
        elif not good and first is not None:
            if index - first >= 12:
                arc = relative[first:index]
                theta = np.unwrap(np.arctan2(arc[:, axes[1]], arc[:, axes[0]]))
                increments = np.diff(theta)
                sweep = abs(float(theta[-1] - theta[0]))
                # Dense samples forbid an endpoint jump masquerading as a circle.
                if np.max(np.abs(increments), initial=0.0) < 0.35:
                    best = max(best, sweep)
            first = None
    return {
        "success": best >= 2 * np.pi - 0.20,
        "swept_radians": best,
        "required_radians": float(2 * np.pi - 0.20),
        "radial_and_plane_tolerance": tolerance,
        "detail": "Measured complete gripper circle."
        if best >= 2 * np.pi - 0.20
        else failed["detail"],
    }
