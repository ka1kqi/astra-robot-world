"""Check the installed simulator, Panda assets, physics, and offscreen rendering."""

from pathlib import Path
import argparse
import platform
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Also open a viewer for 5 seconds (use mjpython on macOS)",
    )
    args = parser.parse_args()
    scene = ROOT / "assets/menagerie/franka_emika_panda/scene.xml"
    if not scene.exists():
        raise SystemExit(
            "Panda assets missing: run uv run python scripts/fetch_assets.py"
        )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        raise RuntimeError("Physics produced non-finite joint positions")
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.4]
    camera.distance = 2.2
    camera.azimuth = 135
    camera.elevation = -25
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    if pixels.std() < 1:
        raise RuntimeError("Renderer returned a blank image")
    output = ROOT / "artifacts/panda-smoke.ppm"
    output.parent.mkdir(exist_ok=True)
    output.write_bytes(b"P6\n640 480\n255\n" + pixels.tobytes())
    print(
        f"Python {platform.python_version()} ({platform.machine()}), MuJoCo {mujoco.__version__}"
    )
    print(
        f"Panda: {model.nq} position DOFs, {model.nu} actuators; 1000 steps, t={data.time:.3f}s"
    )
    print(f"Rendered {pixels.shape[1]}x{pixels.shape[0]} RGB image: {output}")
    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            deadline = time.monotonic() + 5
            while viewer.is_running() and time.monotonic() < deadline:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
        print("Native viewer check completed")


if __name__ == "__main__":
    main()
