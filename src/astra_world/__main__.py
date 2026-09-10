"""Launch with uv run mjpython -m astra_world, or python with --headless."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import threading
import time

from .simulation import SimulationRuntime


def demo(runtime, destination):
    """An explicitly offline tool rehearsal; it does not call a language model."""
    records = []
    commands = [
        ("build_sorting_station", {"seed": 7}),
        ("sort_blocks", {"color": "red", "destination_id": "left_bin"}),
        ("add_obstacle", {}),
        ("retry_last_task", {}),
    ]
    for name, args in commands:
        started = time.monotonic()
        result = runtime.submit(name, args).result(timeout=600)
        records.append(
            {
                "tool": name,
                "arguments": args,
                "result": result,
                "wall_seconds": round(time.monotonic() - started, 3),
            }
        )
        print(
            f"{name}: {'ok' if result['ok'] else result.get('error_code')}", flush=True
        )
    report = {
        "mode": "offline_tool_rehearsal",
        "seed": 7,
        "versions": {
            name: version(name) for name in ("mujoco", "numpy", "scipy", "pydantic")
        },
        "calls": records,
        "final_world": runtime.snapshot(),
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Demo log: {path.resolve()}", flush=True)
    return all(record["result"]["ok"] for record in records)


def main():
    parser = argparse.ArgumentParser(
        description="Astra Robot World: native physics and local chat."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run physics without the native viewer."
    )
    parser.add_argument(
        "--native-viewer",
        action="store_true",
        help="Also open the native MuJoCo window; requires mjpython.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the four offline tool calls, save a log and exit.",
    )
    parser.add_argument(
        "--log",
        default="artifacts/demo.json",
        help="Destination for the offline demo JSON log.",
    )
    args = parser.parse_args()
    runtime = SimulationRuntime(
        headless=not args.native_viewer or args.headless or args.demo,
        render_frames=not args.demo,
    )
    if args.demo:
        runtime.start()
        try:
            ok = demo(runtime, args.log)
        finally:
            runtime.close()
        raise SystemExit(0 if ok else 1)

    import uvicorn
    from .server import create_app

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(runtime),
            host="127.0.0.1",
            port=args.port,
            log_level="info",
            access_log=False,
        )
    )

    def serve():
        runtime.ready.wait()
        if runtime.snapshot()["scene_revision"]:
            try:
                server.run()
            finally:
                runtime.close()

    network = threading.Thread(target=serve, name="local-http", daemon=True)
    network.start()
    print(f"Chat and manual controls: http://127.0.0.1:{args.port}", flush=True)
    try:
        runtime.run()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
        server.should_exit = True
        network.join(timeout=5)
    if runtime._startup_error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
