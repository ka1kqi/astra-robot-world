"""One app process owns localhost:8765, embedded physics, and optional native viewer."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765"


def health():
    try:
        with urlopen(URL + "/health", timeout=1) as response:
            result = json.load(response)
        if (
            result.get("app") == "astra-robot-world"
            and result.get("project") == str(ROOT)
            and isinstance(result.get("pid"), int)
            and result["pid"] > 1
        ):
            return result
    except (OSError, ValueError):
        pass
    return None


def port_busy():
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", 8765)) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native",
        action="store_true",
        help="Also open the native MuJoCo window, connected to this same app.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Gracefully restart this project’s app if already running. Save your scene first.",
    )
    args = parser.parse_args()
    current = health()
    if current and not args.restart:
        print(
            f"Already running: {URL} (process {current['pid']}). Reusing the existing app; no second simulator started."
        )
        if args.native and not current["native_viewer"]:
            print(
                "To enable the desktop window, save your scene and run ./run.sh --restart --native."
            )
        return
    if current and args.restart:
        os.kill(current["pid"], signal.SIGINT)
        deadline = time.monotonic() + 10
        while port_busy() and time.monotonic() < deadline:
            time.sleep(0.1)
    if port_busy():
        raise SystemExit(
            "Port 8765 is occupied by an unrecognized or still-stopping server. Close that process before launching; no extra instance was started."
        )
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with (ROOT / "artifacts" / "launcher.lock").open("a") as lock:
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                existing = health()
                if existing:
                    print(f"Already running: {URL}. No second simulator started.")
                    return
                if time.monotonic() > deadline:
                    raise SystemExit(
                        "Another launcher is starting or stopping. Wait a moment and retry."
                    )
                time.sleep(0.1)
        command = [
            "uv",
            "run",
            "mjpython" if args.native else "python",
            "-m",
            "astra_world",
            "--port",
            "8765",
        ]
        if args.native:
            command.append("--native-viewer")
        print(
            f"Starting one shared world at {URL}. "
            + (
                "Desktop viewer enabled."
                if args.native
                else "Simulation is embedded in the browser."
            ),
            flush=True,
        )
        process = subprocess.Popen(command, cwd=ROOT)
        try:
            raise SystemExit(process.wait())
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
