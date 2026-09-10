"""Opt-in live rehearsal against the running app on8765; uses API credits. Requires a red/green tower."""

import argparse
import httpx, time, json, hashlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scenario", help="Optionally load this saved scenario before the rehearsal."
)
args = parser.parse_args()
c = httpx.Client(base_url="http://127.0.0.1:8765", timeout=10)
for _ in range(100):
    try:
        if c.get("/experiments").status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(0.2)
else:
    raise RuntimeError("new server unavailable")


def wait(tid):
    deadline = time.monotonic() + 310
    while time.monotonic() < deadline:
        state = c.get("/state").json()
        turn = next(t for t in state["turns"] if t["id"] == tid)
        if turn["status"] != "running":
            print("Turn", turn["mode"], turn["status"], flush=True)
            assert turn["status"] == "completed", turn
            return state
        time.sleep(0.2)
    raise RuntimeError("timeout")


def post(path, data):
    r = c.post(path, json=data)
    r.raise_for_status()
    return r.json()


if args.scenario:
    wait(post("/tools/load_scenario", {"name": args.scenario})["turn_id"])
world = c.get("/state").json()["world"]
original = world["state_token"]
request = "Knock over the red and green tower. During experimentation, start with a 5 mm horizontal contact stroke from 9 cm left of the green block, then revise the stroke if that fails."
response = post("/actions/propose", {"message": request, "trial_budget": 5})
state = wait(response["turn_id"])
proposal = next(p for p in state["proposals"] if p["id"] == response["proposal_id"])
print("Proposal", proposal["status"], proposal["interpretation"], flush=True)
assert proposal["status"] == "ready", proposal
assert state["world"]["state_token"] == original
turn_id = post("/actions/start", {"proposal_id": proposal["id"]})["turn_id"]
print("Experiment started", flush=True)
state = wait(turn_id)
turn = next(t for t in state["turns"] if t["id"] == turn_id)
draft_id = next(
    event["result"]["payload"]["draft_id"]
    for event in turn["events"]
    if event["type"] == "tool_result" and event["name"] == "draft_action"
)
trials = [
    trial for trial in c.get("/experiments").json()["trials"]
    if trial["experiment_id"] == draft_id
]
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/visible-learning-live.json").write_text(
    json.dumps(
        {"proposal": proposal, "turn": turn, "trials": trials}, indent=2
    )
)
assert state["world"]["state_token"] == original, "Live physical state changed"
assert len(trials) >= 4, trials
assert trials[0]["goal_success"] is False
assert all(t["goal_success"] for t in trials[-3:])
for trial in (trials[0], trials[-1]):
    assert trial["frame_count"] > 10, trial
    images = [
        c.get(
            "/frame.jpg",
            params={"view": "experiment", "trial_id": trial["id"], "frame": i},
        ).content
        for i in [0, trial["frame_count"] // 2, trial["frame_count"] - 1]
    ]
    assert all(x.startswith(b"\xff\xd8") for x in images)
    assert len({hashlib.sha256(x).hexdigest() for x in images}) >= 2
    Path("artifacts/trial-" + trial["state"] + ".jpg").write_bytes(images[1])
print(
    "Free text -> review -> failed visible trial -> revised/confirmed action passed. Live physics preserved.",
    flush=True,
)
