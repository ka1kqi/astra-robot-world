import pytest
from astra_world.action_notebook import ActionNotebook


def test_notes_and_failed_trials_survive_restart(tmp_path):
    book = ActionNotebook(tmp_path / "notebook.sqlite3")
    eid = "a" * 32
    book.record(eid, "draft", {"name": "topple", "goal": {"kind": "topple"}})
    book.record(
        eid,
        "trial",
        {
            "trial_number": 1,
            "program": {"steps": []},
            "report": {"goal_success": False, "error_code": "no_path"},
        },
    )
    book.add_note(eid, "Approach was blocked; try from the left.", 1)
    reopened = ActionNotebook(tmp_path / "notebook.sqlite3")
    rows = reopened.read(eid)
    assert [r["kind"] for r in rows] == ["draft", "trial", "note"]
    assert rows[1]["source"] == "simulator"
    assert rows[1]["payload"]["report"]["goal_success"] is False
    assert rows[2]["source"] == "astra"
    assert reopened.search("blocked")[0]["experiment_id"] == eid


def test_note_cannot_claim_a_nonexistent_trial(tmp_path):
    book = ActionNotebook(tmp_path / "notebook.sqlite3")
    with pytest.raises(ValueError):
        book.add_note("b" * 32, "Made up trial", 1)
    book.record("a" * 32, "draft", {"name": "test"})
    with pytest.raises(ValueError):
        book.add_note("a" * 32, "Made up trial", 7)
    assert len(book.read("a" * 32)) == 1


def test_runtime_records_failure_and_recalls_notes_after_restart(tmp_path, monkeypatch):
    from astra_world.simulation import SimulationRuntime
    import astra_world.action_lab as module

    monkeypatch.setattr(module, "ACTIONS_DIR", tmp_path)
    runtime = SimulationRuntime(headless=True).start()

    def command(name, args):
        return runtime.submit(name, args).result(20)

    try:
        assert command(
            "create_world", {"name": "Notebook test", "robot": "panda", "entities": []}
        )["ok"]
        draft = command(
            "draft_action",
            {
                "name": "circle_test",
                "goal": {"kind": "circle", "target_position": [0.45, 0, 0.3]},
            },
        )
        eid = draft["payload"]["draft_id"]
        assert command(
            "write_action_note",
            {
                "experiment_id": eid,
                "text": "Hypothesis: waiting alone will not draw a circle.",
            },
        )["ok"]
        trial = command(
            "test_action",
            {"draft_id": eid, "program": {"steps": [{"op": "wait", "seconds": 0.02}]}},
        )
        assert not trial["payload"]["verified"]
        assert command(
            "write_action_note",
            {
                "experiment_id": eid,
                "trial_number": 1,
                "text": "No circle was traced. Generate pose waypoints next.",
            },
        )["ok"]
    finally:
        runtime.close()
    runtime = SimulationRuntime(headless=True).start()
    try:
        rows = command("read_action_notes", {"experiment_id": eid})["payload"][
            "entries"
        ]
        recorded = next(r for r in rows if r["kind"] == "trial")
        assert not recorded["payload"]["report"]["goal_success"]
        assert recorded["payload"]["program"]["steps"][0]["op"] == "wait"
        assert rows[-1]["source"] == "astra"
        assert command("search_action_notes", {"query": "waypoints"})["payload"][
            "entries"
        ]
        assert not command("list_actions", {})["payload"]["actions"]
    finally:
        runtime.close()


def test_related_experiments_use_goal_not_null_measurement_field(tmp_path):
    book = ActionNotebook(tmp_path / "notebook.sqlite3")
    book.record("a" * 32, "draft", {"name": "tower", "goal": {"kind": "topple"}})
    book.record(
        "a" * 32,
        "trial",
        {"trial_number": 1, "report": {"measurements": {"circle": None}}},
    )
    assert not book.search("circle")
    assert not book.related("circle")
    book.record("b" * 32, "draft", {"name": "ring", "goal": {"kind": "circle"}})
    book.add_note("b" * 32, "Keep waypoints close together.")
    related = book.related("circle")
    assert related[0]["experiment_id"] == "b" * 32
    assert related[0]["entries"][-1]["source"] == "astra"
