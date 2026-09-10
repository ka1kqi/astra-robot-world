"""Opt-in Astra extraction rehearsal on a saved scene copy; uses API credits, no HTTP/viewer instance."""
import argparse
import asyncio
import json
from pathlib import Path

from astra_world.astra import AstraAdapter
from astra_world.simulation import SimulationRuntime


async def main():
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', default='before_expandable_trace')
    args = parser.parse_args()
    import astra_world.action_lab as lab_module
    lab_module.ACTIONS_DIR = Path('artifacts/extraction-rehearsal-actions').resolve()
    runtime = SimulationRuntime(headless=True).start()
    calls = []
    allowed = {'observe_world', 'describe_asset', 'search_assets', 'list_assets', 'list_actions',
               'search_action_notes', 'read_action_notes', 'write_action_note', 'check_approaches',
               'draft_action', 'test_action', 'save_action'}
    try:
        loaded = await asyncio.wrap_future(runtime.submit('load_scenario', {'name': args.scenario}))
        assert loaded['ok'], loaded
        initial = runtime.snapshot()['state_token']
        async def execute(name, arguments):
            assert name in allowed, f'Rehearsal forbids live mutation: {name}'
            result = await asyncio.wrap_future(runtime.submit(name, arguments))
            calls.append({'name': name, 'arguments': arguments, 'result': result})
            Path('artifacts/extraction-rehearsal.json').write_text(json.dumps(calls, indent=2))
            payload = result.get('payload', {})
            print(name, result['ok'], result.get('error_code', ''), payload.get('state', ''), flush=True)
            if name == 'check_approaches':
                print([(c['name'], c['clear'], c.get('error_code')) for c in payload.get('candidates', [])], flush=True)
            if name == 'test_action':
                print([(t['trial_number'], t['goal_success'], t.get('error_code')) for t in payload.get('trials', [])], flush=True)
            return result
        history = [{'role': 'user', 'content':
            'Learn how to extract blue_block from underneath red_block in this tray. Red may drop onto tray. '
            'Preserve green and tray. Work only in sandbox copies; do not run an action live or rebuild the scene. '
            'Use the extraction goal, preflight candidate approaches before physical trials, and at most 6 physical trials. '
            'Prior measured attempts: downward-wrist slow +X push .06m/s carried the entire stack 11.6cm; '
            'fast .2m/s and off-center +X pushes collided red with hand; horizontal wrist and -Y approaches were blocked. '
            'Consider a side grasp and horizontal extraction if clearance permits; change strategy using diagnostics. '
            'Save only if verified. If no candidate works, summarize specific measured blockers honestly.'}]
        answer = await AstraAdapter.from_env().run_turn(history, execute)
        assert runtime.snapshot()['state_token'] == initial, 'Source physics changed'
        assert any(c['name'] == 'check_approaches' for c in calls), 'No approach preflight'
        assert any(c['name'] == 'draft_action' and c['arguments']['goal']['kind'] == 'extract' and c['result']['ok'] for c in calls), 'No accepted extraction goal'
        print(answer, flush=True)
        print('Rehearsal completed; source physics unchanged. See artifacts/extraction-rehearsal.json for outcomes.', flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    asyncio.run(main())
