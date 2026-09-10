from fastapi.testclient import TestClient
from urllib.parse import urlsplit
import pytest

from astra_world.astra import AstraAdapter
from astra_world.server import create_app


class Runtime:
    def __init__(self):
        self.stop_calls = 0

    def snapshot(self):
        return {'scene_revision': 1}

    def stop(self):
        self.stop_calls += 1
        return {'ok': True, 'scene_revision': 1}


@pytest.mark.parametrize('host', [
    'evil.example', '127.0.0.1.evil.example', 'localhost.evil.example',
    'localhost@evil.example', 'evil.example@localhost', 'localhost/path',
    'localhost:bad', '[::1]:bad', '[::1]evil.example', 'localhost:0',
])
def test_foreign_or_malformed_host_is_rejected_even_without_origin(host):
    runtime = Runtime()
    with TestClient(create_app(runtime, adapter=AstraAdapter()), base_url='http://127.0.0.1:8765') as client:
        assert client.get('/state', headers={'host': host}).status_code == 400
        assert client.post('/stop', headers={'host': host}).status_code == 400
        assert runtime.stop_calls == 0


@pytest.mark.parametrize('origin', [
    'https://evil.example', 'http://127.0.0.1.evil.example:8765', 'null', '',
    'http://127.0.0.1:9000', 'https://127.0.0.1:8765',
    'http://evil.example@127.0.0.1:8765', 'http://127.0.0.1:8765/path',
])
def test_cross_origin_stop_never_reaches_runtime(origin):
    runtime = Runtime()
    with TestClient(create_app(runtime, adapter=AstraAdapter()), base_url='http://127.0.0.1:8765') as client:
        result = client.post('/stop', headers={'origin': origin})
        assert result.status_code == 403
        assert runtime.stop_calls == 0


@pytest.mark.parametrize('base', ['http://localhost:8765', 'http://127.0.0.1:8765', 'http://[::1]:8765', 'http://localhost'])
def test_same_origin_browser_and_originless_cli_remain_supported(base):
    runtime = Runtime()
    with TestClient(create_app(runtime, adapter=AstraAdapter()), base_url='http://127.0.0.1', headers={'host': urlsplit(base).netloc}) as client:
        assert client.get('/state').status_code == 200
        assert client.post('/stop', headers={'origin': base}).status_code == 200
        assert client.post('/stop').status_code == 200
        assert runtime.stop_calls == 2


def test_duplicate_host_or_origin_is_not_accepted():
    runtime = Runtime()
    with TestClient(create_app(runtime, adapter=AstraAdapter()), base_url='http://127.0.0.1:8765') as client:
        assert client.post('/stop', headers=[('host', '127.0.0.1:8765'), ('host', 'evil.example')]).status_code == 400
        assert client.post('/stop', headers=[('origin', 'http://127.0.0.1:8765'), ('origin', 'https://evil.example')]).status_code == 403
        assert runtime.stop_calls == 0
