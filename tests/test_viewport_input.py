import base64
from io import BytesIO
import json
import time

from PIL import Image
from fastapi.testclient import TestClient
import pytest

from astra_world.server import create_app
from astra_world.astra import AstraAdapter


def viewport():
    output = BytesIO()
    Image.new('RGB', (32, 24), 'green').save(output, format='JPEG')
    return {'image': 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode(),
            'view': 'experiment', 'frame_age_ms': 500}


class Runtime:
    def snapshot(self):
        return {'entities': [], 'status': 'Ready'}

    def stop(self):
        return {'ok': True}


@pytest.mark.parametrize('model,image_type', [('gpt-6-astra', 'input_image'), ('other', 'image_url')])
def test_viewport_reaches_provider_but_not_state_or_later_turns(model, image_type):
    class Adapter(AstraAdapter):
        def __init__(self):
            super().__init__('http://example.invalid', model, 'test')
            self.inputs = []

        async def run_turn(self, history, execute, emit):
            self.inputs.append(json.loads(json.dumps(self.request_body(history))))
            history.append({'role': 'assistant', 'content': 'Seen.'})

    adapter = Adapter()
    shot = viewport()
    with TestClient(create_app(Runtime(), adapter=adapter), base_url='http://127.0.0.1') as client:
        response = client.post('/chat', json={'message': 'What is screen right?', 'viewport': shot})
        assert response.status_code == 202
        for _ in range(100):
            state = client.get('/state').json()
            if not state['busy']:
                break
            time.sleep(.005)
        body = adapter.inputs[0]
        content = (body.get('input') or body['messages'])[-1]['content']
        assert content[-1]['type'] == image_type
        assert shot['image'] in json.dumps(content)
        assert 'experiment' in content[0]['text']
        assert 'not the current live state' in content[0]['text']
        assert shot['image'] not in json.dumps(state)
        assert state['turns'][-1]['viewport']['view'] == 'experiment'
        assert client.post('/chat', json={'message': 'And now?'}).status_code == 202
        for _ in range(100):
            if not client.get('/state').json()['busy']:
                break
            time.sleep(.005)
        assert shot['image'] not in json.dumps(adapter.inputs[-1])


@pytest.mark.parametrize('image', ['https://example.invalid/frame.jpg', 'data:image/jpeg;base64,bad', 'data:image/jpeg;base64,' + base64.b64encode(b'not jpeg').decode()])
def test_invalid_viewport_rejected_before_provider(image):
    with TestClient(create_app(Runtime(), adapter=AstraAdapter('', '', '')), base_url='http://127.0.0.1') as client:
        shot = viewport(); shot['image'] = image
        assert client.post('/chat', json={'message': 'look', 'viewport': shot}).status_code == 422


def test_browser_captures_displayed_image_and_skips_unavailable_or_disabled():
    import subprocess
    from pathlib import Path
    source = Path('web/app.js').read_text()
    capture = source[source.index('function captureViewport()'):source.index('$("chat-form").addEventListener')]
    script = r'''
const assert = require('node:assert/strict');
const frame = {hidden:false,complete:true,naturalWidth:1280,naturalHeight:960,dataset:{view:'action',loadedAt:String(Date.now()-500)}};
const option = {checked:true};
const $ = id => id === 'simulation-frame' ? frame : option;
let drawn;
const canvas = {getContext: () => ({drawImage: (...args) => {drawn=args;}}), toDataURL: () => 'data:image/jpeg;base64,test'};
const document = {createElement: () => canvas};
''' + capture + r'''
let shot=captureViewport();
assert.equal(shot.view,'action');assert.equal(shot.image,'data:image/jpeg;base64,test');
assert.equal(drawn[0],frame);assert.equal(canvas.width,960);assert.equal(canvas.height,720);
assert(shot.frame_age_ms>=500);
option.checked=false;assert.equal(captureViewport(),null);option.checked=true;
frame.hidden=true;assert.equal(captureViewport(),null);frame.hidden=false;
frame.complete=false;assert.equal(captureViewport(),null);frame.complete=true;
canvas.toDataURL=()=>{throw new Error('unavailable')};assert.equal(captureViewport(),null);
'''
    subprocess.run(['node','-e',script],check=True,capture_output=True,text=True)


@pytest.mark.parametrize('cancel', [False, True])
def test_images_removed_after_failure_or_cancel(cancel):
    import asyncio
    from threading import Event
    from astra_world.astra import ProviderError
    entered = Event()

    class Adapter(AstraAdapter):
        async def run_turn(self, history, execute, emit):
            self.history = history
            entered.set()
            if cancel:
                await asyncio.sleep(60)
            raise ProviderError('Provider unavailable')

    adapter = Adapter('http://example.invalid', 'gpt-6-astra', 'test')
    shot = viewport()
    with TestClient(create_app(Runtime(), adapter=adapter), base_url='http://127.0.0.1') as client:
        assert client.post('/chat', json={'message':'look','viewport':shot}).status_code == 202
        assert entered.wait(2)
        if cancel:
            client.post('/stop').raise_for_status()
        for _ in range(100):
            state = client.get('/state').json()
            if not state['busy']:
                break
            time.sleep(.005)
        assert state['turns'][-1]['status'] == ('cancelled' if cancel else 'failed')
        assert shot['image'] not in json.dumps(adapter.history)


def test_truncated_jpeg_is_rejected():
    from astra_world.viewport import ViewportInput
    from pydantic import ValidationError
    shot = viewport()
    raw = base64.b64decode(shot['image'].split(',', 1)[1])
    shot['image'] = 'data:image/jpeg;base64,' + base64.b64encode(raw[:-2]).decode()
    with pytest.raises(ValidationError):
        ViewportInput.model_validate(shot)
