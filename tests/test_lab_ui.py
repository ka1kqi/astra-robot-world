"""Exercise actual browser rendering/polling functions with a minimal DOM."""
import subprocess
from pathlib import Path


def test_lab_progress_updates_and_manual_replay_selection_survives():
    source = Path('web/app.js').read_text()
    rendering = source[source.index('function renderLab('):source.index('async function loadActions(')]
    library = source[source.index('async function loadActions('):source.index('async function request(')]
    polling = source[source.index('async function loadExperiments('):source.index('function resetFrame(')]
    script = r'''
const assert = require('node:assert/strict');
class Element {
  constructor(text = '') { this.textContent = text; this.children = []; this.dataset = {}; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  querySelectorAll(selector) {
    const descendants = this.children.flatMap(c => [c, ...c.querySelectorAll('all')]);
    return selector === 'all' ? descendants : descendants.filter(c => c.open && c.dataset.labKey);
  }
  setAttribute() {}
  get allText() { return this.textContent + this.children.map(c => c.allText || '').join(' '); }
}
const elements = new Map();
const $ = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
const document = {createDocumentFragment: () => new Element()};
const node = (tag, text = '', cls) => new Element(text);
const human = value => String(value || '').replaceAll('_', ' ');
const format = value => String(value);
const programDetails = () => new Element('program');
let libraryPending = false, lastLibraryPoll = 0, busy = false, lastActions = '';
const controls = () => {};
let lastLab = '', labSnapshot, labStatus = '', trialClips = [], selectedTrialId = 'old';
let previewError = '', latestTrialId = 'old', replayIndex = 4, replaying = true, view = 'experiment';
const trialActive = trial => trial && ['testing', 'running', 'recording'].includes(trial.state);
const selectedTrial = () => trialClips.find(trial => trial.id === selectedTrialId);
const trialLabel = trial => `Trial ${trial.trial_number}`;
const setView = next => { view = next; };
const resetFrame = () => {};
const renderPreview = () => {};
let response;
const fetch = async () => ({ok: true, json: async () => response});
'''
    checks = r'''
(async () => {
  labSnapshot = {drafts: [{draft_id: 'exp', name: 'extract', state: 'testing', trials: [{trial_number:1, goal_success:false}], program: {}, trials_used: 1, trial_budget: 6}]};
  labStatus = 'Step 2/4: move_to_pose.';
  trialClips = [{id: 'old', state: 'failed', experiment_id: 'exp', trial_number: 1}];
  $('follow-trials').checked = false;
  response = {latest_trial_id: 'new', trials: [...trialClips, {id: 'new', state: 'testing', experiment_id: 'exp', name: 'extract', trial_number: 2, trial_budget: 6, duration: 1.2}]};
  await loadExperiments();
  assert.equal(selectedTrialId, 'old');
  assert.equal(replayIndex, 4);
  assert.equal(replaying, true);
  assert.equal(String($('action-count').textContent), '1');
  assert.match($('trial-results').allText, /Trial 2/);
  assert.match($('trial-results').allText, /1.2 s/);
  assert.match($('trial-results').allText, /Step 2\/4/);
  const details = $('trial-results').querySelectorAll('all').find(c => c.dataset.labKey === 'exp:trial:1');
  details.open = true;
  response.trials[1].duration = 2.4;
  await loadExperiments();
  assert.match($('trial-results').allText, /2.4 s/);
  assert.equal($('trial-results').querySelectorAll('all').find(c => c.dataset.labKey === 'exp:trial:1').open, true);
  assert.equal(selectedTrialId, 'old');
  $('follow-trials').checked = true;
  response = {latest_trial_id: 'next', trials: [...response.trials.slice(0,1), {id: 'next', state: 'testing', experiment_id: 'exp', name: 'extract', trial_number: 3, trial_budget: 6, duration: 0}]};
  await loadExperiments();
  assert.equal(selectedTrialId, 'next');
  assert.equal(replayIndex, null);
  assert.equal(replaying, false);
  assert.match($('trial-results').allText, /Trial 3/);
  const savedLab = labSnapshot;
  labSnapshot = {drafts: []};
  await loadExperiments();
  assert.equal(String($('action-count').textContent), '1');
  assert.match($('trial-results').allText, /Testing now/);
  labSnapshot = savedLab;
  response.trials[1].state = 'failed';
  await loadExperiments();
  assert.doesNotMatch($('trial-results').allText, /Testing now/);
  response = {payload: {actions: []}};
  await loadActions(true);
  assert.equal(String($('action-count').textContent), '1');
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(['node', '-e', script + rendering + polling + library + checks], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_live_action_replay_uses_recorded_get_frames_and_preserves_trial_selection():
    source = Path('web/app.js').read_text()
    playback = source[source.index('function selectedTrial('):source.index('async function refresh(')]
    script = r'''
const assert = require('node:assert/strict');
class Element {
  constructor(text = '') { this.textContent = text; this.children = []; this.dataset = {}; this.listeners = {}; this.classList = {toggle() {}}; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  setAttribute() {}
  addEventListener(name, fn) { this.listeners[name] = fn; }
  get allText() { return this.textContent + this.children.map(c => c.allText || '').join(' '); }
}
const elements = new Map();
const $ = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
const document = {hidden: false, querySelector: () => new Element(), querySelectorAll: () => [], createTextNode: text => new Element(text)};
const node = (tag, text = '') => new Element(text);
const human = value => String(value || '').replaceAll('_', ' ');
let trialClips = [{id:'trial-old',trial_number:1,state:'failed',frame_count:2}], selectedTrialId = 'trial-old', latestTrialId = 'trial-old';
let actionClip = null, previewError = '', actionReplayError = '', view = 'live', replayIndex = null, replaying = false;
let notebookNotes = [], frameGeneration = 0, frameObjectUrl = null, frameTimer, rendererError = '', labSnapshot, labStatus;
const renderLab = () => {};
const setTimeout = () => 0, clearTimeout = () => {};
URL.createObjectURL = () => 'blob:frame'; URL.revokeObjectURL = () => {};
const requests = [];
let metadata = {clip: {id:'action-one',name:'lift cube',tool:'pick_place',state:'succeeded',ok:true,frame_count:3,duration:1,truncated:true}};
const fetch = async (url, options) => {
  requests.push({url, options});
  return {ok:true,json:async () => metadata,blob:async () => new Blob(['jpeg'])};
};
'''
    checks = r'''
(async () => {
  await loadActionReplay();
  assert.equal($('view-action').disabled, false);
  $('follow-trials').checked = true;
  $('view-action').listeners.click();
  assert.equal(view, 'action');
  assert.equal($('follow-trials').checked, false);
  assert.equal(selectedTrialId, 'trial-old');
  assert.match($('view-badge').allText, /RECORDED LIVE ACTION/);
  assert.match($('preview-result').textContent, /Partial recording/);
  assert.equal($('trial-select').hidden, true);
  await loadFrame();
  const frame = requests.find(request => request.url.startsWith('/frame.jpg'));
  assert.match(frame.url, /view=action/);
  assert.match(frame.url, /clip_id=action-one/);
  assert.match(frame.url, /frame=0/);
  $('replay-frame').value = '2';
  $('replay-frame').listeners.input();
  assert.equal(replayIndex, 2);
  assert.equal(replaying, false);
  await loadFrame();
  assert.match(requests.at(-1).url, /frame=2/);
  metadata = {clip: {...actionClip, id:'action-two', state:'testing'}};
  await loadActionReplay();
  assert.equal(replayIndex, null);
  assert.equal(replaying, false);
  assert.equal($('view-action').disabled, true);
  $('view-experiment').listeners.click();
  assert.equal(view, 'experiment');
  assert.equal(selectedTrialId, 'trial-old');
  $('view-live').listeners.click();
  assert.equal(view, 'live');
  assert(requests.every(request => !request.options.method || request.options.method === 'GET'));
  assert(requests.every(request => !request.url.includes('/tools/') && request.url !== '/chat'));
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run(['node', '-e', script + playback + checks], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
