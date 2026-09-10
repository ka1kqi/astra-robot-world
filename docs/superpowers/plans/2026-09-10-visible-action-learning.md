# Visible action learning implementation plan

User approved implementing the free-text proposal workflow with visible experimentation.

Goal: natural-language request -> reviewable measured-outcome proposal -> isolated visible trials -> notes/results -> verified saved action, with live world preserved.

## Scope and interfaces

- Free text initially compiles to existing topple/displace/circle goals, as first increment of free-text-action-design.md. Unsupported semantics and ambiguity are explicit; broader predicate families remain subsequent work, not silently approximated.
- Proposal endpoints: POST /actions/propose {message,trial_budget:5} returns202 turn_id; GET /state includes proposals [{id,status,interpretation,goal,name,clarification,limitations,scene_revision,trial_budget}]. POST /actions/start {proposal_id} accepts only ready immutable storedproposal against unchanged state; returns202turn_id. Clarification/editing reruns propose from request text, rather than letting clients rewrite evaluator data.
- Existing /actions/create remains backward compatible; new UI uses proposal flow.
- Preview: runtime.latest/frame access remains live bydefault. GET /frame.jpg?view=experiment&trial_id=<id>&frame=<index> selects recorded trial frames; omitting frame getslatest. GET /experiments returns {trials:[{id,experiment_id,trial_number,confirmation,state,frame_count,duration,goal_success,error_code,...}],latest_trial_id,available,error}. Frames stay bounded in memory and are cleared onrestart; notebook isdurable.
- Trial callbacks carry scratch world only on simulationowner thread. Physics context never leavesowner; HTTP gets immutableJPEGbytes/metadata. Renderingerrors cannot grantgoal success or mutate livephysics. Live nativewindow continuesliveworld.
- Render atbounded cadence and retain bounded clips for failure/success replay. Throttle visible trials modestly so human canwatch; headlesstests remainfast. Stop interrupts pacing promptly. Avoid extraHTTPinstances; use8765launcher.

## Tasks

- [x] Previewagent: action_lab optionalpreviewcallback begin/tick/end, SimulationRuntime separatepreviewrenderer/cache, safe boundedclip store, state_token forproposalstaleness, physicalisolation/replay/canceltests. Minimaldispatch previewhook intools.py.
- [x] UIagent: live/experiment switch, auto-follow currenttrial, trialmetadata, selectable clips andplayback; free-textrequest form ->proposalcard ->Start/Edit; notebookalongside; nohardcodedgoal dropdown.
- [x] Parent: proposalmodel/adapter, immutableproposal lifecycle/serverendpoints, independentread-only evaluation scope, previews HTTProutes, testsforstale/unsupported/no-mutation/startfreeze.
- [x] Parent: liveAstra free-textsuccess +visiblefailed/revisedtrial recording; browserproofand fullregressions; review and fixrealbugs; preserve userstate and restartonlysingleapp.

## Verification

Tests must show a proposal cannot execute tools ormutateworld; acceptance rejects stale world/reused IDs/unsupported goals; trial rendercallback sees moving scratch state whileliveqposisunchanged; bothsuccessandfailureclipsretainframes; boundedclip eviction; frameendpointvalidatesindices andview; notebook remainsdurable. Live rehearsal uses failing shortpushthenrevision, withmeasuredreports andrealJPEGframes. Existing122tests remainpassing.


## Completion evidence

- Full regression suite: `uv run pytest -q` — 139 passed (two dependency deprecation warnings).
- JavaScript syntax and whitespace checks passed.
- Live Astra rehearsal on the single app at localhost:8765: read-only proposal, failed short push, revised successful push, and two successful confirmation runs. Full live physics state token stayed unchanged. Failed and successful clips contained changing JPEG frames.
- Browser verification: actual trial following, failed/success replay, play/pause/seek, switching back to the live camera, proposal Edit, and 390 px layout. Browser console was empty.
- Review found an immediate-Stop task startup race; fixed and covered by a regression test.
- Product title is consistently “Astra Robot World”. Notes and saved motion programs persist; bounded in-memory recordings clear on restart.
