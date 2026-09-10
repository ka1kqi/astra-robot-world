"use strict";
const $ = (id) => document.getElementById(id);
let connected = false;
let configured = false;
let busy = false;
let pending = false;
let generalWorld = false;
let pandaWorld = false;
let lastTurns = "";
let proposals = [];
let lastProposal = "";
let lastProposalRequest = "";
let editingProposalId = null;
let trialClips = [];
let selectedTrialId = null;
let latestTrialId = null;
let previewError = "";
let actionClip = null;
let actionReplayError = "";
let view = "live";
let replaying = false;
let replayIndex = null;
let frameTimer;
let frameGeneration = 0;
let frameObjectUrl = null;
let notebookNotes = [];
let lastLab = "";
let labSnapshot;
let labStatus = "";
let lastActions = "";
let lastLibraryPoll = 0;
let pollPromise = null;
let libraryPending = false;
let rendererError = "";

function node(tag, text, cls) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (cls) element.className = cls;
  return element;
}
function human(value) {
  return String(value ?? "").replaceAll("_", " ");
}
function format(value) {
  if (typeof value === "number")
    return Number.isFinite(value) ? value.toFixed(3) : "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.map(format).join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value ?? "—");
}
function controls() {
  const blocked = !connected || busy || pending;
  $("send").disabled = blocked || !configured;
  document
    .querySelectorAll("[data-tool], [data-action-name]")
    .forEach((button) => {
      const tool = button.dataset.tool;
      button.disabled =
        blocked ||
        (["simulate", "reset_world"].includes(tool) && !generalWorld) ||
        (["sort_blocks", "add_obstacle", "retry_last_task"].includes(tool) &&
          generalWorld) ||
        (Boolean(button.dataset.actionName) && !pandaWorld);
    });
  $("create-action").disabled =
    blocked || !configured || !pandaWorld || !$("action-message").value.trim();
  document.querySelectorAll("[data-start-proposal]").forEach((button) => {
    button.disabled = blocked || !configured || !pandaWorld;
  });
  $("stop").disabled = !connected;
  $("composer-hint").textContent = configured
    ? "Enter to send · Shift + Enter for a new line"
    : "Connect a model to chat. Scene tools are available.";
  let hint = "Trials use a scene copy. Saved Run moves your live robot.";
  if (!pandaWorld)
    hint = "Build a Panda workspace in Conversation to create actions.";
  else if (!configured)
    hint = "Connect a model to let Astra create and test an action.";

  $("action-hint").textContent = hint;
}
function selectTab(name, focus = false) {
  for (const tab of ["chat", "actions"]) {
    const active = tab === name;
    $(`${tab}-tab`).setAttribute("aria-selected", String(active));
    $(`${tab}-tab`).tabIndex = active ? 0 : -1;
    $(`${tab}-panel`).hidden = !active;
  }
  if (focus) $(`${name}-tab`).focus();
}
for (const name of ["chat", "actions"]) {
  $(`${name}-tab`).addEventListener("click", () => selectTab(name));
  $(`${name}-tab`).addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let next = name === "chat" ? "actions" : "chat";
    if (event.key === "Home") next = "chat";
    if (event.key === "End") next = "actions";
    selectTab(next, true);
  });
}
$("open-action").addEventListener("click", () => {
  selectTab("actions");
  $("action-message").focus({ preventScroll: true });
  $("actions-panel").scrollIntoView({ block: "nearest", behavior: "auto" });
});

function traceDetails(label, key, openKeys) {
  const details = node("details", undefined, "trace-details");
  details.dataset.traceKey = key;
  details.open = openKeys.has(key);
  details.append(node("summary", label));
  return details;
}
function traceParameters(value, key, openKeys, label = "Parameters") {
  const details = traceDetails(label, key, openKeys);
  details.append(node("pre", JSON.stringify(value, null, 2), "trace-json"));
  return details;
}
function traceProgram(program, key, openKeys, completed) {
  const details = traceDetails(
    `Motion sequence · ${program.steps.length} steps`,
    key,
    openKeys,
  );
  program.steps.forEach((step, index) => {
    let suffix = " · submitted";
    if (step.status) suffix = ` · ${step.status}`;
    else if (Number.isInteger(completed))
      suffix = index < completed ? " · completed" : " · not completed";
    const item = traceDetails(
      `${index + 1}. ${step.op}${suffix}`,
      `${key}/${index}`,
      openKeys,
    );
    const { op, ...parameters } = step;
    item.append(
      traceParameters(parameters, `${key}/${index}/params`, openKeys),
    );
    details.append(item);
  });
  return details;
}
function renderTurns(turns) {
  const serialized = JSON.stringify(turns);
  if (serialized === lastTurns || !turns.length) return;
  lastTurns = serialized;
  const transcript = $("transcript");
  const scrollTop = transcript.scrollTop;
  const openKeys = new Set(
    [...transcript.querySelectorAll("details[open][data-trace-key]")].map(
      (el) => el.dataset.traceKey,
    ),
  );
  const atBottom =
    transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight <
    80;
  const fragment = document.createDocumentFragment();
  for (const turn of turns) {
    const section = node("article", undefined, "turn");
    let label = "YOU";
    if (turn.mode === "manual") label = "MANUAL TOOL";
    if (["action", "action_lab"].includes(turn.mode)) label = "ACTION LAB";
    section.append(
      node("p", `${label} · ${human(turn.status).toUpperCase()}`, "turn-label"),
    );
    section.append(node("p", turn.message, "user-message"));
    for (const [index, event] of (turn.events || []).entries()) {
      let text = event.text || "";
      let cls = `event ${event.type}`;
      const key = `${turn.id}/${event.call_id || index}/${event.type}`;
      if (event.type === "tool_start") {
        const details = traceDetails(`↗ ${human(event.name)}`, key, openKeys);
        if (event.arguments !== undefined) {
          details.append(
            traceParameters(
              event.arguments,
              `${key}/args`,
              openKeys,
              "Tool parameters",
            ),
          );
          if (event.arguments.program?.steps)
            details.append(
              traceProgram(event.arguments.program, `${key}/program`, openKeys),
            );
        } else {
          details.append(
            node(
              "p",
              "Arguments were not recorded for this older call.",
              "muted",
            ),
          );
        }
        section.append(details);
        continue;
      }
      if (event.type === "tool_result") {
        const result = event.result || {};
        text =
          result.detail ||
          result.payload?.detail ||
          result.payload?.message ||
          (result.ok
            ? `${human(event.name)} completed.`
            : human(result.error_code || "Action failed"));
        if (result.payload?.route) text += ` Route: ${result.payload.route}.`;
        if (!result.ok) cls += " failure";
        const details = traceDetails(text, key, openKeys);
        details.classList.add("tool-outcome");
        if (!result.ok) details.classList.add("failure");
        const payload = result.payload || {};
        if (payload.program?.steps) {
          details.append(
            traceProgram(payload.program, `${key}/candidate`, openKeys),
          );
        }
        if (payload.command_trace?.length)
          details.append(
            traceProgram(
              { steps: payload.command_trace },
              `${key}/commands`,
              openKeys,
            ),
          );
        const reports = [
          ...(payload.trials || []).map((t) => ({
            label: `Trial ${t.trial_number}${t.confirmation ? " · confirmation" : ""}`,
            report: t,
          })),
          ...(payload.trial
            ? [{ label: "Preflight trial", report: payload.trial }]
            : []),
          ...(payload.result
            ? [{ label: "Live execution", report: payload.result }]
            : []),
        ];
        reports.forEach(({ label, report }, reportIndex) => {
          if (!report.motion?.program?.steps) return;
          const trial = traceDetails(
            `${label} · ${report.goal_success ? "goal reached" : "goal not reached"}`,
            `${key}/report/${reportIndex}`,
            openKeys,
          );
          trial.append(
            traceProgram(
              report.motion.program,
              `${key}/report/${reportIndex}/program`,
              openKeys,
              report.motion.completed_steps,
            ),
          );
          details.append(trial);
        });
        details.append(
          traceParameters(
            result,
            `${key}/result`,
            openKeys,
            "Full tool result",
          ),
        );
        section.append(details);
        continue;
      }
      if (text) section.append(node("p", text, cls));
    }
    fragment.append(section);
  }
  transcript.replaceChildren(fragment);
  if (atBottom) transcript.scrollTop = transcript.scrollHeight;
  else transcript.scrollTop = scrollTop;
}
function renderEntities(entities) {
  const proposal = proposals.at(-1);
  const boundIds =
    proposal?.status === "ready"
      ? [
          proposal.goal?.object_id,
          proposal.goal?.support_id,
          proposal.goal?.supported_id,
          proposal.goal?.landing_id,
        ]
      : [];
  const fragment = document.createDocumentFragment();
  for (const entity of entities) {
    const row = node(
      "div",
      undefined,
      `entity-row${boundIds.includes(entity.id) ? " proposal-bound" : ""}`,
    );
    const dot = node("span", undefined, "entity-color");
    const color = entity.color;
    if (
      Array.isArray(color) &&
      color.length >= 3 &&
      color.slice(0, 3).every(Number.isFinite)
    ) {
      dot.style.backgroundColor = `rgb(${color
        .slice(0, 3)
        .map((n) => Math.max(0, Math.min(255, Math.round(n * 255))))
        .join(",")})`;
    } else if (typeof color === "string" && CSS.supports("color", color))
      dot.style.backgroundColor = color;
    const title = node("span", entity.id, "entity-title");
    if (entity.fixed) title.append(node("small", "fixed"));
    row.append(
      dot,
      title,
      node(
        "span",
        Array.isArray(entity.position)
          ? entity.position.map((n) => Number(n).toFixed(3)).join(" / ")
          : "—",
        "entity-position",
      ),
    );
    fragment.append(row);
  }
  if (!entities.length)
    fragment.append(
      node("p", "No objects yet. Ask Astra to build a scene.", "small"),
    );
  $("entity-list").replaceChildren(fragment);
}
function programDetails(program) {
  const details = node("details", undefined, "program-details");
  details.append(node("summary", "Inspect motion program"));
  const steps = program?.steps || [];
  const list = node("ol", undefined, "program-list");
  for (const step of steps) {
    const operation = human(step.op || step.operation || step.type || "step");
    const reference = step.reference_id
      ? ` relative to ${step.reference_id}`
      : "";
    const contact = step.contact_ids?.length
      ? ` · contact: ${step.contact_ids.join(", ")}`
      : "";
    list.append(node("li", `${operation}${reference}${contact}`));
  }
  if (steps.length) details.append(list);
  details.append(node("pre", JSON.stringify(program, null, 2)));
  return details;
}
function renderLab(lab, status = "") {
  const active = [...trialClips].reverse().find(trialActive);
  const serialized = JSON.stringify([lab, active, active ? status : ""]);
  if (serialized === lastLab) return;
  lastLab = serialized;
  const drafts = lab?.drafts || [];
  const experimentCount = drafts.length + (
    active && !drafts.some((draft) => draft.draft_id === active.experiment_id) ? 1 : 0
  );
  $("action-count").textContent = experimentCount;
  $("action-count").title = `${experimentCount} experiments`;
  if (!drafts.length && !active) {
    $("trial-results").replaceChildren(
      node(
        "p",
        "Your live world stays in place during trials. Successful motions need two confirmation runs before saving.",
        "small",
      ),
    );
    return;
  }
  const expanded = new Set(
    [...$("trial-results").querySelectorAll("details[open][data-lab-key]")]
      .map((item) => item.dataset.labKey),
  );
  const fragment = document.createDocumentFragment();
  if (active) {
    const progress = node("article", undefined, "trial-draft");
    progress.setAttribute("aria-live", "polite");
    progress.append(
      node("strong", `Testing now · ${human(active.name || active.experiment_id)}`),
      node("p", `Trial ${active.trial_number}${active.trial_budget ? ` / ${active.trial_budget}` : ""}${active.confirmation ? " · confirmation" : ""} · ${Number(active.duration || 0).toFixed(1)} s simulated`, "trial-budget"),
      node("p", status || "Testing motion in a scene copy…", "small"),
    );
    fragment.append(progress);
  }
  const orderedDrafts = [...drafts].reverse().sort((a, b) =>
    Number(b.draft_id === active?.experiment_id) - Number(a.draft_id === active?.experiment_id),
  );
  for (const draft of orderedDrafts) {
    const section = node("article", undefined, "trial-draft");
    const heading = node("div", undefined, "draft-heading");
    heading.append(
      node("strong", human(draft.name)),
      node("span", human(draft.state), "trial-state"),
    );
    section.append(
      heading,
      node(
        "p",
        `${draft.trials_used ?? draft.trials?.length ?? 0} / ${draft.trial_budget} trials · ${draft.verified ? "confirmed on fresh scene copies" : "isolated from your live world"}`,
        "trial-budget",
      ),
    );
    for (const trial of draft.trials || []) {
      const card = node(
        "details",
        undefined,
        `trial-card${trial.goal_success ? " success" : ""}`,
      );
      card.dataset.labKey = `${draft.draft_id}:trial:${trial.trial_number}`;
      card.open = expanded.has(card.dataset.labKey);
      const outcome = trial.goal_success
        ? "Goal reached"
        : trial.detail || human(trial.error_code) || "Goal not reached";
      card.append(
        node(
          "summary",
          `Trial ${trial.trial_number}${trial.confirmation ? " · confirmation" : ""} — ${outcome}`,
        ),
      );
      const measurements = node("dl", undefined, "measurements");
      for (const [key, value] of Object.entries(trial.measurements || {})) {
        const entry = node("div");
        entry.append(node("dt", human(key)), node("dd", format(value)));
        measurements.append(entry);
      }
      card.append(measurements);
      if (trial.motion?.detail) card.append(node("p", trial.motion.detail));
      section.append(card);
    }
    if (!(draft.trials || []).length)
      section.append(
        node(
          "p",
          draft.state === "testing"
            ? "Running the first candidate in the sandbox…"
            : "Waiting for Astra’s first motion candidate…",
          "small",
        ),
      );
    if (draft.program) {
      const program = programDetails(draft.program);
      program.dataset.labKey = `${draft.draft_id}:program`;
      program.open = expanded.has(program.dataset.labKey);
      section.append(program);
    }
    if (draft.best_result && !draft.verified) {
      const best = draft.best_result.measurements?.goal_score;
      if (best !== undefined)
        section.append(
          node("p", `Best measured goal score: ${format(best)}`, "small"),
        );
    }
    fragment.append(section);
  }
  $("trial-results").replaceChildren(fragment);
}
async function loadActions(force = false) {
  if (
    libraryPending ||
    (!force && (busy || Date.now() - lastLibraryPoll < 4000))
  )
    return;
  libraryPending = true;
  lastLibraryPoll = Date.now();
  try {
    const response = await fetch("/actions", { cache: "no-store" });
    if (!response.ok)
      throw new Error(
        "Could not load saved actions. Select Refresh to try again.",
      );
    const result = await response.json();
    if (result.ok === false)
      throw new Error(result.detail || "Could not load saved actions.");
    const payload = result.payload || result;
    const actions = payload.actions || [];
    const serialized = JSON.stringify(payload);
    if (serialized === lastActions) return;
    lastActions = serialized;
    const fragment = document.createDocumentFragment();
    for (const action of actions) {
      const article = node("article", undefined, "saved-action");
      const heading = node("div", undefined, "saved-heading");
      const run = node("button", "▷ Run live");
      run.type = "button";
      run.dataset.actionName = action.name;
      run.addEventListener("click", async () => {
        selectTab("chat");
        await request("/tools/run_action", { name: action.name, bindings: {} });
      });
      heading.append(node("strong", human(action.name)), run);
      article.append(
        heading,
        node(
          "p",
          action.verified
            ? "✓ Verified in this action’s test scene"
            : "Verification unavailable",
          "verified-label",
        ),
      );
      if (action.goal)
        article.append(
          node(
            "p",
            action.goal.kind === "circle"
              ? `Circle · ${action.goal.plane?.toUpperCase() || "XY"} plane · ${format(action.goal.radius)} m radius`
              : `${human(action.goal.kind)} · ${action.goal.object_id || "target"}${action.goal.support_id ? ` / ${action.goal.support_id}` : ""}`,
          ),
        );
      if (action.program) article.append(programDetails(action.program));
      fragment.append(article);
    }
    if (!actions.length)
      fragment.append(
        node(
          "p",
          "Tested actions will appear here, ready to run in the live world.",
          "small",
        ),
      );
    for (const invalid of payload.invalid_actions || [])
      fragment.append(node("p", `${invalid.name}: ${invalid.error}`, "small"));
    $("saved-actions").replaceChildren(fragment);
  } catch (error) {
    lastActions = "";
    $("saved-actions").replaceChildren(node("p", error.message, "small"));
  } finally {
    libraryPending = false;
    controls();
  }
}
async function request(path, body = {}) {
  if (path !== "/stop" && (pending || busy)) return false;
  $("error").textContent = "";
  pending = true;
  controls();
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok || result.ok === false)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "The request could not run. Check the selected objects and try again.",
      );
    await poll();
    return true;
  } catch (error) {
    $("error").textContent =
      error.message || "Could not reach the local simulation.";
    return false;
  } finally {
    pending = false;
    controls();
  }
}
function poll() {
  if (pollPromise) return pollPromise;
  pollPromise = updateState().finally(() => {
    pollPromise = null;
  });
  return pollPromise;
}
async function updateState() {
  try {
    const response = await fetch("/state", { cache: "no-store" });
    if (!response.ok) throw new Error("State unavailable");
    const state = await response.json();
    connected = true;
    configured = Boolean(state.provider?.configured);
    busy = Boolean(state.busy);
    if (!busy) $("proposal-progress").hidden = true;
    $("provider").textContent =
      state.provider?.message || "Provider status unavailable.";
    const world = state.world || {};
    let activity = "Ready when you are";
    if (busy)
      activity = world.busy
        ? world.status || "Running the simulation…"
        : "Waiting for Astra’s response…";
    const lastEvent = state.turns?.at(-1)?.events?.at(-1);
    if (busy && !world.busy && lastEvent?.type === "provider_retry") activity = lastEvent.text;
    $("action-status").textContent = activity;
    generalWorld = world.kind === "general";
    pandaWorld = generalWorld && world.robot?.type === "panda";
    rendererError = world.rendering?.error || state.rendering?.error || "";
    $("world-kind").textContent = generalWorld
      ? "Physics scene"
      : "Panda station";
    $("scene-name").textContent = world.name || "Sorting station";
    $("robot-name").textContent =
      !generalWorld || pandaWorld ? "Franka Panda" : "No robot";
    $("revision").textContent = world.scene_revision ?? world.revision ?? "—";
    $("simulation-time").textContent =
      `t ${Number(world.sim_time || 0).toFixed(2)} s`;
    const objects = generalWorld
      ? world.entities || []
      : world.blocks || world.objects || [];
    const entities = Array.isArray(objects)
      ? objects
      : Object.entries(objects).map(([id, item]) => ({ id, ...item }));
    $("object-count").textContent = entities.length;
    $("world-state").textContent = JSON.stringify(world, null, 2);
    renderProposal(state.proposals || []);
    renderEntities(entities);
    renderTurns(state.turns || []);
    labSnapshot = world.action_lab || state.action_lab;
    labStatus = world.status || "";
    await loadExperiments();
    await loadActionReplay();
    void loadActions();
  } catch (_) {
    connected = false;
    $("action-status").textContent = "Connection interrupted";
    $("provider").textContent =
      "Cannot reach the local server. Start Astra Robot World; this page reconnects automatically.";
  }
  $("connection-label").textContent = connected ? "Connected" : "Reconnecting";
  $("connection-dot").classList.toggle("ready", connected);
  $("provider-dot").classList.toggle("ready", configured && connected);
  controls();
}
$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("message").value.trim();
  if (!message || !configured || busy || pending || !connected) return;
  if (await request("/chat", { message })) $("message").value = "";
});
$("message").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});
$("create-action-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("action-message").value.trim();
  if (!message || $("create-action").disabled) return;
  lastProposalRequest = message;
  if (
    await request("/actions/propose", {
      message,
      trial_budget: Number($("trial-budget").value),
    })
  ) {
    $("proposal-progress").hidden =
      !busy || proposals.at(-1)?.status === "interpreting";
  }
});
$("action-message").addEventListener("input", controls);
document.querySelectorAll("[data-action-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    $("action-message").value = button.dataset.actionPrompt;
    $("action-message").focus();
    controls();
  });
});
$("refresh-actions").addEventListener("click", () => void loadActions(true));
document.querySelectorAll("[data-prompt]").forEach((button) =>
  button.addEventListener("click", () => {
    $("message").value = button.dataset.prompt;
    $("message").focus();
  }),
);
document.querySelectorAll("[data-tool]").forEach((button) =>
  button.addEventListener("click", () => {
    const name = button.dataset.tool;
    let args = {};
    if (name === "sort_blocks")
      args = { color: "red", destination_id: "left_bin" };
    else if (name === "build_sorting_station") args = { seed: 7 };
    else if (name === "simulate") args = { duration: 2 };
    void request(`/tools/${name}`, args);
  }),
);
async function searchAssets() {
  const results = $("asset-results");
  results.replaceChildren(node("p", "Searching the local catalog…", "small"));
  try {
    const response = await fetch(
      `/assets?query=${encodeURIComponent($("asset-query").value.trim())}`,
    );
    if (!response.ok) throw new Error("Could not load assets. Try again.");
    const result = await response.json();
    if (!result.ok) throw new Error(result.detail || "Asset search failed.");
    const fragment = document.createDocumentFragment();
    const assets = result.payload?.assets || [];
    for (const asset of assets) {
      const entry = node("article", undefined, "asset-result");
      entry.append(
        node("code", asset.id),
        node("p", asset.description || asset.name || "Scene asset"),
      );
      const dimensions = Array.isArray(asset.bounds)
        ? asset.bounds.map((n) => Number(n).toFixed(2)).join(" × ") + " m"
        : "";
      entry.append(
        node(
          "p",
          [
            dimensions,
            asset.readiness === "imported"
              ? "Imported · convex collision"
              : "Procedural",
          ]
            .filter(Boolean)
            .join(" · "),
        ),
      );
      fragment.append(entry);
    }
    if (!assets.length)
      fragment.append(
        node("p", "No matching assets. Try a broader search.", "small"),
      );
    results.replaceChildren(fragment);
  } catch (error) {
    results.replaceChildren(node("p", error.message, "small"));
  }
}
$("asset-search").addEventListener("submit", (event) => {
  event.preventDefault();
  void searchAssets();
});
$("stop").addEventListener("click", () => void request("/stop"));
$("fullscreen").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.querySelector(".viewport-panel").requestFullscreen();
  } catch (_) {
    $("error").textContent =
      "Your browser could not expand the simulation view.";
  }
});
function goalMeasurements(goal) {
  const entries = [];
  if (goal.kind === "circle") {
    entries.push(["Center (X / Y / Z)", `${format(goal.target_position)} m`]);
    entries.push([
      "Circle",
      `${goal.plane.toUpperCase()} plane · radius ${format(goal.radius)} m`,
    ]);
    entries.push([
      "Path tolerance",
      `${format(Math.min(0.012, goal.radius * 0.18))} m radial and out of plane`,
    ]);
    entries.push(["Required sweep", `${format(2 * Math.PI - 0.2)} radians`]);
  } else {
    entries.push(["Target object", goal.object_id]);
    if (goal.support_id) entries.push(["Leave this support", goal.support_id]);
    if (goal.kind === "displace") {
      entries.push([
        "Destination (X / Y / Z)",
        `${format(goal.target_position)} m`,
      ]);
      entries.push(["Position tolerance", `${format(goal.tolerance)} m`]);
    }
    if (goal.kind === "topple")
      entries.push([
        "Required contact",
        "Reach the ground and leave the support",
      ]);
    if (goal.kind === "extract") {
      entries.push(["Upper object allowed to drop", goal.supported_id]);
      entries.push(["Upper object lands on", goal.landing_id]);
      entries.push([
        "Minimum extraction distance",
        `${format(goal.min_displacement)} m`,
      ]);
      entries.push([
        "Required outcome",
        "Lower object moves out; upper loses its support and lands on the named surface; both settle.",
      ]);
      if (goal.target_position)
        entries.push([
          "Lower object destination",
          `${format(goal.target_position)} m ± ${format(goal.tolerance)} m`,
        ]);
    }
    entries.push([
      "Settle for",
      "0.5 s · speed ≤ 0.015 m/s · rotation ≤ 0.2 rad/s",
    ]);
  }
  if (goal.preserve_ids?.length) {
    entries.push(["Keep in place", goal.preserve_ids.join(", ")]);
    entries.push([
      "Maximum movement",
      `${format(goal.preserve_tolerance)} m for each preserved object`,
    ]);
  }
  return entries;
}
function renderProposal(items) {
  proposals = items;
  const proposal = items.at(-1);
  const serialized = JSON.stringify(proposal);
  if (serialized === lastProposal) return;
  lastProposal = serialized;
  const container = $("action-proposal");
  container.replaceChildren();
  if (!proposal || proposal.id === editingProposalId) return;
  $("proposal-progress").hidden = true;
  const card = node("article", undefined, `proposal-card ${proposal.status}`);
  const heading = node("div", undefined, "section-heading");
  const statuses = {
    ready: "REVIEW OUTCOME",
    needs_clarification: "DETAIL NEEDED",
    unsupported: "CAPABILITY GAP",
    started: "EXPERIMENT STARTED",
    interpreting: "INTERPRETING",
    stale: "SCENE CHANGED",
    failed: "PROPOSAL FAILED",
    cancelled: "CANCELLED",
  };
  heading.append(
    node("h3", human(proposal.name || "Proposed action")),
    node(
      "span",
      statuses[proposal.status] || human(proposal.status),
      "sandbox-badge",
    ),
  );
  card.append(
    heading,
    node(
      "p",
      proposal.interpretation || "Review the details below.",
      "proposal-interpretation",
    ),
  );
  if (proposal.status === "stale") {
    card.append(
      node(
        "p",
        "The live scene changed after this proposal. Edit the request and propose it again to check the current scene.",
        "proposal-clarification",
      ),
    );
  }
  if (proposal.clarification)
    card.append(node("p", proposal.clarification, "proposal-clarification"));
  if (proposal.goal) {
    const measurements = node(
      "dl",
      undefined,
      "measurements proposal-measurements",
    );
    for (const [label, value] of goalMeasurements(proposal.goal)) {
      const entry = node("div");
      entry.append(node("dt", label), node("dd", value));
      measurements.append(entry);
    }
    card.append(measurements);
  }
  const limitations = Array.isArray(proposal.limitations)
    ? proposal.limitations
    : [proposal.limitations];
  for (const limitation of limitations.filter(Boolean))
    card.append(node("p", limitation, "proposal-limitation"));
  card.append(
    node(
      "p",
      `${proposal.trial_budget} trials maximum · scene revision ${proposal.scene_revision}. Trials use a scene copy; your live world stays unchanged.`,
      "small",
    ),
  );
  const buttons = node("div", undefined, "proposal-buttons");
  if (proposal.status === "ready") {
    const start = node("button", "Start experiments", "primary");
    start.type = "button";
    start.dataset.startProposal = proposal.id;
    start.addEventListener("click", async () => {
      if (await request("/actions/start", { proposal_id: proposal.id })) {
        $("follow-trials").checked = true;
        setView("experiment");
      }
    });
    buttons.append(start);
  }
  const edit = node(
    "button",
    proposal.status === "started" ? "Propose another action" : "Edit request",
    "text-button",
  );
  edit.type = "button";
  edit.addEventListener("click", () => {
    $("action-message").value =
      proposal.message || proposal.request || lastProposalRequest;
    $("trial-budget").value = String(proposal.trial_budget);
    editingProposalId = proposal.id;
    container.replaceChildren();
    $("action-message").focus();
    controls();
  });
  buttons.append(edit);
  card.append(buttons);
  container.append(card);
}
function selectedTrial() {
  return trialClips.find((trial) => trial.id === selectedTrialId);
}
function selectedRecording() {
  return view === "action" ? actionClip : selectedTrial();
}
function trialActive(trial) {
  return trial && ["testing", "running", "recording"].includes(trial.state);
}
function trialOutcome(trial) {
  if (trialActive(trial)) return "Testing motion in a scene copy…";
  if (trial.goal_success) return "Goal reached";
  if (trial.state === "cancelled") return "Cancelled";
  const detail = trial.detail || human(trial.error_code);
  return detail ? `Goal not reached · ${detail}` : "Goal not reached";
}
function trialLabel(trial) {
  return `Trial ${trial.trial_number}${trial.confirmation ? " · confirmation" : ""} · ${human(trial.name || trial.experiment_id)}`;
}
function setView(next, manual = false) {
  view = next;
  replaying = false;
  if ((manual && next === "live") || next === "action") $("follow-trials").checked = false;
  $("view-live").setAttribute("aria-pressed", String(next === "live"));
  $("view-experiment").setAttribute(
    "aria-pressed",
    String(next === "experiment"),
  );
  $("view-action").setAttribute("aria-pressed", String(next === "action"));
  $("experiment-preview").hidden = next === "live";
  document
    .querySelector(".viewport-panel")
    .classList.toggle("show-experiment", next !== "live");
  renderPreview();
  resetFrame();
}
function renderPreview() {
  const trial = selectedRecording();
  const action = view === "action";
  const active = trialActive(trial);
  const experiment = view !== "live";
  let badge = "LIVE WORLD";
  let frameStatus = "LIVE SIMULATION";
  if (experiment) {
    badge = trial && !active ? "TRIAL REPLAY" : "EXPERIMENT COPY";
    frameStatus = trial
      ? `${active ? "TESTING" : "REPLAY"} · TRIAL ${trial.trial_number}`
      : "WAITING FOR TRIAL";
  }
  if (action) {
    badge = "RECORDED LIVE ACTION";
    frameStatus = active ? "RECORDING LIVE ACTION" : "LIVE ACTION REPLAY";
  }
  $("view-badge").replaceChildren(node("i"), document.createTextNode(badge));
  let caption = experiment ? "Scene copy · live world unchanged" : "MuJoCo · live scene";
  if (action) caption = "Recorded frames · playback never moves the robot";
  $("view-caption").textContent = caption;
  $("frame-status").textContent = frameStatus;
  let title = trial ? trialLabel(trial) : "No recorded trials yet";
  if (action) title = trial
    ? `Last live action · ${human(trial.tool)} · ${human(trial.name)}`
    : "No live action recorded yet";
  $("preview-title").textContent = title;
  let previewState = "SCENE COPY";
  if (trial) previewState = active ? "TESTING COPY" : "RECORDED COPY";
  $("preview-state").textContent = action ? "LIVE ACTION RECORDING" : previewState;
  $("trial-select").hidden = action;
  $("replay-description").textContent = action
    ? "Playback only; the robot does not move. The last live action is kept until another action moves the robot. Only actions recorded during this session are available; recordings clear on restart."
    : "Trials run in a scene copy. Recordings clear on restart; experiment notes stay saved.";
  let result = "Start an experiment to watch the robot test a motion.";
  if (trial) {
    result = trialOutcome(trial);
    if (action) {
      result = trial.ok ? "Action completed" : trial.detail || human(trial.error_code) || "Action stopped";
      if (active) result = "Recording the current live action…";
    }
    if (trial.duration != null)
      result += ` · ${Number(trial.duration).toFixed(1)} s simulated`;
  }
  if (trial?.truncated) result += " · Partial recording: only the most recent frames were retained";
  $("preview-result").textContent = (action ? actionReplayError : previewError) || result;
  const note = [...notebookNotes]
    .sort((a, b) => b.id - a.id)
    .find(
      (entry) =>
        entry.experiment_id === trial?.experiment_id &&
        (entry.payload.trial_number === trial?.trial_number ||
          entry.payload.trial_number == null),
    );
  let hypothesis =
    "Astra’s hypothesis and revisions appear in the experiment notebook.";
  if (trial?.hypothesis) hypothesis = `Astra’s hypothesis: ${trial.hypothesis}`;
  else if (note) hypothesis = `Astra’s note: ${note.payload.text}`;
  $("preview-hypothesis").textContent = hypothesis;
  $("preview-hypothesis").hidden = action || !trial;
  const count = trial?.frame_count || 0;
  $("replay-toggle").disabled = !trial || active || count < 2;
  $("replay-toggle").textContent = replaying ? "Pause replay" : "Play replay";
  $("replay-frame").disabled = !trial || active || !count;
  $("replay-frame").max = Math.max(0, count - 1);
  $("replay-frame").value = replayIndex ?? Math.max(0, count - 1);
  $("replay-position").textContent = count
    ? `${(replayIndex ?? count - 1) + 1} / ${count} frames`
    : "0 frames";
  document.querySelectorAll(".step-controls button").forEach((button) => {
    button.title = experiment ? "This control affects the live world" : "";
    if (button.dataset.tool === "simulate")
      button.textContent = experiment ? "▷ Step live 2 s" : "▷ Step 2 s";
    else button.textContent = experiment ? "↺ Reset live" : "↺ Reset";
  });
}
async function loadExperiments() {
  try {
    const response = await fetch("/experiments", { cache: "no-store" });
    if (!response.ok)
      throw new Error(
        "Experiment preview unavailable. It will reconnect automatically.",
      );
    const data = await response.json();
    const prior = selectedTrial();
    trialClips = data.trials || [];
    previewError = data.error || "";
    const nextLatest = data.latest_trial_id;
    const isNew = nextLatest && nextLatest !== latestTrialId;
    const selectionExpired =
      selectedTrialId !== null &&
      !trialClips.some((trial) => trial.id === selectedTrialId);
    latestTrialId = nextLatest;
    if (selectionExpired || (isNew && $("follow-trials").checked)) {
      selectedTrialId = nextLatest || trialClips.at(-1)?.id || null;
      if (view !== "action") {
        replayIndex = null;
        replaying = false;
      }
      if (isNew && $("follow-trials").checked) setView("experiment");
      else if (view === "experiment") resetFrame();
    } else if (view !== "action" && trialActive(prior) && !trialActive(selectedTrial())) {
      replayIndex = null;
    }
    const select = $("trial-select");
    const signature = JSON.stringify(
      trialClips.map(({ id, state, goal_success }) => ({
        id,
        state,
        goal_success,
      })),
    );
    if (select.dataset.signature !== signature) {
      select.dataset.signature = signature;
      select.replaceChildren();
      for (const trial of [...trialClips].reverse()) {
        const option = node(
          "option",
          `${trialLabel(trial)} · ${trialActive(trial) ? "testing" : trial.goal_success ? "goal reached" : human(trial.state)}`,
        );
        option.value = trial.id;
        select.append(option);
      }
      if (!trialClips.length) {
        const placeholder = node("option", "No recordings yet");
        placeholder.value = "";
        select.append(placeholder);
      }
    }
    select.value = selectedTrialId || "";
    select.disabled = !trialClips.length;
  } catch (error) {
    previewError = error.message;
  }
  renderPreview();
  renderLab(labSnapshot, labStatus);
}
async function loadActionReplay() {
  try {
    const response = await fetch("/action-replay", { cache: "no-store" });
    if (!response.ok) throw new Error("Live action replay unavailable.");
    const data = await response.json();
    const changed = actionClip?.id !== data.clip?.id;
    actionClip = data.clip;
    actionReplayError = data.error || "";
    $("view-action").disabled = !actionClip?.frame_count || trialActive(actionClip);
    $("view-action").title = actionClip?.frame_count
      ? "Play recorded frames of the last live action; no robot motion"
      : "Available after the next live robot action is recorded";
    if (changed && view === "action") {
      replayIndex = null;
      replaying = false;
      resetFrame();
    }
  } catch (error) {
    actionReplayError = error.message;
  }
  renderPreview();
}
function resetFrame() {
  frameGeneration++;
  $("simulation-frame").hidden = true;
  $("frame-placeholder").hidden = false;
  $("frame-message").textContent =
    view === "live" ? "Loading live world" : view === "action" ? "Loading recorded live action" : "Loading experiment copy";
  $("frame-detail").textContent = "";
  scheduleFrame(0);
}
function scheduleFrame(delay) {
  clearTimeout(frameTimer);
  frameTimer = setTimeout(loadFrame, delay);
}
async function loadFrame() {
  if (document.hidden) return scheduleFrame(800);
  const generation = frameGeneration;
  const trial = selectedRecording();
  const action = view === "action";
  const experiment = view !== "live";
  if (experiment && !trial?.frame_count) {
    $("simulation-frame").hidden = true;
    $("frame-placeholder").hidden = false;
    $("frame-message").textContent = action
      ? "No live action recording available"
      : trial ? "Waiting for the first trial frame" : "No trial recording selected";
    $("frame-detail").textContent = action
      ? actionReplayError || "Recordings are available after a live robot action; playback never moves the robot."
      : previewError || "Start experiments in Action Lab to see motion in a separate scene copy.";
    return scheduleFrame(500);
  }
  const params = new URLSearchParams({ t: String(Date.now()) });
  if (experiment) {
    params.set("view", action ? "action" : "experiment");
    params.set(action ? "clip_id" : "trial_id", trial.id);
    if (replayIndex !== null && !trialActive(trial))
      params.set("frame", String(replayIndex));
  }
  try {
    const response = await fetch(`/frame.jpg?${params}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Frame unavailable");
    const blob = await response.blob();
    if (generation !== frameGeneration) return;
    const previousUrl = frameObjectUrl;
    frameObjectUrl = URL.createObjectURL(blob);
    const frame = $("simulation-frame");
    frame.src = frameObjectUrl;
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    frame.alt = experiment
      ? `Experiment scene copy, trial ${trial.trial_number}${trialActive(trial) ? " in progress" : " recording"}`
      : "Live rendered MuJoCo simulation of the current scene";
    if (action) frame.alt = `Recorded live action: ${human(trial.name)}`;
    frame.hidden = false;
    $("frame-placeholder").hidden = true;
    renderPreview();
    if (experiment && replaying) {
      if (replayIndex >= trial.frame_count - 1) {
        replaying = false;
        renderPreview();
      } else replayIndex++;
    }
    scheduleFrame(experiment && replaying ? 1000 / (trial.fps || 10) : 150);
  } catch (_) {
    if (generation !== frameGeneration) return;
    $("simulation-frame").hidden = true;
    $("frame-placeholder").hidden = false;
    $("frame-message").textContent = experiment
      ? "Recording view unavailable"
      : "Simulation view unavailable";
    $("frame-detail").textContent =
      (action ? actionReplayError : experiment ? previewError : rendererError) ||
      "Waiting for the renderer. The view reconnects automatically.";
    $("frame-status").textContent = "WAITING FOR RENDERER";
    scheduleFrame(1500);
  }
}
$("view-action").addEventListener("click", () => {
  if (!actionClip?.frame_count || trialActive(actionClip)) return;
  replayIndex = 0;
  setView("action", true);
  replaying = actionClip.frame_count > 1;
  renderPreview();
});
$("view-live").addEventListener("click", () => setView("live", true));
$("view-experiment").addEventListener("click", () =>
  setView("experiment", true),
);
$("follow-trials").addEventListener("change", () => {
  if ($("follow-trials").checked && latestTrialId) {
    selectedTrialId = latestTrialId;
    replayIndex = null;
    $("trial-select").value = selectedTrialId;
    setView("experiment");
  }
});
$("trial-select").addEventListener("change", () => {
  selectedTrialId = $("trial-select").value;
  $("follow-trials").checked = false;
  replayIndex = 0;
  setView("experiment");
});
$("replay-toggle").addEventListener("click", () => {
  const trial = selectedRecording();
  if (!trial || trialActive(trial)) return;
  replaying = !replaying;
  $("follow-trials").checked = false;
  if (replayIndex === null || replayIndex >= trial.frame_count - 1)
    replayIndex = 0;
  renderPreview();
  frameGeneration++;
  scheduleFrame(0);
});
$("replay-frame").addEventListener("input", () => {
  replaying = false;
  $("follow-trials").checked = false;
  replayIndex = Number($("replay-frame").value);
  renderPreview();
  frameGeneration++;
  scheduleFrame(0);
});

async function refresh() {
  await poll();
  setTimeout(refresh, 600);
}
void refresh();
scheduleFrame(0);

let notebookLastPoll = 0;
let notebookPending = false;
let notebookSerialized = "";
let notebookExperiment = null;
async function loadNotebook(force = false, experimentId = undefined) {
  if (notebookPending || (!force && Date.now() - notebookLastPoll < 5000))
    return;
  if (experimentId !== undefined) notebookExperiment = experimentId;
  notebookPending = true;
  notebookLastPoll = Date.now();
  try {
    const params = new URLSearchParams({ query: $("notebook-query").value });
    if (notebookExperiment) params.set("experiment_id", notebookExperiment);
    const response = await fetch("/action-notes?" + params, {
      cache: "no-store",
    });
    if (!response.ok)
      throw new Error("Could not read the experiment notebook.");
    const data = await response.json();
    const serialized = JSON.stringify(data.payload.entries);
    if (!force && serialized === notebookSerialized) return;
    notebookSerialized = serialized;
    const fragment = document.createDocumentFragment();
    const entries = data.payload.entries;
    const notes = new Map(notebookNotes.map((entry) => [entry.id, entry]));
    for (const entry of entries)
      if (entry.kind === "note") notes.set(entry.id, entry);
    notebookNotes = [...notes.values()].slice(-200);
    renderPreview();
    if (!entries.length)
      fragment.append(
        node(
          "p",
          "No matching records yet. New attempts and notes are saved automatically.",
          "small",
        ),
      );
    for (const entry of entries) {
      const card = node("details", "", "provider-details");
      const payload = entry.payload;
      const title =
        entry.kind === "note"
          ? "Astra note · interpretation"
          : `Simulator record · ${entry.kind}`;
      card.append(node("summary", title));
      card.append(
        node("p", new Date(entry.created_at).toLocaleString(), "small"),
      );
      if (entry.kind === "note") card.append(node("p", payload.text));
      else if (entry.kind === "trial")
        card.append(
          node(
            "p",
            `Trial ${payload.trial_number}: ${payload.report.goal_success ? "goal achieved" : "goal not met"}. ${payload.report.detail || ""}`,
          ),
        );
      else if (entry.kind === "draft")
        card.append(node("p", `${payload.name} · ${payload.goal.kind}`));
      const inspect = node("details");
      inspect.append(node("summary", "Inspect recorded data"));
      inspect.append(node("pre", JSON.stringify(payload, null, 2)));
      card.append(inspect);
      const button = node("button", "Read this experiment", "text-button");
      button.type = "button";
      button.addEventListener(
        "click",
        () => void loadNotebook(true, entry.experiment_id),
      );
      card.append(button);
      fragment.append(card);
    }
    $("notebook-entries").replaceChildren(fragment);
  } catch (error) {
    $("notebook-entries").replaceChildren(node("p", error.message, "small"));
  } finally {
    notebookPending = false;
  }
}
$("notebook-search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void loadNotebook(true, null);
});
$("refresh-notebook").addEventListener("click", () => void loadNotebook(true));
setInterval(() => void loadNotebook(), 5000);
void loadNotebook(true);
