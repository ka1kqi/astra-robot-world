"use strict";
const $ = (id) => document.getElementById(id);
let lastTurns = "",
  connected = false,
  configured = false,
  busy = false;
function node(tag, text, cls) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (cls) element.className = cls;
  return element;
}
function controls() {
  $("send").disabled = !connected || !configured || busy;
  document
    .querySelectorAll("[data-tool]")
    .forEach((b) => (b.disabled = !connected || busy));
  $("stop").disabled = !connected;
  $("composer-hint").textContent = configured
    ? "Enter to run · Shift + Enter for a new line"
    : "Connect Astra to use chat. Manual tools are available.";
}
function renderTurns(turns) {
  const serialized = JSON.stringify(turns);
  if (serialized === lastTurns || !turns.length) return;
  lastTurns = serialized;
  const transcript = $("transcript");
  const atBottom =
    transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight <
    70;
  const fragment = document.createDocumentFragment();
  for (const turn of turns) {
    const section = node("article", undefined, "turn");
    section.append(
      node(
        "p",
        `${turn.mode === "manual" ? "MANUAL TOOL" : "YOU"} · ${turn.status.toUpperCase()}`,
        "turn-label",
      ),
    );
    section.append(node("p", turn.message, "user-message"));
    for (const event of turn.events) {
      let text = event.text || "",
        cls = `event ${event.type}`;
      if (event.type === "tool_start")
        text = `↗ ${event.name.replaceAll("_", " ")}`;
      if (event.type === "tool_result") {
        const result = event.result;
        text =
          result.detail ||
          result.payload?.detail ||
          result.payload?.message ||
          (result.ok
            ? `${event.name.replaceAll("_", " ")} completed.`
            : `${result.error_code || "Action failed"}`);
        if (result.payload?.route) text += ` Route: ${result.payload.route}.`;
        if (!result.ok) cls += " failure";
      }
      section.append(node("p", text, cls));
    }
    fragment.append(section);
  }
  transcript.replaceChildren(fragment);
  if (atBottom) transcript.scrollTop = transcript.scrollHeight;
}
async function request(path, body = {}) {
  $("error").textContent = "";
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok)
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "The request was invalid. Check your input.",
      );
    await poll();
    return true;
  } catch (error) {
    $("error").textContent =
      error.message || "Could not reach the local simulation.";
    return false;
  }
}
async function poll() {
  try {
    const response = await fetch("/state", { cache: "no-store" });
    if (!response.ok) throw new Error("State unavailable");
    const state = await response.json();
    connected = true;
    configured = state.provider.configured;
    busy = state.busy;
    $("provider").textContent = state.provider.message;
    $("action-status").textContent = state.busy ? "Action running" : "Ready";
    const world = state.world;
    $("revision").textContent = world.scene_revision ?? world.revision ?? "—";
    const objects = world.blocks ?? world.objects ?? world.payload?.objects;
    let objectCount = "—";
    if (Array.isArray(objects)) objectCount = objects.length;
    else if (objects) objectCount = Object.keys(objects).length;
    $("object-count").textContent = objectCount;
    $("world-state").textContent = JSON.stringify(world, null, 2);
    renderTurns(state.turns);
  } catch (_) {
    connected = false;
    $("action-status").textContent = "Disconnected";
    $("provider").textContent =
      "Cannot reach the local server. Start Astra Robot World and keep this page open.";
  }
  controls();
}
$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("message").value.trim();
  if (!message || !configured || busy || !connected) return;
  $("send").disabled = true;
  if (await request("/chat", { message })) $("message").value = "";
  controls();
});
$("message").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});
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
    request(`/tools/${name}`, args);
  }),
);
$("stop").addEventListener("click", () => request("/stop"));
async function refresh() {
  await poll();
  setTimeout(refresh, 600);
}
refresh();
