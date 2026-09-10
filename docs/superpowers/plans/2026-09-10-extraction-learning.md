# Extraction learning improvements

User authorized implementing the highest-return fixes after reviewing six failed extraction trials.

- Add an extraction goal: lower object moves out from under a named upper object; upper may land on a named tray/support; both settle; unrelated objects remain preserved. Never silently substitute ground contact or preserving the upper object. Preserve existing saved actions and goals.
- Add bounded read-only approach checks on copied physics state. Return reachability and collision diagnostics for candidate poses/routes; never spend a physical trial or mutate the live world. Improve failed motion reports with step/pose/contact information and object displacement summaries.
- Separate experimental/read-only call budget from live mutations, retain finite total limits, and reserve a tool-free summary response so hitting a tool limit cannot mask an already successful physical action. Teach Astra to preflight approaches and change strategy after repeat failures.
- Update proposal interpretation and UI goal descriptions for extraction; verify mocks, physics tests, browser/static checks, and a rehearsal on a copy of the user's tray stack. Restart only when idle, save/restore the current scene, do not reset objects or run speculative motions live.

Ownership: extraction agent owns action_contracts.py/action_lab.py/action_proposals.py and extraction tests. Approach agent owns action_motion.py/tools.py/new approach module and tests. Parent owns astra.py/server.py/web integration/docs/budget tests.

## Verification and additional steering

- Initial integrated suite: 176 passed. New goal/preflight/budget tests include direct-upper-manipulation rejection, named tray contact, both-object settling, live state isolation, collision diagnostics, and tool-free summary enforcement.
- Live Astra rehearsal on saved tray stack: 13 approach candidates checked, one physical trial; accepted extraction goal and returned blue/red measurements. No verified extraction; identified hand/tray clearance failures and left source physics unchanged. This demonstrates better feedback and avoided wasted trials, not guaranteed task success.
- Added guidance for high-clearance route waypoints, interpreting failed waypoint indices, and Panda grasp-frame axes. Second rehearsal pending.
- User additionally requested that experiment cards update live with the currently running trial; UI agent investigating and fixing progress/follow behavior before final restart.
