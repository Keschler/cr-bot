# Simulator Subtree Guidance

## Automatic Delegation

For the active deterministic Clash Royale simulator and physical-fidelity-lab
goal, automatically delegate independent work when two or more workstreams are
available. Spawn up to three subagents using the configured subagent defaults
(`gpt-5.6-luna` with `max` reasoning), keep the primary agent as coordinator
and integrator, and wait for the workers' results before making cross-cutting
decisions.

Use parallel workers for codebase exploration, test analysis, replay or card
audits, and evidence review. During a dual-device physical-lab experiment,
assign exactly one dedicated operator to each phone:

- `phone_a_operator` owns only the explicitly assigned Phone A device serial.
- `phone_b_operator` owns only the explicitly assigned Phone B device serial.

The primary agent must establish and record the serial-to-phone mapping before
delegating device work. Each phone operator may perform calibration, input,
capture, and observation actions only on its own assigned serial. Every `adb`
command must be explicitly scoped to that serial; never use an unscoped device
command when more than one phone is connected. Operators must not control,
reset, capture, or inspect the other phone.

The primary agent owns experiment IDs, synchronized start barriers, shared
timelines, cross-device comparison, and final integration. Phone operators may
run concurrently with each other after the experiment is approved, but shared
worktree writes, destructive storage eviction, and any operation affecting
both phones remain serialized under primary-agent control. Require confirmation
before starting an external physical experiment or performing a destructive
action.

Do not spawn agents for a one-step task, or when another worker is already
editing the same files. Return concise findings with file references, test
commands, provenance, and any unresolved uncertainty to the primary agent.
