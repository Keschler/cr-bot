# Simulator and V4 policy goal

Build a deterministic, versioned Level-11 Clash Royale simulator that supports
reproducible policy training and meaningful evaluation. The learner uses the
fixed base Hog-cycle deck against the supported opponent roster. Mechanics with
insufficient evidence remain provisional.

The current V1 ruleset is executable but has `training_ready: false`. It can
support research runs, but neither live-game fidelity nor policy strength is
established by simulator-only results. There is no promoted V4 checkpoint in a
fresh checkout.

## Runtime boundary

The simulator owns authoritative state, legal actions, rewards, terminal
outcomes, and replay hashes. The V4 actor receives only public observations,
public memory, and legal-action masks. A privileged critic may read simulator
state during training; its features must never enter actor inference. The
phone/video frontend must adapt its visual public observation to the same V4
actor input and checkpoint contract.

Strategic choices stay with the actor. A rules teacher may create supervised
bootstrap labels or a diagnostic baseline, but it must not choose actions in
ordinary actor-controlled PPO rollouts. `WAIT` advances one fixed simulator
decision interval.

## From-scratch workflow

1. Bootstrap a V4 actor from real simulator observations with a seeded teacher
   dataset. Save a plain actor state dict and a manifest with code, ruleset,
   observation, model, generator, seed, and checkpoint identity.
2. Generate disjoint train and sealed frontier pools using that explicit
   checkpoint. Keep source matches, search verdicts, and checkpoint hashes
   together; regenerate labels after simulator or search changes.
3. Audit search labels and run search distillation or controlled PPO as
   separate experiments. Each candidate starts from a declared checkpoint and
   uses only the training split for fitting.
4. Evaluate update zero and candidates on paired held-out seeds, complete
   matches, and the sealed split without tuning on that split. Record wins,
   crowns, truncation, invalid actions, behavior changes, and uncertainty.
5. Preserve the smallest reproducer for any simulator exploit, observation
   leak, invalid mask, replay mismatch, or performance regression. Fix the
   underlying contract, add a regression test, and restart contaminated
   collection before drawing conclusions.

The executable commands and package responsibilities are in
[V4 training and evaluation](../docs/v4_training.md).

## Evidence and release gates

- Deterministic reset, save/restore, and replay hashes must agree for fixed
  seeds and actions.
- Actor inputs and actions must be public-only and legal at the point of use.
  Recurrent state must reset at match boundaries; terminal and truncated
  outcomes must remain distinct.
- Every training and evaluation artifact must identify its code revision,
  ruleset and observation contracts, checkpoint hash, opponent pool, seeds,
  and data split. Changed source requires a new output directory or explicit
  compatible resume validation.
- Simulator reward gains alone do not establish a better policy. Compare
  held-out complete-match outcomes and action consequences against the frozen
  starting actor, with uncertainty and per-opponent results.
- Simulator fidelity and a live deployment claim require separate physical
  evidence through [the physical lab](PHYSICAL_FIDELITY_LAB.md). Provisional
  simulator experiments do not satisfy the `training_ready` gate.

Generated datasets, videos, checkpoints, and reports under `outputs/` are local
artifacts. They are not shipped or treated as current evidence merely because
an older run produced them.
