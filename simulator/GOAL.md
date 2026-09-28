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

The [three-arm comparison plan](../docs/v4_comparison_plan.md) is the concrete
next experiment and awaits user approval. Its prerequisites include correcting
bootstrap sampling weights and separating development from final evaluation.

1. Save a random V4 actor for reward-only PPO. Bootstrap a copy from real
   simulator observations with a seeded teacher dataset for the other two
   arms. Correct mode-loss sampling weights; teacher behavior is a prior whose
   value must be measured, not presumed optimal. Save plain actor state dicts
   and manifests with code, ruleset, observation, model, generator, seed, and
   checkpoint identity.
2. Generate disjoint train and sealed frontier pools using that explicit
   checkpoint. Keep source matches, search verdicts, and checkpoint hashes
   together; regenerate labels after simulator or search changes.
3. Audit search labels, then run search distillation and reward-only PPO from
   the same bootstrap checkpoint. Add a random-initialized PPO arm to measure
   the bootstrap's contribution. Select distillation variants on a separate
   development-validation pool; never select any arm on sealed outcomes or
   select the best replicate seed.
4. Compare all arms on the same held-out opponents, seeds, complete matches,
   and sealed frontier entries. Record wins, crowns, truncation, invalid
   actions, behavior changes, uncertainty, and simulator/search compute cost.
5. Preserve the smallest reproducer for any simulator exploit, observation
   leak, invalid mask, replay mismatch, or performance regression. Fix the
   underlying contract, add a regression test, and restart contaminated
   collection before drawing conclusions.
6. For the large training run, use the versioned catalog of complete real decks
   that the simulator supports. Keep deck identities disjoint
   between training and final evaluation. Use random deck mixtures only for
   stress testing, and report frozen V4 mirror opponents separately from
   non-mirror matchups.

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
  evidence through [the physical lab](physical_lab/PHYSICAL_FIDELITY_GOAL.md). Provisional
  simulator experiments do not satisfy the `training_ready` gate.

Generated datasets, videos, checkpoints, and reports under `outputs/` are local
artifacts. They are not shipped or treated as current evidence merely because
an older run produced them.
