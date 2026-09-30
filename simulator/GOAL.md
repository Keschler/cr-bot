# Simulator and V4 policy goal

The strategic goal is a very strong policy for the fixed Hog 2.6 deck:

```text
Hog Rider, Cannon, Musketeer, Skeletons,
Ice Golem, Ice Spirit, Fireball, The Log
```

It should play well against all enemy decks that the project can represent,
including supported special card forms as the catalog grows. The simulator is
the training and measurement instrument for that goal. A strong simulator
score is useful evidence about the policy, but it is not evidence that the
policy is already strong in the live game.

The current V1 ruleset is executable and contains 124 definitions, including
the 109-card eligible opponent roster. The source-linked real-deck catalog
currently has 40 decks: 30 training decks and 10 evaluation decks across five
training archetypes. That catalog is useful for the current experiment, but it
does not yet represent every enemy deck or every opponent style. Mechanics
with insufficient evidence remain provisional. There is no promoted V4
checkpoint or measured policy strength in a fresh checkout.

The ruleset metadata still reports `training_ready: false`. This records open
sim-to-real and fidelity evidence; it does not block simulator learning,
checkpoint generation, or simulator evaluation. It limits the claims made
from those results.

## Runtime boundary

The simulator owns authoritative state, legal actions, rewards, terminal
outcomes, and replay hashes. The V4 actor receives only public observations,
public memory, and legal-action masks. A privileged critic may read simulator
state during training; those features must never enter actor inference. The
phone and video frontend must adapt its visual public observation to the same
V4 actor input and checkpoint contract.

Strategic choices stay with the actor. A rules teacher may create supervised
bootstrap labels or a diagnostic baseline, but it must not choose actions in
ordinary actor-controlled PPO rollouts. `WAIT` advances one fixed simulator
decision interval.

## Learning path

The current next step is the bounded V4 pilot described in the
[pilot and comparison notes](../docs/v4_comparison_plan.md). It is an
authorized, one-day-scale experiment for checking that bootstrap, PPO,
search, distillation, public-only action execution, and measurement work on a
fresh source revision. It is not a commitment to a large fixed comparison.

1. Save a random V4 actor and fit a public-observation bootstrap actor. Keep
   the exact random and bootstrap state dicts, data counts, weighting, source
   revision, ruleset, observation contract, and checkpoint hashes.
2. Run the bounded pilot on fresh real-deck training and development pools.
   Check legal actions, finite losses, mode coverage, later-match
   observations, independent search corrections, actor exports, and measured
   cost. Treat an empty or contaminated pool as a collection problem.
3. Use the evidence from the pilot to decide which training method deserves
   more compute. If the experiment grows, keep training and development
   identities separate from reserved evaluation identities and compare
   complete matches on the same seeds and opponents.
4. Expand opponent breadth as data and simulator coverage improve. Track the
   current 40-deck catalog separately from the longer-term goal of all
   supported enemy decks. Add learned opponents and self-play only as separate,
   predeclared experiments.
5. Preserve the smallest reproducer for an exploit, observation leak, invalid
   mask, replay mismatch, or performance regression. Fix the contract, add a
   regression test, and regenerate contaminated data before drawing a policy
   conclusion.

The executable commands and package responsibilities are in
[V4 training and evaluation](../docs/v4_training.md).

## Evidence that travels with every experiment

- Fixed seeds and actions must reproduce deterministic reset, save/restore,
  state/event/replay hashes, and terminal versus truncated outcomes.
- Actor inputs and actions must be public-only and legal at the point of use;
  recurrent state must reset at match boundaries.
- Every training and evaluation artifact must identify its code revision,
  ruleset and observation contracts, checkpoint hash, opponent pool, seeds,
  data split, and measured compute. Changed source requires a new output
  directory or explicit compatible-resume validation.
- Compare held-out complete-match outcomes and action consequences against the
  starting actor, with uncertainty, truncation counts, invalid actions, and
  per-opponent results. Reward gains or label losses alone do not establish a
  better policy.
- Generated datasets, videos, checkpoints, and reports under `outputs/` are
  local artifacts. Older outputs are not current evidence merely because they
  exist.

## Sim-to-real boundary

The simulator gives the actor exact public quantities such as tower HP and
applied damage, while the phone path estimates those quantities from video.
This precision gap is public-information mismatch, not a privilege leak, and
it bounds what simulator-only results can claim. The actor also faces known
controller and catalog limits: the current real-deck evaluation uses scripted
opponents and ten reserved deck identities, so it does not establish
robustness against learned adversaries, all human decks, or real players.

The [physical lab](physical_lab/PHYSICAL_FIDELITY_GOAL.md) remains available
for causal, version-sensitive measurements and for transfer validation when a
learned policy provides evidence that such validation is the next useful
step. Physical evidence is required for a live-game claim appropriate to its
scope; it is not a universal prerequisite for learning in the simulator.
