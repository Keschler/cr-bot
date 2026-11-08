# Bait regression diagnosis, October 20, 2026

> Historical record (2026-10-20). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The targeted control's original Bait deficit **did not repeat overall in a
larger, frozen-checkpoint check**. On 90 fresh paired Bait games per actor, the
initializer won **21/90 (23.3%)** and control **25/90 (27.8%)**. This does not
prove a Bait improvement: five match seeds remain a small sample, and the
descriptive seed-cluster uncertainty range includes both gains and losses.

The exact old losses nevertheless reveal real tactical fragility. Three of
the five lost-win pairs can be rescued by changing just one decision, then
letting the unchanged control finish the game. The strongest example is a
Cannon placed in the wrong lane against Skeleton Barrel. The evidence favors
keeping control as the research reference and improving its Bait decisions,
rather than treating the old two-win deficit as proof of general forgetting.
No checkpoint was trained, replaced or promoted during this diagnosis.

## What actually regressed

The supplied [October 20 status PDF](../../outputs/local/reports/pdf/cr_bot_simulator_policy_status_2026-10-20.pdf),
pages 19–22, describes one **failed Bait safeguard**, rather than one lost
game. The original control comparison was:

| Original screen | Initializer | Targeted control | Net wins |
|---|---:|---:|---:|
| All families | 27/120 | 38/120 | +11 |
| Bait | 8/24 | 6/24 | −2 |

There were **five Bait win→loss pairs, three loss→win pairs and 16 pairs with
the same outcome**. The six Bait identities had four games each. Valk Guards
accounted for the net −2: it fell from 3/4 to 1/4. Prince Rascals lost one,
Valk Log Bait gained one, and the other three identities were unchanged in
net wins. See the complete [old paired matrix](../../outputs/v4/bait_regression_20261020/replication/old_pairs.md).

Bait was control's only failed selection safeguard. Hog and P.E.K.K.A also
lost two wins each in that original small screen; X-Bow gained nine, Golem
four, and Giant and Lavaloon two each. The overall gain is substantial in
that observed grid, but Bait was not the only family with a negative count.

## Fresh replication: the deficit is not stable across seeds

Before starting new games, the diagnostic fixed both checkpoint hashes,
runtime bytes, all six existing training Bait deck identities, three opponent
settings, and five fresh seeds. All **90 pairs / 180 matches** were completed;
there was no optional stopping, retraining or checkpoint selection.

| Opponent setting | Pairs | Initializer wins | Control wins | Net |
|---|---:|---:|---:|---:|
| Fixed | 30 | 6 | 5 | −1 |
| Adaptive | 30 | 6 | 9 | +3 |
| Split pressure | 30 | 9 | 11 | +2 |
| Total | 90 | 21 | 25 | +4 |

Seventeen pairs improved, 13 worsened and 60 kept the same outcome. Net wins
by match seed were −3, +4, 0, +4 and −1. Resampling the five entire seed
clusters gives a descriptive 95% range of **−8.9 to +17.8 percentage points**
for the mean score difference. This is a sensitivity calculation with few
clusters, not a calibrated strength guarantee.

No fresh deck identity had a negative net: Skeleton Barrel gained three,
Valk Guards gained one, and the other four tied in wins. Both policies still
lost all 15 fresh Prince Rascals games. Fixed opponents remain slightly worse
for control in this check, so it would be premature to declare Bait retention
solved. These are six familiar deck identities and scripted styles; this is
neither a new 40-deck benchmark nor a human-strength test.

The original failed gate remains the historical result under its declared
rule. The new check changes our assessment of its cause and repeatability;
it does not retrospectively promote control. See the
[pre-outcome protocol](../../outputs/v4/bait_regression_20261020/replication/protocol.md)
and [complete replication results](../../outputs/v4/bait_regression_20261020/replication_driver.md).

## Exact losses and controlled interventions

All eight changed-outcome cells were replayed with both actors: **16/16
reproduced the original complete native record and diagnostics exactly**,
including action, event and final-state hashes. Each policy was also probed
on the other actor's public history using its own chronological GRU memory.
Privileged engine events were used only to attribute damage and inspect
scenes; neither actor received them.

For each intervention, an identity-action replay first reproduced the old
result. The branch preserved the entire original prefix, current opponent
action, recurrent context and original actor draw. It then forced one legal
action and continued the unchanged control normally. These are selected
diagnostic branches, **not additional evaluation wins**.

| Old control loss | Single changed decision | Branch result |
|---|---|---|
| Skeleton Barrel, fixed, seed 2026101922 | 22.50s: move Cannon from left `(1,19)` to right `(14,19)` | **Win 1:0**, previously loss 1:2 |
| Valk Guards, split pressure, seed 2026101923 | 15.25s: play baseline's Cannon `(4,18)` before Ice Spirit | **Win 1:0**, previously loss 0:1 |
| Valk Log Bait, split pressure, seed 2026101923 | 12.75s: play baseline's Ice Golem `(14,18)` instead of waiting | **Win 1:0**, previously loss 0:1 |
| Prince Rascals, fixed, seed 2026101921 | 42.25s: move Ice Spirit from left to right | Still loss 0:1 |
| Skeleton Barrel, same fixed game | 12.00s: place Skeletons left as baseline | Still loss 0:1 |
| Skeleton Barrel, same fixed game | 12.25s: play baseline's Ice Golem left | Still loss 1:2 |
| Skeleton Barrel, same fixed game | 3.00s: restore the first differing Log placement | Still loss 1:2 |
| Valk Guards, fixed, seed 2026101921 | 5.75s: remove initial Log delay using baseline's action | Still loss 0:1 |

Coordinates above are `(column,row)` on the existing simulator grid.

**Skeleton Barrel shows a concrete placement failure.** At 22.50s, an enemy
Skeleton Barrel was in our right lane at approximately `(15.0,19.0)`. Control
spent three elixir on a left Cannon. The payload subsequently dealt **567
damage to our right tower at 26.00–27.75s**. Moving only that Cannon right
made the barrel retarget the building and drop its payload there; it removed
all 567 damage in that window and the unchanged continuation won.

This was not a newly forgotten Cannon choice in isolation. On that exact
control-generated history, the initializer also strongly preferred playing
Cannon and assigned 22.6% conditional probability to the same bad left cell,
versus control's 26.0%. Earlier changes had led into a state where both actors
were vulnerable. Copying the initializer on a finite set of saved states
cannot guarantee recovery from every later state.

**The split-pressure Valk Guards loss was a narrow cycle and crown failure.**
At the first action difference, both policies saw the same public history.
The initializer assigned Cannon 60.1% conditional card probability; control
assigned it 3.9% and preferred Ice Spirit at 70.0%. Both cards were played
eventually, but in different order and positions. Restoring only the first
Cannon decision yielded a win. The original control actually took less total
direct tower damage than the winning initializer, **3,329 versus 3,455**,
but concentrated it on its left tower. It lost that tower while leaving the
enemy's nearly destroyed tower on **94 HP**. Total damage alone concealed the
decisive distribution across lanes and crowns.

**Valk Log Bait illustrates sensitivity to a small timing change.** Control
waited at 12.75s and played Ice Golem at 13.00s. Moving that one play earlier
allowed a winning continuation. Its later states, actions and random sampling
path changed too; this does not prove that every 250ms delay is costly or that
this delay alone explains all subsequent damage.

There is no common first bad move across all losses. In Prince Rascals, the
first changed move was a Musketeer at 28.50s which prevented 453 Dart Goblin
damage suffered by the initializer. Later right-lane Gang/Barrel pressure
caused losses. Moving an Ice Spirit right reduced some Spear Goblin damage
but exposed the left lane to Knight damage and still lost. The fixed Valk
Guards control made only four Hog plays before 180s, versus six for the
initializer, and took 2,279 Princess damage versus 1,344. Removing its first
Log delay did not repair it. A uniquely decisive repair for these two losses
has not been established.

All branch receipts and damage windows are in
[diagnosis.json](../../outputs/v4/bait_regression_20261020/diagnosis.json) and
`outputs/v4/bait_regression_20261020/counterfactuals/`.

## Why the retention objective allowed this

The retention implementation is present and mathematically covers **mode,
conditional card and every legal placement cell**, using separate anchor and
student recurrent states. This is not an omitted-placement-loss or memory
reset bug. It is an average soft constraint on a finite collection of
histories, rather than a promise to preserve every sampled action.

The full chronological audit covered all **24 collected Bait episodes /
19,527 rows**. Only Skeleton Barrel supplied active Bait tactical teaching:
501 training rows, of which the teacher chose 462 WAITs and 39 PLAYs. The
other five Bait identities had no active rows and were anchored on their
recorded histories while shared policy parameters learned other situations.

The 14,246 inactive Bait training rows had mean joint KL **0.0180**, but a
maximum of **4.4318**. Many rows had no legal play and therefore contributed
zero divergence. Restricting to inactive states with a legal PLAY raised
the mean to **0.0402**. Calibration showed the same issue: inactive mean
**0.0301**, maximum **4.8579**, or mean **0.0682** when a play was legal.

Across train and calibration histories, there were 158 mode argmax changes,
46 card changes when both policies greedily chose PLAY, and 395 placement
argmax changes among 1,192 states where both greedily played the same card.
These probes describe distribution changes, not counts of actual mistakes;
evaluation sampled actions stochastically, and many nearby cells are harmless.

On the 16 selected changed-outcome traces, legal-PLAY mean KL was **0.0848 on
control's histories**, versus **0.0397 on collected training actor histories**.
The largest individual divergence was 9.6048. This supports checking the
states the new actor actually reaches. These traces were selected by outcome
change, so their mean is not an unbiased estimate for all Bait games.

The causal account supported by this evidence is: shared updates change a
few timing/card/cell probabilities; sampled choices then change future hand,
elixir, lane pressure and observations; later defenses can be fragile, and
small tower-HP margins turn those changes into different crowns. Those
changes also create wins. The old net −2 combines these real trajectory
changes with substantial variation across seeds, rather than establishing
a uniform loss of Bait skill.

See [retention measurements](../../outputs/v4/bait_regression_20261020/retention_drift.md)
and the complete machine-readable distributions in its sibling JSON.

## Follow-up proposed at the time justified by this diagnosis

Use control as the comparator for the next experiment while retaining the
current selected checkpoint. Collect full chronological Bait histories from
the control itself, emphasizing incoming barrel lanes, Princess removal,
Log availability, cheap-defense ordering and attacking the damaged tower.
Validate corrective actions with short damage windows **and complete-match
outcomes**. Preserve both successful control continuations and difficult
recovery states; the old initializer is not an expert on every new state.

Prioritize placement and cycle teaching over a blanket increase in global
retention strength. Audit divergence tails and legal-PLAY decisions by card
and scenario, alongside the mean. A stronger KL coefficient alone could
suppress useful X-Bow improvements without supplying the missing defenses.
Public lane-threat, building-pull and spell-coverage features are reasonable
separate ablations once tied to these demonstrated failures.

Before promoting any repair, predeclare a fresh comparison against control
and the initializer across all families, including Hog and P.E.K.K.A, with
multiple paired seeds and explicit uncertainty. Preserve the X-Bow gain and
then test broader deck identities and behaviors. The new Bait-only check
does not establish the overall gain on a larger benchmark or human-level play.

## Verification and reproducibility

The [integrated audit](../../outputs/v4/bait_regression_20261020/diagnosis.json)
verified all 153 sealed source files and both checkpoint hashes unchanged,
all 180 fresh records and their byte digests, all 16 exact factual replays,
and eight identity/intervention pairs. Fresh matches had zero invalid actions,
rejected learner actions, truncations, cap hits or opponent warnings.
Independent checks confirmed identical branch prefixes and pre/post-sample
RNG states at each intervention. No production code changed, so pytest was
not rerun for this analysis.

The initial diagnostic question for Skeletons at 12.00s called it the first
divergence. Full trace inspection corrected that description: the first
differing action was Log at 3.00s. Both actions were tested separately and
neither rescued the loss; original protocols are preserved.

Recompute the integrated checks from the repository root:

```bash
outputs/venv/bin/python outputs/v4/bait_regression_20261020/diagnosis_audit.py
```

Other output-only helpers are `trace_replay.py`, `retention_drift.py`,
`replication_driver.py` and `causal_branches.py` under the same artifact root.
They keep original screen records, fresh diagnostic matches and selected
counterfactual branches separate.
