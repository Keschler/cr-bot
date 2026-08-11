# Codex Video Annotation Harness

This workflow creates blinded Clash Royale card-play labels from prepared video
frames by using local Codex subscription workers. It does not call the OpenAI
API.

The current workflow version is v7. Never reuse an older run with the v7
scripts.

## Evidence and blindness

Workers may inspect only:

- the prepared run manifest and extracted frames;
- review sheets produced by `scripts/codex_annotation/`;
- their assigned work package.

The deterministic red-team marker pass is permitted candidate generation. It
tracks card-free red level/deployment UI components in prepared frames; it is
not a detector prediction and never identifies a card.

Workers must not inspect ground truth, human labels, replay caches, prediction
files, detector output, evaluation reports, other runs, or earlier worker
transcripts. Reference labels may be used by an outer development session only
after the blind semantic artifact is checkpointed. Reference data must never
be copied into a worker prompt or package.

## 1. Prepare a fresh v7 run

At 10 fps, frames `[0, 1931)` cover 193.1 seconds:

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/prepare_gpt_annotation_run.py \
  --video dataset_generation/data/video_clips/10_fps_2.6HogCycle.mp4 \
  --start 0 \
  --end 193.1 \
  --output-dir outputs/annotation_runs/2hog-0-1931-v7-trial1
```

Preparation creates numbered frames, a v7 manifest, empty stage documents, and
review/checkpoint indexes. The default enemy arena scan uses non-overlapping
2-second primary windows plus compact boundary windows.

To create deterministic evidence and work packages without starting a model:

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/run_annotation_pipeline.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --profile hybrid-accuracy \
  --prepare-only
```

## 2. Run or resume semantic annotation

The recommended command is:

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/run_annotation_pipeline.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --profile hybrid-accuracy \
  --chunk-frames 200
```

The semantic DAG is:

```text
own hand-slot empty intervals
     ├── deterministic same-card-return/release constraint
     ├── high-resolution departing-card crop
     └── Luna-low card identity only
                deterministic elixir-cost onset alignment
                exact post-release rendering/checkpoint ──┐
                                                         │
red marker bursts ──> Sol sequence-aware existence ──────┤
                       corrected labeled onset frames     │
                       independent Terra-low side check   │
                       same-side temporal deduplication   │
arena windows ──────> independent Terra spell workers
                       exact before/after rendering
                       fresh Terra-low enemy/sequence gate ─┤
                  tracked delayed identity crops          │
                  compact target-labelled deck roster
                  one Terra-medium card assignment pass ──┤
                                                          v
                         merge -> release-review checkpoint
                               -> verification checkpoint
```

Each model worker receives one bounded package in a fresh `codex exec`
conversation. Stable outputs live in `worker_outputs/`; attempts, transcripts,
and result metadata are kept separately. `pipeline_state.json` records the
package hash, every attached review-image hash, prompt-template hash,
worker/validator code hashes, workflow version, model, effort, output hash,
raw tokens, and weighted tokens.

Rerun the exact same command to resume. Valid completed jobs are skipped.
Quota exhaustion records `paused_quota`; a `--max-weighted-tokens` ceiling
records `paused_budget`. Do not delete stable outputs, change a package/prompt,
switch profiles, or change `--chunk-frames` in the same run. Those changes
intentionally invalidate reuse; start a new run for a controlled comparison.

The semantic pipeline stops at `semantic_complete`. It checkpoints
`release_review` and `verification`; it does not localize, perform final
full-interval completeness, evaluate, or finalize.

## 3. Model and cost policy

`hybrid-accuracy` is recommended when semantic accuracy is the objective:

| Stage | Model | Effort | Cost weight |
|---|---|---:|---:|
| own departing-card identity | gpt-5.6-luna | low | 0.1× |
| own release/cancel and onset timing | deterministic | — | 0× |
| enemy unit/building existence + onset | gpt-5.6-sol | medium | 2.5× |
| enemy spell onset | gpt-5.6-terra | medium | 1× |
| independent enemy-spell confirmation | gpt-5.6-terra | low | 1× |
| enemy side | gpt-5.6-terra | low | 1× |
| compact enemy deck/card assignment | gpt-5.6-terra | medium | 1× |

The current Plus UI displays local-message ranges per five-hour window: Sol
`10–100`, Terra `25–200`, and Luna `250–2,000`. Both range endpoints give Sol
`2.5×` and Luna `0.1×` relative to Terra, matching the ratios in the current
token-based Codex rate card. These are accounting ratios for comparing worker
tokens, not guaranteed message allocations for an individual account.

Sol is confined to the sequence-aware existence task where controlled
benchmarks found that Terra missed small actors occluded by an existing large
unit. Side is cheaper and more reliable as a separate direct-evidence task.

Luna is not used indiscriminately. Luna-low correctly identified the departing
own card after deterministic release/cancel constraints were added, at a
0.1× Terra local-message cost weight. Luna-medium was materially less reliable
on isolated enemy identities, so enemy card assignment remains Terra-medium.
The release review is built from the deterministic same-card-return gate and
new exact post-release sheets. Canceled drags cannot be restored by a worker.
Every source worker session, model, effort, package hash, and evidence hash
remains in pipeline provenance.

`terra-efficient` replaces Sol existence with Terra medium. It is the
lower-cost comparison profile, not the accuracy reference. `terra-recall`
retains the legacy own-adjudication model field for resumability, but the
deterministic union does not launch that worker.

`sol-experimental` uses gpt-5.6-sol medium for every stage and assigns a 2.5×
cost multiplier. Equal Sol and Terra token counts are therefore not equal
cost. Sol must remain stage-specific unless its accuracy gain exceeds that
weighted cost on held-out blind runs.

Plus usage is a shared, task-dependent agentic pool rather than a fixed
messages-per-run allowance. Check the account’s Codex usage panel or limit
banner before starting a long run; the pipeline’s weighted-token budget is an
internal comparison guard and cannot predict the exact account limit.

## 4. Semantic evidence contracts

### Own plays

The harness first detects every interval in which a hand slot becomes empty.
It compares several pre-empty card-art frames with several post-interval frames
using HSV histograms. If the same card returns to the slot, the interval is a
canceled drag; that deterministic constraint is placed in the package and the
worker cannot override it. A final interval with no post-release confirmation
also fails closed. This gate covers brief previews, long held placements, and
spells still held at the segment boundary.

Each interval package contains synchronized HUD/arena evidence and a
high-resolution crop of the departing card. Luna-low names only that card and
supplies a visual explanation. It does not decide whether release occurred.

For constrained releases, the merger aligns event time to visible elixir-drop
transitions. It searches up to three adjacent negative transitions within four
frames whose total matches the canonical card cost within one elixir; otherwise
it retains the visually selected fallback. It then renders an exact bounded
`own_confirmation` sheet at least 0.5 seconds later. Thus hand-cycle outcome,
card identity, event time, and confirmation have separate evidence sources.

### Enemy unit/building onsets

Highlighted marker bursts are proposals, not truth. Marker tracking uses a
six-frame continuity gap and 70-pixel maximum association distance. In the
development benchmark this retained all marker-supported reference events
while reducing the number of model proposals substantially.

The existence worker sees sparse long-horizon focused sequences, follows an
active-object ledger, and decides only whether a new actor exists. It is
deliberately side-agnostic: a real own actor remains true until the next
stage. A self-sacrificing unit that moves or jumps coherently and disappears
on impact satisfies the post-onset check. The corrected event frame must be
one of the sheet's labeled samples.

All existence-confirmed proposals then receive an independent full-arena side
check. Enemy requires a direct red indicator plus upper origin or coherent
downward motion. Own requires a blue indicator, lower/own-release origin, or
coherent upward motion. Unresolved results fail closed. Only after side is
known are same-side proposals within five frames deduplicated; this prevents a
simultaneous own and enemy play from being collapsed.

### Enemy spells

Spells are scanned independently from unit markers. A spell needs a coherent
new projectile, rolling object, area effect, or impact sequence. Ongoing
combat and an existing object crossing a package boundary are not new plays.

Broad spell rows are proposals, not accepted events. Each proposal receives
separate exact before/after sheets spanning 1.2 seconds before and up to 3
seconds after the proposed frame. A low-cost independent confirmation worker
must reject own spells, targeting overlays, floating card labels, unit/tower
attacks, abilities, spawn residue, and effects without direct enemy
origin/direction. It may correct the onset to a later labeled enemy sequence
in that exact window. Rejected and unresolved proposals are removed before
card identity.

An additional segment-end sentinel always reviews the final 1.3 seconds. Only
that sentinel may retain a spell on the last source frame without forward
resolution evidence, and only when the first enemy projectile or impact is
already directly visible.

### Side and card identity

Enemy candidates are classified as `own`, `enemy`, or `unresolved` without
naming a card. Side comes from direct team indicator, legal origin, and
direction evidence—not from screen half after a unit has moved.

Each retained enemy troop/building then receives two delayed, grid-free,
dynamically tracked identity sheets:

- the first begins at least 0.5 seconds after onset;
- the second is a distinct later tracked view;
- neither uses `--focus-cell` or a grid;
- the crop follows the marker track nearest the verified onset rather than
  centering the full arena;
- a cyan rectangle marks the deployment clock/level marker directly below or
  overlapping the new body.

The renderer condenses each unit target to three clear frames and stacks three
target-labelled panels per roster sheet. One card-only Terra-medium worker sees
the complete interval roster plus the retained spell sequences. It must infer a
consistent deck of at most eight base card slots and assign every labelled
target. This prevents a large older actor from being copied onto a newly
deployed Ice Spirit or Dart Goblin and prevents one unclear repeat from
inventing a ninth card. Evolution and base forms share one deck slot; temporary
gold/purple enchantment glow is not an evolution identity. Use canonical `log`,
never `the-log`.

### Onset-first residual adjudication

Crowded overlaps can make a nearest delayed track follow a simultaneous own
deployment or an older enemy. For such residuals, use
`prompts/enemy_card_onset_adjudication.txt`: the full-arena temporal sequence
establishes the new red-clock enemy first, and delayed neighbor candidates only
resolve that body's anatomy. Luna-low is the inexpensive first pass; invalid or
genuinely conflicting rows may be escalated one at a time. A full-run deck pass
can be prepared with `prepare_enemy_onset_deck_package.py`, but the 2hog
benchmark found large multimodal batches less reliable than isolated residuals.

If the body remains occluded in the default delayed window, render a copied
target document with later offsets rather than altering the original evidence:

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/render_enemy_identity_neighbor_candidates.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --targets-file <isolated-target-package> \
  --output-targets <copied-target-document> \
  --output-dir <new-review-directory> \
  --sample-offsets 15,19,23,27,31,35,39,43,47
```

Evolution slugs are preserved in semantic output and validated through their
base card's metadata. For controlled comparison,
`merge_enemy_card_attempts.py` can merge ordered blind artifacts, preserve the
exact evidence cited by each selected row, and record per-onset provenance.
This is an outer-session tool: workers must never see its source selections or
evaluation report.

Adaptive residual scoring on one interval is not a held-out accuracy estimate.
Once a threshold is reached, freeze the policy and test it on a fresh clip
without label-informed prompt or evidence changes.

## 5. Localization after semantic completion

Localization may begin only after verification is checkpointed. For each
verified event, render an event-scoped macro sheet and then a tight labeled
grid:

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/render_annotation_review.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --event-id event-own-000694-cannon \
  --start-frame 694 \
  --end-frame 698 \
  --purpose macro \
  --output outputs/annotation_runs/2hog-0-1931-v7-trial1/reviews/cannon-macro.jpg

outputs/venv/bin/python \
  scripts/codex_annotation/render_annotation_review.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --event-id event-own-000694-cannon \
  --start-frame 694 \
  --end-frame 698 \
  --purpose grid \
  --grid-center 9,21 \
  --grid-radius 3 \
  --tile-width 900 \
  --output outputs/annotation_runs/2hog-0-1931-v7-trial1/reviews/cannon-grid.jpg
```

Record `location_frame_index`, the card-specific location rule, and the cell
in `localization.json`, then checkpoint:

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/checkpoint_annotation_stage.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --stage localization
```

The checkpoint validates deployment legality, mirrored for enemy plays. If the
location cannot be supported visually, use the explicit unavailable/unscorable
schema and an adjudication artifact instead of guessing.

For a benchmark sidecar after frozen semantics, use
`prepare_own_localization_packages.py` to render event-scoped macro and clean
axis-grid sheets. The blind worker package may contain the frozen card, its
canonical elixir cost, review frames, and allowed location rules, but never a
reference cell. Keep each worker in its prepared isolated directory and score
only after `validate_own_localization_decisions` accepts the output. Small
batches (at most four events) are preferred; isolate difficult residuals.

The outer session may select validated rows with
`merge_own_localization_attempts.py`. The selection manifest records only the
blind source and its sealed package, not reference coordinates. Re-evaluate
the aggregate localization prediction independently on both cell coordinates;
a `±1` policy means `abs(column error) <= 1` and `abs(row error) <= 1`, not a
Manhattan-distance threshold. Re-run the location-free evaluator on the merged
candidate to prove that side, card, and timing were not changed.

For deployable, label-independent localization, do not use that adaptive
selection workflow. Run `run_label_independent_own_localization.py` in a fresh
output directory. Its frozen policy gives every event the same two Luna-low
primary views and permits escalation only from structural validity, direct
confidence, own-side legality, and coordinate-wise agreement. It has no
ground-truth argument. The command writes `SEALED.json` only after all event
decisions are merged; evaluation is prohibited until that seal exists. Score
with `evaluate_sealed_own_localization.py`, which verifies both seal hashes,
writes `EVALUATED.json`, and registers the semantic-source hash in
`own_localization_label_independent/EVALUATED_RUNS.json`. The runner then
refuses any further cascade over that scored source, including a new output
directory.

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/run_label_independent_own_localization.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --source-file outputs/annotation_runs/2hog-0-1931-v7-trial1/own_semantics.json \
  --output-dir outputs/annotation_runs/2hog-0-1931-v7-trial1/own_localization_label_independent/v1 \
  --chunk-size 4
```

Do not rerun only the rows that fail after scoring, change the frozen policy in
place, or choose among sources using evaluation results. A revised policy
requires a new versioned directory and a different held-out video with manual
locations. On the 2hog benchmark, the first clean v1 run scored only 23/41
(56.1%) despite expensive Terra/Sol escalation; the adaptive 41/41 score must
not be cited as unlabeled-video performance.

## 6. Independent final completeness

Final completeness is distinct from the internal own recall sweep. Run it in a
new conversation that has not read `verification.json`, `localization.json`,
checkpoints, or accepted events:

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/render_annotation_sweep.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --side own \
  --output-dir outputs/annotation_runs/2hog-0-1931-v7-trial1/reviews

outputs/venv/bin/python \
  scripts/codex_annotation/render_annotation_sweep.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --side enemy \
  --output-dir outputs/annotation_runs/2hog-0-1931-v7-trial1/reviews
```

Copy the exact ranges and artifacts printed by the renderer into
`completeness.json`. If the sweep finds a possible missing event, record it as
unresolved and return to semantic verification; do not force completeness.

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/checkpoint_annotation_stage.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --stage completeness
```

## 7. Finalize and evaluate

Keep experimental artifacts under `outputs/`:

```bash
outputs/venv/bin/python \
  scripts/codex_annotation/finalize_gpt_annotation.py \
  --run-dir outputs/annotation_runs/2hog-0-1931-v7-trial1 \
  --output outputs/eval/blind_ground_truth/2hog-0-1931-v7-trial1.json \
  --audit-output outputs/eval/blind_ground_truth/2hog-0-1931-v7-trial1.audit.json
```

Finalization validates every stage hash and writes an immutable lock without
overwriting an existing result. Only after the `.lock.json` exists may a
separate evaluation compare the result with reference ground truth.
