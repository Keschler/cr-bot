# Simulator V1 Main Goal

## Mission

Build a deterministic, headless, versioned Level-11 Clash Royale simulator
that is accurate enough to train production-scale reinforcement-learning
policies. The player uses one fixed base Hog-cycle deck. The opponent may use
every in-scope base card released before **2025-12-01**, including ground
troops, air troops, buildings, and spells. Every implemented mechanic and
interaction must have generated tests and independent sim-to-real evidence.
The controlled `physical_lab` is the primary high-fidelity evidence path:
known actions, current patch/level, calibrated placement, synchronized
captures, and explicit device/capture provenance make it more informative
for version-sensitive behavior than passive public-video analysis. Public
video remains a supplementary discovery and validation source, never a
substitute for a required controlled probe.

Maximum fidelity is a release requirement, not a later polish phase. Unknown
behavior must remain explicitly provisional and must fail the relevant
training-readiness gate; it must never be silently presented as accurate.

Current implementation checkpoints are intentionally visible: the fixed
`v1.json` artifact exposes all 109 classified opponent cards but is marked
`training_ready: false`; `2026-08-04-roster` remains a compatibility/source-
build artifact only. `generate-scenarios` creates deterministic
card/mechanic cases; the current V1 one-per-mechanic run covers 1,071 cases
across 61 mechanics, including passive-spawner lifecycle cases (Goblin Drill,
Goblin Cage, huts and Tombstone), Furnace movement/spawning, Cannon Cart form
change, and nested Golem/Elixir-Golem/Lava-Hound/Goblin-Giant death streams,
with zero repeated-hash failures. The Phase-0 `physical_lab` harness now
provides the primary controlled-evidence workflow; `mine-video-truth` and the
`discover-video-source`, `extract-video`, and `mine-replay-tracks` commands
remain a supplementary public-video discovery path with reproducible,
budget-aware provenance. These are coverage and evidence-pipeline milestones.
None of these results is a claim that the final real-game fidelity gates have
passed.

Large synthetic matrices can be validated with `validate-generated --workers
N`. The current four-variant V1 roster run covers 4,284 cases (109 cards, 61
mechanics) with four isolated workers, final-state invariant checks, two
replays per case, and zero failures or determinism mismatches. Worker count is
not an accuracy shortcut: every process reconstructs the pinned ruleset and
scenario, and rows are sorted by scenario ID before reporting. This remains
synthetic determinism/exercisability evidence, not sim-to-real truth.

The current truth miner additionally rejects detector linger, short-window,
irregular-gap, teleport, zig-zag, and unstable-speed tracks using a
simulator-independent motion-quality gate. Repeated extractor roots can be
merged deterministically per HUD/video,
with the selected cache and discarded candidates retained as provenance; this
keeps resumed high-scale extraction runs from duplicating evidence.

`discover-replay-interactions` is the action-free high-scale discovery pass. It
scans sealed replay caches for detector track-onset candidates, legal bridge
crossings/path topology, Cannon lifetime/HP-decay signatures, and Hog-to-Cannon
approaches. It is deliberately not a truth oracle: candidates carry
`truth_promoted: false`, uncertain rows stay in `rejected`, and malformed or
level-conflicting sources stay in `source_failures`. Cross-level topology may
be selected only with an exact support-tower-HP sentinel; cross-level Cannon
HP/lifetime rows remain rejected because Level-11 values cannot be assumed
level invariant. This keeps action-free mining useful for prioritizing video
audits without weakening held-out readiness.

The generated interaction layer additionally probes every eligible opponent
card against every card in the unchanged fixed player deck (`109 × 8 = 872`
cases). Each probe requires both action-boundary plays to be accepted; the
current matrix has passed all 872 repeated deterministic runs with final-state
invariant validation. A strict per-tick run over the 109 opponent-card/Hog
Rider column also passes `109/109`; the full strict matrix remains an explicit
nightly option because it is substantially more expensive. This synthetic
matrix complements, but never replaces, held-out sim-to-real fidelity
evidence from the primary physical-lab lane and any explicitly permitted
supplementary public-video measurements.

The exhaustive unordered opponent-pair layer now probes every distinct pair of
the 109-card eligible roster (`C(109, 2) = 5,886`) against the unchanged Hog
cycle player deck. Each case fills the match to the ten-elixir cap, plays Hog,
plays both opponent cards from a legal hand slot, and stops immediately after
the second play so the matrix tests action acceptance without pretending to be
a long-horizon fidelity oracle. The pinned report
(`outputs/simulator/generated-opponent-pairs-validation-v1-062.json`) passes
`5,886/5,886`, with zero failures and zero repeated-hash mismatches using four
isolated workers and final-state invariant validation. This is synthetic
coexistence/exercisability coverage; card-specific timing, targeting, path,
damage, and outcome claims still require the independent video/in-game gates.

The policy runtime also exposes a deterministic `VectorSimulatorEnv` process
backend. It runs independent lanes concurrently, restores canonical
state/event logs in the parent, and rebuilds observations using persistent
per-lane observation memory. A parity test covers policy actions, rewards,
observations, state hashes, and event-log hashes against the sequential
reference backend. This is an intermediate parallel-reference baseline: it is
not yet the final structure-of-arrays/JIT backend or a production throughput
claim.

The current fixed-`v1` throughput snapshots are
`outputs/simulator/benchmark-vector-v1-reference-074.json` and
`outputs/simulator/benchmark-vector-v1-process-075.json`: 16 independent
environments completed 20 policy-boundary steps at 41.59 and 39.69
environment-steps/second respectively under `reference-0.28.0`. These numbers
are reproducibility baselines, not a claim that the final production-scale RL
backend is finished; a structure-of-arrays/JIT/vectorized implementation and
long-horizon throughput gate remain future work after fidelity gates pass.

`merge-replay-interactions` is the conservative dual-HUD reconciliation stage.
It pairs standard and alternative candidates from the same source video by
mechanic, card/owner, onset, and observed geometry. Only agreeing pairs are
reported as stronger candidates; unmatched observations are retained as
rejections, and dual-HUD agreement can never satisfy a held-out gate because
both profiles render the same underlying video.

The current engine version is `reference-0.28.0`.  This release adds
first-class deterministic components for shield layers, Royal Ghost stealth,
Miner burrow, mixed Goblin Gang/Rascals child composition, Magic Archer line
piercing, Executioner return passes, Hunter pellets, Bowler knockback, Mega
Knight jumps, Electro/Ice Wizard deployment control, Lumberjack Rage, Mother
Witch conversion, Ram Rider snare, and Witch death Skeletons.  Each branch has
focused regression coverage in `tests/simulator/test_requested_card_mechanics.py`
and a generated scenario obligation where the action-boundary fixture can
observe the event; exact animation/edge timing remains blocked by the fidelity
ledger until high-confidence real footage or a controlled in-game capture
resolves it.

## Frozen V1 scope

## Latest autonomous audit (2026-08-18)

The current `reference-0.28.0` engine passes the regenerated deterministic
roster matrix (`outputs/simulator/generated-roster-validation-v1-per1-041.json`):
1,071/1,071 scenarios pass with zero repeated-run determinism failures. The
fixed-player/opponent interaction matrix also passes 872/872 in
`outputs/simulator/generated-interactions-validation-v1-042.json`.

The vision truth compiler now seals its movement estimator. The current
path-length held-out corpus (`outputs/simulator/fidelity_media/corpus-independent-full-v2-path-044.json`)
has three samples from two video groups and reports 33.3% agreement; Ice
Spirit and Giant Skeleton remain outside tolerance. This is intentionally a
failed/insufficient fidelity result, not a readiness claim. The corresponding
readiness report (`outputs/simulator/training-readiness-current-045.json`)
remains `ready: false` because the declared per-mechanic gates require at
least 20 observations, two independent groups, and 0.98 agreement.

The sealed action-window corpus
(`outputs/simulator/fidelity_media/corpus-action-windows-v2-path-049.json`)
adds 17 held-out movement observations from the same two source-level video
groups without treating HUD variants as independent evidence. Its report
(`outputs/simulator/fidelity_media/fidelity-action-windows-v2-path-heldout-050.json`)
reports 4/17 agreements; the combined readiness audit is still intentionally
failed (`outputs/simulator/training-readiness-current-051.json`).

The stricter quality-gated action-window corpus
(`outputs/simulator/fidelity_media/corpus-action-windows-v3-quality-path-053.json`)
keeps only tracks with at least 0.75 seconds of motion, a path-to-endpoint
ratio no greater than 1.5, and an interquartile speed-spread ratio no greater
than 1.5. It has four held-out samples from two source groups and reports
2/4 agreement (`outputs/simulator/fidelity_media/fidelity-action-windows-v3-quality-path-heldout-054.json`).
This is a more conservative measurement, not a green result: the current
readiness report (`outputs/simulator/training-readiness-current-055.json`)
remains `ready: false` and still requires substantially more independent,
mechanic-specific evidence. The gates are simulator-independent and recorded
in the truth manifest, so no card speed or engine output is used to select
truth.

The Hunter fan fix is included in this engine version: its ten pellets retain
their independent spread instead of being re-homed onto the acquired target
on the following tick. The workspace budget audit
(`outputs/simulator/fidelity_media/media-budget-current-056.json`) reports
164,248,917,543 bytes (165,248,917,543 including the configured reserve),
below the 200 GB cap; no media was deleted.

The exhaustive pair audit was regenerated after shortening its setup horizon
and making its report card accounting include both pair members. The pinned
manifest (`outputs/simulator/generated-opponent-pairs-v1-061.json`) contains
5,886 unordered pairs over all 109 eligible opponent cards; its validation
report (`outputs/simulator/generated-opponent-pairs-validation-v1-062.json`)
passes 5,886/5,886 repeated runs with zero failures or determinism mismatches.

The action-free miner now has a fail-closed Level-11 proof path for bounded
windows whose HUD crop does not contain a full support tower. A proof cache is
accepted only when it comes from the same source-video key and contains the
exact declared Level-11 support-tower HP; its hash and any rejection are
recorded. Applying that contract to both HUD variants produced 58 paired
track-onset candidates in
`outputs/simulator/fidelity_media/autonomous-interactions-action-windows-dual-proven-073.json`.
All 58 remain `truth_promoted: false`, and 24 level-conflicting source rows
remain rejected, so this expands auditable prioritization only and does not
change the failed held-out/readiness gates.

The controlled evidence path is implemented as the Phase-0
[`physical_lab`](PHYSICAL_FIDELITY_LAB.md) package and `python -m simulator lab`.
It seals canonical experiment specifications, calibrated two-device actions,
lifecycle and synchronization reports, observation manifests, simulator
replays, comparisons, first-divergence reports, and artifact hashes. The
offline/fake harness is an integration test only and remains
`candidate_only`; the ADB path now has a hash-verified reviewed-template
lifecycle detector, but connected-device runs remain rejected until continuous
capture, replay-cache extraction, and the downstream observation/readiness
boundaries are complete. No Phase-0 result changes training readiness.

### Physical-lab status (2026-08-20)

Phase 0 of the controlled fidelity lab is implemented and covered by focused
tests. It proves canonical experiment hashing, logical phone calibration,
fail-closed lifecycle handling, clock alignment, split locking, replay-cache
validation, confidence-gated ingest, deterministic simulator replay, and
first-divergence comparison. It does not constitute physical evidence: the
offline run is `candidate_only`, and no connected physical run has yet been
admitted to a validation or held-out corpus. The lifecycle boundary now accepts
only per-device reviewed templates with sealed file/manifest hashes and records
that detector provenance. The next fidelity milestone is one complete
connected `hog_cannon_pull` or isolated Hog probe whose sealed artifacts can
flow into the existing fidelity/readiness reports.

### Player

The only player deck is the base-form classic Hog-cycle deck:

1. Hog Rider
2. Musketeer
3. Ice Golem
4. Ice Spirit
5. Cannon
6. Skeletons
7. Fireball
8. The Log

The policy can play only these cards. V1 does not support choosing another
player deck.

### Opponent

The opponent roster is a versioned manifest containing every eligible card
whose original live release date is strictly earlier than `2025-12-01`.
Eligible cards include:

- ground and air troops;
- single- and multi-unit cards;
- buildings and spawners;
- direct, projectile, area, persistent-area, and troop-producing spells;
- legendary and other rarity classes when their complete behavior is inside
  the V1 mechanic scope.

Release date, external card ID, canonical ID, card kind, and eligibility reason
must be machine-readable. CI must fail when an eligible card is absent from
the ruleset, engine dispatch, scenario generator, observation vocabulary, or
readiness matrix.

The date is an eligibility cutoff, not a runtime balance selector. V1 uses the
single fixed `rulesets/v1.json` Level-11 data artifact and one engine version;
timestamped balance migrations are deferred until V2. Changing V1 data creates
a new immutable version rather than mutating it in place.

### Explicit V1 exclusions

- Evolutions and evolution cycles.
- Heroes and hero mechanics.
- Activated abilities. Consequently, Champions or other forms whose complete
  gameplay requires an activated ability are outside V1; simulating only
  their body would be knowingly inaccurate.
- Alternative tower troops; only the pinned Princess and King Towers exist.
- Alternative game modes, arenas with different gameplay geometry, special
  events, draft rules, and mode-specific objects.
- Alternative balance versions and card levels.
- Alternative player decks.

These exclusions simplify V1; they do not permit an eligible opponent card to
be approximated by a mechanically different card.

When official data and the DeckShop corroboration snapshot do not resolve a
behavior, record it in [`UNKNOWN_BEHAVIORS.md`](UNKNOWN_BEHAVIORS.md) and
[`unknown_behaviors.json`](unknown_behaviors.json). The user can answer those
rows with controlled in-game tests; an answer must include level proof, frame
ranges, and a source hash before it is promoted.

## Required simulator behavior

V1 must run complete unattended matches through the same observation/action
boundary used by the vision policy. Authoritative state must remain integer,
serializable, replayable, deterministic, and free of hidden wall-clock or
thread-order dependence.

The engine must cover, where required by the opponent roster:

- match clock, phases, elixir, legal placement, hands, cycle, crowns, overtime,
  tiebreak, and King activation;
- troop placement must reject both ground and air troops whose footprint
  overlaps a friendly or enemy building/tower; Mirror must reject an
  immediately subsequent Mirror play (no Mirror chaining);
- ground and air navigation, bridge selection, river/terrain restrictions,
  building diversion, repathing, local steering, collision, mass, congestion,
  knockback, pulls, and retargeting;
- ground-only, air-only, building-only, and mixed target legality; sight,
  target persistence, invisibility or untargetability when applicable, and
  target priority exceptions;
- deploy time, first-hit delay, attack wind-up, hit speed, melee and ranged
  attacks, projectile origin/speed/arc, piercing, chains, splash, area shape,
  damage timing, Crown Tower scaling, death, and simultaneous events;
- building lifetime and HP decay, passive and active spawners, waves, shields,
  charge states, transformations, split units, death spawns, reflected or
  redirected damage, and any other mechanic actually needed by an eligible
  card;
- Furnace follows the post-August-2025 rework as a moving medium-speed troop,
  not a stationary building: its body attacks while walking and its
  data-driven component emits one Fire Spirit per configured cadence. The
  rework source describes the cauldron attack as single-target; exact
  projectile timing and Fire Spirit launch geometry remain explicit fidelity
  unknowns until isolated footage or a controlled in-game test resolves them;
- freeze, slow, stun, rage, poison/damage-over-time, heal, knockback and
  immunity/resistance rules;
- shield layers (Dark Prince, Guards, Royal Recruits), stealth/re-cloak
  (Royal Ghost), burrow/emergence (Miner), mixed swarm child compositions
  (Goblin Gang, Rascals), line-piercing and return-path projectiles (Magic
  Archer, Executioner), pellet fans (Hunter), troop impact knockback (Bowler),
  jump/landing attacks (Mega Knight), deployment stun/freeze (Electro Wizard,
  Ice Wizard), death Rage (Lumberjack), Mother Witch curse conversion,
  Ram Rider snare, and Witch death Skeletons;
- spells targeting ground, air, buildings, towers, empty arena positions, or
  combinations thereof, including exact boundary and temporal behavior;
- stable policy observations without privileged opponent state, plus
  authoritative legal-action masks.

Card-specific branches are acceptable only when the real card is genuinely
exceptional. Shared mechanics should be data-driven components so one fix can
be tested across every consumer.

## Physical-first fidelity and test factory

The central implementation pipeline is:

```text
official/current structured data + patch history + release manifest
                              |
                              v
             canonical versioned Level-11 truth database
                              |
                +-------------+-------------+
                |                           |
                v                           v
       generated exact tests       generated interaction graph
                                            |
                                            v
                              deterministic scenario factory
                                            |
                +--------------------------+------------------+
                |                                             |
                v                                             v
       engine invariant/fuzz runs                 physical_lab (primary)
                                                   |
                                  controlled actions + dual-device capture
                                                   |
                                                   v
                                  synchronized, confidence-gated observations
                                                   |
                +----------------------------------+
                |
                v
       per-mechanic sim-to-real comparison/readiness
                ^
                |
 supplementary public-video miner: discovery, natural interactions,
 topology, and candidate/validation observations only
                |
                v
       direct frame/contact-sheet audit and next-experiment planning
```

### 1. Canonical roster and truth ingestion

Create a checked-in opponent roster manifest and generate a coverage failure
for every missing eligible card. Every numeric or semantic ruleset field must
carry provenance, ruleset applicability, confidence, and unresolved conflicts.
Official explicit values override lower-tier sources. Patch application must
be reproducible from a previous snapshot, and the final canonical artifact
must have an immutable content hash.

Do not use a single third-party page as ground truth. Structured game-data
mirrors and independent simulators are differential signals, not authorities.

### 2. Mechanic inventory and coverage graph

Each card declares the mechanics it consumes. Generate a machine-readable
graph such as:

```text
card -> form/spawn -> movement layer -> targeting -> attack/effect -> statuses
     -> victims/targets -> tower interaction -> death/lifetime behavior
```

From this graph, generate a required readiness matrix. A card is incomplete if
even one required mechanic has no implementation, no exact/property test, or
no sufficient real-game validation. Aggregate percentages must not hide a
missing rare mechanic.

### 3. Generated synthetic testing

Generate tests rather than hand-writing one happy path per card:

The factory must also make each case exercise the declared branch.  Active
buildings receive a legal moving target, enemy-targeting spells receive a
legal victim, friendly effects receive a same-owner troop, and exceptional
troops receive the smallest deterministic target setup needed for hooks,
dashes, reflections, or transformations.  Each generated oracle records both
required card plays and required public event kinds (for example
`attack_started`, `projectile_resolved`, `entity_spawned`, `status_applied`,
or `entity_transformed`); a repeated hash is not a pass when the branch event
never occurred.  Lifetime cases run through the full declared lifetime, so
HP-decay/expiry behavior cannot be hidden by an early scenario cutoff.

- schema and stat tests for every canonical field;
- every attacker/defender and spell/target legality combination;
- ground/air/building/tower target matrices;
- placement, range, radius, projectile, splash, line, cone, and spell-boundary
  sweeps at `boundary-epsilon`, `boundary`, and `boundary+epsilon`;
- timing offsets around deploy, first-hit, retarget, death, lifetime, spawn,
  status expiry, overtime, and simultaneous-event boundaries;
- navigation from every legal deployment region to relevant targets, both
  bridges, moving targets, placed buildings, destroyed buildings, and dense
  traffic;
- pairwise tests plus automatically selected three- and four-card scenarios
  for mechanics that interact non-linearly;
- mirror, time-segmentation, save/reload, reference-backend parity, and repeated
  execution metamorphic tests;
- randomized state/action fuzzing and long soak runs with per-tick invariants.

Every discovered engine failure becomes a minimized, immutable regression
scenario. Scenario identity includes ruleset, engine, seed, and evidence
lineage.

### 4. Controlled physical-lab fidelity oracle (primary)

Use the [`physical_lab`](PHYSICAL_FIDELITY_LAB.md) workflow for exact and
version-sensitive sim-to-real evidence. It is the authoritative evidence lane
for deployment and first-hit timing, projectile and damage behavior, targeting
and retargeting, building lifetime and HP decay, spell boundaries and victim
sets, statuses, spawns, transformations, collision, pulls, and controlled
movement. Public video cannot reliably expose the action boundary, current
patch/level, or causal interaction needed for these measurements.

Every probe must be generated from a canonical experiment specification that
records the pinned ruleset and engine hashes, current game patch and card
level, device identities, calibrated placement, logical actions, capture group,
evidence split, and measurement questions. The controlled path is:

```text
lab plan -> lab run -> lifecycle/synchronization gates -> lab ingest
         -> sealed observation manifest -> simulator replay
         -> comparison/first divergence -> fidelity/readiness
```

An accepted connected run requires both devices to complete the verified
battle lifecycle, synchronized and hashed captures, recognized replay caches,
confidence-gated observations with frame/time provenance, the same logical
experiment specification in the simulator, and a comparison report with
first-divergence data. Capture groups and splits are locked before inspection.
Failed, incomplete, inferred-only, or ambiguous observations remain rejected;
the offline/fake Phase-0 harness remains `candidate_only` and can never satisfy
a held-out gate. The connected ADB path remains fail-closed without both
reviewed lifecycle manifests and remains ineligible until continuous capture
integration is verified.

The lab planner should prioritize unresolved `unknown_behaviors`, missing or
failed readiness edges, and first-divergence clusters. Use local boundary
sweeps and isolated probes before expanding to broad card matrices. A physical
probe is not a green result by itself: it becomes usable evidence only after
the normal oracle-audit, independent-group, leakage, and readiness rules pass.

### 4A. Supplementary public-video oracle

Run the repository's vision extractor offline over a large, diverse video
corpus for supplementary discovery and validation. It may process slowly and
use ensembles because validation is not a real-time path. Reconstruct arena
coordinates with homography and mine only segments with strong agreement
between card identity, tracking, motion, action timing, visual effects, and HP
changes. This lane is lower-authority than a controlled physical probe: it is
useful for natural interactions, topology, candidate generation, and
cross-checks, but it cannot substitute for current-version controlled evidence
when the action boundary or causal behavior matters.

Automatically extract evidence for at least:

- isolated ground and air motion, bridge/lane choice, path diversion, and
  repathing;
- pulls, target selection, target persistence, retargets, and invisibility or
  targetability transitions;
- deploy, first-hit, repeat-hit, projectile, impact, damage, death, spawn,
  transformation, lifetime, and status timings;
- spell origin, trajectory, impact, victim set, damage, knockback, and
  persistent-area ticks;
- collision/congestion using both identity tracks and robust aggregate group
  geometry;
- decision outcomes: tower hits, pull success, deaths, surviving HP, King
  activation, and spell kills.

The repository integration is explicit: `discover-video-source` seals source
metadata, `plan-action-windows` selects confidence-gated action windows from
the offline card/action candidates (selection only, never truth), `extract-video`
or its action-window plan runs both HUD profiles under the workspace budget,
`mine-replay-tracks` converts caches into hashed tracks, `mine-video-truth`
confidence-gates them, and `compile-video-truth` creates mid-track fidelity
scenarios. The movement-track adapter excludes stationary buildings and
projectile/spell detections; Cannon lifetime/damage and spell miners consume
the same caches through separate event oracles. Isolated tracks produce
target-independent movement-speed evidence;
absolute positions are not compared unless an action/path reconstruction makes
the target observable. Unsupported cards (for example excluded ability forms)
are recorded as rejected evidence rather than being forced into the corpus.
Every extractor subprocess is bounded by a recorded wall-clock timeout
(`--job-timeout-s`, 30 minutes by default); timeouts remain rejected evidence
and never stall or silently pass an autonomous batch.

Cannon lifetime discovery can run without a manual action file: it then uses
only a non-boundary detector track onset as an explicitly provisional action
hypothesis and preserves the inferred source in the candidate report. A
curated action file remains the stricter path when one exists.

Both HUD runs are now aligned by video frame and cross-checked by card, owner,
and trajectory. The merged manifest records matched-track counts, overlap,
position MAE/max error, and per-track agreement. HUD runs decode the same
frames, so this is explicitly a detector-quality diagnostic—not independent
truth and not a way around confidence or held-out gates. Disagreements remain
auditable and are never silently averaged into a fidelity claim.

Low-confidence clips should normally be discarded because more footage is
cheaper than manual labeling. Human effort is reserved for strategically
important gaps and unknown first-divergence classes.

The primary V1 public-video source for supplementary discovery is the
[`YersonCz` YouTube channel](https://www.youtube.com/@yersoncz6334). A source
video is eligible only when its verified YouTube publication date is strictly
earlier than **2023-06-19**, the declared Evolution cutoff. The downloader must
record channel ID, video ID, canonical URL, title, publication timestamp,
duration, and downloaded-media hash before the video can enter an evidence
split. Videos on or after the cutoff are rejected before download when
metadata is available and rejected before mining in all cases. Playlist order,
title text, or filesystem timestamps are not acceptable substitutes for the
publication date.

When the extractor may be wrong, automatically render the relevant original
frames as images/contact sheets with tracks, IDs, coordinates, HP plateaus,
and simulated overlays. Direct frame inspection can confirm or reject that
candidate, but confirmed evidence remains assigned to its original split and
can never be recycled as held-out truth.

Public-video results may prioritize and help design physical-lab experiments,
and may supplement a gate only when the readiness report explicitly permits
that evidence for the measured behavior. They must not replace a controlled
current-version probe for timing, damage, projectile, status, spawn, or other
causal mechanics.

### 5. Leakage-safe evidence lifecycle

Assign every physical run and public video to immutable capture/source groups
before inspection or mining and split groups—not frames—into calibration,
validation, and held-out partitions. Store device, media, replay-cache, and
observation hashes. For the physical lab, lock the capture group and split
before the first observation is inspected; for public video, assign the source
group before extraction. Any material used for parameter discovery, debugging,
direct frame inspection, or regression construction is forever ineligible as
held-out.

Use calibration to estimate parameters, validation to choose implementations,
and untouched held-out groups only for final fidelity claims. Require multiple
independent physical capture groups per version-sensitive mechanic, with
public-video groups retained as supplementary evidence where appropriate.
Audit a small random sample of automatically accepted cases to estimate
observation/oracle error; do not treat either the physical detector or the
vision oracle as perfect.

### 5A. Storage budget and media retention

The complete `cr-bot` workspace must not exceed **200,000,000,000 bytes
(200 GB decimal)**. Public-video acquisition and physical-lab capture must
measure the whole workspace before and after every download/capture/extraction
job and reserve enough space before starting. They must stop or evict eligible
raw media rather than crossing the limit.

Downloaded public videos and raw physical-lab recordings become
eviction-eligible only after extraction/ingest has completed and the retention
manifest contains their source/device metadata, cryptographic media hash,
evidence-group assignment, generated scenario/measurement paths, and compact
audit artifacts needed to interpret the evidence. The budget enforcer
re-hashes the registered file and refuses to delete it if the bytes no longer
match the sealed hash. When space is needed, delete the oldest
eviction-eligible raw media first until the workspace is below a configured
low-water mark. Keep the extracted observations, provenance, hashes, reports,
minimized regression scenarios, and selected contact-sheet/frame evidence.

Deletion must be path-safe and restricted to raw public-video or physical-lab
media registered by the simulator fidelity-media/retention manifests. Never
delete curated ground truth, replay caches, models, source code, tests,
unrelated user videos, or other repository data merely to meet this budget. If
registered raw media are insufficient to return below 200 GB, fail the
acquisition/capture job and report the deficit instead of deleting outside
that scope.

### 6. First-divergence analysis and active learning

Replay sealed physical-lab experiment actions and initial state, together with
any explicitly admitted supplementary mined actions, in the simulator. Match
real and simulated entities and compute time-indexed position, HP, target,
event, and victim-set error. Report the first decision-relevant divergence and
classify it into a subsystem such as data, geometry, movement, targeting,
collision, attack timing, projectile, status, spell, spawn, or observation
uncertainty.

Cluster failures by mechanic and automatically request another controlled
probe or mine supplementary footage for the highest-impact, lowest-confidence
gaps. Prioritize by:

```text
match frequency * decision impact * uncertainty * current failure rate
```

This closes the loop without asking a person to inspect thousands of clips.

## Fidelity gates

Every card and required mechanic must appear in the readiness report.
Readiness is fail-closed and requires:

- 100% exact rule, schema, determinism, replay, invariant, and backend-parity
  tests;
- zero missing eligible cards or required mechanic edges;
- zero unresolved critical source conflicts;
- no invariant failures in the declared cumulative fuzz/soak budget;
- held-out measurements from at least two independent capture groups per
  mechanic, with a minimum sample count chosen before opening held-out data;
- version-sensitive measurements have accepted physical-lab provenance:
  current patch/level, canonical experiment and ruleset hashes, verified
  dual-device captures, synchronization uncertainty, recognized observation
  caches, and immutable capture-group assignment;
- predeclared per-mechanic tolerances based on measurement noise and gameplay
  relevance, not one global accuracy score;
- at least 98% held-out agreement for discrete decision-critical outcomes,
  with stricter thresholds (normally 99%+) for common Hog interactions;
- confidence intervals, sample/group counts, oracle-audit error, and excluded
  cases reported beside every metric;
- no training-ready status while any required mechanic is missing, provisional,
  under-sampled, leaking across splits, or failing its threshold.

The offline/fake physical-lab harness, inferred-only observations, rejected
captures, and public-video candidates cannot satisfy a controlled evidence
requirement. Public-video evidence may supplement a measurement only when its
version applicability and oracle uncertainty are explicitly recorded.

Trajectory position error, timing error, target agreement, victim-set
agreement, HP/damage error, tower-hit count, pull outcome, and divergence time
must be reported separately. “Overall simulator accuracy” alone is forbidden.

## Production-scale RL architecture

The Python engine is the readable reference oracle, not the final throughput
backend. Preserve it while implementing an optimized, structure-of-arrays,
batched backend suitable for many concurrent environments. Prefer integer
state, preallocated pools, compact component masks, cached navigation data,
and deterministic parallel stepping. Heavy fidelity/mining work runs outside
the policy step.

Before optimization, add reproducible workload benchmarks for full matches,
dense swarm states, projectile/spell-heavy states, and policy observation
construction. Derive the required environments-per-second from the intended
trainer's rollout budget and hardware, then make that budget an enforced
performance gate. Do not claim “production scale” from a microbenchmark.

An optimized backend is acceptable only when differential execution against
the Python reference produces identical canonical state and public-event
hashes for reviewed scenarios, randomized streams, save/reload boundaries,
and long seeded matches. It must expose batched reset/step, legal masks,
terminal outcomes, truncation, reproducible seeding, and asynchronous episode
reset without leaking privileged state into the actor observation.

The current intermediate process backend has reproducible benchmark artifacts
for 16 lanes × 20 policy steps: 45.58 environment-steps/s in the sequential
reference and 43.69 environment-steps/s with four serialized workers after
removing redundant worker-side observation projection. Hash parity is proven,
but this throughput is explicitly below a production gate and full state
serialization remains the next optimization target. The final SoA backend must
beat a trainer-derived target rather than this local baseline.

## Delivery order

1. Freeze the roster manifest, exclusions, ruleset, observation/action contract,
   mechanic vocabulary, and card-to-mechanic coverage graph.
2. Complete the physical-lab Phase-0 integration and run one connected,
   reproducible `hog_cannon_pull` or isolated Hog probe before expanding the
   fidelity matrix. Keep all offline/fake runs `candidate_only`.
3. Generalize the current engine for air movement/targeting and the shared
   mechanics needed across the opponent roster.
4. Implement cards in dependency clusters, not arbitrary card order: simple
   troops; ranged/projectiles; swarms/spawns; buildings/spawners; statuses and
   charge states; simple spells; persistent/complex spells; exceptional cards.
5. For each cluster, generate exact/property/fuzz/scenario tests and add
   controlled physical probes plus supplementary calibration/validation miners
   before moving on.
6. Continuously run targeted physical experiments, inspect only ambiguous
   important cases, tune on calibration/validation, and preserve untouched
   held-out groups. Use public videos for discovery and natural-interaction
   cross-checks.
7. Reach complete per-card/per-mechanic readiness and decision-fidelity gates.
8. Profile representative workloads, implement the batched optimized backend,
   prove reference parity, and meet the trainer-derived rollout budget.
9. Begin serious RL training only for mechanics whose readiness gates pass;
   the full V1 release requires every in-scope opponent card to pass.

## Definition of done

V1 is complete only when:

- the fixed player deck can play complete deterministic matches against any
  legal eight-card deck drawn from the complete eligible opponent manifest;
- all in-scope ground, air, building, and spell mechanics are implemented;
- every card-to-mechanic edge has generated exact/property coverage and
  independent held-out sim-to-real reporting, with physical-lab evidence as
  the primary source for version-sensitive behavior;
- the physical-lab experiment, capture, ingest, replay, comparison, and
  first-divergence chain is reproducible for the required probe matrix;
- every generated scenario and physical/public-video discrepancy is
  reproducible from immutable versions and hashes;
- the fail-closed report passes all per-mechanic gates without leakage;
- direct capture/frame audits show the physical observation and supplementary
  public-video oracle error rates are within their declared bounds;
- the optimized batched backend is hash-identical to the reference engine for
  the parity corpus and meets the trainer-derived production throughput gate;
- the policy sees the same versioned observation/action contract in video,
  reference simulation, and optimized simulation.
