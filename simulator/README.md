# Deterministic Level-11 Simulator

The authoritative V1 scope, autonomous fidelity strategy, production-scale RL
target, and definition of done are in [`GOAL.md`](GOAL.md). The implementation
below is the current base Hog-cycle reference and must expand according to that
goal without weakening its fail-closed evidence rules.

V1 fidelity footage is restricted to verified pre-`2023-06-19` publications
from the YersonCz channel. The entire workspace has a 200 GB decimal hard cap;
see `GOAL.md` for the path-safe raw-video eviction contract.

`simulator/` is a headless, versioned Clash Royale simulation package for
training and evaluating policies against the repository's existing vision
feature boundary. The default CLI ruleset is the fixed `v1` artifact, which
declares the complete 109-card eligible opponent roster. The date-stamped
`2026-08-04` ruleset remains an explicit compatibility artifact for older
manifests and source audits. Its original small exact base interaction set is:

```text
Hog Rider, Cannon, Musketeer, Skeletons,
Ice Golem, Ice Spirit, Fireball, The Log,
Princess Towers, and King Towers
```

The bundled `2026-08-04` ruleset is an **executable provisional baseline**, not
a claim that every private game mechanic has been reverse engineered. Exact
Level-11 values, source provenance, conflicts, assumptions, and unresolved
timing/geometry questions are stored in the pinned ruleset. Fidelity reports
describe agreement with held-out observations; they never relabel those
observations as perfect game truth.

The explicit `2026-08-04-roster` artifact expands that boundary to all 109
eligible opponent cards so generated scenarios cannot omit a card. Its
generated definitions are high-impact provisional values and intentionally
fail the fidelity-ready gate until exact field evidence and held-out tests
replace them; it is the coverage target, not a training release.

Synthetic coverage is reproducibly scalable. `generate-scenarios
--per-mechanic N` creates `N` deterministic geometry/timing variants for every
card component; variant zero is the canonical fixture, while later variants
use the mirrored bridge lane and delayed action ticks. `generate-interactions
--variants N` applies the same perturbation to the fixed eight-card player
deck × opponent matrix. These cases are property/determinism evidence only;
they never satisfy the held-out sim-to-real readiness gate.

`validate-generated --workers N` runs the same case validator in isolated
processes for large matrices. Each worker reconstructs the pinned ruleset and
scenario from JSON, and the report is sorted by scenario ID, so worker count
cannot change state/event hashes or pass/fail outcomes. The default is one
worker for fast commit CI; larger worker counts are intended for PR/nightly
coverage and still run the final invariant check when `--no-tick-validation`
is selected. Add `--require-complete` to require all 109 roster cards and the
complete declared mechanic, fixed-deck interaction, or unordered-pair scope;
focused manifests remain available without that release gate. Strict
roster-mechanic validation also requires every case to carry at least one
behavioral event predicate or final-state check; card-play acceptance alone
cannot satisfy mechanic coverage. Action-boundary interaction and opponent-pair
manifests keep their separate required-play contract.

The current v1 matrix is intentionally red under that strict behavioral gate:
1,076/1,076 cases execute deterministically, while 259 structural cases still
need explicit event/state obligations.

For exhaustive two-card opponent coexistence coverage, generate and validate
the unordered 109-card pair matrix (5,886 cases) against the fixed player
deck:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 \
  generate-opponent-pairs \
  --json-out outputs/simulator/generated-opponent-pairs-v1.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 \
  validate-generated outputs/simulator/generated-opponent-pairs-v1.json \
  --json-out outputs/simulator/generated-opponent-pairs-validation-v1.json \
  --workers 4 --repeats 2 --no-tick-validation --require-complete
```

The pair report is a deterministic synthetic gate. It proves that every
ordered action boundary can instantiate two distinct eligible opponent cards;
it does not count as held-out sim-to-real evidence or make the readiness report
green.

For V1 runtime/training, use the single fixed artifact `rulesets/v1.json`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  outputs/venv/bin/python -m simulator --ruleset v1 run --seed 7
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  outputs/venv/bin/python -m simulator --ruleset v1 reconcile-data \
  --json-out outputs/simulator/v1-card-data-reconciliation.json
```

The date-stamped files remain pinned provenance/compatibility artifacts only;
V1 does not select a balance version or roster dynamically. A later V2 can
introduce timestamped migrations without changing V1 data in place.

## Quick start

### Physical-fidelity lab (offline Phase 0)

The software-only lab harness is available before the two phones are connected.
It seals canonical experiments, exercises fake phone/capture adapters and the
screen-verified lifecycle, aligns capture clocks, replays the logical actions
through the pinned simulator, and keeps the result `candidate_only` until a
real observation extractor supplies evidence:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab plan \
  --hog-cannon-only \
  --json-out outputs/simulator/fidelity_media/physical_lab/plan.jsonl
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab run \
  --experiment outputs/simulator/fidelity_media/physical_lab/plan.jsonl \
  --mode offline \
  --json-out outputs/simulator/fidelity_media/physical_lab/offline-summary.json
```

Use `lab ingest` for detector rows and `lab compare` for a divergence report.
`lab run --mode adb` is available for later device preflight and can use the
hash-verified reviewed lifecycle manifests supplied with
`--lifecycle-templates-a` and `--lifecycle-templates-b`. Without both manifests
it fails closed; with them, only the coarse lifecycle gate is enabled, while
continuous capture, replay-cache, observation, and readiness gates remain
mandatory. Physical recordings and caches remain under the ignored `outputs/`
tree.

### One-phone preparation and fixed Testspiel order

The versioned campaign manager creates the automated card-variant sweep. It
starts with isolated troops and becomes progressively more complex; each case
records its ordered decks and first-four opening hand:

```bash
outputs/venv/bin/python scripts/run_physical_fidelity_campaign.py plan
```

Phone B is the host. Set both explicitly mapped phones to stay awake while
powered (the command is serial-scoped and does not discover unspecified
devices):

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab keep-awake \
  --serial-a PHONE_A --serial-b PHONE_B
```

While only one phone is available, prepare its deck and leave it in the lobby.
The preparation manifest is bound to the phone's hashed ADB serial and can be
validated later when the second phone is connected:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab prepare \
  --serial PHONE_A --side A \
  --json-out outputs/simulator/fidelity_media/physical_lab/preparation-A.json
```

The default fixed Hog cycle is written in this order:
`hog-rider, cannon, musketeer, skeletons, ice-golem, ice-spirit, fireball, log`.
For a campaign case, the autonomous preparation changes the requested card(s)
and records the resulting deck. When both phones are available, prepare both
phones for the selected case. Phone B alone then long-presses the `1v1 Battle`
button, enables the reviewed fixed-order toggle, and hosts the match; Phone A
accepts. Therefore the first four cards in each ordered deck are the opening
hand and the remaining four are the first replacements.
The two pixel points must be reviewed for the installed game build before use:

```bash
outputs/venv/bin/python scripts/run_physical_lab_autonomous.py \
  --serial-a PHONE_A --serial-b PHONE_B --no-prepare \
  --campaign outputs/simulator/fidelity_media/physical_lab/campaigns/physical-fidelity-interaction-sweep-v1/campaign.json \
  --case-id hog-cannon-pull \
  --preparation-a outputs/simulator/fidelity_media/physical_lab/preparation-A.json \
  --preparation-b outputs/simulator/fidelity_media/physical_lab/preparation-B.json \
  --fixed-deck-order \
  --fixed-deck-toggle-point TOGGLE_X,TOGGLE_Y \
  --test-match-start-point START_X,START_Y \
  --calibration-a path/to/A.json --calibration-b path/to/B.json \
  --lifecycle-templates-a path/to/A-templates.json \
  --lifecycle-templates-b path/to/B-templates.json
```

After each admitted case has produced a fidelity corpus, re-evaluate the
whole stored campaign after a simulator change. Use a new output path for each
evaluation snapshot because case evidence is write-once:

```bash
outputs/venv/bin/python scripts/run_physical_fidelity_campaign.py evaluate \
  --campaign outputs/simulator/fidelity_media/physical_lab/campaigns/physical-fidelity-interaction-sweep-v1/campaign.json \
  --results-root outputs/simulator/fidelity_media/physical_lab/campaigns/physical-fidelity-interaction-sweep-v1/results \
  --json-out outputs/simulator/fidelity_media/physical_lab/campaign-evaluation-v2.json
```

The coordinator writes a sealed `run.json` and
`observation-handoff.json` under the run output directory. After a reviewed
detector has produced normalized observations and the handoff's recognized
primary replay cache, compile the standard fidelity corpus/report with:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab fidelity \
  outputs/simulator/fidelity_media/physical_lab/<run-id>/observations.json \
  --run outputs/simulator/fidelity_media/physical_lab/<run-id>/run.json \
  --replay-cache outputs/simulator/fidelity_media/physical_lab/<run-id>/replay-cache-A.pkl.gz \
  --corpus-out outputs/simulator/fidelity_media/physical_lab/<run-id>/fidelity-corpus.json \
  --json-out outputs/simulator/fidelity_media/physical_lab/<run-id>/fidelity-report.json
```

This bridge remains fail-closed and does not turn a candidate-only run into
training readiness.

Do not use `--start-test-match` during the one-phone deck-only preparation:
fixed order is a Testspiel option for the host challenge and is applied by the
two-phone coordinator once both devices are ready.

Run commands from the repository root with the project environment:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 ruleset
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset 2026-08-04-roster ruleset
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 roster
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 run --seed 7
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 check-determinism --seed 7
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 audit --seeds 4 --max-ticks 1000
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 benchmark-vector \
  --envs 16 --steps 20 --backend process --workers 4
PYTHONPATH=.:src outputs/venv/bin/python -m simulator scenario \
  simulator/scenarios/hog_cannon_pull.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator mine-video-truth \
  outputs/simulator/fidelity_media/source-manifest.json \
  --json-out outputs/simulator/fidelity_media/truth.json \
  --retention-out outputs/simulator/fidelity_media/retention.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator discover-video-source \
  --source data/audio_classifier/mined/sources.jsonl \
  --json-out outputs/simulator/fidelity_media/source-manifest.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator extract-video \
  outputs/simulator/fidelity_media/source-manifest.json \
  --json-out outputs/simulator/fidelity_media/extractor-plan.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator plan-action-windows \
  outputs/simulator/fidelity_media/source-manifest.json \
  data/audio_classifier/mined/candidates \
  --json-out outputs/simulator/fidelity_media/action-window-plan.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 \
  validate-generated outputs/simulator/generated-roster-scenarios.json \
  --json-out outputs/simulator/generated-roster-validation.json \\
  --workers 4 --require-complete
PYTHONPATH=.:src outputs/venv/bin/python -m simulator reconcile-data \
  --json-out outputs/simulator/card-data-reconciliation.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator mine-replay-tracks \
  outputs/simulator/fidelity_media/source-manifest.json \
  --video-id F3lqHvlPfOU \
  --cache outputs/simulator/fidelity_media/extractor/F3lqHvlPfOU/standard/replay-cache.json \
  --hud-variant standard \
  --json-out outputs/simulator/fidelity_media/tracks.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator mine-replay-batch \
  outputs/simulator/fidelity_media/source-manifest.json \
  --extractor-root outputs/simulator/fidelity_media/extractor-f3-20 \
  --extractor-root outputs/simulator/fidelity_media/extractor-gx-20 \
  --hud-variant both \
  --json-out outputs/simulator/fidelity_media/tracks-merged.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 \
  discover-replay-interactions \
  outputs/simulator/fidelity_media/extractor-f3-20/F3lqHvlPfOU/standard/replay-cache.json \
  --source-level 11 \
  --json-out outputs/simulator/fidelity_media/autonomous-interactions.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 \
  merge-replay-interactions \
  outputs/simulator/fidelity_media/autonomous-interactions-standard.json \
  outputs/simulator/fidelity_media/autonomous-interactions-alternative.json \
  --require-both-hud \
  --json-out outputs/simulator/fidelity_media/autonomous-interactions-dual-hud.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator --ruleset v1 \
  compile-video-truth outputs/simulator/fidelity_media/truth.json \
  --json-out outputs/simulator/fidelity_media/corpus.json
```

The batch miner also consumes bounded action-window roots emitted by
`plan-action-windows --execute`, such as
`<video-id>:action-window:000/{standard,alternative}/replay-cache.json`.
Each window is retained as a distinct `source_group_id`, so same-video windows
cannot overwrite one another during HUD merging or truth mining; split
assignment still keeps all evidence from that video in one leakage-safe split.

`discover-video-source` is the only command that contacts the source resolver;
it requires exact YouTube metadata and rejects every publication on or after
`2023-06-19`. `extract-video` is a dry-run by default and schedules both
standard and shifted-HUD profiles for every accepted video. Add `--execute`
only when the local media path and the 200 GB budget have been checked; bounded
jobs can use `--video-start-time` and `--video-duration`. Failed or ambiguous
jobs stay in the JSON report and are never silently promoted to truth.
`mine-replay-tracks` is the deterministic adapter from those caches to the
confidence-gated `mine-video-truth` manifest; it retains cache hashes and
discards unstable IDs, stationary buildings, and crowded detections before any
movement-fidelity case exists. `mine-replay-batch` accepts repeated
`--extractor-root` values so resumed/high-scale extraction runs are merged per
source group without duplicate evidence; full videos use the video ID and
bounded windows use their window ID. The selected cache and all alternatives
are recorded in provenance. Building lifetime/damage miners consume those same
caches separately. `plan-action-windows` uses the repository's
confidence-gated card/action candidates only to select short, reproducible
windows (it never promotes a candidate to truth), then schedules both HUD
profiles for each window. Unsupported forms, unanchored timestamps, and low
confidence rows are retained as rejection reasons.
`discover-replay-interactions` is the action-free companion for high-scale
mining. It scans one or more sealed caches, infers only detector track-onset
action candidates, and searches for bridge/path-topology crossings, Cannon
lifetime/HP-decay signatures, and Hog-to-Cannon approaches. It never writes an
action label or a gold truth: every candidate carries `truth_promoted: false`,
and every ambiguous window is retained in `rejected`. A source with a different
card level may be included only with `--level-invariant-current-ruleset` and an
exact `--expected-support-tower-hp`; cross-level Cannon HP/lifetime rows remain
explicitly rejected while topology evidence can still be reviewed.
When a bounded window omits the tower frame, a separately hashed full cache
from the same video can prove its level with repeated `--level-proof-cache`
arguments. The miner derives a stable video key from the cache paths, permits
inheritance only for that exact key, and records every proof hash; a proof from
another video or a cache without the exact declared support-tower HP remains
rejected.
`merge-replay-interactions` reconciles standard and alternative HUD candidate
reports from the same source video. It pairs only matching
card/owner/mechanic/onset/geometry observations, leaves unmatched rows in
`rejected`, and keeps `truth_promoted: false`; the two HUD renders are a
quality cross-check, never independent held-out truth. `--require-both-hud`
is useful in CI when a source must have both profiles before it contributes
paired candidates.
When both HUD jobs are available, `mine-replay-batch --hud-variant both` also
records frame-aligned trajectory agreement (position MAE/max error and matched
track IDs).  The selected source still contains only one HUD interpretation;
the second run is a quality cross-check, never independent evidence and never
a bypass around the confidence or held-out gates.  A disagreement is retained
as an auditable candidate failure rather than being averaged away.
`mine-video-truth` also applies a simulator-independent motion-quality gate:
tracks must have enough elapsed time and endpoint displacement, continuous
frame sampling, a bounded generic step speed, and a sufficient fraction of
moving intervals. The gate rejects detector linger/teleport artifacts and
stores every reason in `discarded`; it never compares against the card's
expected speed, so held-out evidence cannot leak simulator constants. For
high-fidelity runs, `--maximum-path-to-displacement-ratio` rejects tracks whose
detector path is implausibly zig-zagged, while `--maximum-speed-iqr-ratio`
rejects unstable interval-speed measurements using only the observed track.
Both thresholds and the resulting quality metrics are sealed in the truth
manifest; rejected rows remain auditable and are never silently promoted.
`compile-video-truth` then creates mid-track scenarios and target-independent
movement-speed measurements for the existing `fidelity`/`readiness` gates. It
does not invent an unseen target from an isolated trajectory; absolute position
checks require a separately reconstructed action/path scenario.

The compiler keeps the speed oracle explicit with `--speed-estimator`: the
legacy `endpoint` measurement is retained for reproducibility, while
`path_length` sums the ordered detector trajectory and is preferred for curved
lane movement; `median_step` is available as a robust diagnostic. The selected
estimator is sealed in every displacement observation and corpus, so changing
it cannot silently rewrite an earlier truth set.

Extractor jobs are resumable: an opaque/legacy replay cache is treated as a
sealed artifact and skipped, while a recognized cache is checked against its
requested start/end coverage. A valid-but-truncated prefix is retained with
its hash and automatically repaired on resume; `--rerun-existing` remains the
explicit opt-in for complete-cache recomputation. A source manifest may carry
`analysis_start_time_s` and `analysis_duration_s` per video, which lets an
upstream gameplay-span detector avoid menus without applying one global
timestamp to every recording. When raw downloads live outside the simulator's
default media directory, pass the same workspace-relative root to
`mine-video-truth --raw-root-relative` and `media-budget --raw-media-root`.
Eviction validates the registered SHA-256 against the file before deletion.
Each executed job also has a wall-clock timeout (30 minutes by default;
override with `--job-timeout-s`). A timeout is recorded as a failed, auditable
job and cannot be promoted to truth, so an unreadable codec or stalled model
process cannot block an autonomous batch indefinitely.

`reconcile-data` audits the complete 109-card opponent roster by default. It
reads the pinned Level-11 snapshot in `sources/level11_card_stats.json`, applies
only the field-level official patch rows in
`sources/official_level11_overrides.json`, and records every mismatch. Use
`--strict` in CI: missing structured data and unresolved conflicts return a
non-zero status. A structured source is comparison evidence, not authority;
official values win, and a missing value remains blocked rather than being
inferred from the old level-16 repository table.

Run the simulator suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  outputs/venv/bin/python -m pytest -q -p no:cacheprovider tests/simulator
```

## What is implemented

- Integer-only authoritative state:
  - microseconds for time;
  - milli-elixir for resources;
  - milli-tiles for world positions;
  - integer HP and damage.
- Stable entity/projectile UIDs, canonical JSON, SHA-256 state hashes, explicit
  SplitMix64 state, deterministic UID iteration, and a pinned engine algorithm
  version (`reference-0.29.0`).
- Full regulation, overtime, double/triple elixir, crowns, King activation,
  sudden death, and raw-HP tiebreak termination.
- Eight-card hands/cycle, exact costs, legal placement checks, rejected-action
  events, and simultaneous two-player action input.
- Ground and air troop footprints cannot overlap either owner’s buildings or
  Crown Towers at deployment. A Mirror immediately after a Mirror is rejected
  so the fixed V1 action boundary cannot create recursive Mirror chains.
- Furnace follows the post-August-2025 rework as a moving medium-speed troop:
  its Level-11 body attacks while walking and emits one Fire Spirit through an
  unbounded deterministic cadence while it lives. The source describes the
  cauldron attack as single-target; exact projectile/launch geometry and any
  hidden live-child cap are kept in `UNKNOWN_BEHAVIORS.md` until isolated
  evidence resolves them.
- Heal Spirit is an explicit suicide-impact support component: its Level-11
  official Spirit HP override remains `215`, while the DeckShop Level-11
  reference supplies a provisional `532` heal for friendly air/ground troops
  inside a `1.5`-tile impact radius. Damage resolves before healing and
  buildings, Crown Towers, enemies, full-health troops, and the dead source
  body are excluded; edge/animation semantics remain in the open-behavior
  ledger.
- Cannon Cart uses an in-place health-threshold transform: at 50% of its
  shared Level-11 HP pool it becomes a stationary building without allocating
  a second UID, then follows building lifetime/linear-decay rules. The
  transform trigger is official; first stationary shot and exact decay remain
  held-out validation targets.
- Compound death streams are data-driven and recursive: Golem emits two
  Golemites, Elixir Golem emits two Elixir Golemites which each emit two
  Elixir Blobs, Lava Hound emits six airborne Lava Pups, and Goblin Giant
  carries two single Spear Goblins that attack while attached and releases
  those same bodies on death. Child forms are hidden, Level-11 definitions
  with their own targeting/attack/death components; split offsets, carried-body
  timing remain explicit video/in-game validation targets. Elixir Golem's
  opponent award is applied at every death stage in milli-elixir units.
- Obstacle-aware visibility-graph bridge routing, building pulls,
  target legality/persistence/retarget, sourced mass-weighted local separation, deployment
  delay, targetable deploying buildings, placement-started linear building
  HP/lifetime decay, attacks, projectiles,
  splash, structure/terrain-clipped knockback, slow/freeze, and death effects.
- A serialized, UID-ordered persistent-area component for temporal damage,
  status, displacement, and troop-spawn zones. The roster artifact currently
  uses it for Arrows' three waves and provisional Poison, Earthquake,
  Graveyard, Tornado, Rage, and
  Goblin Curse definitions; Goblin Curse also carries a death-transform
  status that emits a caster-owned one-body Goblin. Tornado uses a deterministic
  two-pulse damage schedule plus a pull-active tail, while Rage combines a one-shot damage schedule with a
  friendly speed/hit-speed aura. Each pulse re-evaluates entrants and expires
  deterministically rather than freezing the impact-time victim set.
- Current base-card components also cover Goblin Hut's visible-enemy proximity
  clock, Tesla underground concealment and spell exceptions, Electro Spirit's
  delayed nine-target chain, Zappies stun, Bomb Tower/Tombstone/Hut death
  payloads, river jumping with an airborne river layer, and Barbarian Barrel's
  swept rolling collision plus endpoint Barbarian.
- Clone is an explicit impact component: it snapshots eligible friendly troop
  bodies in its three-tile radius, excludes buildings/enemies/existing clones,
  and emits one-HP copied entities whose ordinary card mechanics continue to
  run. Spawn offset and edge/target exceptions remain held-out unknowns.
- Electro Dragon uses a bounded nearest-neighbour chain component (three total
  targets, three-and-a-half-tile hops) rather than single-target damage; each
  bolt emits a replay event and applies the shared electric stun/reset.
- Shields are first-class damage layers for Dark Prince, Guards, and Royal
  Recruits; an incoming hit is fully consumed by the shield before body HP is
  touched, with deterministic break events. Royal Ghost now has an explicit
  reveal/re-cloak lifecycle, and Miner has an anywhere-ground tunnel phase
  that is hidden until emergence.
- Mixed swarm cards materialize their real child bodies (Goblin Gang's Goblins
  and Spear Goblins; Rascals' Rascal Boy and Rascal Girls) rather than cloning
  aggregate parent stats. Magic Archer line sweeps, Executioner outbound and
  return passes, Hunter's ten-pellet fan, and Bowler projectile-direction
  knockback are serialized projectile components.
- Mega Knight jump/landing damage, Electro Wizard deployment stun, Ice Wizard
  deployment freeze, Lumberjack death Rage, Mother Witch curse conversion to
  a caster-owned Cursed Hog, Ram Rider snare, and Witch's three-Skeleton death
  spawn are executable and covered by deterministic component tests. Their
  exact animation/edge timing remains a fidelity-gated unknown where noted in
  `UNKNOWN_BEHAVIORS.md`.
- Electro Wizard uses a discrete two-target bolt component. Its three-tile
  source radius is not treated as splash, so a third nearby legal target is
  left untouched; selection, bolt spacing, and exact tie breaks remain held-out
  fidelity questions.
- Electro Giant uses a reactive reflection component: a nearby concrete
  attacker receives the separate Level-11 reflected body/Crown-Tower damage,
  stun/reset, and an auditable `reflected_damage` event. Spell/no-source and
  recursive reflection edge cases remain held-out questions.
- Prince, Dark Prince, Battle Ram, and Ram Rider use one generic movement-charge
  component: Battle Ram arms after 3.5 tiles, Prince and Ram Rider after 2.5
  tiles, and Dark Prince after 3.0 tiles. Charging doubles medium movement
  speed and resets on impact, hard crowd control, knockback, or retarget. Exact
  live trigger/reset frames remain in the open-behavior ledger.
- Bandit, Fisherman, and Firecracker use reusable dash, hook/pull, and
  splash/recoil components respectively. Their boundary geometry and timing
  remain explicitly provisional until high-confidence video traces fit them.
- Structured projectile muzzle offsets for Musketeer, Cannon, Princess Tower,
  and King Tower, so flight timing starts at the weapon rather than the entity
  center.
- Card components needed by the base Hog-cycle interaction set, including
  video-calibrated Skeleton formation, Ice Golem death slow, Ice Spirit
  jump/suicide/freeze, Fireball, and a continuous piercing Log.
- Integer boundary regressions pin Fireball victim geometry and The Log's
  continuous lateral collision at the exact target-collision edge; these are
  executable simulator contracts while their real-game widths remain explicit
  held-out fidelity requirements.
- The official June 1, 2026 blanket spell nerf is applied before the August
  ruleset: Fireball deals `172` Crown Tower damage and The Log `35`. These
  Tier-A values override the older third-party-derived `207`/`40` baseline.
- Phoenix has a data-driven one-time rebirth lifecycle: its provisional
  Level-11 death burst creates a targetable 317-HP egg for 3.8 s, which hatches
  one full-stat 1,052-HP/217-damage Phoenix; the reborn body cannot recurse.
- Ice Spirit uses the official July 2025 `1.1s` freeze duration in addition to
  the August 2026 `215` HP and unassisted Crown Tower connection behavior.
- Cannon uses the official April 2026 Level-11 damage nerf (`212 → 202`).
- Complete unattended matches through deterministic controllers.
- Roster-wide executable smoke coverage for every eligible opponent card,
  including air navigation, passive spawners, status/DoT payloads, and death
  spawns. These shared components remain training-blocked until card-specific
  evidence passes.
- A generated-scenario validation report that runs every roster card/mechanic
  case twice and requires identical state, event-log, and replay hashes; the
  generated oracle also requires the scheduled setup cards and branch events
  needed to exercise attacks, victims, projectiles, statuses, spawns,
  transformations, and lifetime expiry rather than accepting targetless
  deployments;
  current fixed V1 109-card artifact covers 4,304 four-variant scenarios
  across 66 mechanics, including passive-spawner lifecycle cases, Inferno
  ramp stages, Phoenix rebirth, Heal Spirit impact healing, Furnace
  movement/spawning, Cannon Cart transformation, and recursive split/death
  streams, with zero synthetic failures. This proves deterministic executable
  coverage, not real-game fidelity.
- An exhaustive unordered opponent-pair action-boundary matrix covers all
  `C(109, 2) = 5,886` distinct eligible-card pairs against the fixed Hog-cycle
  player deck. The pinned V1 report passes `5,886/5,886` repeated runs with
  zero determinism failures. Pair coverage is synthetic exercisability only;
  it never substitutes for mechanic-specific held-out fidelity.
- A second generated interaction matrix covers all `109 × 8 = 872` eligible
  opponent/card combinations against the unchanged eight-card player deck.
  Each case requires both scheduled cards to produce a `card_played` event;
  the current matrix passes `872/872` repeated-hash runs with final-state
  invariant validation. A strict per-tick sample covers all 109 opponent cards
  against Hog Rider (`109/109` repeated-hash passes). Use strict per-tick
  validation on smaller samples and `validate-generated --no-tick-validation`
  for the large nightly matrix.
- Canonical JSON scenarios shared by tests, regressions, benchmarks, and
  observed validation cases.
- A two-player training environment and batched wrapper with a 250 ms default
  decision cadence independent of the 50 ms provisional physics tick.
- Per-mechanic sample/trace fidelity metrics with MAE, p95, event divergence,
  agreement rate, and Wilson confidence intervals.
- A sealed replay-cache → video-truth → fidelity-corpus path that preserves
  video/cache hashes, capture groups, HUD profile, and split provenance; a
  calibration smoke corpus has been run against the roster ruleset and its
  disagreement remains visible instead of being promoted.

## Architecture

```text
rulesets/2026-08-04.json
          │
          ▼
      ruleset.py ─────────────── provenance + immutable content hash
          │
scenario.py ──► engine.py ───── authoritative state + public events
                  │   │
                  │   ├────────► env.py ──► policy decisions / rewards
                  │   │
                  │   └────────► observation.py ──► vision-v1 tensors
                  │
observed corpus ──┴────────────► validation.py / fidelity.py ──► JSON report
```

The simulator's `BattleState` is deliberately separate from
`cr_bot.domain.game_state.GameState`. The latter is a lossy observed DTO built
from video. `observation.py` is the one-way adapter between them. A policy does
not receive exact opponent elixir/hand/cycle, entity targets, cooldowns, status
timers, RNG state, or other privileged physics fields.

The player hand/action vocabulary remains the fixed eight-card Hog deck, while
eligible non-deck opponent plays are still charged and encoded in the existing
seen-opponent-card channel. Internal split forms (for example Golemites and
Lava Pups) use a public-card feature profile only for aggregate board channels;
their authoritative IDs and mechanics remain in `BattleState`.

## Ruleset and truth contract

`simulator/rulesets/2026-08-04.json` is network-free and immutable. Its hash and
the engine algorithm version are embedded into every state and scenario.
Loading fails if either contract differs. Any behavior-changing engine edit
must bump `ENGINE_VERSION`; source-only refactors that preserve canonical
states and public events do not.

Every card/tower field includes source IDs. Sources carry:

- confidence tier;
- URL or local lineage;
- retrieval/publication time;
- optional content hash;
- notes about shared derivation.

The file records source conflicts explicitly. For example, official August
2026 Spirit HP (`215`) overrides an older third-party `230` value. Unknowns are
not omitted: provisional movement conversion, the 50 ms physics tick, tower
timings, and spell trajectories all have uncertainty and resolution records.
Skeleton formation is pinned to a locally measured Level-11 video layout with
its evidence recorded in the ruleset provenance.

The generated roster now takes Level-11 scalar stats from a checked-in,
content-hashed structured snapshot rather than the repository's level-16
`CARD_METADATA` fallback. Current explicit official corrections include the
July Baby Dragon/Electro Giant/P.E.K.K.A/Ram Rider changes, June building and
spell Crown-Tower changes, and the August Spirit, Void, X-Bow, Mortar,
Barbarian, Goblin, and spell/mechanic changes. This still does not make the
roster training-ready: unsupported child units, geometry, timings, and
card-specific behavior stay visible in the reconciliation report until
held-out video evidence resolves them.

DeckShop is retained as an independent Level-11 corroboration source for card
pages (Battle Healer is 1920 HP, 268 damage, 2.0 s hit speed; the Furnace
snapshot is 896 HP, 135 damage, 1.7 s hit speed, 5.5-tile range, and medium
movement). It is lower priority than official patch notes and is never used to
hide a conflict: the stale structured Furnace row remains visible in the base
comparison, while the current DeckShop snapshot resolves the scalar conflict
in the CLI's multi-source audit. Exact rework mechanics remain open. Questions
that need an in-game answer are tracked in
[`UNKNOWN_BEHAVIORS.md`](UNKNOWN_BEHAVIORS.md) and
[`unknown_behaviors.json`](unknown_behaviors.json).

The V1 opponent roster is checked separately from mechanics implementation:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator roster \
  --require-release-verification --require-coverage \
  --json-out outputs/simulator/roster-contract.json
```

This command is intentionally fail-closed today: exact per-card release-date
lineage and implementation coverage must be completed before the full roster
can be declared training-ready.

The roster-complete artifact is executable but not ready:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator \
  --ruleset 2026-08-04-roster roster \
  --require-release-verification --require-coverage
```

It must exit non-zero until generated definitions are replaced by
field-level Level-11 evidence and held-out validation. `implemented` means
dispatchable; `fidelity_ready` means evidence-gated.

Inspect them without loading game/vision models:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator ruleset
```

## Policy boundary

`PolicyObservationV1` preserves the current feature schema:

```text
board          float32 [21, 32, 18]  channel, row, column
global_vector  float32 [768]
spatial_masks  bool    [4, 32, 18]   exact legacy vision-v1 masks
legal_play     bool    [4, 32, 18]   authoritative legality in SimulatorEnv
legal_wait     bool
```

The imported channel order, normalizers, action grid, ground mask, base card
IDs/metadata, and tensor shapes are sealed as observation contract
`vision-v1-exact-1` with SHA-256
`fa42f20ddce8a6d9d5ee2089e415d610834ba324d2fc8195dc618b556f206d79`.
Import fails if those unversioned live feature dependencies drift without a
reviewed contract bump. `exact_policy_inputs()` returns precisely the live
stack's board/global/spatial-mask tensors; `legal_play` and `legal_wait` are
additional simulator-side action-safety signals.

Actions remain:

```text
Wait
Play(card_slot=0..3, cell=(column, row))
```

Cells use `(column, row)`, with `(0, 0)` at the top-left, rows `15/16` at the
river, and the bottom player's deployment starting at row `17`. Viewer 1 is
rotated 180 degrees on observation and action decode.

The old Cannon spatial mask interpreted a cell as a 3×3 footprint's top-left
anchor. Real action labels and this engine use the deployment center.
`spatial_masks` retains the old tensor for checkpoint/debug parity;
`legal_play` contains the corrected center-based action mask and is the mask
new training code should use. `SimulatorEnv` supplies the engine legality
callback, including elixir, dynamic post-tower territory, footprints, and
occupied buildings. Direct adapter callers that need that guarantee must pass
the same callback; otherwise the adapter intentionally uses a conservative
geometry-only fallback.

Policy-v1 has IDs only for the eight base cards. Evo Cannon, Evo Skeletons,
Hero Musketeer, Hero Ice Golem, and abilities fail explicitly instead of being
folded into base IDs or encoded as zero. The core action schema already has
`UseAbilityAction`, but the base ruleset rejects it until a versioned form and
policy-v2 vocabulary are pinned.

### Additive public hybrid observation contract

[`observation_v2.py`](observation_v2.py) adds `PolicyObservationV2` without
changing the sealed V1 tensors. It snapshots the V1 raster/global inputs and
adds up to 128 public entity rows with 32 normalized features plus an entity
mask. Rows must come from the same public observation boundary; authoritative
simulator state and private opponent hand/elixir values are not accepted. The
schema and hash are validated at construction time, and all arrays are
read-only snapshots. [`observation_v2_adapter.py`](observation_v2_adapter.py)
now provides the trusted simulator-side public projection: hidden entities and
private fields are dropped before rows are built, while the existing V1
legality and memory boundary is reused. `SimulatorEnv.observe_v2()` exposes it
as an explicit opt-in; legacy `observe()` and the smoke trainer remain V1.
The current public DTO does not expose velocity, target, status, or temporal
features, so those V2 slots are intentionally zero rather than reconstructed
from authoritative state.

## Python training API

```python
from cr_bot.domain.game_state import Action
from simulator.env import SimulatorEnv

env = SimulatorEnv(decision_interval_us=250_000)
player_0_obs, player_1_obs = env.reset(seed=42)

transition = env.step((
    Action(kind="Play", card_idx=0, cell=(3, 23)),
    Action(kind="Wait"),
))

next_observations = transition.observations
rewards = transition.rewards
done = transition.terminated
legal_actions = next_observations[0].legal_play

# Explicit opt-in to the additive public hybrid contract:
hybrid_observations = env.observe_v2()
```

Default `info` exposes only version identifiers, cadence, terminal outcome,
and reward version. Exact target/damage events, state/event/replay hashes, and
the authoritative state are absent unless `expose_privileged_info=True` is
explicitly selected for deterministic debugging or a privileged critic.

`SimulatorEnv` uses the engine's training mode, which skips the expensive
full-state schema walk after every physics tick. Reset/load boundaries are
still validated, and strict `BattleEngine(validate_every_tick=True)` remains
the default for tests, audits, and fidelity evaluation. A regression test
requires strict and training modes to produce identical replay hashes.
`VectorSimulatorEnv` keeps the same policy boundary in three modes. The
default `backend="reference"` executes parent-owned lanes sequentially. The
explicit `backend="process"` mode advances independent serialized lanes in
isolated workers, while `backend="packed-process"` crosses the same boundary
with the bounded, deterministic fixed-stride ABI in
[`packed_batch.py`](packed_batch.py). Both worker modes reinstall canonical
states in the parent and rebuild observations with the parent's temporal
memories. All modes must produce the same state/event hashes;
`close()` or a context manager shuts down workers. The packed mode is a
transport milestone and deterministic scaling baseline, not yet the final
structure-of-arrays physics implementation.

### Bounded PPO smoke training

The repository now includes a NumPy-only, masked PPO smoke trainer. It is
intentionally a small factorised linear actor/critic, so it can run without
PyTorch and is best used to catch simulator regressions before investing in a
large neural policy. The learner sees only `PolicyObservationV1`, masks every
illegal card/cell, and records the ruleset, engine, reward, and observation
contract in an `.npz` checkpoint.

The trainer's default reward is the sparse `terminal-outcome-v1` objective
(`+1` win, `-1` loss, and `0` for non-terminal decisions). The environment
still exposes the explicit `tower-damage-crowns-v1` shaping preset for smoke
experiments and regression comparisons; its reward version is sealed in the
checkpoint metadata.

The bundled `v1` ruleset is still marked `training_ready: false`. A bounded run
must opt into provisional smoke mode explicitly:

```bash
PYTHONPATH=.:src python3 -m simulator --ruleset v1 train \
  --steps 10000 --envs 8 --rollout-steps 128 \
  --opponent scripted --allow-provisional-smoke \
  --checkpoint-out outputs/simulator/training/ppo-smoke.npz \
  --json-out outputs/simulator/training/summary.json
```

The command can resume a compatible checkpoint with `--checkpoint`. Evaluate a
saved policy on fixed seeds with:

```bash
PYTHONPATH=.:src python3 -m simulator --ruleset v1 evaluate \
  --checkpoint outputs/simulator/training/ppo-smoke.npz \
  --episodes 8 --seed 10000
```

`scripted` uses the deterministic cycle controller; `self-play` uses the same
policy for both players. This is a simulator smoke test, not a live-device
bot: connecting the visual extractor and physical card placer still requires
an observation/action bridge and separate physical-fidelity validation.

Evaluation caps are explicit censoring boundaries. Omitting
`--eval-max-decisions` or `--max-decisions` evaluates through the ruleset's
regulation-plus-overtime horizon; a smaller cap reports truncated episodes
separately and excludes them from win/loss/draw rates. The Python
`PPOConfig` defaults to an undiscounted match result (`gamma=1`); its
`discount_time_constant_us` option can derive a real-time discount when a
finite horizon is preferred.

Serious runs can provide `--training-profile profile.json`. A
`TrainingProfile` fixes the Hog player deck, opponent scope, ruleset identity,
and required cards/mechanics. `smoke` profiles are explicitly provisional;
`training` and `evaluation` profiles require a matching, ready evidence report.
The global V1 roster remains fail-closed even when a narrower scoped report is
ready.

The optional [`rl/`](rl/) package contains the recurrent policy foundation: it combines
the V2-compatible hybrid raster/entity/card encoder, a Transformer over public
entity and hand-card tokens, a GRU over time, and WAIT/PLAY → card → placement
masked action heads. In the strategic variant, each hand slot is decoded from
the stable public card table, mapped through a learned card-identity embedding,
and given a slot embedding before entering that shared Transformer. It stores
recurrent trajectory tensors and reset masks, and includes training-only
opponent-belief heads, a detached-feature privileged critic, and clipped PPO/
behavior-cloning objective functions. `RecurrentPPOLearner` now provides
sequence-preserving PPO updates, optional temporal chunks with hidden-state
snapshots, rollout continuation, and complete checkpoint continuation.
`RecurrentRolloutCollector` bridges initialized `SimulatorEnv` lanes to those
tensors while keeping actor inputs public; an optional callback supplies
privileged critic features, and exact enemy elixir/hand/next-card targets can
be recorded for belief losses. The generalized runner uses the larger
strategic actor by default, stages opponent sampling, and can update a
frozen-checkpoint PFSP payoff book from completed lane outcomes. Serializable
confidence-weighted BC curriculum and deck-conditioned historical/exploiter
league sampling are also available in `rl.curriculum` and `rl.league`.
Persistent actor/learner services, full league artifact loading,
evidence-backed physics-parameter variants, and tactical search remain later
milestones. `rl.domain_randomization` already provides deterministic
cadence/latency/V2-token-noise variants; it does not silently perturb
uncertain collision, targeting, or attack-timer mechanics.
PyTorch is an optional dependency.

### Runnable recurrent PPO prototype

The first end-to-end neural prototype is available as
[`rl/prototype.py`](rl/prototype.py). It trains the public V2 actor with an
entity/hand-card Transformer, GRU, masked autoregressive action head, and
recurrent PPO.
The default critic may use exact simulator state only through a separate
training-only privileged callback; the checkpoint records that asymmetry and
fails closed unless the actor is explicitly marked public-only. The opponent
is the deterministic cycle controller, and the reward is terminal
win/draw/loss rather than a simulator-specific shaping signal.
The collector uses `SimulatorEnv.reset_v2()`/`step_v2()` so legacy V1
observations are not constructed on the actor-facing path.

The pinned `v1` ruleset is executable but not fidelity-ready, so prototype
training requires an explicit provisional flag:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../outputs/venv/bin/python \
-m rl.prototype train --allow-provisional \
  --updates 1 --envs 2 --horizon 128 \
  --checkpoint-out outputs/simulator/training/recurrent-prototype.pt
```

For `rl.prototype`, one PPO update collects `envs × horizon` environment
decisions (`2 × 128 = 256` in the smoke command).  When a checkpoint is
resumed, `starting_update` is loaded from the checkpoint and the report's
`final_update` advances by the number of `--updates` requested in that run;
`--checkpoint-out` can preserve the input artifact.

Resume with `--checkpoint` and evaluate deterministic actions with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../outputs/venv/bin/python \
-m rl.prototype evaluate \
  --checkpoint outputs/simulator/training/recurrent-prototype.pt \
  --episodes 1 --policy actor
```

For several full matches, evaluation runs independent episodes in concurrent
CPU workers (the simulator physics is Python-side) and keeps short capped
checks on the batched policy path. Use `--no-parallel-episodes` to force the
in-process path. To inspect every decision, add `--trace-out path.json`; each
`episodes[].trace[]` row records the proposed action, the hand and elixir
before the decision, accepted/rejected action events, simulator time, and
`troop_positions_after`.  Those positions are authoritative post-step
snapshots of living troops, with stable entity IDs, HP, integer
`x_mtile`/`y_mtile`, and `[column, row]` `world_cell` coordinates. Trace mode
is intentionally slower and is disabled from the parallel path so snapshots
cannot be lost.

The same command is exposed through the repository CLI as
`python -m simulator recurrent-prototype ...`. From the repository parent,
the equivalent invocation is
`PYTHONPATH=.:src outputs/venv/bin/python -m simulator recurrent-prototype ...`;
from this `simulator/` directory, keep the `PYTHONPATH=.:..:../src` form above.
A short evaluation cap is reported as censored/truncated, never as a draw.
Interface-only cadence, latency, and public-token-noise randomization are
opt-in; uncertain physics parameters are not silently changed.

For a fixed deterministic-cycle regression, run the actor and both explicit
baselines with the same checkpoint, seeds, and full-match cap:

```bash
for policy in actor public-counter strategic-counter; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:..:../src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../outputs/venv/bin/python -m rl.prototype evaluate \
    --checkpoint outputs/simulator/training/generalized-coverage-ppo-v2.pt \
    --episodes 8 --seed 10000 --max-decisions 1200 --device cpu \
    --policy "$policy" --no-parallel-episodes \
    --json-out "outputs/simulator/training/generalized-coverage-ppo-v2-${policy}-fixed-eval.json"
done
```

The `public-counter` and `strategic-counter` runs take the same public
observations and simulator opponent as the actor run, but their action source
is explicit code. This is the quickest way to detect the common mistake of
reporting a baseline's wins as if they belonged to the neural checkpoint.

### Strategic learning redesign (2026-08-28)

The generalized RL harness now defaults to an actor-controlled strategic
learner rather than a rule-teacher-centered experiment. The old teacher
warm-start commands later in this section remain reproducibility fixtures for
historical checkpoints; they are not evidence of learned strategy and must
not be used as the mainline training recipe.

The policy may receive only public game information. The simulator remains the
authority for card mechanics and legal action masks. The actor must sample its
own legal action from the factorized policy:

```text
mode/timing -> card slot -> card-conditioned placement cell
```

`WAIT` is a legal strategic action whenever the match is active. A card may be
masked only when it is not in hand, unaffordable, or has no legal placement or
target according to the authoritative simulator. A human preference for
Musketeer, Cannon, Fireball, Hog, a lane, or a timing window is not a mask.

The mainline settings are:

```text
expert_execution_probability = 0
imitation_only = false
entropy_coef > 0
terminal outcome = +1 / 0 / -1
temporary shaping = normalized potential difference only
actor = public recurrent policy
critic = optional isolated privileged current-state value function
```

Use `--expert-guidance` and imitation only for a separately named bootstrap
experiment. If replay-derived spell locations or timings are uncertain, omit
that factor's loss rather than teaching a false one-hot action. Do not feed
teacher-executed transitions into ordinary on-policy PPO.

The staged plan is documented in the [strategic architecture decision in
GOAL.md](GOAL.md#strategic-learning-architecture-decision-2026-08-28): validate
the simulator, learn short mechanics scenarios, progress through randomized
scripted opponents, expand across validated meta archetypes, then train
against a historical/PFSP pool and a small league. The executable harness now
includes the non-prescriptive curriculum, recurrent PPO sequence chunks,
temporary potential-reward annealing, a public spatial placement path, and
optional payoff-updated PFSP over frozen actors. The 2.6 Hog Cycle reference is
a sanity source for scenario coverage and grave-error checks, not a policy
label source.

The implementation intentionally keeps segment resets in this local runner:
each segment samples a reproducible lane set and resumes the same policy and
optimizer checkpoint. This bounds memory and makes curriculum provenance easy
to audit. Persistent episode lanes are the next throughput-oriented runner
upgrade; they are not silently claimed by the current reports.

New training reports must retain the ruleset/engine hash, observation-contract
hash, exact opponent deck and strategy, seeds, action source, terminal versus
truncated outcome, and whether the actor controlled every action. Promotion
requires paired frozen evaluation across decks and seeds; success against one
deterministic enemy is only a regression gate.

### Generalized opponent training and held-out evaluation

[`rl/generalized.py`](rl/generalized.py) adds the next training loop. It
restarts the simulator lanes at each rollout segment, samples reproducible
deck archetypes and simulator-side controllers from [`rl/opponent_pool.py`](rl/opponent_pool.py),
and resumes the same PPO checkpoint between segments. Lane zero can remain a
deterministic Hog-cycle regression lane while the other lanes cover pressure,
defense, beatdown, air, siege/bait, and random legal play. Unsupported cards
in an opponent's private hand are masked only in the unused opponent-view
projection; the opponent controller retains the authoritative hand and deck.

The segment-reset behavior and deterministic lane are retained here only for
historical reproducibility. The redesign will replace them in the mainline
curriculum with persistent episode lanes, actor-controlled actions on every
learner step, and explicitly mixed scripted/frozen opponents. Do not interpret
this legacy command as the recommended route to strategic competence.

The commands in this section assume the shell is in `simulator/`, the
directory containing this README. The project environment is
`../outputs/venv`; use that interpreter consistently. ROCm-enabled PyTorch
uses the `torch.cuda` API, so verify both the wheel and device visibility
before starting a long run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:..:../src \
../outputs/venv/bin/python -c '
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch_file:", torch.__file__)
print("hip:", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
'
```

On the development machine this resolves to `torch 2.12.0+rocm7.14.0` with
HIP `7.14.60850` inside `../outputs/venv`. `torch.version.hip` being non-empty
proves that the ROCm build is installed; `torch.cuda.is_available()` and a
positive `device_count` are additionally required to show that the process can
see an AMD device. A sandbox, container, missing `/dev/kfd`, or missing
`/dev/dri/renderD*` can make visibility `False` even when the ROCm wheel is
installed. `--device cuda` is not a promise of GPU execution, so stop and
diagnose a run when this check is false rather than interpreting CPU activity
as GPU training.

For the RX 6700/gfx1031 setup, add
`HSA_OVERRIDE_GFX_VERSION=10.3.1` only when the installed ROCm stack requires
that compatibility override. It is not a general acceleration switch and
does not prove kernel correctness. The MIOpen
`libMIOpenCKGroupedConv_gfx1031.so` message is a kernel-library fallback
warning; record it with the run because it can affect performance, but do not
use the warning alone to infer the execution device. Keep
`/usr/lib/python3.14/site-packages` out of this verification command so a
system PyTorch cannot shadow the ROCm wheel; the printed `torch_file` is the
authoritative wheel check.

For a fresh generalized training job, use the following. On the development
RX 6700, retain the first environment assignment only if the visibility check
above requires it; otherwise remove that line:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../outputs/venv/bin/python \
-m rl.generalized train --allow-provisional \
  --updates 20 --envs 4 --horizon 512 --seed 0 --device cuda \
  --rollouts-per-scenario 3 \
  --sequence-length 128 \
  --potential-reward-weight 0.1 \
  --potential-reward-anneal-segments 32 \
  --threat-stratified \
  --archetypes aggressive-pressure,defensive-cycle,beatdown,air-beatdown,siege-bait,random-legal \
  --strategies aggressive-pressure,defensive-cycle,beatdown,siege-bait,random-legal \
  --checkpoint-out outputs/simulator/training/generalized-recurrent-prototype.pt \
  --json-out outputs/simulator/training/generalized-recurrent-prototype.json
```

Each generalized segment samples one reproducible scenario per environment
lane. It keeps that scenario for `--rollouts-per-scenario` consecutive
rollouts, which is important for exposing opening, mid-match, and late-match
states to a recurrent policy before changing the opponent. Here, generalized
`--updates 20` means **20 scenario segments**, not 20 cumulative PPO updates.
Each segment performs `rollouts_per_scenario` prototype updates, so the
transition and update arithmetic is:

```text
transitions = segments × rollouts_per_scenario × envs × horizon
final_update (fresh run) = segments × rollouts_per_scenario
```

The example therefore collects `20 × 3 × 4 × 512 = 122,880` transitions and
advances the fresh learner to `final_update=60`. The same learner and
optimizer state continue between segments. Use `--checkpoint` to continue
from an earlier compatible checkpoint; select a new `--checkpoint-out` path to
preserve the earlier artifact:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:..:../src \
../outputs/venv/bin/python -m rl.generalized train --allow-provisional \
  --updates 20 --envs 4 --horizon 512 --seed 0 --device cuda \
  --rollouts-per-scenario 3 --explicit-hand-features \
  --checkpoint outputs/simulator/training/generalized-recurrent-prototype.pt \
  --segment-offset 20 \
  --checkpoint-out outputs/simulator/training/generalized-recurrent-prototype-continuation.pt \
  --json-out outputs/simulator/training/generalized-recurrent-prototype-continuation.json
```

If the first report was written beside the input checkpoint as
`<checkpoint-stem>.json`, the generalized runner advances its scenario cursor
from that sidecar automatically. In the continuation above, the sidecar makes
the next run use `starting_segment=20` and `segment_indices=20..39`; with the
same settings its run-local transition count is still `122,880`, while the
loaded `starting_update=60` advances to `final_update=120`. The schedule cursor
and learner update counter are different quantities when
`rollouts_per_scenario > 1`. Set `--segment-offset 20` explicitly, as shown,
when the sidecar was moved, is unavailable, or a deliberate branch is wanted;
otherwise omit it and let the sidecar provide the cursor. Keep `--json-out`
beside the checkpoint if future automatic resume inference is required.

Generalized training defaults to the larger strategic model: learned public
hand-card identity tokens, board-aligned spatial placement features, a 256-unit
GRU, and a 128-step recurrent PPO chunk. Use `--small-model` only for quick smoke
tests or when resuming a legacy small checkpoint. These architecture choices
are serialized and must match when resuming; an old checkpoint remains useful
as an evaluation or self-play opponent, but cannot be resumed into the new
model without a compatible conversion. The lower-level `rl.prototype train`
command keeps its small legacy default for compatibility; pass
`--strategic-model` there for a fresh strategic artifact.

`--env-backend process` or `--env-backend packed-process` exposes the existing
CPU lane-parallel backends; set `--env-workers` to bound worker count. Benchmark
them against `reference` on the target machine because process serialization
can outweigh the benefit for short horizons. These are throughput options,
not changes to the policy's observation or legality contract.

The staged curriculum changes only the sampled opponent distribution. It does
not force a learner card or placement. Temporary potential shaping is reduced
linearly by global segment and reaches zero at
`--potential-reward-anneal-segments`; set that option to `0` to keep the
coefficient fixed. Frozen-checkpoint lanes can be made payoff-aware after
enough completed matches with, for example,
`--opponent-checkpoints old-a.pt,old-b.pt --pfsp
--pfsp-payoff-book payoffs.json --pfsp-payoff-book-out payoffs-next.json`.
Unseen opponents retain exploration weight, and simulator-controller lanes are
not entered into the checkpoint payoff book.

#### Legacy teacher-only reproduction

The following command reproduces the earlier rule-teacher/imitation
experiments. It is intentionally not the mainline training recipe: it executes
the teacher on every step, sets imitation to full weight, and disables entropy.
Use it only to reproduce historical artifacts or compare the teacher itself.

#### 2.6 Hog Cycle as a sanity reference

The [2.6 Hog Cycle reference](https://clashroyale.fandom.com/wiki/Deck%3A2.6_Hog)
is used only as an external, high-level sanity reference for grave role,
timing, and placement errors in the simulator and the learned policy. Its
useful checks are that Hog Rider is the win condition, Cannon is a defensive
building that pulls attackers, Musketeer is a key air/ground support unit, Ice
Golem tanks and kites, and Skeletons/Ice Spirit provide cheap defensive cycle.
It also describes tracking the opponent's rotation, avoiding unnecessary
single-elixir overcommitment, and turning good defenses into Hog pressure as
elixir income increases.

These are audits and curriculum labels, not a scripted policy. The actor still
receives only public observations and must learn when a Hog push, Cannon,
Musketeer, spell, or cycle card is appropriate for the observed opponent and
placement. In particular, a run that wins by repeatedly cycling Cannon or that
never plays Hog Rider/Musketeer is not accepted as evidence of learning 2.6
Hog; it is flagged for action-slot, hand-conditioning, placement, and
defense-to-offense regression checks. The public counter is a reproducible
teacher/baseline, while actor evaluation remains the quality gate. The page is
mutable and is not an authoritative source for the pinned V1 ruleset, exact
balance values, reward labels, or a guaranteed placement sequence; those must
come from versioned simulator artifacts and evaluation traces.

For a public-teacher warm start, enable balanced factor imitation so
WAIT/PLAY, card-slot, and placement errors contribute separately.  The legacy
`public-counter` teacher is useful for the deterministic regression.  The
`strategic-counter` teacher is a stronger public-only demonstration source: it
holds Fireball until a damaged tower or visible crossing, prioritizes Hog
pressure, and defends visible threats.  Neither teacher receives hidden
opponent state, and neither is evidence that the actor has learned the rule:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m rl.generalized train \
  --allow-provisional --updates 8 --rollouts-per-scenario 3 \
  --envs 4 --horizon 512 --seed 314159 --device cuda \
  --explicit-hand-features --expert-guidance --expert-teacher strategic-counter \
  --imitation-only --expert-execution-probability 1.0 \
  --bc-coef 1.0 --bc-factor-coef 1.0 --learning-rate 1e-4 \
  --update-epochs 4 --sequence-minibatch-size 1 \
  --entropy-coef 0 --belief-coef 0 --no-belief-targets \
  --checkpoint-out outputs/simulator/training/generalized-public-hand.pt \
  --json-out outputs/simulator/training/generalized-public-hand.json
```

After the teacher warm start, use a lower teacher-execution probability only
with the authoritative-state `deterministic-counter` teacher, for example
`--expert-teacher deterministic-counter --expert-execution-probability 0.25`.
That DAgger-style phase exposes actor-induced states while keeping the
teacher's cycle count aligned with actions that were actually accepted. The
public teachers are suitable for pure imitation (`1.0`) and evaluation, but
their fallback cadence should not be used as a hidden state signal.  Expert
action weights are deliberately moderate: WAIT labels remain meaningful while
rare Hog/Fireball labels are still visible to the card head.

The current retained generalized lineage has two distinct stages. The
`outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt`
sidecar records a narrow strategic-counter warm start: four segments, three
rollouts per scenario, four lanes, 24,576 transitions, and
`final_update=24`, all against the deterministic-cycle deck. The retained
teacher-free continuation is
`outputs/simulator/training/generalized-coverage-ppo-v2.pt`: eight segments,
one rollout per scenario, four
lanes, 16,384 transitions, and `final_update=32`. It samples the six
non-regression archetypes and five non-regression strategies listed above;
training-time segment boundaries are not full-match quality evaluations.

The exact CPU continuation that produced the retained coverage artifact is:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../outputs/venv/bin/python -m rl.generalized train --allow-provisional \
  --updates 8 --rollouts-per-scenario 1 --envs 4 --horizon 512 \
  --seed 0 --device cpu \
  --player-deck hog-rider,cannon,musketeer,skeletons,ice-golem,ice-spirit,fireball,log \
  --checkpoint outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt \
  --explicit-hand-features \
  --direct-public-action-features \
  --direct-public-card-features \
  --contextual-public-card-features \
  --direct-public-mask-features \
  --direct-public-context-features \
  --learning-rate 3e-4 --update-epochs 4 --sequence-minibatch-size 2 \
  --entropy-coef 0 --belief-coef 0 --no-belief-targets \
  --archetypes aggressive-pressure,defensive-cycle,beatdown,air-beatdown,siege-bait,random-legal \
  --strategies aggressive-pressure,defensive-cycle,beatdown,siege-bait,random-legal \
  --checkpoint-out outputs/simulator/training/generalized-coverage-ppo-v2.pt \
  --json-out outputs/simulator/training/generalized-coverage-ppo-v2.json
```

Because the input checkpoint has its generalized JSON sidecar beside it, the
runner infers `starting_segment=16` and uses segments `16..23`. If that
sidecar is moved or unavailable, add `--segment-offset 16`; an explicit offset
overrides sidecar inference. The command collects
`8 × 1 × 4 × 512 = 16,384` transitions. The retained checkpoint metadata
uses the architecture flags shown above; it does not enable
`--direct-public-slot-card-features`. That option is available for fresh
architectures, requires `--explicit-hand-features`, and must not be added when
resuming an incompatible checkpoint.

Reproduce the retained fixed-opponent actor audit with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../outputs/venv/bin/python -m rl.prototype evaluate \
  --checkpoint outputs/simulator/training/generalized-coverage-ppo-v2.pt \
  --episodes 8 --seed 10000 --max-decisions 1200 --device cpu --policy actor \
  --no-parallel-episodes \
  --json-out outputs/simulator/training/generalized-coverage-ppo-v2-fixed-8.json
```

Use `--trace-out path.json` for a slower per-decision audit. The fixed
deterministic-cycle result and the generalized held-out matrix below are
separate gates; success on the fixed deck does not establish generalization.

The retained six-deck held-out smoke audit for the current actor is reproduced
with:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
../outputs/venv/bin/python -m rl.generalized evaluate \
  --checkpoint outputs/simulator/training/generalized-coverage-ppo-v2.pt \
  --policy actor --device cpu --batch-size 6 --seed 0 --max-decisions 1200 \
  --archetypes aggressive-pressure,defensive-cycle,beatdown,air-beatdown,siege-bait,random-legal \
  --strategies deterministic-cycle --seeds 10000 \
  --training-report outputs/simulator/training/generalized-coverage-ppo-v2.json \
  --no-match-results \
  --json-out outputs/simulator/training/generalized-coverage-ppo-v2-heldout-smoke.json
```

The six archetype decks are evaluation-only variants. The
`deterministic-cycle` archetype is deliberately absent: its only template is
the fixed learner deck, so it cannot be disjoint from a training report. Test
that fixed deck separately with `rl.prototype evaluate` above. The smoke
matrix has `6 × 1 × 1 = 6` matches. A broader run can use the same command with
the five additional strategies and seeds `10000,10001,10002,10003`, giving
`6 × 6 × 4 = 144` matches.

To compare the two retained actors on exactly the same held-out deck set,
use the coverage training report as the common exclusion source:

```bash
for checkpoint in \
  outputs/simulator/training/generalized-coverage-ppo-v2.pt \
  outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt
do
  name=$(basename "$checkpoint" .pt)
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:..:../src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../outputs/venv/bin/python -m rl.generalized evaluate \
    --checkpoint "$checkpoint" --policy actor --device cpu --batch-size 6 \
    --seed 0 --max-decisions 1200 \
    --archetypes aggressive-pressure,defensive-cycle,beatdown,air-beatdown,siege-bait,random-legal \
    --strategies deterministic-cycle --seeds 10000 \
    --training-report outputs/simulator/training/generalized-coverage-ppo-v2.json \
    --no-match-results \
    --json-out "outputs/simulator/training/${name}-on-coverage-heldout-smoke.json"
done
```

`actor` is the checkpoint's neural policy. The generalized evaluator also
accepts `public-counter`, `strategic-counter`, and
`deterministic-counter`, but those are explicit diagnostics rather than
measurements of checkpoint weights; only `policy_mode=actor` with
`actor_controls_actions=true` is actor evidence.

The matrix size is `number of archetypes × number of strategies × number of
seeds`. It records the selected deck cards, strategy, seed, outcome, decision
count, terminal reason, accepted target cards, proposed target cells, and
aggregate summaries for every deck/strategy cell. A supplied
`--training-report` provides the exclusion list, but the report is held-out
evidence only when `held_out_audit.disjointness_verified=true` and its
`source` is non-null. The current CLI permits an omitted training report; that
leaves disjointness unverified and must not be labeled a held-out result.
`--no-match-results` keeps only aggregates when the JSON would otherwise be
large. `--no-shuffle` is available when exact ordered deck inputs are part of a
comparison.

The matrix report is schema version 2 with kind
`recurrent_public_ppo_evaluation_matrix`. Its top-level `total` is the overall
summary, `matchups` groups results by deck and strategy, and optional
`matches` contains the individual seed rows. Per-match diagnostics such as
`target_plays_by_card`, `target_play_trace`, `tower_hp_end`, and
`troop_positions_end` are nested under `matches[].metrics`; they are not
top-level counters. `target_play_trace` contains target `PLAY` attempts only;
it is not the full per-decision state trace. The matrix runner keeps one
terminal/cap-time `troop_positions_end` snapshot, whereas a prototype
evaluation with `--trace-out` keeps `episodes[].trace[]` for every decision.

To compare two recurrent checkpoints, run the exact same held-out command
with the same `--seed`, archetypes, strategies, evaluation seeds, shuffle mode,
device, and decision cap, changing only `--checkpoint` and `--json-out`:

```bash
for checkpoint in \
  outputs/simulator/training/generalized-coverage-ppo-v2.pt \
  outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt
do
  name=$(basename "$checkpoint" .pt)
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:..:../src \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../outputs/venv/bin/python -m rl.generalized evaluate \
    --checkpoint "$checkpoint" --policy actor --device cpu --batch-size 6 \
    --seed 0 --max-decisions 1200 \
    --archetypes aggressive-pressure,defensive-cycle,beatdown,air-beatdown,siege-bait,random-legal \
    --strategies deterministic-cycle --seeds 10000 \
    --training-report outputs/simulator/training/generalized-coverage-ppo-v2.json \
    --no-match-results \
    --json-out "outputs/simulator/training/${name}-on-coverage-heldout-smoke.json"
done
```

To evaluate the current actor against frozen earlier actors, use
`--opponent-checkpoints`. Each opponent receives only its own public V2
observation, so this is true checkpoint-versus-checkpoint play rather than a
metadata comparison. The self-play matrix uses the fixed prototype player
deck and reports `held_out=false`; diverse opponent decks remain in the
held-out matrix above:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m rl.generalized evaluate \
  --checkpoint outputs/simulator/training/generalized-coverage-ppo-v2.pt \
  --opponent-checkpoints outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt \
  --policy actor --device cpu --batch-size 1 --max-decisions 1200 \
  --seeds 10000,10001 \
  --json-out outputs/simulator/training/generalized-coverage-ppo-v2-v7-self-play.json
```

Training can also place frozen actors in non-regression lanes with the same
public-only boundary:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m rl.generalized train \
  --allow-provisional --updates 8 --envs 4 --horizon 512 \
  --device cpu --explicit-hand-features --opponent-checkpoints \
  outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt \
  --checkpoint-out outputs/simulator/training/generalized-selfplay-ppo-v3.pt \
  --json-out outputs/simulator/training/generalized-selfplay-ppo-v3.json
```

The `--json-out` sidecar is important here: without it the report is printed
but cannot later supply the generalized schedule cursor automatically.

The smaller NumPy smoke trainer also exposes same-policy self-play for smoke
testing:

```bash
PYTHONPATH=.:..:../src ../outputs/venv/bin/python -m simulator --ruleset v1 train \
  --steps 10000 --envs 8 --rollout-steps 128 \
  --opponent self-play --allow-provisional-smoke \
  --checkpoint-out outputs/simulator/training/ppo-self-play-smoke.npz \
  --json-out outputs/simulator/training/ppo-self-play-smoke.json
```

#### Reading training and evaluation metrics

The binary `.pt` checkpoint is the model/optimizer artifact and records
`checkpoint_format=recurrent-public-ppo-prototype-v1`. A prototype
training JSON report has `update_rows`; a generalized training JSON sidecar
has `segment_reports` and `scenario_schedule`, and also includes `segments`,
`envs`, `horizon`, and `transitions`. `transitions` is the number of
environment decisions collected in that invocation, while `final_update` is
the learner's cumulative update counter and can be larger than this run's
segment count when resuming a checkpoint. The generalized sidecar is also the
source for automatic `starting_segment` recovery on the next run.

| Field | Meaning |
| --- | --- |
| `kind`, `schema_version` | Identify the report family and its schema; prototype training/evaluation/trace kinds are `recurrent_public_ppo_prototype`, `recurrent_public_ppo_prototype_evaluation`, and `recurrent_public_ppo_prototype_evaluation_trace`; generalized training is `recurrent_public_ppo_generalized_training` v1, and matrix evaluation is `recurrent_public_ppo_evaluation_matrix` v2. |
| `segments`, `segment_indices`, `starting_segment`, `segment_offset_source` | Generalized schedule accounting. These are scenario-segment indices, not learner update numbers. |
| `starting_update`, `final_update`, `updates` | Prototype learner-update accounting. On a fresh prototype run, `final_update` advances by `updates`; generalized segments multiply that count by `rollouts_per_scenario`. |
| `aggregate_outcomes` / `outcomes` | Generalized-run or prototype-segment outcome totals; rollout-boundary truncation is kept separate from terminal results. |
| `match_outcomes` | Optional lane-indexed terminal outcomes emitted by a rollout; generalized PFSP uses these to attribute a result to the exact frozen checkpoint assigned to that lane. |
| `wins`, `losses`, `draws` | Terminal outcomes from the learner's perspective. |
| `completed_matches` / `completed` | Matches that reached a terminal result; excludes truncated matches. |
| `truncated` / `truncated_matches` | Matches stopped by the decision cap or rollout boundary; never count as draws. |
| `win_rate` | `wins / completed`; it is not diluted by censored matches. |
| `completion_rate` | `completed / matches` in a matrix report. |
| `all_wins` | `true` only when every reported matrix match is a completed win; it is not a universal-deck claim. |
| `policy_loss`, `value_loss` | PPO actor objective and critic regression terms; compare trends under the same configuration, not absolute values across different rewards. |
| `belief_loss` | Optional training-only prediction loss for hidden opponent state; it does not give the actor privileged inputs. |
| `entropy` | Policy action-distribution uncertainty; falling entropy can mean sharper choices, but not necessarily stronger play. |
| `approx_kl`, `clip_fraction` | PPO update size and the fraction of ratios clipped by the trust-region objective. |
| `gradient_norm` | Gradient magnitude before/around clipping; `Infinity` or `NaN` indicates an unstable update that must not be treated as a successful training result. |
| `optimization_steps`, `minibatches`, `epochs` | The amount of learner optimization performed for that segment. |
| `mean_decisions`, `max_decisions` | Match length and the censoring cap used by evaluation. |
| `policy_mode`, `actor_controls_actions` | Whether actions came from the neural `actor` or an explicit `public-counter`, `strategic-counter`, or `deterministic-counter`; `actor_controls_actions` is `true` only for the neural actor. |
| `potential_reward_weight`, `potential_reward_anneal_segments` | Temporary normalized tower/crown potential coefficient and its global-segment annealing schedule; a zero coefficient after annealing means terminal outcome is the only reward. |
| `pfsp_matches_recorded`, `pfsp_payoff_book_state` | Number of completed frozen-checkpoint matches attributed to the learner and the resulting directional payoff history; simulator-controller lanes are excluded. |
| `target_plays_by_card` | In a matrix match's `metrics`, counts accepted `card_played` events for the target player by card ID; it is the reliable card-count audit. |
| `target_play_trace` | Per-PLAY attempt/accepted-play diagnostics with decision, physics time, card ID, `accepted`, and viewer-local cell; inspect `accepted` before counting a proposal. |
| `tower_hp_before`, `tower_hp_after`, `tower_hp_end` | `tower_hp_before`/`tower_hp_after` are per-decision snapshots in a full prototype trace; `tower_hp_end` is the final or cap-time king/princess snapshot in an episode or matrix result, keyed by `player_0`/`player_1` and `king`/`left`/`right`. None is a continuous damage timeline. |
| `position_schema`, `troop_positions_after`, `troop_positions_end` | `position_schema` and `troop_positions_after` describe the full prototype trace's authoritative living-troop snapshots; `troop_positions_end` is only the final or cap-time snapshot in an episode or matrix result. These are not interpolated paths. |

For a prototype trace, `episodes[].trace[]` contains one row per decision.
When the actor proposes `PLAY`, `card_slot` indexes `hand_before`, so
`card_id` is the selected card at proposal time; `played_card_id` is populated
from an accepted simulator `card_played` event. Use `accepted` and
`action_events` to distinguish a legal play from a rejected proposal. A
`WAIT` row is accepted by definition. In a matrix report, use
`matches[].metrics.target_plays_by_card` for accepted card counts and
`target_play_trace[].accepted` for attempt-level diagnostics.

`policy_cell` is the action cell in the acting player's perspective;
`world_cell` is the canonical arena cell. For player 1 the latter is mirrored
from the former. Cells use `[column, row]`; `x_mtile`/`y_mtile` use `[x, y]` in
integer milli-tiles (`1,000` milli-tiles per tile), and the current schema
declares an `18,000 × 32,000` milli-tile arena. `viewer_cell` in matrix
`target_play_trace` is the viewer-local equivalent. `troop_positions_after`
contains living troops after the step, not a separately interpolated path;
track a troop across rows by its stable `uid`.

Matrix `target_play_trace` rows retain the target's attempted `PLAY` actions,
accepted simulator events, cards, timing, and cells, but omit the expensive
per-decision troop-position snapshots. Use `troop_positions_end` when only an
episode-end/cap snapshot is needed; use a prototype `--trace-out` artifact
when reconstructing the position sequence.

`tower_hp_end` is the post-terminal snapshot when a match finishes, or the
state at the evaluation cap when it is truncated. Therefore a row with
`outcome=truncated` and `winner=null` is censored evidence, not a draw or a
loss; the matrix runner uses `terminal_reason=evaluation_cap`, while some
prototype traces from a cap may leave that field null. At a tiebreak ending,
the terminal reason and final tower HP explain why the winner was selected.

The `rl.prototype` progress bar reports both PPO updates and collected
transitions. The generalized wrapper's outer bar reports collected transitions
during training and matrix cells during evaluation; all of these are
throughput indicators, not quality metrics. `device: cuda` moves neural
inference and optimization to the ROCm device when visible, but simulator
physics and opponent controllers remain Python-side CPU work, so low average
GPU utilization is expected for this small recurrent model.
For small CPU evaluation runs, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` often
reduces thread-launch overhead; it changes throughput only, not the model or
the match rules.

#### Latest measured status and limitations

The following are local development measurements from 2026-08-28 on the
provisional `v1` simulator, not exhaustive evidence. The retained actor
artifacts report engine `reference-0.31.0`, observation contract
`public-hybrid-v2-1` with contract hash
`sha256:3bd60d514eb919de1ace676e8a2faa14f6130cecfd217cb78a5d3c538e01689e`,
and ruleset hash
`sha256:84be7cc2c6476a6609fce80dbe0c834b648b4791919c4d492cb6bfb609c4234b`.

| Evaluation | Scope | Result |
| --- | --- | --- |
| Current coverage actor ([checkpoint](outputs/simulator/training/generalized-coverage-ppo-v2.pt), [report](outputs/simulator/training/generalized-coverage-ppo-v2-fixed-8.json)) | 8 fixed deterministic-cycle matches, `--policy actor`, full 1,200-decision cap | 8 wins, 0 losses, 0 draws, 0 truncated; `actor_controls_actions=true`. This is a narrow fixed-deck regression, not diverse-deck evidence. |
| Current coverage actor ([checkpoint](outputs/simulator/training/generalized-coverage-ppo-v2.pt), [report](outputs/simulator/training/generalized-coverage-ppo-v2-heldout-smoke.json)) | 6 held-out archetype variants × deterministic-cycle strategy × seed `10000`, full 1,200-decision cap | 1 win, 5 losses, 0 draws, 0 truncated; `actor_controls_actions=true`, `held_out_audit.disjointness_verified=true`. |
| v7 actor on the same six held-out cells ([checkpoint](outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt), [report](outputs/simulator/training/v7-on-coverage-heldout-smoke.json)) | Same deck set, strategy, seed, cap, and exclusion report as the preceding row | 0 wins, 6 losses, 0 draws, 0 truncated; `actor_controls_actions=true`. This paired result is a finite regression comparison, not proof of broad generalization. |
| v7 actor broad matrix ([checkpoint](outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt), [report](outputs/simulator/training/generalized-strategic-context-card-v7-factor-full-matrix-10000.json)) | 6 held-out archetypes × 6 simulator strategies × 1 seed, 36 completed matches | 1 win, 35 losses, 0 draws, 0 truncated; `actor_controls_actions=true`. |
| v7 against itself ([checkpoint](outputs/simulator/training/generalized-strategic-context-card-v7-factor-full.pt), [report](outputs/simulator/training/generalized-strategic-context-card-v7-selfplay-2.json)) | 2 fixed player-deck self-play matches, `held_out=false` | 2 wins, 0 losses, 0 draws, 0 truncated. This identity-style self-play check does not measure diverse-deck strength or a distinct previous policy. |

These measurements show that the current neural actor has not yet generalized.
The declared `v1` roster has 109 eligible opponent cards, but this evaluation
pool is only a small curated set of archetypes, variants, strategies, and
seeds; it does not test every legal deck, card interaction, strategy, seed, or
previous checkpoint. The simulator is marked `training_ready: false`, several
mechanics are provisional, and the actor receives only public observations
while the critic may use training-only privileged state. Consequently, no
finite `all_wins=true` result here supports a claim of winning against any
deck or of live-game performance. Treat a result as a regression gate only
when the exact deck, strategy, seed set, ruleset/engine identity, decision cap,
held-out audit, and `policy_mode` are recorded in the JSON report. The training
sidecar's `aggregate_outcomes` can be zero completed matches when segment
boundaries stop rollouts; that is not a win-rate measurement.

Use prototype `--trace-out` for every decision, card proposal, accepted event,
troop position, and per-decision tower snapshot. Matrix reports retain compact
PLAY-attempt and terminal/cap-time diagnostics. Check `accepted` before
counting a proposed card, and do not treat `troop_positions_end` or
`tower_hp_end` as a continuous trajectory.

### Observation-only testing on MP4 and real matches

The `shadow` subcommand runs the checkpoint through the existing visual
pipeline and records the actions it would choose. It does not advance the
simulator, use privileged critic inputs, call ADB, or send taps. An MP4 and a
live stream therefore produce an action trace and diagnostics, not a new
win-rate measurement.

Shadow inference defaults to CPU (`--device cpu`). On a compatible CUDA/ROCm
installation, override this explicitly with `--device cuda` (and set any
required ROCm environment override, such as `HSA_OVERRIDE_GFX_VERSION`). If a
checkpoint predates the active ruleset artifact, pass
`--allow-stale-ruleset` explicitly for read-only shadow inspection only;
`train` and `evaluate` remain fail-closed, so retrain the checkpoint for normal
evaluation.

For an already extracted replay cache (the fastest path and one that does not
rerun YOLO), run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../outputs/venv/bin/python \
-m rl.prototype shadow \
  --checkpoint outputs/simulator/training/recurrent-prototype.pt \
  --replay-cache outputs/simulator/fidelity_media/physical_lab/<run-id>/replay-cache-A.pkl.gz \
  --device cpu \
  --max-frames 300 \
  --json-out outputs/simulator/training/shadow-replay.json
```

To process an MP4 directly, use the same command with `--video` instead:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../outputs/venv/bin/python \
-m rl.prototype shadow \
  --checkpoint outputs/simulator/training/recurrent-prototype.pt \
  --video /absolute/path/to/match.mp4 \
  --device cpu \
  --sample-interval 0.1 \
  --json-out outputs/simulator/training/shadow-mp4.json
```

`--video-start-time`, `--video-end-time`, `--frame-stride`, and
`--max-frames` bound a recorded run. `--max-seconds` bounds duration from the
selected start (or from zero when no start is supplied). The output contains
`predictions`, `invalid_observations`, match-boundary counters, source hash,
and `taps_sent: 0`. A `PLAY` prediction uses the viewer-local action cell
`[column, row]`; it is a suggestion only.

For a real match, mirror the phone display into a V4L2 loopback device with an
explicit serial, then point shadow mode at that device. Keep this in one
terminal:

```bash
sudo modprobe v4l2loopback video_nr=37 card_label=cr-bot-shadow exclusive_caps=1
scrcpy --serial PHONE_SERIAL --video-source=display \
  --no-control --no-window --no-audio \
  --v4l2-sink=/dev/video37 --v4l2-buffer=0
```

In another terminal, run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:..:../src \
../outputs/venv/bin/python \
-m rl.prototype shadow \
  --checkpoint outputs/simulator/training/recurrent-prototype.pt \
  --video-device /dev/video37 \
  --device cpu \
  --sample-interval 0.1 \
  --max-seconds 30 \
  --json-out outputs/simulator/training/shadow-live.json
```

The live runner is deliberately read-only; stopping it with `Ctrl-C` writes
the partial trace. Run the command with the project's vision dependencies in
the configured `../outputs/venv` environment (`opencv-python`, the detector
runtime, and their dependencies) before using media input. The training
checkpoint can still be loaded with the PyTorch environment alone, but actual
MP4/V4L2 inference needs the visual stack as well.

## Deterministic tick order

Each physics step has explicit ordering:

1. elixir regeneration;
2. actions and deployments;
3. deployment, persistent-area, status, and lifetime clocks;
4. target invalidation, acquisition, and defensive retargeting;
5. movement and deterministic local separation;
6. attack wind-up/cooldown and projectile creation;
7. projectile movement and impacts;
8. damage, status, deaths, and death effects;
9. tower/crown/victory resolution;
10. phase and match clock transition.

Changing this order is a rules/mechanics change and should produce new
regression evidence, not a drive-by refactor.

## Scenarios

Scenarios contain a schema version, immutable engine/ruleset versions, seed,
two decks, optional deterministic shuffle, scheduled actions at physics ticks,
a bounded stop tick, a preassigned evidence split, and oracle metadata. For
automatically mined mid-match clips, `initial_state` may contain a complete
canonical `BattleState`; its tick, entities, HP, hands, cycle, and phase are
then the replay boundary, so the preceding match need not be reconstructed.

Automatically found discrepancies must enter as `candidate`; they are never
promoted to `regression` or used as expected truth by a failing simulator run.

```json
{
  "schema_version": 1,
  "scenario_id": "example",
  "engine_version": "reference-0.29.0",
  "ruleset_id": "2026-08-04",
  "ruleset_hash": "sha256:...",
  "seed": 1,
  "shuffle_decks": false,
  "decks": [["... eight cards ..."], ["... eight cards ..."]],
  "actions": [
    {"tick": 0, "action": {
      "kind": "play", "player": 0, "card_slot": 0, "cell": [3, 23]
    }}
  ],
  "max_ticks": 200,
  "split": "synthetic",
  "tags": ["movement"],
  "oracle": {"promoted": false}
}
```

## Sim-to-real fidelity

`fidelity.py` separates four evidence roles assigned before evaluation:

```text
calibration   fit provisional mechanics/tolerances
validation    select implementations without touching held-out data
regression    reviewed failures which must not recur
heldout       final unbiased reporting only
```

An observed scalar or trace always includes source/method provenance,
confidence metadata, and measurement tolerances. Confidence is not treated as
truth probability and does not weight a score. A held-out report excludes all
other splits and reports those exclusions.

Run a corpus and atomically write a report:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator fidelity \
  path/to/corpus.json --split heldout --json-out outputs/sim/fidelity.json \
  --expected-corpus-hash sha256:... --min-observations 1000 \
  --min-agreement-rate 0.98 --require-mechanic hog_cannon_targeting
```

Compile offline detector/tracker output into that corpus with no uncertain
sample rescue step:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator mine-corpus \
  outputs/vision/observed_tracks.json \
  --json-out outputs/sim/heldout-corpus.json \
  --discarded-out outputs/sim/discarded-clips.json
```

The compiler rejects occluded/low-confidence clips, assigns entire capture
groups to calibration/validation/heldout using a stable salted hash when no
split is predeclared, records the media hash and frame range, creates pinned
mid-match scenarios, and generates x/y/HP/alive measurements for every clean
track sample. If every clip is ambiguous it fails instead of producing an
empty “successful” corpus.

For the repository's existing `ReplayCacheWriter` output, the first autonomous
miner needs no intermediate manifest:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator mine-replay \
  outputs/replays/match.pkl.gz --corpus-id hog-movement-2026-08 \
  --group-id match-video-sha --source-level 11 \
  --evidence-split calibration \
  --json-out outputs/sim/movement-corpus.json
```

It considers every supported stable-ID, high-confidence **troop** independently
and keeps a contiguous run only while no other detected unit is inside the
larger of the configured local radius or that card's sight radius. It requires
meaningful displacement, maps the
detection's ground-contact anchor through the existing calibrated action grid
into radius-safe milli-tiles, starts mid-track observations already deployed,
scales detector Level-16-normalized HP ratios back to the pinned Level-11
ruleset, hashes the replay artifact, and discards crowded,
occluded, identity-unstable, stationary, or low-confidence frames. This
provides a fully automatic isolated-movement oracle while leaving ambiguous
clips unused. `--source-level` is mandatory because compact caches discard
level badges and substitute the live-capture maximum for inactive King
Towers. Support-tower OCR is used as a fail-closed contradiction check: any
value above the pinned Level-11 maximum rejects the complete cache, and at
least one exact full support-tower reading is required to confirm the level.
Use `--evidence-split calibration` for any source already viewed or used to
change the engine. If omitted, the whole group receives its stable salted
split; that automatic result is valid as held-out only while the group remains
untouched.

Contamination is deliberately asymmetric: a truth candidate needs the high
confidence gate and a stable track ID, while a nearby object invalidates the
candidate at the lower `--contamination-confidence-threshold` even when that
object has no stable ID. This prevents detector uncertainty about a defender
from becoming false “isolated movement” truth.

Ordinary locally isolated tracks emit displacement-speed evidence only. They
do not emit absolute x/y fidelity because an off-screen target is not known,
and automatically choosing a Princess Tower would manufacture path errors.
Tracks visibly occupying a bridge retain x/y samples: bridge occupancy is a
target-independent topological observation and therefore valid pathfinding
evidence.
The speed observation is compared directly with the versioned card movement
stat; it is not compared with simulated displacement toward a guessed target.
Calibration may use the expected-speed consistency gate to discard obvious
combat/status tracks. Validation and held-out mining refuse that selector
because it leaks the expected answer into evaluation; they require
`--kinematic-only-gate`, which checks pauses, speed discontinuities, reversals,
and path looping without reading the simulator card speed.

Movement speed, bridge routing, and collision-free trajectory geometry do not
scale with card level. The movement miner can therefore accept a current-
ruleset capture from another level only with both
`--level-invariant-current-ruleset` and an exact
`--expected-support-tower-hp`. The latter must be observed in the cache and
prevents a declared level from silently substituting for evidence. This escape
hatch applies only to the isolated movement/path corpus; HP, damage, lifetime,
and combat miners remain strictly Level 11.

Hog/Cannon pulls use the stricter action-anchored miner:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator mine-pulls \
  data/eval/replay_cache/match.pkl.gz \
  --ground-truth data/eval/ground_truth/actions.json --source-level 11 \
  --group-id match-video-sha --evidence-split calibration \
  --corpus-id hog-cannon-pulls-2026-08 \
  --json-out outputs/sim/hog-cannon-pulls.json
```

It requires a detected Cannon onset after a labeled Cannon play, checks the
localized deployment cell when available, follows an opposing Hog with a
stable ID, rejects clips with any third unit near either the Hog or Cannon,
and infers the pull
from the Hog approaching the Cannon. Interior points receive only a fixed
three-frame median filter. Detector/action disagreement is discarded rather
than sent to a person or silently treated as game truth.

Action files and replay caches may use different frame cadences. Their frame
indices are never compared directly: an explicit annotation `fps` maps labels
to video time, while caches retain their native frame/time axis. If nearby
troops contaminate the path but the Hog conclusively reaches the only detected
building, the miner may emit a targeting-only case containing the initial
state and discrete Hog→Cannon target event. It never reports that clip as a
trajectory oracle.

Cannon lifetime discovery is deliberately a candidate report rather than an
automatic gold promotion:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator \
  discover-cannon-lifetime outputs/replays/match.pkl.gz \
  --ground-truth data/eval/ground_truth/actions.json \
  --source-level 11 --json-out outputs/sim/cannon-lifetime-candidates.json
```

With `--ground-truth` it uses the localized action anchor; without that option
it infers only a detector track-onset hypothesis and rejects cache-boundary
onsets. Both modes require a continuous high-confidence track, exact Level-11
support-tower confirmation, no nearby enemy detections, an absence tail after
expiry, and an HP curve consistent with at least one declared lifetime-start
hypothesis. The report scores both `placement` and `post_deploy`, records the
action source, and never silently chooses a ruleset constant when footage is
ambiguous.

Lifetime rejection reports list every failed gate, not only the first. In the
current `spell3` calibration clip, the corrected 10 FPS-label/native-cache
alignment anchors the Cannon play at 80.600 seconds, but direct inspection and
the report agree that combat contamination, detector gaps, and a 21.5-second
disappearance make it unsuitable for natural-expiry calibration.

Stable Princess-Tower HP plateaus can independently mine exact damage and
repeat-hit timing without hand labels:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator \
  discover-tower-damage outputs/replays/match.pkl.gz \
  --source-level 11 --json-out outputs/sim/tower-damage-candidates.json
```

The miner requires two sufficiently long OCR plateaus, an exact damage delta,
and a unique supported attacker visible for the correct side near the damaged
tower. A repeat interval additionally requires the same stable detector track
at both impacts. It emits a candidate report only. Spells, suicide impacts,
and death-effect damage are excluded from repeat-attack timing, and
source-version mismatches are retained as rejected evidence instead of
changing simulator constants.

Ground-projectile speed has a separate action-anchored candidate miner:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator \
  discover-log-motion outputs/replays/match.pkl.gz \
  --ground-truth data/eval/ground_truth/actions.json \
  --source-level 11 --json-out outputs/sim/log-motion-candidates.json
```

The Log detector can retain static deployment artwork before the rolling
object is visible. The miner therefore measures only the longest consecutive,
direction-consistent moving segment, reports detector onset separately, and
does not claim that the selected segment start is the launch timestamp. It
also rejects large lateral drift, cadence gaps, low confidence, absent action
anchors, and caches without exact Level-11 confirmation.

Fireball uses a different oracle because airborne screen coordinates cannot
be interpreted as ground-plane speed:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator \
  discover-fireball-flight outputs/replays/match.pkl.gz \
  --ground-truth data/eval/ground_truth/actions.json \
  --source-level 11 --json-out outputs/sim/fireball-flight-candidates.json
```

It requires a localized labeled cast, a direction-consistent moving track,
and a later compact effect track near the selected cell. The first confirmed
effect frame supplies action-to-impact timing; the miner deliberately does
not derive airborne world speed from a ground homography.

The corpus runner executes each pinned scenario, derives declared measurements
from final state or public events, compares observed values/ticks, and emits:

- per-mechanic count and simulated count;
- numeric MAE and p95 absolute error;
- timing MAE and p95 tick error;
- decision/event agreement and Wilson confidence interval;
- first divergent event tick and reason;
- source/group count, evidence methods, split-exclusion counts, per-case
  state/event/replay hashes, and explicit scalar/trace failure records.

Version 1 provides deterministic extractors for:

```text
outcome_winner / outcome_reason
event_count / first_event_tick / last_event_tick / first_event_field
final_tower_hp / final_tower_alive
final_entity_hp / final_entity_alive
entity_x_mtile_at_tick / entity_y_mtile_at_tick
entity_displacement_speed_mtile_per_s (two pinned snapshot ticks)
entity_hp_at_tick / entity_alive_at_tick
tower_hp_at_tick / tower_alive_at_tick
```

Tick extractors capture immutable pre-action state boundaries and make dense
movement, path, collision, HP, and survival curves measurable without adding
debug-only movement events to every simulation tick. Numeric samples feed the
same per-mechanic MAE/p95 and timing-error summaries as atomic outcomes.

Event filters use stable enriched fields such as `card_id`, `source_card_id`,
`target_card_id`, `owner`, and `target_role`, so observed traces do not depend
on run-specific UIDs. A minimal measurement looks like:

```json
{
  "sample_id": "capture-0042:hog-pull-tick",
  "mechanic": "hog_cannon_targeting",
  "observed_value": 49,
  "observed_tick": 49,
  "tolerance": {"absolute": 1, "ticks": 1},
  "extractor": {
    "type": "first_event_tick",
    "event_kind": "target_changed",
    "filters": {
      "card_id": "hog-rider",
      "target_card_id": "cannon"
    }
  }
}
```

Each enclosing case has one inline scenario or a safe relative scenario path,
a split assigned before execution, and evidence containing `source_id`,
`method`, `confidence`, and optional notes. Missing simulator output is a
failed comparison. Ambiguous entity selectors are configuration errors. The
loader requires the engine version, expands relative scenarios once, seals the
complete evidence corpus with SHA-256, and includes `engine_version`,
`corpus_id`, and `corpus_hash` in the report; later file changes cannot alter
an already loaded evaluation.

No bundled video is called “gold.” Controlled 60/120 FPS Level-11 captures are
still required for attack/projectile timing, while the existing 10 FPS action
labels are suitable only for coarser deployment evidence.

### Local evidence snapshot (2026-08-13)

The automated miners were exercised against the repository's local replay
caches; generated corpora/reports remain under local output paths and are not
source-controlled truth. A positively Level-11-gated Hog/Cannon candidate
initially reproduced the target switch and sampled trajectory, but targeted
frame inspection found Skeletons around the interaction. It was invalidated,
and the pull miner now rejects third units near either participant. The
stricter miner currently finds no clean pull in that cache rather than
reporting a contaminated metric.

The current sight-isolated movement miner accepts multiple visible units only
when every other detection is outside both the configured isolation radius
and the candidate card's sight range. It also rejects tracks outside 50--150%
of the sourced base speed, which removes unobserved status/attack pauses and
tracking jumps from the base-movement oracle. On the positively Level-11-gated
0--195-second cache group it retained five clips and 114 x/y observations.
The calibration-only aggregate was MAE `158` milli-tiles, p95 `461`, and
`67.5%` agreement at the deliberately tight sample tolerance. Per-card x/y
MAE was Hog `116/235` (32 observations per axis), Ice Golem `24/59` (9 per
axis), and Musketeer `211/162` (16 per axis). These are calibration
descriptives from one source group, not independent held-out accuracy claims,
so no speed constant was fitted to them.

The miner now reports bridge paths separately. A directly inspected Level-11
Hog left-bridge calibration track had x/y MAE `31/51` milli-tiles and p95
`57/92` over five observations per axis. The classification came from arena
coordinates and was verified against the source frames, not manually assigned.

A separate one-FPS discovery scan over video seconds 485--814 found several
Cannon tracks, and direct inspection confirmed Level 11, but every long track
was damaged, obscured, or involved in combat. None is eligible to calibrate an
undamaged 30-second decay curve. The discovery scan is candidate selection,
not fidelity truth. Running the conservative lifetime candidate miner on the
Level-11 0--195-second cache likewise promoted zero cases: the one
action-anchored track had a detector gap, lasted only about 15 seconds, and
missed both provisional HP curves by over 20%; four other tracks lacked an
action anchor.

A 30 FPS extraction from a native 60 FPS source was also inspected around an
early Cannon defense. It produced a stable Cannon track, but the interaction
contains an attacking Hog and several Skeletons. The detector additionally
misclassified visible units and emitted impossible HP estimates. Direct frame
inspection confirms deployment-time visibility and targetability, but the clip
is rejected for lifetime and repeated-shot calibration; no attack timing is
derived from its noisy HP series.

The first frozen `spell2` holdout exposed two oracle failures: an Ice Golem
track at only 20% base speed and a detector jump at 291%. Its original
422-observation report remains unchanged (`56.9%` tight-tolerance agreement),
but that group was opened for diagnosis and is not reused as final holdout.
After freezing the base-speed gate, then-independent `spell3` produced 14 cases
and 402 position observations: aggregate MAE `160` milli-tiles, p95 `570`, and
`75.6%` tight-tolerance agreement. That group has since been opened for Cannon,
Tower, and Log diagnosis and is therefore calibration evidence now, not a
current held-out claim. The version-0.7 corpus adds one
endpoint-displacement speed measurement per trajectory, for 416 total
observations and `76.0%` aggregate tight-tolerance agreement. Decision-relevant bridge subsets were
stronger: Ice Golem x/y MAE `163/63`, p95 `299/130` over 61 observations per
axis; Ice Spirit x/y MAE `99/94`, p95 `241/201` over 20 per axis. Direct source
inspection confirmed the long Ice Golem track crosses the left bridge. This
remains useful movement/path calibration evidence, but not evidence for combat,
targeting, spell, lifetime, or unbiased final accuracy.

The endpoint-speed metrics agree on all six Ice Golem tracks
(non-bridge MAE `51` milli-tiles/s; bridge error `16`) and both Ice Spirit
bridge tracks (MAE `118`). Four of six ordinary Ice Spirit tracks pass; one is
335 milli-tiles/s slower than the simulator and one simulator trace stops at
attack range while the observed track continues, producing a 2,191
milli-tiles/s discrepancy. That failure remains visible as trajectory/path
divergence rather than being averaged into a fitted base-speed constant.

An independent native 126 FPS Level-11 `spell3` capture supplies stronger
atomic combat evidence. Stable OCR plateaus and direct frame inspection show
seven Hog hits of 317 damage and four Musketeer hits of 217 damage. Three clean
adjacent Hog intervals measure 1600, 1596, and 1605 ms against the declared
1600 ms interval. An older repository capture repeatedly produces obsolete
318/218 deltas; the miner rejects those deltas as ruleset-version mismatches.
The current-video evidence pins exact damage and Hog repeat timing, but does
not by itself validate first-hit delay, projectile flight, or target choice.
Another native 123 FPS Level-11 capture contains one directly inspected
Musketeer continuously tracked as ID 301. Four consecutive tower plateaus fall
by exactly 217 HP, and the three same-track repeat intervals are 1002, 1004,
and 995 ms. This independently pins the declared 1.0-second Musketeer hit
interval while leaving its first-hit and projectile-flight timing provisional.

A subsequently audited Level-15 ladder cache is usable only for mechanics that
do not scale with level. The source was confirmed by exact 4,424-HP support
towers and retained as validation, not held-out, because its frames had already
been opened. Direct inspection showed that detector-silent combatants made
ordinary target-guessed x/y comparisons invalid. The miner now emits x/y only
for visible bridge occupancy and emits target-independent card-speed comparisons
for ordinary tracks. It also forbids expected-speed selection for validation or
held-out sets. Under the unbiased kinematic gate, only three of seventeen prior
candidates survive: Ice Golem speed agrees within tolerance (1/1), while one
Hog and one Musketeer observation fail. These sparse results are diagnostic and
do not satisfy readiness.

The cadence-corrected mirror-match calibration also yields one targeting-only
Hog/Cannon case. Direct frames show the Hog reach and attack the uniquely
detected Cannon while nearby Skeletons invalidate exact path comparison. The
simulator agrees on the Hog→Cannon target event (`1/1` trace); this is useful
calibration evidence, not a held-out accuracy claim or a clean pull trajectory.

The same Level-11 calibration group contains one isolated ordinary Hog track
and one isolated bridge track. Their endpoint speeds are 2178 and 2630
milli-tiles/s, bracketing the declared 2400 and averaging 2404; errors of 217
and 227 both pass the predeclared ±240 gate. This reduces the raw-speed
conversion uncertainty but does not replace group-independent held-out Hog
movement evidence.

Two independent action-anchored Level-11 Log tracks measure 3842 and 4301
milli-tiles/s. Their mean is 4072 against the declared 4000, with individual
errors of 4.0% and 7.5%. Direct contact-sheet inspection confirms the Log
advances through both selected segments. This reduces rolling-speed
conversion uncertainty; it does not establish launch delay, the leading-edge
position, collision width, range, or target-hit timestamps.

Two localized Fireball casts in the same Level-11 recording have directly
inspected explosion frames. Observed action-to-impact times are 2002 and 1806
ms versus 2150 and 1700 ms simulated, giving 127 ms MAE; their detected
effects are 37 and 911 milli-tiles from the labeled target centers. This is
calibration evidence for the combined origin/speed/tick approximation, not
independent identification of projectile speed or visual origin.

Action-aligned Level-11 frames also replaced the generic Skeleton formation:
the selected policy cell anchors the leading Skeleton, with the rear pair at
`[-750, 500]` and `[750, 500]` milli-tiles. Structured masses now make Ice
Golem, Musketeer, Hog, Skeleton, and Ice Spirit collision displacement
mass-weighted rather than 50/50.

The `reference-0.29.0` regression suite covers every walkable Hog policy-grid
deployment to both opposing Princess Towers (380 routes), requiring legal
segments and exact full-arena mirror equivalence. A strict complete seeded
match runs through tick 6,000 twice with identical state and replay hashes. A
separate four-seed lockstep audit covered 4,800 randomized ticks, 109 card
plays, and 2,116 events without divergence or invariant failure.

The training path caches visibility edges whose bridge/structure topology is
unchanged and computes only the original sampled river indices that can lie
inside the river. A 200,000-case differential check found these terrain tests
exactly equivalent to the prior exhaustive sampler, and complete-match state
hashes are unchanged. On this machine the same three-match reference
benchmark improved from about 497 to 755 physics ticks/s; this is a useful
Python baseline, not a claim of production-scale vectorized RL throughput.

## Automation and CI

Fast tests are media-free and run in the normal repository `pytest` command.
Use the deterministic audit for generated action sequences and soak testing:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator check-determinism --seed 918251
PYTHONPATH=.:src outputs/venv/bin/python -m simulator audit \
  --seeds 16 --max-ticks 6000 --json-out outputs/sim/audit.json
PYTHONPATH=.:src outputs/venv/bin/python -m simulator soak \
  --seeds 32 --tick-budget 1000000 --json-out outputs/sim/soak.json
```

Recommended automation tiers:

- every commit: `tests/simulator`, ruleset hash/schema, core interactions,
  state round-trip, policy shape/parity, small deterministic fuzz sample;
- every PR: larger randomized action streams, all reviewed scenarios,
  held-out metadata/split checks, reference performance guard;
- nightly/self-hosted: multi-million-tick soak audit and full local
  video/replay-derived corpus.

Large videos and replay caches are ignored repository artifacts. A hosted CI
job must use an explicit artifact store or self-hosted runner; it must not
silently skip media and report fidelity success.

## Current fidelity limits

The engine is suitable for integration, deterministic testing, scenario
generation, behavior-policy plumbing, and collecting calibration evidence.
Serious RL conclusions should wait until the relevant held-out report has
enough independently audited cases. In particular, these remain provisional:

- real internal tick rate and tick ordering;
- speed-code and projectile-speed conversions other than the now
  two-recording-calibrated Log rolling speed;
- attack wind-up/first-hit timing;
- exact collision radii, bridge steering, and dense-unit behavior;
- Cannon's 30-second linear HP decay begins at placement and it is targetable
  while deploying, but clean no-damage video calibration of the exact decay
  curve and sub-frame start boundary is still missing;
- mass-weighted dense-unit collision behavior beyond the sourced masses;
- Log and Fireball flight/collision geometry;
- Princess/King Tower acquisition, projectile, and activation timing (their
  Level-11 HP/damage are pinned to 3052/109 and 4824/109 respectively);
- all Evolution, Hero, Champion, tower-troop, and ability mechanics.

The correct response to missing evidence is an empty/failed metric with a
recorded uncertainty—not a guessed 99% accuracy value.

Generate the fail-closed, per-mechanic training-readiness summary from any
number of already generated fidelity reports:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator readiness \
  outputs/sim/calibration-fidelity.json outputs/sim/heldout-fidelity.json \
  --candidate-report outputs/sim/log-motion-candidates.json \
  --json-out outputs/sim/training-readiness.json \
  --min-heldout-observations 20 --min-heldout-agreement-rate 0.98 \
  --min-heldout-groups 2
```

The command exits with status 2 until every declared decision-relevant
mechanic has enough passing held-out observations. Calibration evidence is
shown as `calibrated_only` and can never satisfy that gate. It also fails if a
capture group presented as held-out appears in a calibration, validation, or
regression report. Cache/media hashes are checked as well, so renaming a group
cannot hide prior candidate-mining use. Motion/path requirements are per card, and compound systems
such as Log motion and collision are separate gates. The default also requires
at least two independent capture groups per mechanic, because thousands of
correlated frames from one match are not thousands of independent tests. With
the repository's current local evidence the expected
result is **not ready**: the useful clips have already been inspected and used
for calibration, while the untouched recordings are not the pinned current
Level-11 ruleset.

Conservative discovery outputs supplied with `--candidate-report` appear as
`candidate_only`; they document useful measurements but never count as
held-out fidelity. A miner that ran but rejected every clip appears as
`candidate_rejected`, which is distinct from never attempting that mechanic.
