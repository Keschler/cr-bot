# Physical Differential-Testing Lab

## Status and purpose

This document describes the controlled in-game evidence system for Simulator
V1. It is adapted to the repository's existing ruleset, replay-cache, vision,
fidelity, retention, and readiness boundaries. Phase 0 is implemented in
`physical_lab/`: it provides canonical experiment records, fake devices,
lifecycle and synchronization gates, logical replay, comparison, planning,
ingest, and artifact sealing. Device-specific lifecycle detection, continuous
video capture, and detector extraction remain explicit integration boundaries
for the connected-phone phases.

The goal is not to make an AI play Clash Royale. The goal is to make the game a
repeatable black-box reference implementation for the deterministic simulator.
The software-only path is runnable while the two phones are disconnected:

```text
experiment specification
        -> controlled friendly battle
        -> sealed screen/action recordings
        -> constrained state extraction
        -> real observation corpus
        -> simulator replay
        -> first-divergence comparison
        -> targeted next experiment
```

The lab must preserve the fail-closed rules in [GOAL.md](GOAL.md): a detector
candidate, a same-video HUD cross-check, a validation result, or an ambiguous
capture is never silently promoted to held-out truth.

## How this fits the current repository

The physical lab adds a controlled source of observations; it does not replace
the existing public-video and synthetic pipelines.

| Existing component | Physical-lab integration |
| --- | --- |
| `simulator/rulesets/v1.json` | The exact ruleset used for the simulator replay. The run records its content hash and engine version. |
| `simulator/engine.py`, `scenario.py`, `runner.py` | Execute the logical experiment again with the same actions and initial conditions. |
| `simulator/video_pipeline.py` | Reuse source manifests, bounded extraction, cache hashes, retention records, workspace budgeting, and job timeouts. |
| `src/cr_bot/features/action_space.py` and arena calibration | Convert logical arena cells to screen coordinates and preserve the existing coordinate conventions. |
| `src/cr_bot/replay/cache.py` | Store detector output as a sealed replay cache. A replay cache remains an observation artifact, not authoritative truth. |
| `src/cr_bot/vision/` and `src/cr_bot/trackers/` | Extract card identity, owner, position, tower HP, unit HP, and stable tracks from the phone recordings. |
| `simulator/mining.py` and `video_pipeline.py` | Mine movement, targeting, lifetime, damage, spell, and interaction candidates from compatible caches. |
| `simulator/fidelity.py` and `validation.py` | Compare real observations with simulator measurements using stable card/owner/event selectors. |
| `simulator/readiness.py` | Enforce minimum observations, independent capture groups, thresholds, leakage checks, and the final fail-closed status. |
| `simulator/storage.py` and `media-budget` | Seal hashes and retention metadata; keep raw recordings subject to the 200 GB workspace contract. |
| `unknown_behaviors.json` and `UNKNOWN_BEHAVIORS.md` | Generate targeted probes for unresolved timing, geometry, spawn, status, and interaction questions. |

The existing YersonCz source policy remains applicable to the public-video
lane. Controlled current-Level-11 recordings are a separate evidence method and
must carry their own patch, level, device, and capture provenance. Old public
footage can help with version-invariant movement or topology, but it cannot
substitute for current-version damage, timing, projectile, or status evidence.

## Two evidence lanes

### Candidate and validation lane

The existing public-video workflow remains useful for discovering natural
interactions:

```text
discover-video-source
    -> plan-action-windows
    -> extract-video / mine-replay-batch
    -> discover-replay-interactions
    -> inspect candidates
```

This lane may use low-cost scans, inferred action onsets, one HUD profile, or
lower frame rates when selecting windows. Its outputs are `candidate_only`,
`candidate_rejected`, or `validation`. They can guide implementation and
choose a controlled probe, but they cannot satisfy an untouched held-out gate.

### Controlled evidence lane

The physical lab is used for exact, version-sensitive behavior:

- deployment and first-hit timing;
- projectile origin, flight, and impact timing;
- target acquisition and retargeting;
- damage and Crown Tower damage;
- building lifetime and HP decay;
- spell victim sets and boundary geometry;
- status duration, slow, freeze, rage, knockback, and immunity;
- passive spawns, death spawns, splits, transformations, and charge states;
- controlled movement, bridge selection, collision, and pull behavior.

Every controlled run is assigned a capture group and evidence split before its
observations are inspected. A run from a group already used to tune the engine
is calibration or validation forever; it cannot later become held-out by being
renamed or moved.

## Hardware target

Use two dedicated Android phones connected to one controller workstation:

```text
Phone A: controlled player
Phone B: controlled opponent
Workstation: ADB control, screen capture, timestamps, orchestration, storage
```

The first implementation should hide device-specific details behind small
interfaces:

```python
class PhoneController(Protocol):
    def screenshot(self) -> Frame: ...
    def tap_screen(self, x_px: int, y_px: int) -> ActionReceipt: ...
    def press_back(self) -> None: ...
    def device_info(self) -> DeviceInfo: ...

class ScreenCapture(Protocol):
    def start(self) -> CaptureHandle: ...
    def stop(self) -> CaptureManifest: ...
```

An ADB-backed adapter may use `adb exec-out screencap`, a video-capable
screen-streaming path, or a calibrated external camera. The experiment layer
must not depend on one capture transport. Each frame records its source device,
controller monotonic timestamp, frame index, presentation timestamp when
available, and capture uncertainty.

The logical API must never expose raw screen coordinates to the experiment
planner:

```python
phone_a.select_card(slot=2)
phone_a.place_card("hog-rider", arena_cell=(3, 20))
phone_b.place_card("cannon", arena_cell=(8, 13))
```

The device adapter converts the logical card slot and arena cell to pixels
using a versioned calibration artifact. The arena mapping must reuse the
repository's existing action-grid and homography conventions; it must not
introduce a second `(x, y)` or row-origin convention.

### Staged one-phone preparation

The deck setup is intentionally separable from the two-phone run. With only
one handset connected, prepare that handset and record its serial-bound
manifest:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab prepare \
  --serial PHONE_A --side A \
  --json-out outputs/simulator/fidelity_media/physical_lab/preparation-A.json
```

The reviewed deck contract is:

```text
slot 0  hog-rider       opening card 1
slot 1  cannon          opening card 2
slot 2  musketeer       opening card 3
slot 3  skeletons       opening card 4
slot 4  ice-golem       first replacement
slot 5  ice-spirit      second replacement
slot 6  fireball        third replacement
slot 7  log             fourth replacement
```

When the second handset is available, prepare it as side B and pass both
manifests to `scripts/run_physical_lab_autonomous.py`. The host side uses a
real long press on the Friendly Battle `1v1 Battle` button, taps the reviewed
fixed-deck-order toggle, and then taps the reviewed start control. Phone B is
the host and Phone A accepts. The toggle and start points
are explicit because their positions vary by game build; missing or malformed
points reject the run before unsafe UI guessing. The resulting run is a
candidate until the lifecycle, capture, synchronization, observation, and
readiness gates all pass.

The versioned simple-to-complex campaign is documented in
`physical_lab/PHYSICAL_FIDELITY_GOAL.md`. Create it with
`scripts/run_physical_fidelity_campaign.py plan`; its deck mutations and
first-four opening hands are immutable. After physical corpora are admitted,
`scripts/run_physical_fidelity_campaign.py evaluate` re-runs every stored case
against the current simulator and writes a new evaluation snapshot.

To prevent idle phones during a prepared run, use `lab keep-awake` with the
two explicitly mapped serials. It applies the maximum screen timeout and
powered-device stay-awake settings, then verifies both readbacks. The command
does not run ADB device discovery.

The one-phone preparation command returns to the lobby after deck verification.
It does not start a waiting challenge; fixed starting-hand order is selected at
the coordinated host start so the later two-phone run begins from a verified
lobby on both devices.

## Experiment specification

Every physical run is generated from a canonical, hashed specification. A
minimal `physical_experiment_v1` record should contain:

```json
{
  "schema_version": 1,
  "experiment_id": "hog_cannon_pull_0142",
  "ruleset_id": "v1",
  "ruleset_hash": "sha256:...",
  "engine_version": "reference-...",
  "capture_group_id": "lab-session-2026-08-18-a",
  "evidence_split": "calibration",
  "devices": {
    "A": {"serial_hash": "sha256:...", "role": "player"},
    "B": {"serial_hash": "sha256:...", "role": "opponent"}
  },
  "initial_conditions": {
    "tower_state": "default",
    "requested_elixir_milli": {"A": 10000, "B": 10000}
  },
  "actions": [
    {
      "side": "A",
      "card_id": "hog-rider",
      "arena_cell": [3, 20],
      "trigger": {"type": "match_time_us", "value": 0}
    },
    {
      "side": "B",
      "card_id": "cannon",
      "arena_cell": [8, 13],
      "trigger": {
        "type": "after_observation",
        "event": "hog_crosses_y_mtile",
        "value": 17000
      }
    }
  ],
  "measurements": [
    "hog_isolated_movement",
    "hog_cannon_targeting",
    "hog_cannon_pull_trajectory",
    "cannon_lifetime_hp_decay",
    "tower_hit_count"
  ]
}
```

The specification is the shared input to the physical runner and simulator
runner. It records logical actions, never inferred actions. The simulator run
must preserve the same action boundary and initial-state assumptions; if the
physical run deviates, the result is a rejected or separately classified
observation, not a silently modified scenario.

## Match lifecycle state machine

The controller should verify each transition from the screen rather than use
fixed sleeps:

```text
RECOVERY
  -> LOBBY
  -> CHALLENGE_SENT
  -> CHALLENGE_ACCEPTED
  -> LOADING
  -> BATTLE
  -> RESULT
  -> ARCHIVED
  -> RECOVERY
```

Each state has an entry detector, timeout, and recovery action. Examples:

- `LOBBY`: expected battle/challenge controls are visible;
- `CHALLENGE_ACCEPTED`: both devices show the challenge acknowledgement;
- `BATTLE`: arena, match timer, and tower regions are visible;
- `RESULT`: result overlay is visible and the arena is no longer advancing;
- `RECOVERY`: dismiss stale overlays, return to lobby, and verify both phones
  are ready for the next run.

If one phone fails to enter `BATTLE`, the run is archived as a lifecycle
failure. It must not be included in a mechanic corpus.

## Time synchronization

The repository distinguishes video time, match time, and frame index. The lab
must preserve all three:

```text
video_time_us  --capture alignment--> frame_index
match_time_us  --game timer/OCR-->   video_time_us
```

Do not assume that the simulator's provisional 50 ms step is the real game
tick. It is a simulator coordinate for comparison until controlled evidence
measures the real timing.

The controller should:

1. start both captures before entering the battle;
2. timestamp frame arrival with one workstation monotonic clock;
3. record phone/device timestamps when available;
4. create a visible synchronization marker or use a common countdown edge;
5. estimate per-device offset and uncertainty;
6. reject any run whose synchronization uncertainty exceeds the measurement's
   declared timing tolerance.

The target for timing-sensitive probes is less than 10 ms alignment uncertainty,
but the measured uncertainty—not the target—is what enters the evidence record.

## Observation extraction

The controlled experiment provides strong priors, but those priors narrow the
detector; they do not turn a detector guess into truth.

For a Hog/Cannon probe, the expected classes, owners, spawn regions, action
times, and possible target relations are known. The observation stack can use
those constraints to reduce false matches:

```text
screen stream
  -> arena crop / homography
  -> expected-class detection
  -> stable-ID tracking
  -> tower/unit HP OCR and change detection
  -> event reconstruction
  -> confidence-gated observation record
```

The output should be compatible with the repository's replay-cache boundary,
with additional lab provenance alongside it:

```json
{
  "source_id": "lab-session-2026-08-18-a:run-0142",
  "video_hash": "sha256:...",
  "frame_index": 1837,
  "video_time_us": 3125483272,
  "match_time_us": 10483272,
  "entity": {
    "stable_observation_id": "A:hog-rider:0",
    "card_id": "hog-rider",
    "owner": "A",
    "x_mtile": 3180,
    "y_mtile": 16470,
    "hp_observed": 720,
    "confidence": 0.97
  }
}
```

Tower HP is a high-value observable. Record every confident plateau transition
with the raw crop and OCR confidence. Unit HP should use a hybrid estimate:
health-bar change, known damage transitions, and attack/projectile evidence.
Never convert an uncertain bar length directly into exact ground truth.

The physical lab should emit normalized events such as:

```text
SPAWN, TARGET_CHANGED, ATTACK_STARTED, PROJECTILE_SEEN,
DAMAGE_OBSERVED, STATUS_APPLIED, ENTITY_SPAWNED,
ENTITY_TRANSFORMED, DEATH_OBSERVED, TOWER_HP_CHANGED
```

Each event carries evidence references, uncertainty, and whether it is directly
observed or inferred. Inferred events remain ineligible when the declared gate
requires direct timing evidence.

Before a physical observation can enter validation, held-out, or regression
evidence, the ingest boundary also requires a replay cache that has passed the
existing reader/recognition check and whose sealed SHA-256 is recorded in the
manifest. Missing, malformed, or unrecognized caches remain rejected; an
offline `candidate_only` comparison may still be used to plan the next probe.

## Simulator replay and comparison

For each accepted physical run:

```python
real_observation = lab_runner.run(experiment_spec)
simulated_run = simulator_runner.run(
    experiment_spec,
    ruleset="v1",
    seed=experiment_spec.seed,
)
comparison = fidelity.compare(real_observation, simulated_run)
```

The comparison must use stable selectors such as `card_id`, `source_card_id`,
`owner`, and `target_role`, not run-specific UIDs. Report separately:

- position MAE and p95;
- path length and bridge/lane outcome;
- velocity and timing error;
- target and retarget agreement;
- victim-set agreement;
- HP/damage error;
- tower-hit count and Crown Tower damage;
- alive/dead and spawn/transform agreement;
- first decision-relevant divergence time.

The comparison should produce a compact `divergence.json` rather than only a
final score:

```json
{
  "first_divergence": {
    "video_time_us": 31420000,
    "match_time_us": 12100000,
    "real": "hog-rider target_changed cannon -> princess-tower",
    "simulator": "hog-rider target remained cannon",
    "subsystem": "targeting",
    "confidence": 0.91
  },
  "follow_up": {
    "parameter": "building_acquisition_radius_mtile",
    "sweep": {"start": 5000, "stop": 6000, "step": 50}
  }
}
```

The first-divergence record should link to a minimized simulator scenario when
the failure is reproducible. This converts a physical observation into a
regression test without treating the raw detector output as universal truth.

## Experiment selection and active learning

The readiness report already exposes missing or failed card/mechanic edges. The
lab planner should turn those rows into experiment templates using:

- `unknown_behaviors.json` and `UNKNOWN_BEHAVIORS.md` for unresolved questions;
- card and mechanic coverage from the pinned ruleset;
- current held-out failure clusters;
- match frequency and decision impact;
- observation uncertainty and estimated execution cost.

A practical priority score is:

```text
priority = frequency * decision_impact * uncertainty * failure_rate
           / estimated_run_cost
```

When a boundary is uncertain, generate a local sweep rather than another
random match:

```text
candidate boundary
  - 0.50, -0.25, -0.10, -0.05,
   0.00,
  +0.05, +0.10, +0.25, +0.50
```

Use this for range, splash radius, aggro radius, bridge placement, pull
distance, knockback, lifetime, status duration, and timing boundaries. A
binary-search or sequential-design strategy can then reduce the number of
physical runs.

The AI agent may propose experiments, cluster failures, and explain likely
subsystems. Deterministic code must retain control of taps, timestamps,
simulation, metrics, split assignment, and readiness decisions.

## Evidence and leakage rules

Every run must preserve:

```text
experiment spec hash
ruleset and engine hash
device/capture metadata
capture-group ID and split
raw media hash
action log
replay-cache hash
observation corpus hash
simulator state/event/replay hashes
comparison and divergence report
```

A capture group is a complete independent session, not an individual frame.
Repeated runs on the same phones in one uninterrupted session do not create
independent groups. Device reset, game version, patch, operator/session, and
source media must be recorded well enough to detect correlated evidence.

The physical lab must keep three datasets:

```text
calibration  -> may change parameters and extractors
validation   -> may select implementations and thresholds
heldout      -> opened only for final measurement
regression   -> known failures that must not recur
```

Held-out data is never used by the active-learning planner to tune the
simulator before the report is sealed. A failed extraction, one-HUD run,
timeout, incomplete cache, or ambiguous observation remains rejected evidence.

Raw recordings and large replay caches belong under ignored output paths such
as:

```text
outputs/simulator/fidelity_media/physical_lab/<run_id>/
```

They must be registered with the existing retention manifest and checked by
`media-budget`. No physical-lab recording, cache, or generated report should
be committed to source control merely because it was produced by a run.

The storage guard counts the entire `cr-bot` workspace, not only the media
directory, and rejects any configured cap above 200,000,000,000 bytes. It is
checked before capture and after capture; eviction is opt-in and limited to
already finalized hash-verified raw videos.

## Verification layers

The lab is one layer in a larger verification stack:

| Layer | What it can establish |
| --- | --- |
| Official/current structured data | Costs, HP, damage, release metadata, and explicitly documented rules |
| Controlled physical probe | Hidden timing, geometry, targeting, status, damage, spawn, and lifecycle behavior |
| Public held-out video | Natural movement, pathing, congestion, and decision outcomes |
| Replay/cache observation | Detector-derived trajectories and event candidates, subject to oracle audit |
| Differential simulator test | Implementation disagreement and regression localization |
| Synthetic/property/fuzz test | Determinism, invariants, boundaries, save/reload, and backend parity |
| Full-match replay comparison | Whether small atomic errors accumulate into decision-relevant divergence |

No one layer is sufficient for every mechanic. For example, a Cannon damage
value may use official data plus a controlled HP-transition probe; bridge
pathing needs controlled placements plus natural held-out trajectories; and
rare death-spawn behavior needs a dedicated isolated probe through the complete
spawn lifecycle.

## Implemented Phase-0 workflow

The harness is available through the simulator CLI. It does not require ADB
or connected phones:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab plan \
  --hog-cannon-only \
  --json-out outputs/simulator/fidelity_media/physical_lab/plan.jsonl
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab run \
  --experiment outputs/simulator/fidelity_media/physical_lab/plan.jsonl \
  --mode offline \
  --json-out outputs/simulator/fidelity_media/physical_lab/offline-summary.json
```

The offline run exercises both logical phones, calibrated cell/slot mapping,
the full lifecycle state machine, action receipts, capture hashes, clock
alignment, and a deterministic simulator replay. Its status is
`candidate_only`; it is not physical evidence. Run artifacts are written below
`outputs/simulator/fidelity_media/physical_lab/<run_id>/` and are registered
with the shared retention manifest when `--retention-manifest` is supplied.
Each run also writes `observation-handoff.json`. It is a sealed index of the
run manifest, A/B capture IDs and hashes, the standard replay-cache extractor
commands, and the expected ingest/comparison/fidelity output paths. It is a
handoff plan, not evidence or a truth label.

`lab ingest` accepts a detector-produced JSON observation document and retains
low-confidence, inferred-timing, synchronization-failed, or otherwise
ambiguous rows in `rejected`; `lab compare` emits the stable-selector metrics
and first-divergence report. `lab run --mode adb` performs device provenance
preflight and fails closed if either phone is absent. For lifecycle admission,
pass both `--lifecycle-templates-a` and `--lifecycle-templates-b`. Each manifest
must contain at least one reviewed PNG template for every lifecycle state,
relative paths, per-file `sha256:` hashes, matching `device_id`, and optional
canonical `manifest_hash`; score and runner-up-margin thresholds are recorded
in the lifecycle report. The detector is deliberately a coarse screen gate,
not an observation oracle, and its manifest/template hashes are retained as
provenance. Omitting either manifest keeps the connected path fail-closed.
Connected evidence still requires verified continuous video capture, recognized
replay-cache extraction, and the normal observation/readiness gates.

Physical raw recordings are registered as non-evictable provenance when
`lab run` writes its artifacts. After a recognized replay cache and a
non-rejected observation manifest have been sealed, finalize that same ingest
with:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab ingest observations.json \
  --run outputs/simulator/fidelity_media/physical_lab/<run-id>/run.json \
  --replay-cache outputs/simulator/fidelity_media/physical_lab/<run-id>/replay-cache-A.pkl.gz \
  --retention-manifest outputs/simulator/fidelity_media/retention.json \
  --json-out outputs/simulator/fidelity_media/physical_lab/<run-id>/observations.json
```

Finalization re-hashes both captures and every retained audit/scenario path,
checks the run's passed complete lifecycle and reviewed detector hashes, the
two device identities, capture group, split, synchronization, and replay-cache
hash, then marks only the matching raw videos eviction-eligible. Rejected or
incomplete ingest never unlocks deletion. `lab run` measures the whole
workspace before capture and reserves the requested `--reserve-bytes`;
`--evict` can remove only older finalized registered media. The shared
`media-budget` command now scans the complete `fidelity_media` tree so public
and physical raw recordings use the same path-safe cap.

After ingest has produced a non-rejected observation manifest, the physical
fidelity bridge verifies the run/cache bindings and writes the standard corpus
and report schemas:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab fidelity \
  outputs/simulator/fidelity_media/physical_lab/<run-id>/observations.json \
  --run outputs/simulator/fidelity_media/physical_lab/<run-id>/run.json \
  --replay-cache outputs/simulator/fidelity_media/physical_lab/<run-id>/replay-cache-A.pkl.gz \
  --corpus-out outputs/simulator/fidelity_media/physical_lab/<run-id>/fidelity-corpus.json \
  --json-out outputs/simulator/fidelity_media/physical_lab/<run-id>/fidelity-report.json
```

The bridge requires a recognized cache file, accepted synchronization,
acknowledged direct timing, passed lifecycle, verified A/B captures, and
matching sealed hashes. One probe can contribute evidence but cannot satisfy
the training-readiness minimums. The first probe uses the A cache as the
primary observation-cache admission while retaining B as auxiliary capture
provenance; a side-by-side cache schema extension remains required before
claiming dual-cache completeness.

## Proposed repository workflow

The physical-lab CLI is present as `python -m simulator lab`. It composes with
the current commands rather than creating a second evidence format:

```text
lab plan
  -> physical experiment JSONL

lab run
  -> raw phone captures, action log, lifecycle report

lab ingest
  -> sealed replay-cache roots and lab observation manifest

mine-replay-batch / specialized miners
  -> confidence-gated movement, combat, spell, and lifecycle observations

compile-video-truth
  -> fidelity corpus with explicit capture groups and splits

fidelity
  -> per-measurement and first-divergence report

readiness
  -> fail-closed training-readiness summary
```

Before any physical run, use the existing `ruleset`, `reconcile-data`, and
`media-budget` checks. After every run, require media-stream validation,
cache-completeness validation, source/hash sealing, and a non-empty lifecycle
result. A successful subprocess exit alone is not evidence success.

## Rollout plan

### Phase 0: software-only lab harness (implemented)

- Experiment schemas and canonical hashing are in `physical_lab/schema.py`.
- Fake/ADB phone and capture adapters are in `physical_lab/devices.py`.
- Lifecycle recovery, hash-verified reviewed screen templates, action logging,
  split locking, clock conversion, timeout handling, cache
  recognition/rejection, observation ingest, differential comparison, and
  deterministic simulator replay are covered by the package and focused tests.

### Phase 1: one complete probe

Use two devices to validate one controlled `hog_cannon_pull` or isolated Hog
movement experiment from action to readiness report. Do not expand the card
matrix until the complete artifact chain is reproducible.

### Phase 2: core Hog-cycle evidence

Cover Hog movement/bridge path, Cannon placement and lifetime, Musketeer
projectile timing, Skeletons lifecycle, Ice Golem death slow, Ice Spirit
connection, Fireball, and Log. Use separate capture groups for calibration and
held-out measurement.

### Phase 3: mechanic clusters

Add air navigation, buildings/spawners, spells, status effects, collision,
death/split streams, and exceptional cards in dependency clusters. Generate
boundary sweeps from the first-divergence reports.

### Phase 4: full V1 readiness

Run the complete card/mechanic ledger, independent capture groups, oracle audit,
leakage checks, cumulative fuzz/soak checks, and final `readiness`. Only then
use the evidence-qualified mechanics for serious RL training.

## Per-run definition of done

A physical experiment is complete only when:

- the requested actions were acknowledged and timestamped;
- the battle lifecycle completed or failed with an explicit reason;
- both captures have verified video streams and hashes;
- synchronization uncertainty is recorded;
- the replay caches are complete and recognized by the existing reader;
- every observation has confidence, source frame/time, and provenance;
- simulator and real runs share the same logical experiment specification;
- comparison metrics and first divergence are generated;
- the evidence split and capture group are immutable;
- ambiguous or failed cases are retained as rejection records;
- the output is eligible for the stated gate, or explicitly marked
  `candidate_only`, `validation`, `calibrated_only`, or `rejected`.

The lab is successful when it makes the next experiment more informative and
the simulator's remaining uncertainty smaller—not when it produces a green
metric by relaxing the evidence contract.
