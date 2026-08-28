# Physical-fidelity lab goal and current implementation

## Goal

Build a repeatable sim-to-real test corpus for the deterministic Clash Royale
simulator. Two physical phones run the same reviewed interaction cases as the
simulator, with the physical opening hand fixed by deck order. The campaign
starts with isolated one-troop interactions and grows to paired, multi-card,
and complex interactions. Each captured case keeps its raw captures,
provenance, replay cache, normalized observations, comparison, and fidelity
report so a simulator change can be evaluated against all previously gathered
physical data.

Phone B is the friendly-match host. On B, the operator opens Friendly Battle,
holds the `1v1 Battle` button, enables `Fixed Deck Order`, and then hosts the
match. With that option enabled, the first four ordered deck cards are the
starting hand. Phone A accepts the challenge.

The host must challenge the controlled online account `KeschlerHD`. The
automation scans the rendered online-player names with bounded OCR, taps the
matching row, verifies that the selected popup names `KeschlerHD`, and only
then opens `Testspiel`. It never assumes that the first online row is the
correct account. The fixed-order switch is also read back from the rendered
blue active half before tapping, so a rerun does not accidentally turn an
already-enabled switch off.

The current device mapping is explicit: Phone A is the ASUS AI2302 at
1080x2400, and Phone B is the Samsung SM-G970F/beyond0 at 1080x2280. Physical
UI coordinates use each phone's native dimensions. The A extractor uses the
ASUS-calibrated 1080x2400 identity path; the B extractor selects the existing
alternative HUD profile on the extractor's supported 1080x2400 processing
canvas. Passing `--no-normalize` to the B stream is invalid because the
detector rejects the Samsung aspect ratio before the alternative ROIs run.
Every run records the native source dimensions, device identity, extractor
profile, and normalization transform contract.
The reviewed B deck rows in the top editor are centered near normalized
y=0.400 and y=0.590; these native coordinates are separate from the normalized
ASUS processing frame and are used only for B's deck-editor input.

Phone A has an additional observed rule: regular Musketeer must be in human
deck slots 4-8, because human slots 1-3 produce Hero Musketeer. The v4 A deck
therefore uses `Hog Rider, Cannon, Skeletons, Musketeer, Ice Golem, Ice Spirit,
Fireball, Log`; the v4 B deck remains
`Hog Rider, Fireball, Log, Cannon, Skeletons, Musketeer, Ice Golem, Ice Spirit`.

Samsung B has a card-identity constraint: `Cannon`, `Skeletons`, `Musketeer`,
`Ice Golem`, and `Ice Spirit` must all be in human deck slots 4-8 (zero-based
slot >=3). Putting any of them in human slot 1-3 activates a Hero/Evolution
variant. Therefore the B fixed deck is ordered as `Hog Rider, Fireball, Log,
Cannon, Skeletons, Musketeer, Ice Golem, Ice Spirit`; its fixed-order opening
hand is the first four of that order. Clone is not a permitted target card and
must be removed from its actual human slot 8 before the deck is accepted.

## Current implementation

- `campaign.py` defines the versioned `InteractionCampaign`, `CampaignCase`,
  and `DeckMutation` records. Decks are validated against the fixed ruleset,
  mutations are deterministic, and the first four ordered cards become
  `hand_slots`.
- `build_default_campaign()` creates five ordered cases:
  isolated Hog, isolated Archers, Hog/Cannon pull, Hog/support/Cannon, and a
  three-card pressure case. The cases contain one or more deck changes and
  immutable hashes.
- `scripts/run_physical_fidelity_campaign.py plan` writes the campaign and
  write-once per-case artifacts. Existing artifacts cannot be overwritten with
  different contents.
- `scripts/run_physical_fidelity_campaign.py evaluate` loads every stored
  `fidelity-corpus.json`, runs the current simulator, applies each case's
  mechanic gate, and writes a new sealed evaluation snapshot. Reuse a new
  output path for a new simulator evaluation; physical evidence is not
  rewritten.
- `scripts/run_physical_lab_autonomous.py --campaign ... --case-id ...`
  selects a campaign case, prepares its exact decks, validates the preparation
  manifests against those decks, and uses the case's actions for the physical
  run. The autonomous action loop now supports ordered match-time actions and
  reviewed after-observation boundaries. Unsupported observations still fail
  closed.
- Fixed-deck preparation in the two-phone path arms the option on B when both
  phones are prepared. The connected run already sends B's challenge and has A
  accept it.
- The connected coordinator defaults its friendly-match target to
  `KeschlerHD`. It uses OCR to locate that name despite online-list reordering,
  verifies the selected-player popup, locates the localized `Testspiel` button,
  and records the target row/menu coordinates in the action result. Detection
  first scans one bounded online panel, then retries bounded row crops across
  animated frames and uses a second OCR mode only as a local fallback; the
  adjacent `pwn_keschler` clan label is rejected by both the required name
  prefix and the `HD` suffix. A missing, ambiguous, or post-tap-mismatched
  name fails closed before a challenge is sent. If the current screen is not a
  verified lobby, the retry path does not navigate the bottom bar blindly.
- Fixed-deck setup is idempotent on the named-player path: the B switch is
  inspected after the long press and is tapped only when it is visibly off;
  the enabled state is verified before hosting.
- Preparation carries a Samsung-specific guard: a B deck that puts Cannon,
  Skeletons, Musketeer, Ice Golem, or Ice Spirit in human slots 1-3 is rejected
before card input, preventing Hero/Evolution substitutions. The default v4
campaign encodes the safe A/B orders and records both constraints in immutable
metadata.
- Preparation also carries the ASUS-specific regular-Musketeer guard and a
  donor-slot preflight: if a desired card is already in another deck slot,
  that misplaced copy is removed first so the game exposes it in the
  collection picker instead of silently hiding it as a duplicate.
- Collection selection uses a high-confidence identity and margin gate. The
  reviewed B card-upgrade tutorial (`Verstärke deine Karten!`) is detected as
  a modal and closed only by its upper-right control; the candidate is then
  rejected and preparation stops for review rather than committing an
  ambiguous card.
- The reviewed regular Musketeer capture on B has a lower art-correlation
  floor (`0.50`) because of its current frame/level rendering; the ASUS
  capture has an evidenced `0.48` floor. Both still need the normal `0.12`
  winner-over-runner-up identity margin. Other cards retain the `0.60`
  collection floor.
- Physical action timing is anchored to the workstation monotonic clock after
  both reviewed lifecycle detectors report `BATTLE`. The in-game clock remains
  diagnostic visual evidence only; it is never used to schedule actions or
  assign match timestamps.
- The authoritative timestamp for a played card is the accepted placement
  receipt's completion time relative to the BATTLE barrier. The post-placement
  screenshot is deliberately not used as the action timestamp, because an ADB
  read can add seconds of latency. Older runs with the former field are
  corrected in memory when their placement receipt is present.
- After a complete candidate capture, `extract_physical_run()` invokes the
  pre-existing `cr_bot` extractor separately for A and B, seals both recognized
  replay caches, and stores the extractor command/output, cache hash, native
  geometry, timeline, hand/elixir/tower state, unit samples,
  spawn/disappearance/tower-damage events, and acknowledged runner action
  receipts. The visual extractor's game clock is retained only as a diagnostic
  field.
- Card identity is now adjudicated with the physical experiment's known card
  universe. An accepted placement receipt can override a detector class near
  that placement; every unreceipted class must belong to the observed owner's
  declared deck. Impossible or unmapped classes are excluded from normalized
  entities/events but retained under `identity_rejections` with their raw class,
  frame, confidence, and reason. The stream records `identity_source`, raw
  detector identity, and matched action ID for every accepted correction. This
  prevents a detector-only `hunter` label from entering a deck that contains
  only `hog-rider`, while preserving the raw evidence for later model work.
- Lifecycle normalization now requires three consecutive missing frames before
  promoting a disappearance to a confirmed transition. A same-card track that
  reappears within 500 ms and within the calibrated movement distance is merged
  back into its prior logical track. The initial edge remains readable as a
  `tentative` event and is excluded from simulator scoring unless it is
  independently corroborated.
- Overlapping A/B lifecycle and tower observations are matched one-to-one on
  the internal synchronized time axis. Matched events are stored once with
  both capture references and cross-phone timing delta; two tentative views
  can promote a transition to inferred evidence.
- Cache video timestamps are mapped onto the internal axis using each capture's
  monotonic start and the reviewed monotonic BATTLE barrier. Acknowledged input
  receipts are direct-timing observations on that same axis; no comparison
  timing is computed from the displayed in-game clock.
- The per-device synchronization alignment is applied while normalizing each
  stream. In particular, Phone B's measured offset is subtracted from its
  video-derived match times, and the applied offset is retained in stream
  provenance. The native source dimensions and extractor processing canvas
  remain separate throughout this mapping.
- Comparison normalizes extractor lifecycle/card vocabulary to simulator event
  families (`unit_spawn_observed`/`entity_created`, disappearance/death, and
  transforms), removes simulator-only bookkeeping rows from first-divergence
  ordering, and keeps input card-play rows out of mechanics agreement. Event
  multiplicity is counted without reusing one simulator event for every visual
  detector flicker.
- `physical-lab evaluate-stored` discovers every immutable `extracted-case.json`,
  replays its exact logical actions through the current simulator, and writes a
  new aggregate snapshot with a bounded fidelity score, component metrics, and
  first-divergence time. Physical captures, caches, and observations are never
  rewritten, so simulator changes can be evaluated across the complete corpus.
- Battle hand selection uses the reviewed native `hand_px` rectangle from each
  phone's calibration artifact. It does not assume four equal screen quarters;
  this matters on the ASUS and Samsung layouts where the rendered card centers
  are inset and differ from the visual-extractor's normalized processing frame.
- `AdbPhoneController.set_keep_awake()` applies and verifies the maximum
  Android screen timeout, `stay_on_while_plugged_in=3`, and `svc power stayon
  true`. The `lab keep-awake` command requires two explicit, distinct serials;
  it never discovers or addresses an unspecified device.
- The workspace guard counts regular files below the whole `cr-bot` root,
  including `.git`, `outputs`, datasets, environments, and generated files. A
  hard 200,000,000,000-byte ceiling is enforced before physical capture and
  checked again after capture. Eviction remains opt-in and is limited to
  already finalized, hash-verified raw videos.

## Required operator commands

Create the immutable case matrix:

```bash
outputs/venv/bin/python scripts/run_physical_fidelity_campaign.py plan
```

Set both phones to stay awake after their serial-to-phone mapping has been
confirmed:

```bash
PYTHONPATH=.:src outputs/venv/bin/python -m simulator lab keep-awake \
  --serial-a PHONE_A_SERIAL \
  --serial-b PHONE_B_SERIAL
```

The autonomous coordinator runs the extractor automatically after a complete
candidate capture. For an already sealed run, the same boundary can be run
explicitly with `physical-lab extract-run --run ... --json-out ...`; stored
cases can then be re-scored with:

```bash
PYTHONPATH=.:src outputs/venv-cpu/bin/python -c \
  'from simulator.physical_lab.cli import main; raise SystemExit(main(["evaluate-stored", "--json-out", "outputs/simulator/fidelity_media/physical_lab/stored-evaluation.json"]))'
```

Prepare a selected case, with B hosting fixed deck order:

```bash
outputs/venv/bin/python scripts/run_physical_lab_autonomous.py \
  --serial-a PHONE_A_SERIAL --serial-b PHONE_B_SERIAL \
  --campaign outputs/simulator/fidelity_media/physical_lab/campaigns/\
  physical-fidelity-interaction-sweep-v4/campaign.json \
  --case-id hog-cannon-pull \
  --prepare-only --prepare-side both --fixed-deck-order \
  --fixed-deck-toggle-point X,Y --keep-awake
```

The live run additionally requires the reviewed calibration, lifecycle
manifests, preparation manifests, and the reviewed host start point. No live
ADB command is issued by the software-only planning or evaluation commands.

## Evidence boundary

Planning and simulator re-evaluation are software operations. A physical run
is only a candidate until both captures are sealed, synchronization and
lifecycle checks pass, a recognized replay cache and normalized observation
manifest are admitted, and the fidelity/readiness gates pass. The campaign
does not turn a candidate capture into ground truth automatically.
