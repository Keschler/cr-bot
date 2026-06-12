# Flowcharts

This document maps the first-party runtime, extraction, tracking, dataset, and debug scripts. Diagrams use Mermaid so they render directly in GitHub Markdown.

Generated CVAT serverless functions, vendored code, and virtualenv files are intentionally excluded.

## Script Index

| Script | Diagram section |
| --- | --- |
| `capture/capture.py` | `src/cr_bot/app/cli.py` and `capture/capture.py` |
| `src/cr_bot/app/cli.py` | `src/cr_bot/app/cli.py` and `capture/capture.py` |
| `src/cr_bot/app/main.py` | `src/cr_bot/app/main.py`: Top-Level Runtime, `src/cr_bot/app/pipeline.py`: `process_frame()` |
| `scripts/build_card_classifier_imagefolder.py` | Card Classifier Dataset Scripts |
| `scripts/build_seed_annotations.py` | Seed Detector Dataset And Training Scripts |
| `scripts/clean_preannotations.py` | Seed Detector Dataset And Training Scripts |
| `scripts/download_ryleycr1_archive.py` | Utility Scripts |
| `scripts/fetch_royaleapi_data.py` | Utility Scripts |
| `scripts/generate_cvat_labels.py` | Utility Scripts |
| `scripts/import_card_classifier_frames.py` | Card Classifier Dataset Scripts |
| `scripts/mine_enemy_audio_audit.py` | Enemy Audio Mining Scripts |
| `scripts/mine_enemy_audio_dataset.py` | Enemy Audio Mining Scripts |
| `scripts/prepare_card_classifier_dataset.py` | Card Classifier Dataset Scripts |
| `scripts/prepare_seed_dataset.py` | Seed Detector Dataset And Training Scripts |
| `scripts/propose_card_classifier_split.py` | Card Classifier Dataset Scripts |
| `scripts/run_seed_inference.py` | Seed Detector Dataset And Training Scripts |
| `scripts/setup_seed_detectors.py` | Seed Detector Dataset And Training Scripts |
| `scripts/train_audio_classifier.py` | Audio Classifier Training And Evaluation |
| `scripts/train_card_classifier.py` | Card Classifier Dataset Scripts |
| `scripts/train_seed_baseline.py` | Seed Detector Dataset And Training Scripts |
| `scripts/debug/debug_actions_output_selected_actions.py` | Debug Scripts |
| `scripts/debug/debug_own_action_clock_cells.py` | Debug Scripts |
| `scripts/debug/render_predicted_cells.py` | Debug Scripts |
| `scripts/debug/debug_skeleton_clock_cell.py` | Debug Scripts |
| `scripts/debug/debug_spell_purple_detector.py` | Debug Scripts |
| `dataset_generation/scripts/detect_in_game.py` | Data Generation Scripts |
| `dataset_generation/scripts/download_videos.py` | Data Generation Scripts |
| `dataset_generation/scripts/process_frame.py` | Data Generation Scripts |

## `src/cr_bot/app/cli.py` And `capture/capture.py`

```mermaid
flowchart TD
    A["python capture/capture.py or PYTHONPATH=src python -m cr_bot.app.cli"] --> B["parse CLI args"]
    B --> C{"mode"}
    C -->|"--debug or --debug-frame"| D["main(debug=True, debug_frame_path=...)"]
    C -->|"--video"| E["main(debug=False, video=...,<br/>frame_stride or video_sample_interval_s)"]
    C -->|"default"| F["main(debug=False): live video device"]
    B --> G["pass normalize and yolo_detections flags"]
    G --> D
    G --> E
    G --> F
```

## `src/cr_bot/app/main.py`: Top-Level Runtime

```mermaid
flowchart TD
    A["main()"] --> B["build_detector()"]
    B --> C["create trackers: EnemyCardTracker, OwnActionTracker, MatchClockFilter, TowerHPFilter"]
    C --> D{"runtime mode"}

    D -->|"debug frame"| E["read debug image"]
    E --> F["normalize_frame() if enabled"]
    F --> G["process_frame()"]
    G --> H["TowerHPFilter.update()"]
    H --> I["EnemyCardTracker.start_match()"]
    I --> J["build_game_state()"]
    J --> K["OwnActionTracker.update()"]
    K --> L["EnemyCardTracker.reconcile_own_actions()"]
    L --> M["EnemyCardTracker.update()"]
    M --> N["print/debug windows"]

    D -->|"video file"| O["open cv2.VideoCapture(video)"]
    O --> P["decode frames sequentially<br/>sample by frame_stride or exact timestamp interval"]
    P --> Q["normalize_frame() if enabled"]
    Q --> R["process_frame()"]
    R --> S{"game_started?"}
    S -->|"no"| T{"game_start() or visible timer?"}
    T -->|"yes"| U["initialise clock, tower HP, enemy elixir"]
    U --> V["build_game_state() and OwnActionTracker.update()"]
    T -->|"no"| W["skip frame"]
    S -->|"yes"| X["MatchClockFilter.update()"]
    X --> Y["TowerHPFilter.update()"]
    Y --> Z["build_game_state()"]
    Z --> AA["OwnActionTracker.update()"]
    AA --> AB["EnemyCardTracker.reconcile_own_actions()"]
    AB --> AC["EnemyCardTracker.update()"]
    AC --> AD{"game_end_from_result() streak >= 20?"}
    AD -->|"yes"| AE["reset trackers and game_started"]
    AD -->|"no"| AF["print_frame_result(), optional display"]

    D -->|"live capture"| AG["open VIDEO_DEVICE or dummy v4l2 device"]
    AG --> AH["warmup process_frame()"]
    AH --> AI["loop live frames"]
    AI --> AJ["same started/in-game tracker loop as video mode"]
```

## `src/cr_bot/app/pipeline.py`: `process_frame()`

```mermaid
flowchart TD
    A["input frame"] --> B["optional draw_rois()"]
    B --> C["ratio2name() validates aspect ratio"]
    C --> D["process_part(frame, part=2) extracts arena crop"]
    D --> E["compute arena_px in original frame coordinates"]
    E --> F["detector.infer(arena)"]
    F --> G["result.get_data(): YOLO boxes"]
    G --> H["remap_boxes_to_frame()"]
    H --> I["extract_clock_boxes()"]
    H --> J["convert_yolo(): troops and bars"]
    J --> K["estimate_health(frame, bars)"]
    K --> L["match_troops_to_bars()"]
    L --> M["match_from_dict(): typed Match objects"]

    A --> N["extract_elixir()"]
    A --> O["extract_time(), is_overtime(), parse_time_left_s()"]
    O --> P["total_remaining_seconds()"]
    A --> Q["extract_hand_state()"]
    A --> R{"tower HP mode"}
    R -->|"YOLO bars"| S["extract_tower_hp(frame, tower_hp_yolo_boxes)"]
    R -->|"fixed ROIs"| T["extract_tower_hp(frame)"]

    M --> U["return result dict"]
    I --> U
    N --> U
    P --> U
    Q --> U
    S --> U
    T --> U
```

## Runtime State Assembly

```mermaid
flowchart TD
    A["process_frame() result"] --> B["build_game_state()"]
    B --> C["HudState"]
    C --> C1["time, overtime, elixir_self"]
    C --> C2["hand_cards and next_card"]
    C --> C3["tower HP and princess tower alive flags"]
    B --> D["own_units = matches with team ally"]
    B --> E["enemy_units = matches with team enemy"]
    B --> F["seen_enemy_cards from EnemyCardTracker"]
    B --> G["elixir_enemy_est from EnemyCardTracker"]
    B --> H["GameState consumed by policy/features/debug code"]
```

## Own Action Tracking

`OwnActionTracker` detects the player's actions incrementally. A hand-slot change
creates a `PendingOwnPlay`; later frames attach timing evidence and a placement
cell. The action is emitted only when the evidence required by that card type is
available. Mirror is represented as the repeated card with
`played_via="mirror"`; it is not emitted as a separate `mirror` action.

```mermaid
flowchart TD
    subgraph Inputs["Per-frame inputs"]
        I1["game_state<br/>hud.hand_cards: list[str | None]<br/>hud.elixir_self: float<br/>total_remaining_s: float<br/>own_units: list[Match]"]
        I2["arena_px<br/>(x, y, width, height)"]
        I3["frame<br/>BGR image or None"]
        I4["clock_boxes<br/>list[{team, confidence,<br/>center_x, center_y, ...}]"]
        I5["elixir_change<br/>{covered, white, pink, edges}<br/>from digit-overlay detector"]
        I6["video_time_s<br/>float or None"]
        I7["own_actions_blocked<br/>bool, true during emote clutter"]
    end

    I1 --> U["OwnActionTracker.update(...)"]
    I2 --> U
    I3 --> U
    I4 --> U
    I5 --> U
    I6 --> U
    I7 --> U

    U --> B{"own_actions_blocked?"}
    B -->|"yes"| B1["Preserve pending plays<br/>Update last_hand, last_elixir,<br/>recent hand and slot history<br/>Return None"]
    B -->|"no"| D["_detect_slot_drops(hand, elixir, now,<br/>optional visual-overlay timestamp)"]

    D --> D1{"How did a hand slot change?"}
    D1 -->|"previous card -> None"| P0["_resolve_drop_card()<br/>stabilize label from slot history"]
    D1 -->|"Log or Barbarian Barrel -> another label"| P0
    D1 -->|"other replacement"| D2["Ignore as OCR churn"]
    D1 -->|"no relevant change"| C1["Continue"]
    P0 --> PM{"Dropped card is Mirror?"}
    PM -->|"yes, prior confirmed action exists"| PM1["Use previous action.card<br/>played_via = mirror"]
    PM -->|"yes, no prior action"| PM2["Ignore unresolved Mirror drop"]
    PM -->|"no"| P1["Use dropped card<br/>played_via = None"]
    PM1 --> P2
    P1 --> P2
    P2["Append PendingOwnPlay carrying<br/>card, slot_idx, started_at_s, elixir_before,<br/>played_via, visual/numeric elixir timestamps,<br/>spell state and rolling-spell state"] --> C1
    PM2 --> C1
    D2 --> C1

    C1 --> V{"elixir_change.covered?"}
    V -->|"yes"| V1["_attach_elixir_change_to_pending(now, video_time_s)<br/>Return None<br/>Attach calibrated visual-overlay timestamp<br/>to closest eligible pending play"]
    V -->|"no"| C2["Continue"]
    V1 --> C2

    C2 --> CP["_confirm_pending(...)"]
    CP --> T1["Remember tracked own units<br/>Input: game_state.own_units<br/>Update recent_ally_tracks<br/>Output: new_tracks"]
    T1 -.-> F1["_record_new_track_actions(new_tracks, hand, arena_px, now, clock_boxes)<br/>Only when no pending play existed<br/>DIRECT_UNIT_TO_CARD maps track class to card<br/>Allow only explicit track-fallback cards"]
    F1 --> F2["_infer_cell_from_clock([match], arena_px, clock_boxes, card)<br/>Return deploy-clock cell or allowed troop-center fallback"]
    F2 --> A
    T1 --> E1["_pending_for_current_elixir_drop(elixir, now, preferred_pending)<br/>Compare current drop with required card cost<br/>Mirror uses repeated-card cost + 1<br/>Output: one PendingOwnPlay or None"]
    E1 --> E2{"Numeric elixir drop selected<br/>a pending play?"}
    E2 -->|"yes"| E3["Latch numeric_elixir_drop_time_s<br/>and numeric_elixir_drop_video_time_s<br/>on PendingOwnPlay"]
    E2 -->|"no"| K{"Pending card type?"}
    E3 --> K

    K -->|"normal troop or building"| N1["_recent_tracks_for_pending(pending, now)<br/>Return recent list[Match]"]
    N1 --> N2["_infer_cell_from_clock(tracks, arena_px, clock_boxes, card)<br/>Match card track to nearby ally deploy clock<br/>Return cell: tuple[int, int] or None"]
    N2 --> N3{"Cell and visual or latched<br/>numeric elixir evidence?"}

    K -->|"other spell"| S1["_confirm_pending_spell(pending, arena_px, frame, elixir_confirms)<br/>SpellDeployLocator uses radius metadata to find<br/>white aim ellipse and purple release marker<br/>Return (cell or None, keep_pending)"]
    S1 --> S2{"Release cell and latched<br/>spell elixir evidence?"}

    K -->|"Log or Barbarian Barrel"| R1["_confirm_pending_rolling_spell(...)<br/>Use first visible ally rolling-object center<br/>and latch spell elixir evidence<br/>Return (cell or None, keep_pending)"]
    R1 --> R2{"First rolling cell and latched<br/>spell elixir evidence?"}

    N3 -->|"yes"| A
    S2 -->|"yes"| A
    R2 -->|"yes"| A
    N3 -->|"not yet, within timeout"| W["Keep PendingOwnPlay for a later frame"]
    S2 -->|"not yet, keep_pending"| W
    R2 -->|"not yet, keep_pending"| W
    N3 -->|"stale"| X["Remove pending play without emitting action"]
    S2 -->|"cancelled"| X
    R2 -->|"stale or mismatched"| X

    A["_append_action(now, card, slot_idx, cell,<br/>video_time_s, rolling_spell_track_id, played_via)<br/>Reject pre-start actions<br/>Reject recent duplicates except confirmed Mirror"] --> A1["Append OwnActionEvent<br/>{time_left_s, video_time_s, card, slot_idx, cell,<br/>rolling_spell_track_id, played_via}"]
    A1 --> OUT["OwnActionTracker.actions<br/>list[OwnActionEvent] consumed by enemy-card<br/>reconciliation, debug output and state pipelines"]
    W --> OUT2["Update tracker memory and return None"]
    X --> OUT2
```

### Own Action Data Contracts

| Value | Producer | Shape | Purpose |
| --- | --- | --- | --- |
| `game_state.hud.hand_cards` | `build_game_state()` | Four-element `list[str \| None]` | Detect the card that disappeared from a hand slot. |
| `game_state.hud.elixir_self` | `build_game_state()` from `extract_elixir()` | `float` | Compare adjacent frames and latch numeric elixir-drop evidence. |
| `elixir_change` | `detect_elixir_change()` | `{covered: bool, white: float, pink: float, edges: float}` | Detect the visual digit overlay and preserve its timestamp. |
| `game_state.own_units` | YOLO conversion and `build_game_state()` | `list[Match]`, each with `match.troop.class_name`, `track_id`, center and team | Associate a pending play with a visible own unit or rolling spell. |
| `clock_boxes` | `extract_clock_boxes()` | `list[dict]` with track ID, team, confidence, box and center coordinates | Estimate the placement cell for normal troops and buildings. |
| `PendingOwnPlay` | `_detect_slot_drops()` | Stateful dataclass including card, slot, evidence timestamps, spell state, rolling-spell state and `played_via` | Join asynchronous hand, elixir, clock and spell evidence. |
| `OwnActionTracker.actions` | `_append_action()` | `list[OwnActionEvent]` with `time_left_s`, `video_time_s`, `card`, `slot_idx`, `cell`, `rolling_spell_track_id`, `played_via` | Final confirmed own-action stream. Mirror retains the repeated card ID. |

### Rolling-Spell Own Action Detection

```mermaid
flowchart TD
    A["hand slot has Log or Barbarian Barrel"] --> B{"slot disappears or changes"}
    B -->|"yes"| C["create PendingOwnPlay<br/>{card, slot_idx, started_at_s, elixir_before}"]
    C --> D["watch next frames"]
    D --> E{"first matching ally YOLO rolling-object track visible?"}
    E -->|"no"| F["keep pending until timeout"]
    E -->|"yes"| G["choose one matching pending rolling spell<br/>within 1.0s window"]
    G --> H["store first rolling-object track id and cell"]
    H --> I{"current elixir drop >= 0.8<br/>or spell elixir evidence already latched?"}
    I -->|"yes"| J["cell = first visible rolling-object box center<br/>mapped through ACTION_GRID"]
    J --> K["append OwnActionEvent<br/>including rolling_spell_track_id<br/>and optional played_via"]
    K --> L["consume YOLO track id"]
    I -->|"no"| M["keep selected pending, retain first rolling-spell cell"]
```

## Enemy Card Tracking

`EnemyCardTracker` combines deploy-clock claims, multi-frame detector evidence,
motion direction, own-action reconciliation and spell-specific continuation
tracking. Recorded plays may be revised later when class votes or projectile
evidence improve.

```mermaid
flowchart TD
    A["EnemyCardTracker.update(...)"] --> B["Update enemy elixir estimate<br/>remember enemy clock boxes<br/>remember own Log claims<br/>expire stale projectile observations"]
    B --> C["For each YOLO match"]
    C --> R{"Rolling Log object?"}
    R -->|"yes"| R1["RollingSpellTracker.observe()<br/>accumulate trajectory and assign own claims"]
    R1 --> R2{"Direction confirms enemy<br/>and not own claimed?"}
    R2 -->|"yes"| R3["Record enemy Log<br/>suppress later same-lane fragments"]
    R2 -->|"no"| NEXT["Continue"]
    R3 --> NEXT

    R -->|"no"| T{"team != enemy?"}
    T -->|"yes, ally-labelled Fireball"| F1["ProjectileSpellTracker.observe_ally_fireball()<br/>compare own actions, active enemy trajectories<br/>and vertical motion direction"]
    F1 --> F2{"ownership/result"}
    F2 -->|"own or continuation"| NEXT
    F2 -->|"enemy"| REC
    F2 -->|"unresolved/explosion"| NEXT
    T -->|"yes, other class"| NEXT

    T -->|"no"| M{"track_id exists?"}
    M -->|"no"| NEXT
    M -->|"yes"| TM["Create/update TrackMemory<br/>class/team votes, confidence, centers,<br/>motion centers and first video timestamp"]
    TM --> C1{"claim current/recent enemy deploy clock?"}
    C1 -->|"yes"| C2["clock_confirmed=True<br/>store deploy-clock center<br/>consume clock for this track"]
    C1 -->|"no"| FC{"configured frame-confirm class<br/>with enough frames, votes and confidence?"}
    FC -->|"yes"| C3["frame_confirmed=True"]
    FC -->|"no"| NEXT
    C2 --> MP
    C3 --> MP
    MP{"Track already counted?"}
    MP -->|"yes"| REV["_maybe_revise_recorded_play()<br/>late class-vote changes update card, cost,<br/>event ID, cell and projectile event"]
    MP -->|"no"| REL{"Reliable enemy play?"}
    REL -->|"no"| NEXT
    REL -->|"yes"| MAP{"DIRECT_UNIT_TO_CARD mapping?"}
    MAP -->|"none"| DROP["Mark track counted without a play"]
    MAP -->|"card"| OWN{"Explained by recent own spell<br/>or active projectile continuation?"}
    OWN -->|"yes"| DROP
    OWN -->|"no"| DUP{"Recent same-card duplicate?"}
    DUP -->|"yes, independently confirmed before 4-card cycle"| MIR["Treat as Mirror<br/>card remains repeated card<br/>played_via=mirror, cost=base+1"]
    DUP -->|"yes, ordinary duplicate"| DROP
    DUP -->|"no"| REC["Create EnemyCardPlay"]

    MIR --> REC
    REC --> CELL{"Cell source"}
    CELL -->|"claimed clock"| CELL1["Deploy-clock center -> ACTION_GRID<br/>raise enemy row by 2"]
    CELL -->|"object/effect"| CELL2["Detected center -> ACTION_GRID<br/>raise enemy row by 2"]
    CELL1 --> SAVE
    CELL2 --> SAVE
    SAVE["Append detected_card_plays<br/>update seen card IDs, including Mirror<br/>subtract inferred cost<br/>register projectile event when supported<br/>mark track counted"] --> NEXT
    DROP --> NEXT
    REV --> NEXT

    NEXT --> POST["After all matches:<br/>assign projectile continuations<br/>confirm delayed Fireball explosions<br/>store recent arena frames and impact observations<br/>reconcile projectile target cells<br/>drop stale tracks/candidates"]
```

Enemy troops and buildings normally need a deploy-clock box that can be claimed
by their specific YOLO track. A nearby `clock:enemy` summary alone is not enough:
the clock must pass confidence/team filters, be geometrically near the troop, and
not already be consumed by another track. Frame-only confirmation is limited to
spell-like detector classes and configured exceptions such as Electro Wizard.

Some detector classes intentionally do not map directly to playable cards. If a
confirmed detector class still maps to `DIRECT_UNIT_TO_CARD == None`, the tracker
marks the track counted and does not append an enemy play. Skeleton and Skeleton
Evolution detections are exceptions: once confirmed, they map to the `skeletons`
card.

Enemy spells are generally frame-confirmed without a deploy clock. Fireball
ownership is resolved from vertical motion and recent own Fireball actions;
later detections can be attached to an existing enemy projectile event and move
its cell farther along the trajectory. The tracker also retains recent arena
frames and spell-target observations for projectile reconciliation. Log uses
trajectory direction plus explicit own-action claims to avoid counting the
player's rolling spell as an enemy action.

Before each enemy update, `MatchSession` calls
`EnemyCardTracker.reconcile_own_actions()`. It removes previously recorded enemy
Log, Fireball or other spell events that are later explained by confirmed own
actions, releases their projectile records and refunds the enemy elixir
estimate.

Mirror cannot be seen directly in the enemy hand. The tracker infers it only
when the same card appears again before four intervening enemy plays and the
second play has independent evidence, such as a claimed deploy clock or a
distinct spell target. The event keeps the repeated card ID, sets
`played_via="mirror"`, charges one extra elixir and adds Mirror to the seen-card
set. A repeat after four intervening plays is treated as a normal cycle.

The current enemy cell calibration raises every emitted enemy play cell by two
rows after grid conversion. In practical terms, `(7, 7)` becomes `(7, 5)`,
clamped at row `0`.

## Enemy Log Detection

```mermaid
flowchart TD
    A["YOLO detects enemy the-log with track_id"] --> B["TrackMemory votes class/team/confidence"]
    B --> C{"at least 3 frames and avg conf >= 0.65?"}
    C -->|"no"| D["wait"]
    C -->|"yes"| E{"class vote ratio >= 0.6 and enemy team ratio >= 0.8?"}
    E -->|"no"| D
    E -->|"yes"| F["frame_confirmed=True"]
    F --> G["map the-log -> log"]
    G --> H{"matches claimed recent own Log<br/>by source track, time, lane or trajectory?"}
    H -->|"yes"| I["veto as own spell duplicate"]
    H -->|"no"| J{"recent enemy Log duplicate within 0.75s?"}
    J -->|"yes, independently confirmed pre-cycle repeat"| M["record mirrored Log<br/>played_via=mirror, subtract 3 elixir"]
    J -->|"yes, fragment or ordinary duplicate"| K["suppress duplicate"]
    J -->|"no"| L["record enemy Log and subtract 2 elixir"]
```

## Vision And YOLO Runtime

```mermaid
flowchart TD
    A["build_detector()"] --> B["AppDetector(DEFAULT_DETECTOR_WEIGHTS)"]
    B --> C["load two YOLO_CR models"]
    C --> D["configure device and ByteTrack"]
    D --> E["infer(arena frame)"]
    E --> F["run each detector model"]
    F --> G["map model class ids to KataCR unit ids"]
    G --> H["merge predictions"]
    H --> I["NMS"]
    I --> J["CRResults"]
    J --> K["ByteTrack postprocess"]
    K --> L["filter out top-screen HUD clutter"]
    L --> M["result.get_data() consumed by process_frame()"]
```

## Extractors

```mermaid
flowchart TD
    A["frame"] --> B["extract_hand_state()"]
    A --> C["extract_elixir()"]
    A --> D["extract_time() / is_overtime()"]
    A --> E["extract_tower_hp()"]
    A --> F["estimate_health()"]
    A --> G["match_troops_to_bars() after YOLO conversion"]

    B --> H["hand card labels + next card"]
    C --> I["estimated elixir + displayed digit"]
    D --> J["match clock and total remaining seconds"]
    E --> K["tower HP dictionary"]
    F --> L["bar estimated HP"]
    G --> M["Match(troop, bar) list"]
```

## Feature Pipeline

```mermaid
flowchart TD
    A["GameState"] --> B["board_rasterizer.rasterize_units()"]
    B --> C["ally/enemy presence, threat, HP channels"]
    A --> D["global_features.build_global_vector()"]
    D --> E["time, elixir, towers, seen cards, hand state"]
    A --> F["action_masks"]
    F --> G["legal deploy cells from card metadata and action_space masks"]
    C --> H["model-ready board features"]
    E --> I["model-ready global features"]
    G --> J["model-ready legal-action mask"]
```

## Card Classifier Dataset Scripts

```mermaid
flowchart TD
    A["raw or selected gameplay frames"] --> B["import_card_classifier_frames.py"]
    B --> C["extract_hand_state() labels hand and next cards"]
    C --> D["crop hand-slot art and next-card ROI"]
    D --> E["append metadata/labels_normalized.csv"]

    E --> F["propose_card_classifier_split.py"]
    F --> G["metadata/split.csv"]
    G --> H["build_card_classifier_imagefolder.py"]
    H --> I["ImageFolder layout: hand/train, hand/val, next/train, next/val"]
    I --> J["train_card_classifier.py"]
    J --> K["trained card classifier weights"]

    E --> L["prepare_card_classifier_dataset.py"]
    L --> M["older labels.csv based crop dataset path"]
```

## Seed Detector Dataset And Training Scripts

```mermaid
flowchart TD
    A["local JPG clip frames"] --> B["prepare_seed_dataset.py"]
    B --> C["copy frames into data/seed_dataset/images/part2/<clip>/<round>"]
    C --> D["manual Labelme annotation"]
    D --> E["clean_preannotations.py optional scrub"]
    E --> F["build_seed_annotations.py"]
    F --> G["KataCR LabelBuilder converts labels"]
    G --> H["setup_seed_detectors.py"]
    H --> I["seed detector configs/checkpoints"]
    I --> J["train_seed_baseline.py"]
    J --> K["runs/detector*_baseline_seed*/weights"]
    K --> L["run_seed_inference.py"]
    L --> M["rendered inference video/images and optional Labelme preannotations"]
```

## Enemy Audio Mining Scripts

```mermaid
flowchart TD
    A["scripts/mine_enemy_audio_dataset.py"] --> B["resolve_video_manifest()<br/>default source: @yersoncz6334/videos<br/>filter upload date before cutoff"]
    B --> C["download_video()<br/>yt-dlp with browser-cookie and format fallbacks"]
    C --> D["ensure_opencv_compatible_video()<br/>transcode analysis copy when needed"]
    D --> E["extract_mono_wav_from_video()<br/>16 kHz mono gameplay WAV"]
    E --> F["analyze_video_for_mining()<br/>default target_analysis_fps=10"]
    F --> G["write states/<video>.jsonl"]
    F --> H["write spans/<video>.json"]
    F --> I["export_enemy_candidate_rows()<br/>phase metadata + quality tier"]
    I --> J["write candidates/<video>.jsonl"]
    J --> K["split_manifest_rows()<br/>train/val/test by video/span"]
    K --> L["extract_candidate_windows()<br/>1.0s windows, start_offset_s=-0.3"]
    L --> M["write manifests/<split>.jsonl"]
    M --> N["compute_coverage()<br/>coverage.json + spell_coverage.json"]
    N --> O["prune_raw_video_if_derived_complete()"]

    P["scripts/mine_enemy_audio_audit.py"] --> Q["sample mined train manifest"]
    Q --> R["group spell/gold/silver/bronze rows<br/>into audit/sample.json"]
```

## Audio Classifier Training And Evaluation

```mermaid
flowchart TD
    A["scripts/train_audio_classifier.py"] --> B{"mode"}
    B -->|"synthetic"| C["build_synthetic_loaders()<br/>MixedSFXCardDataset + optional background mix"]
    B -->|"real"| D["build_real_loaders()<br/>ManifestAudioDataset from mined manifests"]
    B -->|"hybrid"| E["synthetic pretrain<br/>then real fine-tune"]
    C --> F["train_model()"]
    D --> F
    E --> F
    F --> G["save audio_card_classifier checkpoint"]
    G --> H["evaluate_manifest()<br/>real mined test manifest"]
    G --> I["evaluate_real_ground_truth()<br/>existing GT json + aligned wav/video"]
    I --> J["summarize_predictions()<br/>accuracy, confusion, phase breakdowns"]
```

## Data Generation Scripts

```mermaid
flowchart TD
    A["YouTube URL or channel"] --> B["dataset_generation/scripts/download_videos.py"]
    B --> C["yt-dlp downloads video"]
    C --> D["OpenCV decode or ffmpeg conversion"]
    D --> E["sample/process frames"]
    E --> F["process_frame() + build_game_state()"]
    F --> G["build_global_vector()"]
    G --> H["dataset_generation outputs"]

    I["existing video clip"] --> J["dataset_generation/scripts/detect_in_game.py"]
    J --> K["extract timer, elixir, hand, tower signals"]
    K --> L["score in-game likelihood"]
    L --> M["identify usable gameplay segments"]

    I --> N["dataset_generation/scripts/process_frame.py"]
    N --> O["sparse frame loop"]
    O --> P["process_frame()"]
    P --> Q["MatchClockFilter + TowerHPFilter + EnemyCardTracker"]
    Q --> R["build_game_state()"]
    R --> S["write frames and states.jsonl"]
```

## Utility Scripts

```mermaid
flowchart TD
    A["fetch_royaleapi_data.py"] --> B["download RoyaleAPI JSON into local cache"]
    C["generate_cvat_labels.py"] --> D["read card template asset names"]
    D --> E["write CVAT label JSON"]
    F["download_ryleycr1_archive.py"] --> G["yt-dlp channel/video download"]
    G --> H["extract frame folders and manifest"]
```

## Debug Scripts

```mermaid
flowchart TD
    A["debug_own_action_clock_cells.py"] --> B["run video through process_frame() and trackers"]
    B --> C["render action-cell overlays for target own actions"]

    D["debug_skeleton_clock_cell.py"] --> E["seek fixed frames"]
    E --> F["draw ACTION_GRID, skeleton boxes, clock boxes, clock-cell scores"]

    G["debug_spell_purple_detector.py"] --> H["seek target spell frames"]
    H --> I["SpellDeployLocator masks/candidates/purple release diagnostics"]

    J["debug_actions_output_selected_actions.py"] --> K["seek selected action frames"]
    K --> L["draw grid and selected action cells"]

    M["render_predicted_cells.py"] --> N["parse txt predictions with parse_predictions_txt()"]
    N --> O["seek source video frames"]
    O --> P["draw ACTION_GRID and predicted cells"]
    P --> Q["write overlay JPGs, index.html, TSV and json summary"]
```

## Cross-Cutting Data Objects

```mermaid
flowchart LR
    A["raw frame"] --> B["process_frame result dict"]
    B --> C["GameState"]
    C --> D["OwnActionTracker.actions"]
    C --> E["EnemyCardTracker.detected_card_plays"]
    C --> F["feature builders"]
    D --> E
    E --> C
    B --> G["debug render output"]
    B --> H["dataset_generation states.jsonl"]
```
