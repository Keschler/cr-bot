# Flowcharts

This document maps the first-party runtime, extraction, tracking, dataset, and debug scripts. Diagrams use Mermaid so they render directly in GitHub Markdown.

Generated CVAT serverless functions, vendored code, and virtualenv files are intentionally excluded.

## Script Index

| Script | Diagram section |
| --- | --- |
| `capture/capture.py` | `capture/cli.py` and `capture/capture.py` |
| `capture/cli.py` | `capture/cli.py` and `capture/capture.py` |
| `capture/main.py` | `capture/main.py`: Top-Level Runtime, `capture/main.py`: `process_frame()` |
| `capture/scripts/build_card_classifier_imagefolder.py` | Card Classifier Dataset Scripts |
| `capture/scripts/build_seed_annotations.py` | Seed Detector Dataset And Training Scripts |
| `capture/scripts/clean_preannotations.py` | Seed Detector Dataset And Training Scripts |
| `capture/scripts/download_ryleycr1_archive.py` | Utility Scripts |
| `capture/scripts/fetch_royaleapi_data.py` | Utility Scripts |
| `capture/scripts/generate_cvat_labels.py` | Utility Scripts |
| `capture/scripts/import_card_classifier_frames.py` | Card Classifier Dataset Scripts |
| `capture/scripts/prepare_card_classifier_dataset.py` | Card Classifier Dataset Scripts |
| `capture/scripts/prepare_seed_dataset.py` | Seed Detector Dataset And Training Scripts |
| `capture/scripts/propose_card_classifier_split.py` | Card Classifier Dataset Scripts |
| `capture/scripts/run_seed_inference.py` | Seed Detector Dataset And Training Scripts |
| `capture/scripts/setup_seed_detectors.py` | Seed Detector Dataset And Training Scripts |
| `capture/scripts/train_card_classifier.py` | Card Classifier Dataset Scripts |
| `capture/scripts/train_seed_baseline.py` | Seed Detector Dataset And Training Scripts |
| `capture/scripts/debug/debug_actions_output_selected_actions.py` | Debug Scripts |
| `capture/scripts/debug/debug_own_action_clock_cells.py` | Debug Scripts |
| `capture/scripts/debug/debug_skeleton_clock_cell.py` | Debug Scripts |
| `capture/scripts/debug/debug_spell_purple_detector.py` | Debug Scripts |
| `dataset_generation/scripts/detect_in_game.py` | Data Generation Scripts |
| `dataset_generation/scripts/download_videos.py` | Data Generation Scripts |
| `dataset_generation/scripts/process_frame.py` | Data Generation Scripts |

## `capture/cli.py` And `capture/capture.py`

```mermaid
flowchart TD
    A["python capture/capture.py or python capture/cli.py"] --> B["parse CLI args"]
    B --> C{"mode"}
    C -->|"--debug or --debug-frame"| D["main(debug=True, debug_frame_path=...)"]
    C -->|"--video"| E["main(debug=False, video=..., frame_stride=...)"]
    C -->|"default"| F["main(debug=False): live video device"]
    B --> G["pass normalize and yolo_detections flags"]
    G --> D
    G --> E
    G --> F
```

## `capture/main.py`: Top-Level Runtime

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
    O --> P["loop frames with frame_stride"]
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

## `capture/main.py`: `process_frame()`

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

```mermaid
flowchart TD
    A["OwnActionTracker.update(game_state, arena_px, frame, clock_boxes)"] --> B["read hand, elixir, now"]
    B --> C["_detect_slot_drops()"]
    C --> D{"slot changed"}
    D -->|"prev -> None"| E["append PendingOwnPlay"]
    D -->|"prev log -> other card"| F["append Log PendingOwnPlay"]
    D -->|"other change"| G["ignore as OCR churn"]

    E --> H["_confirm_pending()"]
    F --> H
    G --> H

    H --> I["collect new ally YOLO tracks"]
    I --> J{"pending card kind"}
    J -->|"normal troop/building"| K["infer cell from deploy clock near matching track"]
    J -->|"spell"| L["SpellDeployLocator aim ellipse + release marker"]
    J -->|"log"| M["first visible ally the-log YOLO + single-use elixir drop"]

    K --> N{"confirmed?"}
    L --> N
    M --> N
    N -->|"yes"| O["_append_action(card, slot_idx, cell, time_left_s)"]
    N -->|"no but keep"| P["pending survives"]
    N -->|"no and stale/cancelled"| Q["pending removed"]
    O --> R["actions list"]
```

## Log-Specific Own Action Detection

```mermaid
flowchart TD
    A["hand slot has Log"] --> B{"slot disappears or changes"}
    B -->|"yes"| C["create pending log with slot_idx and started_at_s"]
    C --> D["watch next frames"]
    D --> E{"first ally YOLO class the-log visible?"}
    E -->|"no"| F["keep pending until timeout"]
    E -->|"yes"| G["choose one pending log within 1.0s window"]
    G --> H["cancel older Log OCR-jitter pendings"]
    H --> I{"current elixir drop >= 1.5 or selected pending already latched?"}
    I -->|"yes"| J["cell = first visible Log box center mapped through ACTION_GRID"]
    J --> K["append own action"]
    K --> L["consume YOLO track id"]
    I -->|"no"| M["keep selected pending, retain first Log cell"]
```

## Enemy Card Tracking

```mermaid
flowchart TD
    A["EnemyCardTracker.update(time_left, enemy_matches, clock_boxes, own_actions)"] --> B["regen enemy elixir estimate"]
    B --> C["remember recent enemy deploy clocks"]
    C --> D["for each enemy YOLO match with track_id"]
    D --> E["TrackMemory.add_observation()"]
    E --> F{"near enemy deploy clock?"}
    F -->|"yes"| G["clock_confirmed=True"]
    F -->|"no"| H{"frame-confirm class and enough frames/confidence?"}
    H -->|"yes"| I["frame_confirmed=True"]
    H -->|"no"| J["wait for more observations"]
    G --> K["_maybe_record_play()"]
    I --> K
    K --> L{"reliable enemy play?"}
    L -->|"no"| J
    L -->|"yes"| M["DIRECT_UNIT_TO_CARD maps YOLO class to card"]
    M --> N{"recent own spell duplicate?"}
    N -->|"yes"| O["mark track counted, do not record enemy play"]
    N -->|"no"| P{"recent duplicate enemy play?"}
    P -->|"yes"| O
    P -->|"no"| Q["append detected_card_plays, add seen card, subtract cost"]
```

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
    G --> H{"matches recent own Log within 3s and <= 3 cells?"}
    H -->|"yes"| I["veto as own spell duplicate"]
    H -->|"no"| J{"recent enemy Log duplicate within 0.75s?"}
    J -->|"yes"| K["suppress duplicate"]
    J -->|"no"| L["record enemy log play and subtract 2 elixir"]
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
