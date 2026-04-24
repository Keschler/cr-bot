# Expert Offline RL Dataset Pipeline

This document describes the pipeline for turning expert Clash Royale gameplay videos into an offline dataset for behavior cloning or offline reinforcement learning.

The target dataset should not contain only moments where the expert plays a card. It must also contain decision points where the expert waits, because waiting is often the correct action.

## Goal

Each final training row should represent one decision point:

```text
observation_t, action_t, reward_t, observation_t+1, done
```

Where `action_t` is either:

```text
WAIT
```

or:

```text
PLAY(card, deploy_cell)
```

For behavior cloning, `reward_t`, `observation_t+1`, and `done` can be stored but ignored at first.

## Pipeline Overview

```text
video links
  -> downloaded videos
  -> frame extraction
  -> per-frame state extraction
  -> expert play event detection
  -> fixed-rate decision sampling
  -> dataset rows
  -> validation/debug viewer
  -> training
```

## 1. Download And Normalize Videos

For every expert gameplay video, store metadata:

- `video_id`
- source URL
- deck used, for example `2.6 hog cycle`
- player side if known
- match start/end timestamps
- video FPS
- resolution
- any crop or resize parameters

Normalize videos into a consistent format if possible:

- fixed FPS for analysis, for example `30 FPS`
- consistent resolution or known screen transform
- one match per clip when practical

## 2. Extract Frames

Extract frames at a rate high enough to detect actions reliably.

Recommended:

- use `10-30 FPS` for action/event detection
- use lower fixed decision ticks later, for example `0.25s` or `0.5s`

Store frame references, not necessarily every image in the final dataset:

```json
{
  "video_id": "abc123",
  "frame_idx": 12345,
  "video_time_s": 411.5,
  "image_path": "frames/abc123/0012345.jpg"
}
```

## 3. Per-Frame State Extraction

For each analyzed frame, extract the current visible game state.

Required observation fields:

- own hand cards: `card_1`, `card_2`, `card_3`, `card_4`
- next card
- own elixir, preferably fractional
- match timer
- overtime / double elixir / triple elixir state if available
- own tower HP: left princess, king, right princess
- enemy tower HP: left princess, king, right princess
- own/enemy princess tower alive flags
- own/enemy king tower active flags
- visible own units
- visible enemy units
- unit class, team, position, confidence, and HP estimate
- known enemy cards seen so far
- estimated enemy elixir

The current codebase already has most of this structure:

- `main.py` creates the raw frame result
- `state_builder.py` builds `GameState`
- `features/global_features.py` builds the global vector
- `features/board_rasterizer.py` builds the board tensor
- `features/action_masks.py` builds legal deploy masks

## 4. Detect Expert Play Events

A play event is the exact moment where the expert plays a card.

Each event should contain:

```json
{
  "video_id": "abc123",
  "frame_idx": 12345,
  "video_time_s": 411.5,
  "match_time_s": 128.5,
  "card": "hog-rider",
  "card_slot": 0,
  "deploy_cell": [14, 23],
  "deploy_pixel": [812, 1740],
  "confidence": 0.92
}
```

Useful signals for detecting play events:

- a card disappears from the hand
- hand order shifts after a play
- own elixir drops by the card cost
- a new own troop/building appears on the board
- a spell effect appears
- tower/unit HP changes shortly after a spell

Best practical method:

1. Track hand cards over time.
2. Detect elixir drops.
3. Detect new own units/buildings/spells.
4. Match the missing hand card to the new board object or spell effect.
5. Estimate the deploy location from first appearance or effect location.

Some cards are harder than others:

- Hog Rider, Musketeer, Cannon, Ice Golem, Ice Spirit, Skeletons are usually visible as units/buildings.
- Fireball and The Log need spell-effect detection or inference from damaged units/towers.

## 5. Build Fixed-Rate Decision Samples

Do not train only on play events.

Sample the game at a fixed decision interval:

```text
decision_tick = 0.25s or 0.5s
```

For each tick:

- find the observation at or just before the tick
- check whether a play event happened inside that tick window
- if yes, label the row as `PLAY(card, deploy_cell)`
- if no, label the row as `WAIT`

Example:

```text
tick 128.50s -> WAIT
tick 128.75s -> WAIT
tick 129.00s -> PLAY(skeletons, [8, 24])
tick 129.25s -> WAIT
```

This teaches both:

- what the expert plays
- when the expert chooses not to play

Avoid storing every raw video frame as a training decision, because that creates too many `WAIT` labels and can make the model overly passive.

## 6. Handle Forced Waits

Some waits are not meaningful expert choices.

Example:

- own elixir is too low for every card in hand
- no legal deploy location exists for the selected card type
- the game is in a transition/menu frame

Keep these rows, but mark them:

```json
{
  "action": {"type": "WAIT"},
  "forced_wait": true
}
```

During training, forced waits can be downweighted or filtered.

## 7. Final Dataset Row Shape

A final row should include both human-readable debug fields and model-ready tensors.

Example:

```json
{
  "sample_id": "abc123_000512",
  "video_id": "abc123",
  "frame_idx": 15360,
  "video_time_s": 512.0,
  "match_time_s": 128.0,
  "decision_tick_s": 0.25,
  "observation": {
    "board_tensor_path": "features/abc123/000512_board.npy",
    "global_vector_path": "features/abc123/000512_global.npy",
    "hand": ["hog-rider", "ice-spirit", "skeletons", "fireball"],
    "next_card": "ice-golem",
    "elixir_self": 8.6,
    "elixir_enemy_est": 5.2,
    "tower_hp_self": [2534, 7032, 4848],
    "tower_hp_enemy": [3100, 7032, 2100]
  },
  "action": {
    "type": "PLAY",
    "card": "skeletons",
    "card_slot": 2,
    "deploy_cell": [8, 24]
  },
  "action_mask_path": "features/abc123/000512_action_mask.npy",
  "reward": 0.0,
  "next_sample_id": "abc123_000513",
  "done": false,
  "forced_wait": false,
  "confidence": {
    "state": 0.91,
    "action": 0.88
  }
}
```

For `WAIT`:

```json
{
  "action": {
    "type": "WAIT",
    "card": null,
    "card_slot": null,
    "deploy_cell": null
  }
}
```

## 8. Rewards

For behavior cloning, rewards are optional.

For offline RL, store enough information to compute rewards later:

- tower HP delta
- enemy tower damage
- own tower damage taken
- tower destroyed events
- crown result
- win/loss
- estimated elixir trades
- visible units killed or surviving after a window

Start with a simple reward:

```text
reward =
  enemy_tower_damage
  - own_tower_damage_taken
  + tower_destroy_bonus
  - own_tower_destroyed_penalty
  + win_bonus
```

Keep reward computation versioned, because reward design will change.

## 9. Validation Viewer

Before training, build a debug viewer for the extracted dataset.

For each decision sample, it should show:

- original frame
- detected hand
- elixir
- tower HP
- detected units and health bars
- board grid
- chosen label: `WAIT` or `PLAY(card, cell)`
- deploy cell overlay for play actions
- confidence scores

This is essential because small action-label errors will hurt training more than imperfect reward design.

## 10. Recommended First Milestone

First build a behavior cloning dataset:

```text
observation_t -> expert_action_t
```

Use:

- fixed decision ticks
- `WAIT` labels
- `PLAY(card, cell)` labels
- action masks
- no reward optimization yet

After action labels are reliable, extend the dataset with:

- `observation_t+1`
- rewards
- terminal outcomes
- enemy elixir/cycle uncertainty

## 11. Common Failure Cases

Watch for these during extraction:

- hand classifier flickers between two cards
- elixir digit is correct but fractional estimate is noisy
- spell actions are missed
- deploy location is taken from unit position after it has already moved
- multiple cheap cards are played within one decision window
- opponent units are mistaken for own units
- tower HP OCR briefly fails
- video crop changes after transitions
- actions during replay overlays or menus are included by mistake

Rows with low confidence should be kept separate so they can be reviewed, filtered, or downweighted.
