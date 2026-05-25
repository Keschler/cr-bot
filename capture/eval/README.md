# Action Detection Evaluation

This evaluates detected own/enemy actions against hand-labeled ground truth.

Ground truth is JSON:

```json
{
  "video": "2hog_cycle_champion.mp4",
  "events": [
    {
      "side": "own",
      "card": "ice-golem",
      "video_time_s": 133.5,
      "time_left_s": 159.0,
      "cell": [7, 19]
    },
    {
      "side": "own",
      "card": "fireball",
      "frame_index": 1572
    },
    {
      "side": "enemy",
      "card": "dart-goblin",
      "video_time_s": 42.6,
      "time_left_s": 252.0
    }
  ]
}
```

If labels are based on frame numbers, add a top-level `fps` value and use
`frame_index`; the evaluator converts it to `video_time_s`. Omit `cell` when
the placement cell is not confirmed.

Run it from the repository root against the txt output from `capture/main.py`:

```bash
python3 -m capture.eval.action_eval \
  --ground-truth capture/eval/ground_truth/2hog_cycle_champion.json \
  --predictions capture/video_output/2hog_cycle_champion.txt
```

If your shell is currently inside the `capture/` directory, either `cd ..` first
or run the file directly with paths relative to `capture/`:

```bash
python3 eval/action_eval.py \
  --ground-truth eval/ground_truth/2hog_cycle_champion.json \
  --predictions video_output/2hog_cycle_champion.txt
```

The report includes:

- precision, recall, and F1 for own and enemy actions
- misses and false positives
- `time_left_error_s`: detected action `time_left_s` minus labeled `time_left_s`
- `added_video_time_error_s`: first txt frame where the script added the action minus labeled `video_time_s`
- own-action cell distance when both expected and predicted cells are available

The txt output repeats cumulative action lists every frame. The evaluator deduplicates actions and keeps the first frame where each action appears as the script's added video time.
Predicted txt actions with impossible `time_left` values above `300` are ignored; this keeps pasted frame-index notes from being counted as script detections.

## Cell Visualization

Use `visualize_cells.py` to render the action grid on the labeled frames. It
highlights the ground-truth cell when present and the matched predicted cell
when `--predictions` is provided. It also writes `cell_suggestions.tsv`, which
is useful when filling cells back into the ground-truth JSON.

From the repository root with extracted frames:

```bash
capture/venv/bin/python3 capture/eval/visualize_cells.py \
  --ground-truth capture/eval/ground_truth/2hog_cycle_champion.json \
  --predictions capture/video_output/2hog_cycle_champion.txt \
  --frames-dir dataset_generation/data/video_clips/2frames \
  --output-dir capture/eval/cell_visualizations/2hog_cycle_champion \
  --side own
```

The default frame filename pattern is `{frame_index:06d}.jpg`; if your frames
look like `frame_000029.png`, the script finds them through a numeric fallback.
You can also provide an explicit pattern:

```bash
--frame-pattern 'frame_{frame_index:06d}.png'
```

To render from a video instead of extracted frames:

```bash
capture/venv/bin/python3 capture/eval/visualize_cells.py \
  --ground-truth capture/eval/ground_truth/2hog_cycle_champion.json \
  --predictions capture/video_output/2hog_cycle_champion.txt \
  --video dataset_generation/data/video_clips/10_fps_2.6HogCycle.mp4 \
  --output-dir capture/eval/cell_visualizations/2hog_cycle_champion \
  --side own
```

Open the generated `index.html` in the output directory to scan all overlays.
Use `--no-cell-labels` when you only want the grid and highlighted cells.
