# Full 40-deck evaluation — 2026-10-04

> Historical record (2026-10-04). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The subsequent [initializer recovery and bounded PPO evaluation](v4_recovery_results.md)
is complete, with a fresh paired comparison under the repaired runtime.
The results below preserve the earlier benchmark and its original seeds.

All five frozen historical pilot policies were evaluated against every deck
in the current catalog, with ten matched fresh seeds per deck. Each policy
played 300 games against the 30 training decks and 100 games against the ten
reserved decks: 2,000 new games overall. There was no new training or
checkpoint selection.

| Policy | Training wins / 300 | Reserved wins / 100 | Total wins / 400 |
| --- | ---: | ---: | ---: |
| Random initialization | 0 | 0 | 0 |
| Random initialization + PPO | 0 | 0 | 0 |
| Rules-teacher bootstrap | 0 | 0 | 0 |
| Bootstrap + PPO | 1 | 0 | 1 |
| Bootstrap + search distillation | 0 | 0 | 0 |

The single win was bootstrap + PPO against `xbow-icewiz-rocket` with seed
`2026100401`. Its overall win rate was 0.25%. The paired training-score gain
over bootstrap was 0.33 percentage points with clustered standard error
0.33 points. This one win does not establish a superior training method.
All five policies lost all reserved-deck games.

The broader evaluation shows that these minimally trained policies are not
competitive against the current scripted opponents. It does not establish
how PPO or distillation would perform after substantially more training.
The strategic goal remains a very strong Hog 2.6 policy against all enemy
decks.

## Execution and integrity

The run completed in 44.9 minutes with three isolated CPU workers, one Torch
thread per worker, and roughly 2.24 CPU-hours. Every actor received the same
deck/controller/opening/sampling conditions. The learner was player 0 with
the fixed Hog 2.6 deck, seeded stochastic actions, 250 ms decisions, and a
1,440-decision cap.

All 2,000 matches terminated: 1 win, 1,999 losses, zero draws, and zero
truncations. Both sides had zero invalid actions. Independent coverage
review found no duplicate or missing cells and verified all ten actor pairs
for each reporting split. Runtime source, execution helper, catalog, and
checkpoint hashes stayed unchanged throughout the run.

This run used the current audited controller and sampling fixes documented
in [the evaluation audit](v4_evaluation_audit.md). Historical five-deck rows
were not reused; old and new win rates should not be interpreted as a
training regression because the evaluation conditions changed. Opponents
are scripted controllers using published base-card deck lists, rather than
real-player decision sequences.

Local generated artifacts are under `outputs/v4/catalog_comparison_20261004/`:

- `results.md` and `summary.json`: concise outcomes, paired uncertainty, and cost;
- `per_deck.csv` and `win_rates.png`: results for each deck;
- `results/comparison.json` and ten raw JSONL files: complete records and provenance;
- `manifest.json`, `command.json`, and `run_parallel.py`: frozen inputs and execution;
- `results/timings.jsonl`: individual game timings;
- `catalog_evaluation_cost.json`: measured wall time, CPU time, and memory;
- `verification.json`: focused, process-parity, and default test results.

No runtime source was changed for this execution; the process orchestration
helper lives under `outputs/`. It reproduced complete serial records in a
four-game process parity probe. Focused checks passed 48 tests. The default
suite produced 1,313 passed, 14 skipped, and the same three existing failures
(annotation matching, missing spell-video fixture, and KataCR tracker import).
