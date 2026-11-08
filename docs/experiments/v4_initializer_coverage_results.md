# Hog 2.6 catalog initializer results

> Historical record (2026-10-06). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The bounded experiment is complete with one scripted-opponent diagnostic warning. The strategic goal of a very strong policy against all enemy decks remains open.

The revised-teacher initializer won **92/400 (23.0%)**, compared with **13/400 (3.25%)** for the recovered baseline: **79 additional wins, +19.75 percentage points**. PPO was rejected on fresh training-deck confirmation (31 wins versus its initializer's 42 in 150 games each).

Checkpoint selected in this experiment: `/home/keschler/Dokumente/cr-bot/outputs/v4/initializer_coverage_20261006/fit_saving/actor_epoch_128.pt`

SHA256: `ba15ded00742ec68b509e89da8ad88bcc12b0e0c7bf2f76d8edd25e869903c3b`

Selection used training decks only and was frozen before final evaluation. Final catalog results did not change the checkpoint.

## Fresh training-deck initializer confirmation

| Actor | Wins | Losses | Games |
|---|---:|---:|---:|
| Recovered baseline | 3 | 147 | 150 |
| Catalog / current teacher | 11 | 139 | 150 |
| Catalog / revised teacher | 40 | 110 | 150 |

The two catalog fits share inputs, weights, initialization, optimizer and dose. Their 40-versus-11 win difference isolates the revised labels on this experiment; the comparison with the recovered baseline also changes data coverage and training dose.

## Final fixed 40-deck catalog

| Actor | Overall | Training 30 decks | Reserved 10 decks |
|---|---:|---:|---:|
| Catalog / revised teacher | 92/400 | 68/300 | 24/100 |
| Recovered baseline | 13/400 | 8/300 | 5/100 |

All matches finished, with no draws or truncation. All 800 native action traces match the diagnostic traces; learner invalid actions, play misses and rejections are zero.

## Opponent diagnostic and reporting recovery

The original strict analysis stopped on one opponent invalid action, so its all-players zero-error check remains **failed**. This was not a training failure. At 165 seconds in the recovered-baseline match against reserved `xbow-hunter`, seed `2026100662`, the scripted opponent attempted X-Bow at cell `[10,19]` and received `illegal_placement`. An exact diagnostic replay reproduced the entire native record and instrumentation, including all action, event and state hashes.

The affected match is a loss for both policies. Keeping all recorded matches gives 92 versus 13 wins in 400 games each. Excluding that paired cell from both policies gives **92/399 versus 13/399**, the same **79 net wins**, and a **+19.80 percentage-point** difference. Reserved results become 24/99 versus 5/99. The conclusion survives this sensitivity check, but this run is not an entirely error-free scripted-opponent benchmark.

The recovery only validates and reports saved evidence. The original ten sealed helpers, simulator source, actor bytes, 800 raw records, train-only selections and original failure logs are preserved. `analysis_recovery.json` records the before/after evidence hashes; `final_catalog/paired_analysis.json` contains the warning, full-grid statistics and paired exclusion sensitivity.

A separate GPT-6.1 Sol/xhigh audit verified all 800 records, paired identities and sealed hashes. Reporting-recovery regression checks: 15 passed, including rejection of learner errors, other opponent anomalies and mismatched diagnostic replays.

## Paired differences

| Scope | Pairs | Net wins | Difference | Seed-cluster 95% interval | Deck-cluster 95% interval |
|---|---:|---:|---:|---:|---:|
| All 40 | 400 | 79 | 19.75 pp | 17.00 to 22.50 pp | 13.25 to 26.75 pp |
| Training 30 | 300 | 60 | 20.00 pp | 17.33 to 22.67 pp | 13.33 to 27.33 pp |
| Reserved 10 | 100 | 19 | 19.00 pp | 11.00 to 26.00 pp | 4.00 to 37.00 pp |

These descriptive intervals measure sensitivity across this fixed catalog and ten seeds, without multiple-comparison adjustment. They do not establish strength against every deck or real players.

## PPO decision

The 16-update reward-only PPO run selected update 4 on its native development curve. Separate fresh training-deck confirmation gave the initializer 42/150 wins and PPO update 4 31/150; the declared rule retained the initializer. There is no measured PPO improvement from this budget.

## Interpretation and continuation

Broader collection and more fitting improved the current-teacher control, and revising teacher decisions improved it further. The final policy still loses 77% of these scripted-catalog games. It is a better starting point, not a strong finished policy. Future work should inspect remaining defensive and spending failures and fix the reproduced scripted X-Bow placement bug before a new benchmark; this report does not start another run.

The subsequent [remaining-loss diagnosis](v4_remaining_losses_diagnosis.md) identifies weak air/siege defense, a remote X-Bow teacher blind spot, imperfect card/placement generalization, and an initializer recurrent-training mismatch. The same-grid teacher comparison wins 99/300 versus the selected policy's 68/300; this supports improving both the teaching decisions and their learned execution.

![Experiment results](../../outputs/v4/initializer_coverage_20261006/initializer_results.png)

## Protocol and audit details

Goal: a very strong Hog 2.6 policy against all enemy decks. This bounded comparison is evidence toward that goal.

## Controls

The recovered initializer is frozen at `outputs/v4/recovery_20261004/ppo_curve/checkpoint_update_000000.pt` (SHA256 `4a54afbc84a5240fcb3e896b5cccf78e8a3760ff2aa32a95f34492849320ffd1`). Native simulator source is sealed at `f2d010233812d8f99f25d60683841f383de34c38d15346f191546cabfa7f7577`.

The training catalog has 30 decks and the reserved catalog has 10. Opponents use exact deck cards and native scripted controller identities. This evaluates the catalog under those controllers; it does not establish strength against real human players or every possible enemy deck.

Both new label arms use exactly the same public observations, inclusion weights, random starting tensors, minibatch order, architecture, native loss settings, optimizer, learning rate, batch size, and 128-epoch dose. The current-label arm isolates broader collection/dose from the revised labels. Comparisons with the recovered initializer combine more data, more optimizer steps, and optionally teacher changes.

Each new fit has 7,680 training observations, 1,920 development observations and 15,360 optimizer steps. The recovered fit had 1,152 observations and 2,304 steps. Retention preserves every observed driver/card/WAIT stratum, with uniform within-stratum sampling and at most64 states per episode. Inclusion weights reconstruct the decision population for mode loss; card, duration and placement objectives still use retained conditional label frequencies.

## Teacher audit

The existing teacher can label ground-only cards as air defense and Log for air clusters, and ignores existing own defenders. The opt-in revised teacher uses air-capable answers, ground/air spell classes, urgent defense before remote clusters, conservative healthy Cannon/Musketeer coverage for isolated ground threats, Hog pressure at four elixir when safe, and cheap cycling only at high elixir. Static coverage is a testable approximation; public observation rows lack target locks and usable dynamic combat flags. Historical default teacher behavior is preserved as control.

Across60 paired training-deck matches each, the current rules teacher won7 and the revised teacher won18. All120 matches completed, with zero invalid actions/misses and native action-trace parity. Revised teacher mean elixir was2.268 versus2.356; Hog plays/game4.983 versus2.483, Musketeer3.283 versus0.933, Fireball2.267 versus3.450. Winning more did not imply holding more elixir.

## Label coverage

Current labels on shared inputs: Hog65, Musketeer51, Fireball140, play-eligible WAIT2938. Revised labels: Hog356, Musketeer235, Fireball54, play-eligible WAIT3095. All30 training decks contribute256 retained training states. The predeclared one-round top-up triggered because the initial revised Fireball count28 was below50; it was decided from label coverage, not outcomes. No labels were forced.

Across the shared training inputs,550 current PLAY labels become revised WAIT. In268 states satisfying conservative complete ground coverage, current labels play104 times versus revised5. Air-threat states also become much less likely to spend, with explicit air-capable cards available when legal. Coexistence of air and ground units does not make every ground play erroneous.

## Selection and evaluation

Binding initializer selection uses fixed128-epoch checkpoints and the frozen recovered baseline, on all30 train decks × five fresh paired match seeds,150 games per actor. Highest completed-match score wins; ties retain baseline then current-label control. The10 reserved decks never enter collection or selection. Mode/card calibration from isolated one-step examples is diagnostic; it is not a substitute for full-match strength. By-card mode recall means choosing PLAY on that card's examples, not predicting that exact card.

PPO is conditional on at least3 net wins-equivalent on train confirmation, an improved declared spending/defense proxy, no archetype with at least3 net losses, and technical validity. If run, its native16-update budget and hyperparameters remain unchanged. A selected positive PPO update must beat its initializer on a separate fresh150-game train confirmation; ties keep the initializer. Final frozen actors use all40 catalog decks ×10 additional fresh seeds each. Final outcomes cannot change selection.

Separate seed/deck cluster-bootstrap intervals describe sensitivity for this fixed catalog. They have only5 confirmation or10 final seed clusters; no multiple-comparison adjustment is applied. A small or degenerate interval does not prove universality.

## Validation and provenance

Focused tests:49 passed. Default suite:1,348 passed,14 skipped, three previously observed unrelated failures (annotation matching, missing spell-evaluation media, KataCR tracker constructor). Core source was frozen before games. Draft analysis helper identity/order validation and fit timing/weighted diagnostic reporting were hardened before their first use, with companion revision hashes; training labels, losses, selection rule and hyperparameters were unchanged. Subagents use GPT-6.1 Sol with xhigh reasoning, as requested.

Complete: fitted-policy confirmation, conditional PPO decision, final catalog evaluation and frozen checkpoint recommendation, with the opponent diagnostic above.
