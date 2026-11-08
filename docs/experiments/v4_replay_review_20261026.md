# October 26 replay review

> Historical record (2026-10-26). Results, checkpoint identities, verdicts and
> proposed follow-ups below apply to this experiment and its sealed runtime.
> See [project status](../../STATUS.md) for the current goal, selected actor and
> latest decision.

The October 20 recordings confirm missed X-Bow defense windows, some poor
Ice Golem spending, and Cannon placements that fail to answer Princess.
Balloon responses need a distinction: the actor usually lacks the elixir
or hand to play Musketeer/Fireball when it chooses a cheap card. Earlier
resource decisions need correction too.

This review concerns selected targeted control, SHA
`123f3ab04c10c19157e1d37f0f05b4c669bdb5d7478c654fe6d942d5319d01bd`.
All 120 recording hashes were verified. Simulator source and policy bytes
remain unchanged; no training, promotion or new match evaluation occurred.
The viewer gained result filtering and sorting.

## Confirmed examples

| Observation | Exact replay and match time | Finding |
|---|---|---|
| X-Bow connects despite available Cannon | `xbow-30__adaptive__2026101923__loss`, 95.25–98.00s | Cannon is in hand, affordable at 3.017 elixir, with legal placement support. WAIT probability is 99.985%. X-Bow locks at 97.50s; the actor plays Hog at 98.00s. |
| Low-elixir Ice Golem in the wrong lane | Same replay, 152.50s | Spends two of 2.071 elixir on a left Golem while a right X-Bow is already damaging the tower. Delays recovery to a four-elixir answer by approximately 2.8s at that elixir rate. |
| Cannon makes no direct combat contribution | `valk-log-bait__split-pressure__2026101923__loss`, 73.00s | Cannon at `(12,19)` never attacks, receives no damage or enemy attacks, and expires at 102.95s while Princess stays on a tower. The reconstructed cycle teacher chooses Log. |
| Cheap card with an unaffordable air answer | `lavaloon-inferno-dragon__fixed__2026101922__loss`, 164.25s | Musketeer and Fireball are in hand, but elixir is only 1.125. Skeletons is the only affordable card. The preceding spending sequence needs review; commanding Musketeer at this moment is impossible. |

Coordinates are `(column,row)`. Search the distinctive deck/style/seed portion
in the Evaluation library. These are decision-level findings, not proof that
a different single action would win each match.

## X-Bow timing and elixir reservation

Of 53 enemy X-Bow deployments that damage towers in the 24 X-Bow games,
43 have an affordable Cannon or Golem in hand before the first in-range
tower lock, and 42 have at least one second of opportunity. These are review
opportunities, not 43 proven preventable locks: placement legality and competing
threats still matter. A stronger example, `xbow-30__split-pressure__2026101923__loss`
at 227.75–230.75s, waits with 6.821–8.964 elixir and every answer in hand.

The teacher recognizes X-Bow but often prioritizes Hog ahead of
Musketeer/Cannon, and lacks explicit pre-lock/post-lock planning. The actor
also fails to reproduce useful teacher choices on its own histories.
In this simulator, a new unit does not automatically pull X-Bow off an
existing valid tower lock. Preventing connection and answering an existing
connection require different decisions.

Low elixir alone does not make Golem wrong. At 14.50s in the adaptive example,
a Golem with 2.178 elixir tanks Archers and Electro Spirit. Many low-elixir
Golems have observable defensive roles. A blanket ban below 3.5 elixir would
remove valid defenses. The problem is spending without enough immediate
value to justify delaying the needed answer. The special air-cycle rescue
preserves four elixir; other emergency ground/siege fallbacks do not share
that general reserve rule.

[Full X-Bow and cycle audit](../../outputs/v4/replay_review_20261026/xbow_cycle/REPORT.md)
includes definitions, all counts, source references and exact snapshots.

## Cannon range and Bait coverage

Cannon's range is 5.5 tiles and Princess's is 9 in the pinned simulator,
measured between collision edges. Cannon cannot walk forward to close a gap.
Across 171 Cannons in 24 Bait games, 49 are directly outranged by a Princess
that damages them, and 13 of those deal no damage to anything.

An out-of-range Cannon can still buy time or defend another threat.
`bait-knight-dart__adaptive__2026101923__loss` at 83.25s places Cannon left
at `(2,19)`: it never fires, but absorbs four Princess shots before dying
10.85s later. Avoided tower damage needs a counterfactual; that tanking is
not proof of optimality. Other outranged Cannons genuinely defend Knights
or Goblins. The 73.00s example above has no observed direct combat contribution,
although zoning/cycle value is not excluded.

Ground-only Princess/Knights states do not activate this tactical fit's
corrective teaching loss, which targets air and X-Bow. They retain the old
actor's distribution on recorded histories without new Princess corrections.
The current teacher also lacks an explicit persistent Princess reachability
decision. The actor sometimes chooses a different card or lane from the
reconstructed public teacher, so weak coverage and policy approximation
must be treated separately.

[Full Bait Cannon audit](../../outputs/v4/replay_review_20261026/bait_cannon/report.md)
contains lifetime attack/damage events and teacher-reconstruction limits.

## Balloon: affordable versus merely in hand

Among 47 Skeletons plays with a deployed Balloon at or beyond y=13 in
24 Lavaloon games, **none** had Musketeer or Fireball both in hand and
affordable. In 37, one was in hand but four elixir was unavailable. Some
other cases cycle toward a missing card. These air-match counts do not
identify an unnamed Bait/X-Bow replay.

Ice Spirit can freeze Balloon in this simulator. The one cheap-card example
with affordable Fireball, `lavaloon-skdragons__fixed__2026101921__loss` at
9.75s, actually freezes Balloon at 10.85s. That stall is not automatically
a mistake.

There is a concrete teaching concern: 55 saved active Balloon-state training
labels choose Skeletons; 19 are below four elixir and all 19 include a ground
threat. When an air answer is unaffordable, the ground-defense fallback can
spend without preserving the future air answer's reserve. Twelve saved public
labels were exactly reproduced with the pinned teacher. This supports a
multi-threat resource repair without proving training-label causality for
every learned mistake.

[Full Balloon and training review](../../outputs/v4/replay_review_20261026/balloon/REVIEW.md)
includes thresholds, earlier card spending and legitimate cycling examples.

## Repair proposed at the time

Prioritize earlier siege interception, deadline-aware elixir reservation,
range-aware Cannon placement and explicit ranged-ground Bait coverage.
Correct full histories from the actor itself, including decisions before
resource starvation, while preserving useful tanks and stalls. Then compare
the candidate with the unchanged control on fresh complete games across all
families. Cleaner labels alone are insufficient evidence of stronger play.
Privileged target identities in this audit remain diagnosis-only; they must
not silently become actor inputs.

## Sorting the evaluations

Restart an already-running viewer server after this update and reload the
browser. Use **Evaluation → Result → Losses/Wins** to filter, or **Order →
Losses first/Wins first** to group results. Combine these with deck search.
The original screen contains **38 wins and 82 losses**; X-Bow has 13 losses.
Known results also appear on match cards. Before restarting, searching
`__loss__` or `__win__` works with the existing export filenames.

Results require completed terminal recordings, relative to player zero.
Partial, capped, stopped and unproven outcomes remain unknown. Filtering and
sorting run before the display limit. Focused viewer checks: **52 passed,
1 skipped**. Optional HTTP tests need absent `httpx`; actual recording counts
and combined filters were checked directly. JavaScript syntax passed Node.
A browser surface was unavailable for visual checking. The full default
suite finished with **1,976 passed, 45 skipped and the same eight existing
failures**: five sandbox forkserver permission errors, annotation matching,
missing spell media and KataCR tracker import. Its log is
[here](../../outputs/v4/replay_review_20261026/default_tests.log).
