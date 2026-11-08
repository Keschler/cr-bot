"""Replica provenance must not count one outcome more than once."""

import pytest

from simulator.rl.v4.search_distillation.targets import paired_advantage


def _verdict() -> dict:
    return {
        "discovery_replica_indices": [0, 1],
        "confirmation_replica_indices": [2, 3],
        "branches": [
            {
                "source": "sim",
                "mode": 0,
                "discovery_values": [0.0, 0.0],
                "confirmation_values": [0.0, 0.0],
            },
            {
                "source": "search",
                "mode": 1,
                "slot": 0,
                "row": 1,
                "col": 2,
                "discovery_values": [1.0, 1.0],
                "confirmation_values": [1.0, 1.0],
            },
        ],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda verdict: verdict.update(confirmation_replica_indices=[2, 2]),
        lambda verdict: verdict.update(discovery_replica_indices=[0, 0]),
        lambda verdict: verdict["branches"][1].update(
            confirmation_replica_indices=[2, 2]
        ),
    ],
    ids=("duplicate-confirmation", "duplicate-discovery", "duplicate-branch-confirmation"),
)
def test_duplicate_replica_indices_cannot_certify_independent_correction(mutate) -> None:
    verdict = _verdict()
    mutate(verdict)

    with pytest.raises(ValueError, match="lacks independent validation replicas"):
        paired_advantage(verdict, require_independent=True)
