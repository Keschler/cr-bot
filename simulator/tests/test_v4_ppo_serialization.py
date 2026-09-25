"""Keep saved PPO artifacts readable after moving their dataclass definitions."""
import pickle

import pytest

pytest.importorskip("torch")

from simulator.rl.v4.v4_ppo import (
    V4OpponentLane,
    V4PPOConfig,
    V4RolloutBatch,
    V4UpdateReport,
)


@pytest.mark.parametrize("record", [V4OpponentLane, V4PPOConfig, V4RolloutBatch, V4UpdateReport])
def test_pre_refactor_pickle_class_reference_resolves(record):
    # Protocol 0 GLOBAL records encode the original module, even though newly
    # written artifacts identify the focused config/records module instead.
    original_reference = f"csimulator.rl.v4.v4_ppo\n{record.__name__}\n.".encode()
    assert pickle.loads(original_reference) is record
