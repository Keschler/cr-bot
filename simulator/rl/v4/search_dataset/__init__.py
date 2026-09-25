"""Outcome-labeled search datasets.

Public entry points remain here; focused modules own collection, target
adoption, sequence batching and worker execution.
"""

from .config import (
    SEARCH_GENERATOR_VERSION,
    ADOPT_MARGIN_ACTION,
    ADOPT_MARGIN_PLACE,
    OPPONENT_STRATEGY,
    _SPELL_COL,
    _stable_seed,
    SearchDatasetConfig,
    _hash_order,
)
from .batching import (
    group_search_sequences,
    iter_search_sequence_batches,
    to_torch_sequence_batch,
)
from .inference import (
    load_champion_actor,
    _record_to_inference_sample,
    _predict_records,
)
from .records import (
    _hand_keys,
    _enemy_count,
    _spell_legal,
    _base_label,
    _new_sim_env,
    _record_from_env,
    _disagrees,
    _spell_opp,
    _ambiguous,
)
from .targets import (
    _search_live,
    _adopt_target,
    _to_search_sample,
)
from .active import (
    _sync_public_memory,
    _active_record_memory,
    _generate_active_records,
    _generate_active_search_dataset,
)
from .generation import (
    generate_search_dataset,
)
from .selection import (
    _search_stats,
    _assert_search_balance,
)
from .workers import (
    TimingSearchTask,
    SimSearchTask,
    _worker_init,
    _run_timing_task,
    _run_sim_task,
    run_search_tasks,
)

__all__ = [
    "ADOPT_MARGIN_ACTION",
    "ADOPT_MARGIN_PLACE",
    "SEARCH_GENERATOR_VERSION",
    "SearchDatasetConfig",
    "SimSearchTask",
    "TimingSearchTask",
    "generate_search_dataset",
    "group_search_sequences",
    "iter_search_sequence_batches",
    "load_champion_actor",
    "run_search_tasks",
    "to_torch_sequence_batch",
]
