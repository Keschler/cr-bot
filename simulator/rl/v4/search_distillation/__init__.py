"""Search-policy improvement with conservative retention.

Public experiment APIs and cached record types remain available at this import
path so existing runners and serialized prefix caches continue to load.
"""

from .records import (
    ActionTarget,
    CachedPrefix,
    CacheConfig,
    FactorizedTarget,
    SEARCH_DISTILLATION_VERSION,
    DEFAULT_CONTEXT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MIN_LCB,
)
from .targets import (
    action_from_record,
    build_soft_search_target,
    factorized_distributions,
    hard_factorized_target,
    paired_advantage,
    validated_correction,
)
from .preparation import (
    load_frontier_entries,
    match_json_path,
    parse_entry_id,
    prepare_records,
    replay_selected_prefixes,
)
from .losses import (
    compute_batch_loss,
)
from .evaluation import (
    evaluate_records,
)
from .artifacts import (
    DistillationDeadline,
    file_sha256,
    git_revision,
)
from .training import (
    train_one,
)

__all__ = [
    "ActionTarget",
    "CachedPrefix",
    "CacheConfig",
    "FactorizedTarget",
    "SEARCH_DISTILLATION_VERSION",
    "DEFAULT_CONTEXT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MIN_LCB",
    "DistillationDeadline",
    "build_soft_search_target",
    "compute_batch_loss",
    "evaluate_records",
    "factorized_distributions",
    "file_sha256",
    "git_revision",
    "hard_factorized_target",
    "load_frontier_entries",
    "match_json_path",
    "paired_advantage",
    "parse_entry_id",
    "prepare_records",
    "replay_selected_prefixes",
    "train_one",
    "validated_correction",
    "action_from_record",
]
