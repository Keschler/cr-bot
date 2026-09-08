"""V4 Stage-0 evaluation harness: sealed champion-challenger protocol."""

from .benchmark_manifest import (
    EvalCell,
    SealedManifest,
    is_retired,
    manifest_cells,
    retire_manifest,
    sample_sealed_manifest,
)
from .champion_registry import ChampionEntry, ChampionRegistry
from .contracts import (
    ACTION_SEMANTICS_VERSION,
    ExperimentContract,
    contract_manifest,
    seal_current_contracts,
)
from .full_match_suite import (
    MatchOutcome,
    crown_differential,
    macro_match_score,
    match_score,
    micro_match_score,
    side_swap_blocks,
)
from .paired_statistics import (
    holm_reject,
    paired_differences,
    required_sample_size,
    stratified_paired_bootstrap_lcb,
)
from .placement_metrics import (
    per_sample_placement_stats,
    placement_error_stats,
    soft_region_mass,
    top1_cells,
)
from .promotion_report import (
    GateResult,
    PromotionReport,
    PromotionThresholds,
    evaluate_promotion,
)
from .tactical_suite import (
    TACTICAL_FAMILIES,
    catastrophic_regression_rate,
    consequence_primary_score,
    consequence_vector,
    macro_tactical_score,
)

__all__ = [
    "ACTION_SEMANTICS_VERSION",
    "EvalCell",
    "SealedManifest",
    "ChampionEntry",
    "ChampionRegistry",
    "ExperimentContract",
    "GateResult",
    "MatchOutcome",
    "PromotionReport",
    "PromotionThresholds",
    "TACTICAL_FAMILIES",
    "catastrophic_regression_rate",
    "consequence_primary_score",
    "consequence_vector",
    "contract_manifest",
    "crown_differential",
    "evaluate_promotion",
    "holm_reject",
    "is_retired",
    "macro_match_score",
    "macro_tactical_score",
    "manifest_cells",
    "match_score",
    "micro_match_score",
    "paired_differences",
    "per_sample_placement_stats",
    "placement_error_stats",
    "required_sample_size",
    "retire_manifest",
    "sample_sealed_manifest",
    "seal_current_contracts",
    "side_swap_blocks",
    "soft_region_mass",
    "stratified_paired_bootstrap_lcb",
    "top1_cells",
]
