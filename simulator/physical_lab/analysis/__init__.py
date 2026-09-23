"""Comparison, stored-case evaluation, and simulator-fidelity bridges."""

from .comparison import ComparisonReport, NumericMetric, compare_observation_to_replay
from .evaluation import EVALUATION_SCHEMA_VERSION, evaluate_stored_cases, write_stored_evaluation
from .fidelity_bridge import build_fidelity_corpus_payload, write_fidelity_corpus_payload

__all__ = [
    "ComparisonReport",
    "EVALUATION_SCHEMA_VERSION",
    "NumericMetric",
    "build_fidelity_corpus_payload",
    "compare_observation_to_replay",
    "evaluate_stored_cases",
    "write_fidelity_corpus_payload",
    "write_stored_evaluation",
]
