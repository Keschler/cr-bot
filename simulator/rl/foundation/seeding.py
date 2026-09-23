"""Stable seeds shared by simulator training utilities."""
import hashlib
import json


def deterministic_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 64-bit seed without process-randomized hash()."""
    if type(base_seed) is not int:
        raise ValueError("base_seed must be an integer")
    try:
        encoded = json.dumps([base_seed, *parts], sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("deterministic seed parts must be JSON-serializable") from error
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
