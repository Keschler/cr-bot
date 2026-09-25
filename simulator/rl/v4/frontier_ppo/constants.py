"""Protocol versions and fixed evaluation thresholds."""

FRONTIER_PPO_VERSION: str = "frontier-ppo-v4-2"
FRONTIER_PPO_EPISODE_CAP: int = 128

# Paired-uncertainty reporting for sealed choice-regret.
# Deltas are policy-minus-champion continuation values under common random
# numbers (matched replica seeds), so per-replica paired differences are the
# correct unit for standard errors -- not the unpaired group means.
PAIRED_SIGN_K: float = 2.0
ADAPTIVE_REPLICA_CAP: int = 32
CATASTROPHIC_DELTA: float = -0.5
TRIM_FRACTION: float = 0.1
