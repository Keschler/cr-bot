from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_importing_rl_keeps_v4_package_boundary_lazy() -> None:
    """The V4 entry point must not pull the historical RL stack into memory."""

    code = """
import sys
import simulator.rl as rl

if rl.TORCH_AVAILABLE:
    from simulator.rl.v4 import frontier_ppo  # noqa: F401
    from simulator.rl import ModelConfigV4

    assert ModelConfigV4.__module__ == 'simulator.rl.v4.model_v4'

loaded = {
    name.rsplit('.', 1)[-1]
    for name in sys.modules
    if name.startswith('simulator.rl.')
}
legacy = {
    'basic_scenarios', 'collector', 'curriculum', 'domain_randomization',
    'evaluation_matrix', 'league', 'learner', 'model', 'opponent_pool',
    'prototype', 'self_play', 'shadow',
}
assert not (loaded & legacy), sorted(loaded & legacy)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "simulator"), str(ROOT / "src")))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
