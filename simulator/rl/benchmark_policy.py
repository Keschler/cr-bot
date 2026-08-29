"""Measure recurrent actor decision latency on a fixed public-observation workload.

This benchmark intentionally keeps the model weights and tensor shapes fixed. It
reports the full public actor step (policy forward plus deterministic masked
action selection), and separates the forward and selection portions so an
inference fast path can be compared with the reference path.

Example from ``simulator/``::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:..:../src \
      ../capture/.venv-train/bin/python -m rl.benchmark_policy \
      --device auto --batch-size 16 --iterations 100
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from typing import Any, Callable

from ._compat import TORCH_AVAILABLE, TorchUnavailableError

if TORCH_AVAILABLE:
    import torch

    from .learner import _deterministic_action, resolve_policy_device
    from .model import ModelConfig, RecurrentHybridPolicy
    from .trajectory import ActionMasks
else:  # pragma: no cover - exercised only without optional torch
    torch = None  # type: ignore[assignment]
    _deterministic_action = Any  # type: ignore[assignment]
    resolve_policy_device = Any  # type: ignore[assignment]
    ModelConfig = Any  # type: ignore[assignment,misc]
    RecurrentHybridPolicy = Any  # type: ignore[assignment,misc]
    ActionMasks = Any  # type: ignore[assignment,misc]


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise TorchUnavailableError(
            "rl.benchmark_policy requires PyTorch in the training environment"
        )
    return torch


def _strategic_config() -> Any:
    from cr_bot.features.channels import GLOBAL_SCALAR_IDX
    from cr_bot.features.global_features import CARD_COUNT

    return ModelConfig(
        raster_channels=21,
        raster_height=32,
        raster_width=18,
        global_dim=768,
        entity_dim=32,
        max_entities=128,
        model_dim=128,
        encoder_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=256,
        gru_hidden_dim=256,
        gru_layers=1,
        card_slots=4,
        belief_card_count=128,
        placement_rows=32,
        placement_cols=18,
        hand_feature_offset=len(GLOBAL_SCALAR_IDX),
        hand_card_count=CARD_COUNT,
        spatial_placement_features=True,
    )


def _workload(config: Any, batch_size: int, device: Any, seed: int) -> tuple[Any, ...]:
    torch = _require_torch()
    from cr_bot.features.global_features import CARD_COUNT

    generator = torch.Generator(device="cpu").manual_seed(seed)
    raster = torch.randn(
        batch_size,
        1,
        config.raster_channels,
        config.raster_height,
        config.raster_width,
        generator=generator,
        dtype=torch.float32,
    )
    global_features = torch.zeros(
        batch_size,
        1,
        config.global_dim,
        dtype=torch.float32,
    )
    hand_start = config.hand_feature_offset
    for slot in range(config.card_slots):
        card_id = (seed + slot) % CARD_COUNT
        global_features[:, 0, hand_start + slot * CARD_COUNT + card_id] = 1.0
    entities = torch.randn(
        batch_size,
        1,
        128,
        config.entity_dim,
        generator=generator,
        dtype=torch.float32,
    )
    entity_mask = torch.ones(batch_size, 1, 128, dtype=torch.bool)
    masks = ActionMasks(
        mode=torch.ones(batch_size, 1, 2, dtype=torch.bool),
        card=torch.ones(batch_size, 1, config.card_slots, dtype=torch.bool),
        placement=torch.ones(
            batch_size,
            1,
            config.card_slots,
            config.placement_rows,
            config.placement_cols,
            dtype=torch.bool,
        ),
    )
    return (
        raster.to(device=device),
        global_features.to(device=device),
        entities.to(device=device),
        entity_mask.to(device=device),
        ActionMasks(
            mode=masks.mode.to(device=device),
            card=masks.card.to(device=device),
            placement=masks.placement.to(device=device),
        ),
    )


def _sync(device: Any) -> None:
    torch = _require_torch()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _timed(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    batch_size: int,
    device: Any,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        fn()
        _sync(device)
        samples.append(perf_counter() - started)
    samples.sort()
    median = samples[len(samples) // 2]
    p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
    return {
        "median_ms": median * 1000.0,
        "p95_ms": p95 * 1000.0,
        "decisions_per_second": float(batch_size) / median,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch = _require_torch()
    if args.batch_size < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError("batch-size and iterations must be positive; warmup must be non-negative")
    device = resolve_policy_device(args.device)
    torch.manual_seed(args.seed)
    config = _strategic_config()
    policy = RecurrentHybridPolicy(config).to(device).eval()
    raster, global_features, entities, entity_mask, masks = _workload(
        config,
        args.batch_size,
        device,
        args.seed,
    )
    hidden = policy.initial_hidden(args.batch_size, device=device)
    reset_mask = torch.ones(args.batch_size, 1, dtype=torch.bool, device=device)

    def forward(*, include_beliefs: bool) -> Any:
        return policy(
            raster,
            global_features,
            entities,
            entity_mask,
            reset_mask=reset_mask,
            hidden=hidden,
            action_masks=masks,
            include_beliefs=include_beliefs,
        )

    def select(*, include_beliefs: bool) -> Any:
        output = forward(include_beliefs=include_beliefs)
        return _deterministic_action(policy, output, masks)

    def fast_select() -> Any:
        actions, _final_hidden = policy.act_deterministic(
            raster,
            global_features,
            entities,
            entity_mask,
            masks,
            reset_mask=reset_mask,
            hidden=hidden,
        )
        return actions

    with torch.inference_mode():
        forward_stats = _timed(
            lambda: forward(include_beliefs=True),
            warmup=args.warmup,
            iterations=args.iterations,
            batch_size=args.batch_size,
            device=device,
        )
        select_stats = _timed(
            lambda: select(include_beliefs=True),
            warmup=args.warmup,
            iterations=args.iterations,
            batch_size=args.batch_size,
            device=device,
        )
        actor_forward_stats = _timed(
            lambda: forward(include_beliefs=False),
            warmup=args.warmup,
            iterations=args.iterations,
            batch_size=args.batch_size,
            device=device,
        )
        actor_select_stats = _timed(
            lambda: select(include_beliefs=False),
            warmup=args.warmup,
            iterations=args.iterations,
            batch_size=args.batch_size,
            device=device,
        )
        fast_stats = _timed(
            fast_select,
            warmup=args.warmup,
            iterations=args.iterations,
            batch_size=args.batch_size,
            device=device,
        )
    return {
        "device": str(device),
        "requested_device": args.device,
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "model": "strategic",
        "forward": forward_stats,
        "forward_and_deterministic_select": select_stats,
        "forward_without_beliefs": actor_forward_stats,
        "forward_without_beliefs_and_deterministic_select": actor_select_stats,
        "fast_deterministic_action": fast_stats,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
