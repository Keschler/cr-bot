from __future__ import annotations

from types import SimpleNamespace


def test_rollout_shared_memory_attach_supports_python312_signature(
    monkeypatch,
) -> None:
    import multiprocessing.resource_tracker as resource_tracker
    import multiprocessing.shared_memory as shared_memory

    import simulator.rl.rollout_farm as rollout_farm

    class LegacySharedMemory:
        def __init__(self, *, name: str, create: bool) -> None:
            self._name = f"/{name}"

    unregistered: list[tuple[str, str]] = []
    monkeypatch.setattr(shared_memory, "SharedMemory", LegacySharedMemory)
    monkeypatch.setattr(
        rollout_farm.inspect,
        "signature",
        lambda _constructor: SimpleNamespace(parameters={"name": object()}),
    )
    monkeypatch.setattr(
        resource_tracker,
        "unregister",
        lambda name, kind: unregistered.append((name, kind)),
    )

    handle = rollout_farm._attach_shared_memory("rollout-segment")

    assert isinstance(handle, LegacySharedMemory)
    assert unregistered == [("/rollout-segment", "shared_memory")]


def test_rollout_shared_memory_schema_preserves_successor_values() -> None:
    import simulator.rl.rollout_farm as rollout_farm

    config = SimpleNamespace(
        envs=2,
        horizon=3,
        gru_hidden_dim=4,
        gru_layers=1,
        use_privileged_critic=False,
        collect_belief_targets=False,
    )
    storage = rollout_farm._SharedRolloutStorage.create(config)
    try:
        assert "next_values" in storage.arrays
        assert storage.arrays["next_values"].shape == (2, 3)
        assert "next_values_present" in storage.arrays
        assert storage.arrays["next_values_present"].shape == (2,)
        assert "bootstrap_values" in storage.arrays
    finally:
        storage.close(unlink=True)
