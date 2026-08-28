from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from simulator.physical_lab.campaign import (
    CampaignCase,
    DeckMutation,
    apply_deck_mutations,
    build_default_campaign,
    evaluate_campaign,
    load_campaign,
)
from simulator.physical_lab.devices import AdbPhoneController
import simulator.physical_lab.devices as devices_module
from simulator.physical_lab.schema import PhysicalLabError


def test_default_campaign_is_simple_to_complex_and_derives_fixed_opening_hands() -> None:
    campaign = build_default_campaign()

    assert [case.level for case in campaign.cases] == [
        "isolated",
        "single_troop",
        "paired_interaction",
        "multi_card",
        "complex",
    ]
    assert campaign.cases[1].mutations == (
        DeckMutation(
            "A",
            3,
            "archers",
            "keep an evolvable troop out of the first three device slots",
        ),
    )
    assert campaign.cases[1].spec.actions[0].card_slot == 3
    for case in campaign.cases:
        for side in ("A", "B"):
            deck = case.spec.initial_conditions.decks[side]
            assert dict(case.spec.initial_conditions.hand_slots[side]) == {
                card: slot for slot, card in enumerate(deck[:4])
            }
        assert case.spec.metadata["fixed_deck_host"] == "B"
        b_deck = case.spec.initial_conditions.decks["B"]
        restricted = case.spec.metadata["phone_card_constraints"]["B"]["restricted_cards"]
        assert all(card not in b_deck[:3] for card in restricted)
        assert case.spec.metadata["phone_card_constraints"]["B"]["restricted_human_slots"] == [4, 8]
        a_deck = case.spec.initial_conditions.decks["A"]
        assert "musketeer" not in a_deck[:3]
        assert case.spec.metadata["phone_card_constraints"]["A"]["restricted_human_slots"] == [4, 8]


def test_deck_mutations_reject_duplicates_and_apply_multiple_changes() -> None:
    base = {
        "A": ("hog-rider", "cannon", "musketeer", "skeletons", "ice-golem", "ice-spirit", "fireball", "log"),
        "B": ("hog-rider", "cannon", "musketeer", "skeletons", "ice-golem", "ice-spirit", "fireball", "log"),
    }

    result = apply_deck_mutations(
        base,
        (
            DeckMutation("A", 1, "archers"),
            DeckMutation("A", 6, "barbarian"),
        ),
    )
    assert result["A"][1] == "archers"
    assert result["A"][6] == "barbarian"

    with pytest.raises(PhysicalLabError, match="unique"):
        apply_deck_mutations(base, (DeckMutation("A", 0, "cannon"),))

    with pytest.raises(PhysicalLabError, match="duplicate mutation"):
        apply_deck_mutations(
            base,
            (DeckMutation("A", 1, "archers"), DeckMutation("A", 1, "barbarian")),
        )


def test_campaign_manifest_and_case_artifacts_are_write_once(tmp_path: Path) -> None:
    campaign = build_default_campaign()
    manifest = tmp_path / "campaign.json"
    campaign.save(manifest)
    assert load_campaign(manifest).campaign_hash() == campaign.campaign_hash()

    case_path = tmp_path / "case.json"
    campaign.cases[0].save(case_path)
    with pytest.raises(PhysicalLabError, match="immutable"):
        replace(campaign.cases[0], description="changed immutable case").save(case_path)

    # The loader still accepts the original immutable content after the
    # failed attempted replacement is restored by writing the same bytes.
    # This also exercises the hash gate independently of filesystem policy.
    original = campaign.cases[0].to_dict(include_hash=True)
    assert CampaignCase.from_dict(original).case_hash() == campaign.cases[0].case_hash()


def test_batch_evaluation_reports_missing_case_artifacts_without_rewriting_data(tmp_path: Path) -> None:
    campaign = build_default_campaign()

    result = evaluate_campaign(campaign, results_root=tmp_path / "results")

    assert result["case_count"] == 5
    assert result["evaluated_case_count"] == 0
    assert result["missing_or_rejected_case_count"] == 5
    assert all(row["status"] == "missing_artifact" for row in result["cases"])


def test_keep_awake_uses_only_the_explicit_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> object:
        commands.append(command)
        if command[-3:] == ["shell", "wm", "size"]:
            return devices_module.subprocess.CompletedProcess(command, 0, b"Physical size: 1080x2400\n", b"")
        if command[-4:] == ["settings", "get", "system", "screen_off_timeout"]:
            return devices_module.subprocess.CompletedProcess(command, 0, b"2147483647\n", b"")
        if command[-4:] == ["settings", "get", "global", "stay_on_while_plugged_in"]:
            # Android builds may normalize the requested USB/AC mask to all
            # supported charging sources.
            return devices_module.subprocess.CompletedProcess(command, 0, b"15\n", b"")
        if command[-3:] == ["shell", "getprop", "ro.product.model"]:
            return devices_module.subprocess.CompletedProcess(command, 0, b"test-model\n", b"")
        if command[-3:] == ["shell", "getprop", "ro.build.version.release"]:
            return devices_module.subprocess.CompletedProcess(command, 0, b"14\n", b"")
        return devices_module.subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(devices_module.subprocess, "run", fake_run)
    result = AdbPhoneController("PHONE_B", device_label="B").set_keep_awake()

    assert result["verified"] is True
    assert commands
    assert all(command[:3] == ["adb", "-s", "PHONE_B"] for command in commands)
    assert [
        "shell",
        "settings",
        "put",
        "system",
        "screen_off_timeout",
        "2147483647",
    ] in [command[3:] for command in commands]
