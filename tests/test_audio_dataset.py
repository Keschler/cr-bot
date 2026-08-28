from pathlib import Path

import numpy as np
from scipy.io import wavfile

from cr_bot.audio.dataset import (
    MixedSFXCardDataset,
    allowed_sample_ranges,
    collect_sfx_files,
    is_deploy_like,
    is_deploy_voice_like,
    split_sfx_samples,
)
from cr_bot.audio.features import AudioFeatureConfig
from cr_bot.audio.labels import audio_card_classes, normalize_card_key, sfx_path_to_card_keys


def test_allowed_sample_ranges_can_limit_samples_to_overtime_ranges():
    ranges = allowed_sample_ranges(
        num_samples=1_000,
        sample_rate=10,
        excluded_s=[(18.0, 20.0), (62.0, 64.0)],
        window_samples=10,
        track_idx=3,
        included_s=[(17.0, 22.0), (60.0, 65.0)],
    )

    assert ranges == [
        (3, 170, 180),
        (3, 200, 220),
        (3, 600, 620),
        (3, 640, 650),
    ]


def test_allowed_sample_ranges_preserves_full_track_behavior_without_included_ranges():
    ranges = allowed_sample_ranges(
        num_samples=100,
        sample_rate=10,
        excluded_s=[(2.0, 4.0)],
        window_samples=10,
        track_idx=0,
    )

    assert ranges == [
        (0, 0, 20),
        (0, 40, 100),
    ]


def test_spirit_empress_sfx_files_are_split_by_elixir_variant():
    folder = "card_legendary_spirit_empress"

    assert sfx_path_to_card_keys(f"{folder}/card_legendary_spirit_empress_3elixir_deploy.wav") == [
        "spirit-empress-3-elixir"
    ]
    assert sfx_path_to_card_keys(f"{folder}/card_legendary_spirit_empress_deploy_6cost_vo_01.wav") == [
        "spirit-empress-6-elixir"
    ]
    assert sfx_path_to_card_keys(f"{folder}/card_legendary_spirit_empress_atk_vo_01.wav") == []


def test_audio_classes_replace_base_spirit_empress_with_variants():
    classes = audio_card_classes(["knight", "spirit-empress"])

    assert classes == [
        "knight",
        "spirit-empress-3-elixir",
        "spirit-empress-6-elixir",
    ]


def test_deploy_filter_accepts_placement_sounds_but_not_spawn_cast_or_land():
    assert is_deploy_like("megaknight_dep_sfx_01.wav")
    assert is_deploy_like("building_place_01.wav")
    assert is_deploy_like("card_rare_goblin_hut_deloy_sfx.wav")
    assert not is_deploy_like("fire_spirit_spawn_01.wav")
    assert not is_deploy_like("attack_sfx_cast.wav")
    assert not is_deploy_like("evo_mega_knight_dash_land_01.wav")
    assert not is_deploy_like("building_destroyed_05.wav")
    assert is_deploy_like("card_epic_witch/summon_witch_01.wav")
    assert not is_deploy_like("card_epic_witch/witch_summon_02.wav")


def test_deploy_voice_filter_detects_followup_voice_lines():
    assert is_deploy_voice_like("card_legendary_princess/princess_archer_dep_vo_01.wav")
    assert is_deploy_voice_like("card_rare_wizard/wiz_deploy_vo_02.wav")
    assert not is_deploy_voice_like("card_legendary_princess/princess_archer_atk_vo_01.wav")
    assert not is_deploy_voice_like("card_legendary_princess/p_archer_dep_01.wav")


def test_evolution_folder_uses_a_separate_audio_class(tmp_path):
    raw_sfx_dir = tmp_path / "raw_sfx"
    folder = raw_sfx_dir / "card_evolution_mega_knight"
    folder.mkdir(parents=True)

    assert sfx_path_to_card_keys(folder / "evo_mega_knight_deploy_sfx_01.wav") == [
        "evo-mega-knight"
    ]
    assert "evo-mega-knight" in audio_card_classes(["mega-knight"], raw_sfx_dir=raw_sfx_dir)
    assert normalize_card_key("evo-mega-knight") == "evo-mega-knight"


def test_evolution_folder_does_not_expand_multi_card_aliases():
    assert sfx_path_to_card_keys("card_evolution_barbarian/evo_barb_dep_sfx_02.wav") == [
        "evo-barbarians"
    ]
    assert sfx_path_to_card_keys("card_evolution_musketeer/evo_musketeer_deploy_sfx_01.wav") == [
        "evo-musketeer"
    ]


def test_deploy_only_collection_does_not_fall_back_to_unrelated_sfx(tmp_path):
    folder = tmp_path / "card_common_knight"
    folder.mkdir()
    (folder / "knight_attack_01.wav").touch()

    samples, _, _ = collect_sfx_files(tmp_path, deploy_only=True, known_cards={"knight"})

    assert samples == []


def test_generated_sample_split_keeps_every_source_wav_in_training():
    samples = [
        ("evo-mega-knight", "deploy_sfx.wav"),
        ("evo-mega-knight", "deploy_vo.wav"),
    ]

    train, val = split_sfx_samples(samples, samples_per_sfx=8, seed=0)

    assert set(samples) <= set(train)
    assert len(train) == 13
    assert len(val) == 3


class _ZeroBackground:
    available = True

    def __init__(self, config: AudioFeatureConfig) -> None:
        self.config = config

    def sample_window(self, rng) -> np.ndarray:  # noqa: ARG002
        return np.zeros(self.config.num_samples, dtype=np.float32)


def test_positive_mixture_can_append_matching_deploy_voice(tmp_path):
    config = AudioFeatureConfig(sample_rate=1000, window_s=1.0, n_fft=128, win_length=64, hop_length=16, n_mels=16)
    deploy_path = tmp_path / "p_archer_dep_01.wav"
    voice_path = tmp_path / "princess_archer_dep_vo_01.wav"
    _write_test_wav(deploy_path, np.ones(80, dtype=np.float32))
    _write_test_wav(voice_path, np.full(60, 0.5, dtype=np.float32))

    dataset = MixedSFXCardDataset(
        [("princess", deploy_path), ("princess", voice_path)],
        classes=["no_event", "princess"],
        config=config,
        background=_ZeroBackground(config),
        samples_per_sfx=1,
        positive_samples=[("princess", deploy_path)],
        no_event_count=0,
        seed=0,
    )

    waveform = dataset._compose_positive_waveform("princess", Path(deploy_path), np.random.default_rng(0))
    assert len(_nonzero_runs(waveform, threshold=1e-3)) == 2


def _write_test_wav(path: Path, waveform: np.ndarray, sample_rate: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.int16(np.clip(waveform, -1.0, 1.0) * 32767))


def _nonzero_runs(waveform: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    mask = np.abs(waveform) > threshold
    runs: list[tuple[int, int]] = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs
