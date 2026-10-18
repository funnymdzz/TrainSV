import os

import numpy as np
import pytest

from opensv_train import data


@pytest.fixture
def counting(monkeypatch):
    calls = {"extract": 0}

    def fake_extract(audio, floor, ceiling, voicing_threshold=0.6):
        calls["extract"] += 1
        return np.arange(4, dtype=np.float64), np.full(4, 200.0)

    monkeypatch.setattr(data, "read_audio", lambda path: np.asarray([os.path.getsize(path)], dtype=np.float32))
    monkeypatch.setattr(data, "extract_f0", fake_extract)
    return calls


def _wav(tmp_path, payload=b"0123456789"):
    path = tmp_path / "a.wav"
    path.write_bytes(payload)
    return path


def test_cache_is_reused_while_the_wav_is_unchanged(tmp_path, counting):
    wav, cached = _wav(tmp_path), tmp_path / "a.tag.npz"
    first = data._cached_features(cached, wav, 65.0, 1100.0, 0.6)
    second = data._cached_features(cached, wav, 65.0, 1100.0, 0.6)
    assert counting["extract"] == 1
    assert np.array_equal(first[0], second[0])


def test_replacing_the_wav_invalidates_the_cache(tmp_path, counting):
    wav, cached = _wav(tmp_path), tmp_path / "a.tag.npz"
    data._cached_features(cached, wav, 65.0, 1100.0, 0.6)
    wav.write_bytes(b"a different, longer recording")
    audio, _, _ = data._cached_features(cached, wav, 65.0, 1100.0, 0.6)
    assert counting["extract"] == 2
    assert audio[0] == len(b"a different, longer recording")


def test_corrupt_cache_entry_is_rebuilt(tmp_path, counting):
    wav, cached = _wav(tmp_path), tmp_path / "a.tag.npz"
    cached.write_bytes(b"PK\x03\x04 truncated")
    data._cached_features(cached, wav, 65.0, 1100.0, 0.6)
    assert counting["extract"] == 1
    data._cached_features(cached, wav, 65.0, 1100.0, 0.6)
    assert counting["extract"] == 1


def test_legacy_entry_without_fingerprint_is_rebuilt(tmp_path, counting):
    wav, cached = _wav(tmp_path), tmp_path / "a.tag.npz"
    with cached.open("wb") as stream:
        np.savez(stream, audio=np.zeros(1, np.float32), f0_times=np.arange(3.0), f0_values=np.ones(3))
    data._cached_features(cached, wav, 65.0, 1100.0, 0.6)
    assert counting["extract"] == 1


def test_no_temporary_files_are_left_behind(tmp_path, counting):
    wav, cached = _wav(tmp_path), tmp_path / "a.tag.npz"
    data._cached_features(cached, wav, 65.0, 1100.0, 0.6)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.tag.npz", "a.wav"]
