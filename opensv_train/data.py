from __future__ import annotations

import csv
import json
import math
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .acoustic import TimedPhoneme
from .audio_io import read_wav_frames
from .duration import PhonemeDuration, Syllable
from .f0 import F0_PROFILES, f0_cache_tag, process_f0_track, resolve_f0_profile
from .f0_tracking import (
    FRAME_SECONDS,
    SAMPLE_RATE,
    _empty_f0_track,
    _stable_f0_mask,
    _validate_f0_settings,
    extract_f0,
)
from .pitch import PITCH_CONTEXT_SECONDS, PitchNote

REST_CONTEXT_SECONDS = 0.1
PHRASE_BREAK_SECONDS = 0.2
SPECIAL_PHONEMES = {"SP": "sil", "AP": "br"}
LANGUAGE_CODES = {"ja": "japanese", "zh": "mandarin", "en": "english", "yue": "cantonese", "es": "spanish"}
NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

@dataclass
class Phone:
    language: str
    symbol: str
    start: float
    end: float
    word: int = -1

    @property
    def silent(self) -> bool:
        return self.symbol in ("sil", "pau")


@dataclass
class Note:
    start: float
    end: float
    pitch: float | None
    slur: bool = False
    word: int = -1

    @property
    def rest(self) -> bool:
        return self.pitch is None


@dataclass
class Utterance:
    name: str
    audio: np.ndarray
    phones: list[Phone]
    notes: list[Note] | None
    f0_times: np.ndarray
    f0_values: np.ndarray
    tuning: float = 0.0
    style: str | None = None

    def f0_at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(self.f0_values, dtype=np.float32).reshape(-1)
        track_times = np.asarray(self.f0_times, dtype=np.float64).reshape(-1)
        voiced = np.isfinite(values) & (values > 0)
        if not values.size or not track_times.size or values.size != track_times.size or not voiced.any():
            # Valid consonant-only and silence recordings can have no measurable
            # F0 at all. Keep a neutral conditioning pitch for the acoustic model,
            # while marking every frame unvoiced so the voicing target stays zero.
            return np.full(times.shape, 150.0, dtype=np.float32), np.zeros(times.shape, dtype=bool)
        log_f0 = np.interp(times, track_times[voiced], np.log(values[voiced]))
        nearest = np.clip(np.searchsorted(track_times, times), 0, len(track_times) - 1)
        return np.exp(log_f0).astype(np.float32), voiced[nearest] & (np.abs(track_times[nearest] - times) < 2 * FRAME_SECONDS)


@dataclass
class Phrase:

    utterance: Utterance
    start: float
    end: float
    phones: list[Phone]
    notes: list[Note] | None = None
    syllables: list[Syllable] | None = None
    syllable_targets: list[float] | None = None
    phone_frames: list[int] = field(default_factory=list)
    problem: str | None = None
    _frame_f0_cache: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False, compare=False)
    _pitch_f0_cache: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def frames(self) -> int:
        return sum(self.phone_frames)

    def timed_phonemes(self) -> list[TimedPhoneme]:
        return [TimedPhoneme(phone.language, phone.symbol, count) for phone, count in zip(self.phones, self.phone_frames)]

    def frame_f0(self) -> tuple[np.ndarray, np.ndarray]:
        # Phrase timing and its utterance F0 target are immutable during a run.
        # Cache this frame-aligned contour because acoustic training revisits
        # the same phrases many times.
        if self._frame_f0_cache is None:
            times = self.start + (np.arange(self.frames) + 0.5) * FRAME_SECONDS
            self._frame_f0_cache = self.utterance.f0_at(times)
        return self._frame_f0_cache

    def audio(self) -> np.ndarray:
        first = int(round(self.start * SAMPLE_RATE))
        length = self.frames * int(SAMPLE_RATE * FRAME_SECONDS)
        audio = self.utterance.audio[max(first, 0):first + length]
        return np.pad(audio, (max(-first, 0), length - len(audio) - max(-first, 0)))

    def audio_segment(self, start_sample: int, sample_count: int) -> np.ndarray:
        """Return only a requested phrase-local audio window, with matching edge padding.

        Acoustic training usually consumes a random segment rather than the
        entire phrase. Avoid materializing and padding the full phrase array
        just to immediately slice that segment back out.
        """
        start_sample, sample_count = int(start_sample), int(sample_count)
        if start_sample < 0 or sample_count < 0:
            raise ValueError("audio segment offsets and lengths must be non-negative")
        phrase_origin = int(round(self.start * SAMPLE_RATE))
        source_start = phrase_origin + start_sample
        source_stop = source_start + sample_count
        audio = self.utterance.audio
        valid_start = max(0, source_start)
        valid_stop = max(valid_start, min(len(audio), source_stop))
        segment = audio[valid_start:valid_stop]
        left_pad = min(sample_count, max(0, -source_start))
        right_pad = max(0, sample_count - left_pad - len(segment))
        if left_pad or right_pad:
            segment = np.pad(segment, (left_pad, right_pad))
        return segment

    def pitch_example(self) -> tuple[list[PitchNote], list[PhonemeDuration], np.ndarray, np.ndarray] | None:
        if not self.syllables:
            return None
        score = self.notes[1:-1]
        first, last = score[0], score[-1]
        origin, stop = first.start - PITCH_CONTEXT_SECONDS, last.end + PITCH_CONTEXT_SECONDS
        language = self.phones[0].language

        def note(pitch, seconds, rest=False, breath=False, slur=False):
            symbol = "br" if breath else "sil" if rest else "a"
            return PitchNote(Syllable(language, [symbol], seconds, pitch, slur), silence=rest and not breath, breath=breath, continuation=slur)

        notes = [note(self.syllables[1].pitch, PITCH_CONTEXT_SECONDS, rest=True)]
        for syllable, entry in zip(self.syllables[1:-1], score):
            breath = entry.rest and "br" in syllable.phonemes
            notes.append(note(syllable.pitch, entry.end - entry.start, rest=entry.rest, breath=breath, slur=entry.slur))
        notes.append(note(self.syllables[-2].pitch, PITCH_CONTEXT_SECONDS, rest=True))
        phonemes = []
        for index, phone in enumerate(self.phones):
            begin = origin if index == 0 else phone.start
            end = stop if index == len(self.phones) - 1 else phone.end
            phonemes.append(PhonemeDuration(phone.language, phone.symbol, 0, 0, end - begin))
        if self._pitch_f0_cache is None:
            frames = int(round((stop - origin) / FRAME_SECONDS))
            times = origin + np.arange(frames) * FRAME_SECONDS
            hz, voiced = self.utterance.f0_at(times)
            inside = (times >= first.start) & (times < last.end) & (times >= 0) & (times < len(self.utterance.audio) / SAMPLE_RATE)
            midi = 69.0 + 12.0 * np.log2(hz / 440.0) - self.utterance.tuning
            self._pitch_f0_cache = (midi.astype(np.float32), voiced & inside)
        midi, pitch_mask = self._pitch_f0_cache
        return notes, phonemes, midi, pitch_mask


def parse_note(name: str) -> float | None:
    if name.lower() == "rest":
        return None
    cents = 0.0
    for sign in "+-":
        if sign in name[2:]:
            name, offset = name[:2] + name[2:].split(sign)[0], name[2:].split(sign, 1)[1]
            cents = float(offset) * (1 if sign == "+" else -1)
            break
    letter, rest = name[0].upper(), name[1:]
    semitone = NOTE_NAMES[letter]
    while rest and rest[0] in "#b":
        semitone += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return 12 * (int(rest) + 1) + semitone + cents / 100.0


def read_audio(path: Path) -> np.ndarray:
    # Share decoding policy with LAB generation so preprocessing and training
    # cannot disagree about whether the same WAV is readable.
    audio, rate = read_wav_frames(path)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    return audio


def map_phoneme(token: str, language: str, mapping: dict[str, str]) -> tuple[str, str]:
    if token in mapping:
        token = mapping[token]
    if "/" in token:
        prefix, token = token.split("/", 1)
        language = LANGUAGE_CODES.get(prefix, prefix)
    return language, SPECIAL_PHONEMES.get(token, token)



def _source_fingerprint(path: Path) -> np.ndarray:
    """Size and mtime of a WAV, stored in its cache entry to detect replaced recordings."""
    stat = path.stat()
    return np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64)


def _read_cache_entry(cached: Path, fingerprint: np.ndarray):
    """Return (audio, times, values) from a valid, current cache entry, else None."""
    if not cached.is_file():
        return None
    try:
        with np.load(cached) as stored:
            if not np.array_equal(stored["source"], fingerprint):
                return None
            audio, times, values = stored["audio"], stored["f0_times"], stored["f0_values"]
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        # Missing fingerprint (legacy entry), truncated or corrupt file: rebuild it.
        return None
    if not times.size or not values.size or times.size != values.size:
        # Incomplete entries must not turn a valid WAV into an IndexError later.
        return None
    return audio, times, values


def _write_cache_entry(cached: Path, fingerprint: np.ndarray, audio, times, values) -> None:
    """Write atomically so an interrupted run never leaves a half-written entry."""
    temp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as stream:
            np.savez(stream, audio=audio, f0_times=times, f0_values=values, source=fingerprint)
        os.replace(temp, cached)
    finally:
        temp.unlink(missing_ok=True)


def _cached_features(cached: Path, wav_path: Path, floor: float, ceiling: float, threshold: float):
    """Audio plus F0 track for a recording, reusing the cache only while the WAV is unchanged."""
    fingerprint = _source_fingerprint(wav_path)
    entry = _read_cache_entry(cached, fingerprint)
    if entry is not None:
        return entry
    audio = read_audio(wav_path)
    times, values = extract_f0(audio, floor, ceiling, voicing_threshold=threshold)
    _write_cache_entry(cached, fingerprint, audio, times, values)
    return audio, times, values


def load_dataset(
    directory,
    language: str,
    phoneme_map: dict[str, str] | None = None,
    cache: Path | None = None,
    f0_range: tuple[float, float] = (65.0, 1100.0),
    f0_voicing_threshold: float = 0.60,
    f0_smoothing: int = 0,
    f0_octave_correction: bool = False,
    f0_shift_semitones: float = 0.0,
    f0_reference_cents: float = 0.0,
) -> list[Utterance]:
    if not np.isfinite(f0_reference_cents) or abs(f0_reference_cents) > 1200.0:
        raise ValueError("F0 reference correction must be finite and between -1200 and +1200 cents")
    directory = Path(directory)
    mapping = phoneme_map or {}
    floor, ceiling, threshold = resolve_f0_profile(
        "standard", floor=f0_range[0], ceiling=f0_range[1], voicing_threshold=f0_voicing_threshold
    )
    _validate_f0_settings(floor, ceiling, threshold)
    cache = cache or directory / ".opensv-cache"
    cache.mkdir(exist_ok=True)
    utterances = []
    with open(directory / "transcriptions.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["name"]
            tokens = row["ph_seq"].split()
            durations = [float(value) for value in row["ph_dur"].split()]
            if len(tokens) != len(durations):
                raise ValueError(f"{name}: ph_seq and ph_dur lengths differ")
            boundaries = np.concatenate([[0.0], np.cumsum(durations)])
            phones = [Phone(*map_phoneme(token, language, mapping), float(boundaries[i]), float(boundaries[i + 1])) for i, token in enumerate(tokens)]
            notes = None
            if row.get("note_seq", "").strip() and row.get("ph_num", "").strip():
                counts = [int(v) for v in row["ph_num"].split()]
                if sum(counts) != len(phones):
                    raise ValueError(f"{name}: ph_num does not cover ph_seq")
                for word, first in enumerate(np.cumsum([0] + counts[:-1])):
                    for phone in phones[first:first + counts[word]]:
                        phone.word = word
                note_bounds = np.concatenate([[0.0], np.cumsum([float(v) for v in row["note_dur"].split()])])
                slurs = [int(v) for v in row["note_slur"].split()] if row.get("note_slur", "").strip() else None
                notes, word = [], -1
                for index, symbol in enumerate(row["note_seq"].split()):
                    slur = bool(slurs[index]) if slurs else bool(index and not _starts_word(row, index))
                    word += 0 if slur else 1
                    notes.append(Note(float(note_bounds[index]), float(note_bounds[index + 1]), parse_note(symbol), slur, word))
                if word + 1 != len(counts):
                    raise ValueError(f"{name}: {word + 1} note groups for {len(counts)} ph_num words")
            # Include analysis parameters in the cache key: switching to a
            # different F0 preset must never silently reuse an old contour.
            cache_tag = f0_cache_tag(floor, ceiling, threshold)
            wav_path = directory / "wavs" / f"{name}.wav"
            audio, times, values = _cached_features(
                cache / f"{name}.{cache_tag}.npz", wav_path, floor, ceiling, threshold
            )
            tuning = 0.0
            if notes is not None:
                tuning = tuning_offset(notes)
                for note in notes:
                    note.pitch = None if note.rest else note.pitch - tuning
            processed_values = process_f0_track(
                values,
                smoothing_frames=f0_smoothing,
                correct_octave=f0_octave_correction,
                shift_semitones=f0_shift_semitones,
            )
            # Apply the reference calibration after loading raw F0 from cache;
            # changing a cents offset does not require re-running audio analysis.
            if f0_reference_cents:
                processed_values = processed_values * np.float32(2.0 ** (f0_reference_cents / 1200.0))
            utterances.append(
                Utterance(
                    name, audio, phones, notes, times, processed_values, tuning,
                    (row.get("style") or "").strip() or None,
                )
            )
    return utterances


def tuning_offset(notes: list[Note]) -> float:
    sung = [note for note in notes if not note.rest]
    phase = sum((note.end - note.start) * np.exp(2j * np.pi * note.pitch) for note in sung)
    return float(np.angle(phase) / (2 * np.pi)) if sung and abs(phase) > 0 else 0.0


def _starts_word(row: dict, note_index: int) -> bool:
    word_bounds = np.cumsum([0.0] + _word_durations(row))
    note_bounds = np.cumsum([0.0] + [float(v) for v in row["note_dur"].split()])
    return bool(np.min(np.abs(word_bounds - note_bounds[note_index])) < 1e-4)


def _word_durations(row: dict) -> list[float]:
    durations = [float(v) for v in row["ph_dur"].split()]
    counts = [int(v) for v in row["ph_num"].split()]
    words, position = [], 0
    for count in counts:
        words.append(sum(durations[position:position + count]))
        position += count
    return words


def quantize(boundaries: np.ndarray) -> list[int]:
    counts, previous = [], 0
    for boundary in boundaries[1:]:
        frame = max(previous + 1, int(round((boundary - boundaries[0]) / FRAME_SECONDS)))
        counts.append(frame - previous)
        previous = frame
    return counts


def make_phrases(utterance: Utterance) -> list[Phrase]:
    phones = utterance.phones
    breaks = [index for index, phone in enumerate(phones) if phone.silent and phone.end - phone.start >= PHRASE_BREAK_SECONDS]
    edges = [-1] + breaks + [len(phones)]
    phrases = []
    for left, right in zip(edges, edges[1:]):
        inner = phones[left + 1:right]
        while inner and inner[0].silent:
            inner = inner[1:]
        while inner and inner[-1].silent:
            inner = inner[:-1]
        if not inner:
            continue
        language = inner[0].language
        onset = _first_nucleus_time(inner, utterance)
        start = min(onset - REST_CONTEXT_SECONDS, inner[0].start - 0.02)
        end = inner[-1].end + REST_CONTEXT_SECONDS
        tiled = [Phone(language, "sil", start, inner[0].start), *inner, Phone(language, "sil", inner[-1].end, end)]
        boundaries = np.array([tiled[0].start] + [phone.end for phone in tiled])
        phrase = Phrase(utterance, start, end, tiled, phone_frames=quantize(boundaries))
        if utterance.notes is not None:
            try:
                _attach_notes(phrase)
            except ValueError as error:
                phrase.problem = str(error)
        phrases.append(phrase)
    return phrases


def _first_nucleus_time(phones: list[Phone], utterance: Utterance) -> float:
    if utterance.notes:
        words = {phone.word for phone in phones}
        sung = [note for note in utterance.notes if not note.rest and note.word in words]
        if sung:
            return min(phone.start for phone in phones if phone.word == sung[0].word)
    return phones[0].start


def _attach_notes(phrase: Phrase) -> None:
    inner = phrase.phones[1:-1]
    notes_of = {}
    for note in phrase.utterance.notes:
        notes_of.setdefault(note.word, []).append(note)
    words = []
    for phone in inner:
        if not words or words[-1][0] != phone.word:
            words.append((phone.word, []))
        words[-1][1].append(phone)
    scored = [index for index, (word, members) in enumerate(words) if members[0].symbol == "br" or any(not note.rest for note in notes_of.get(word, []))]
    sung = [note for word, _ in words for note in notes_of.get(word, []) if not note.rest]
    if not scored or not sung:
        raise ValueError(f"{phrase.utterance.name}: phrase has no sung notes")
    language = phrase.phones[0].language
    leading = [phone for _, members in words[:scored[0]] for phone in members]
    pitch = int(round(sung[0].pitch))
    syllables = [Syllable(language, ["sil"] + [p.symbol for p in leading], REST_CONTEXT_SECONDS, pitch)]
    targets = [REST_CONTEXT_SECONDS - sum(p.end - p.start for p in leading)] + [p.end - p.start for p in leading]
    onset = notes_of[words[scored[0]][0]][0].start
    timeline = [Note(onset - REST_CONTEXT_SECONDS, onset, None)]
    for word, members in words[scored[0]:]:
        notes = notes_of[word]
        for position, note in enumerate(notes):
            pitch = pitch if note.rest else int(round(note.pitch))
            if len(notes) == 1:
                chosen, seconds = members, [p.end - p.start for p in members]
            else:
                nucleus = members[0]
                portion = min(nucleus.end, note.end) - max(nucleus.start, note.start)
                final = position == len(notes) - 1
                chosen = members if final else [nucleus]
                seconds = [portion] + ([p.end - p.start for p in members[1:]] if final else [])
            symbols = [p.symbol for p in chosen]
            syllables.append(Syllable(language, symbols, note.end - note.start, pitch, note.slur))
            targets.extend(seconds)
            timeline.append(note)
    last = timeline[-1]
    syllables.append(Syllable(language, ["sil"], REST_CONTEXT_SECONDS, pitch))
    targets.append(REST_CONTEXT_SECONDS)
    timeline.append(Note(last.end, last.end + REST_CONTEXT_SECONDS, None))
    phrase.notes = timeline
    phrase.syllables = syllables
    phrase.syllable_targets = [max(target, FRAME_SECONDS) for target in targets]


def split(phrases: list[Phrase], validation: float, seed: int = 0) -> tuple[list[Phrase], list[Phrase]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(phrases))
    if validation >= 1:
        count = min(int(validation), len(phrases) - 1)
    else:
        count = max(1, int(round(len(phrases) * validation))) if len(phrases) > 1 and validation > 0 else 0
    held = set(order[:count].tolist())
    return [p for i, p in enumerate(phrases) if i not in held], [p for i, p in enumerate(phrases) if i in held]


def read_phoneme_map(path) -> dict[str, str]:
    return json.loads(Path(path).read_text("utf-8")) if path else {}


def word_spans(tokens: list[str], vowels: set[str]) -> list[int]:
    spans = []
    for token in tokens:
        if not spans or token in ("SP", "AP") or token in vowels:
            spans.append(1)
        else:
            spans[-1] += 1
    return spans


def read_lab(path: Path) -> list[tuple[int, int, str]]:
    return [(int(start), int(end), symbol) for start, end, symbol in (line.split() for line in Path(path).read_text("utf-8").splitlines() if line.strip())]


def lab_recordings(labs, wavs) -> list[tuple[str, Path, list[tuple[int, int, str]]]]:
    labs, wavs = Path(labs), Path(wavs)
    return [(lab.stem, wavs / f"{lab.stem}.wav", read_lab(lab)) for lab in sorted(labs.glob("*.lab")) if (wavs / f"{lab.stem}.wav").exists()]


def corpus_recordings(corpus) -> list[tuple[str, Path, list[tuple[int, int, str]]]]:
    corpus = Path(corpus)
    if corpus.suffix == ".json":
        return [(entry["id"], Path(entry["wav"]), [(label["start_100ns"], label["end_100ns"], label["phoneme"]) for label in entry["labels"]])
                for entry in json.loads(corpus.read_text("utf-8"))["recordings"]]
    if (corpus / "lab").is_dir():
        return lab_recordings(corpus / "lab", corpus / "wav")
    return lab_recordings(corpus, corpus)


def import_labels(recordings: list[tuple[str, Path, list[tuple[int, int, str]]]], output: Path, mapping: dict[str, str], vowels: set[str] | None = None) -> list[str]:
    import shutil

    output = Path(output)
    (output / "wavs").mkdir(parents=True, exist_ok=True)
    rows, notes = [], []
    for name, wav, entries in recordings:
        for previous, current in zip(entries, entries[1:]):
            if previous[1] != current[0]:
                raise ValueError(f"{name}.lab: gap or overlap at {int(current[0]) / 1e7:.3f}s")
        symbols = [mapping.get(symbol, symbol) for _, _, symbol in entries]
        tokens = [{"pau": "SP", "sil": "SP", "br": "AP"}.get(symbol, symbol) for symbol in symbols]
        durations = [(int(end) - int(start)) / 1e7 for start, end, _ in entries]
        rows.append({"name": name, "ph_seq": " ".join(tokens), "ph_dur": " ".join(f"{value:.7f}" for value in durations)})
        if vowels is not None:
            rows[-1]["ph_num"] = " ".join(str(count) for count in word_spans(tokens, vowels))
        notes.append(f"{name}: {len(entries)} phonemes, {sum(durations):.1f}s")
        shutil.copyfile(wav, output / "wavs" / f"{name}.wav")
    with open(output / "transcriptions.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return notes


def read_dictionary(path) -> dict[str, list[str]]:
    entries = {}
    for line in Path(path).read_text("utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            entries[parts[0]] = parts[1:]
    return entries


def convert_dictionary(source: Path, output: Path, source_dictionary: dict[str, list[str]], target_dictionary: dict[str, list[str]], nuclei: set[str], coda_share: float = 0.3) -> tuple[int, dict[str, int]]:
    source, output = Path(source), Path(output)
    spellings = {tuple(phones): syllable for syllable, phones in source_dictionary.items()}
    longest = max(len(phones) for phones in spellings)
    finals = {phones[-1] for phones in source_dictionary.values()}
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "wavs").exists():
        (output / "wavs").symlink_to((source / "wavs").resolve())
    unknown: dict[str, int] = {}
    rows = []
    with open(source / "transcriptions.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            tokens = row["ph_seq"].split()
            durations = [float(value) for value in row["ph_dur"].split()]
            symbols, seconds = [], []
            index = 0
            while index < len(tokens):
                if tokens[index] in ("SP", "AP"):
                    symbols.append(tokens[index])
                    seconds.append(durations[index])
                    index += 1
                    continue
                for size in range(min(longest, len(tokens) - index), 0, -1):
                    syllable = spellings.get(tuple(tokens[index:index + size]))
                    if syllable is not None and syllable in target_dictionary:
                        break
                else:
                    unknown[tokens[index]] = unknown.get(tokens[index], 0) + 1
                    syllable, size = None, 1
                if syllable is None:
                    symbols.append(tokens[index])
                    seconds.append(durations[index])
                    index += 1
                    continue
                group = tokens[index:index + size]
                times = durations[index:index + size]
                split = next(position for position, phone in enumerate(group) if phone in finals)
                onset_time, rhyme_time = sum(times[:split]), sum(times[split:])
                target = target_dictionary[syllable]
                nucleus = next(position for position, phone in enumerate(target) if phone in nuclei)
                onset, codas = target[:nucleus], target[nucleus + 1:]
                symbols.extend(target)
                if onset:
                    seconds.extend([onset_time / len(onset)] * len(onset))
                    nucleus_time = rhyme_time
                else:
                    nucleus_time = rhyme_time + onset_time
                coda_time = rhyme_time * coda_share if codas else 0.0
                seconds.append(nucleus_time - coda_time)
                seconds.extend([coda_time / len(codas)] * len(codas) if codas else [])
                index += size
            rows.append({"name": row["name"], "ph_seq": " ".join(symbols), "ph_dur": " ".join(f"{value:.6f}" for value in seconds)})
    with open(output / "transcriptions.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), unknown
