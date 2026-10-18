from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

from .audit import audit_dataset
from .acoustic import TimedPhoneme
from .calibration import fit_input_ranges
from .data import F0_PROFILES, LANGUAGE_CODES, SAMPLE_RATE, load_dataset, make_phrases, read_phoneme_map, resolve_f0_profile, split
from .f0 import F0_SMOOTHING_OPTIONS, resolve_f0_processing, resolve_pitch_loss_options
from .train import MODELS, TrainConfig, Trainer, apply_gain, level_gain, mode_levels, mode_weights
from .voice import ROLES, Voice, initialize_pitch_from_file
from .packaging import build_svpk
from .sfpk import build_sfpk
from .timbre import summarize_timbre

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _validate_pitch_options(arguments) -> None:
    try:
        (
            arguments.f0_smoothing,
            arguments.f0_octave_correction,
        ) = resolve_f0_processing(
            arguments.f0_profile,
            smoothing_frames=arguments.f0_smoothing,
            correct_octave=arguments.f0_octave_correction,
        )
        (
            arguments.pitch_smoothing_hz,
            arguments.pitch_melody_weight,
            arguments.pitch_motion_weight,
        ) = resolve_pitch_loss_options(
            pitch_smoothing_hz=arguments.pitch_smoothing_hz,
            pitch_melody_weight=arguments.pitch_melody_weight,
            pitch_motion_weight=arguments.pitch_motion_weight,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not math.isfinite(arguments.f0_reference_cents) or not -1200.0 <= arguments.f0_reference_cents <= 1200.0:
        raise SystemExit("--f0-reference-cents must be finite and between -1200 and +1200 cents")
    if not math.isfinite(arguments.f0_shift_semitones) or not -24.0 <= arguments.f0_shift_semitones <= 24.0:
        raise SystemExit("--f0-shift-semitones must be finite and between -24 and +24 semitones")


def finetune(arguments) -> None:
    _validate_pitch_options(arguments)
    voice = Voice.load(arguments.voice)
    models = tuple(arguments.models.split(","))
    if unknown := set(models) - set(MODELS):
        raise SystemExit(f"unknown models: {', '.join(sorted(unknown))}")
    if arguments.pitch_model:
        if "pitch" not in models:
            raise SystemExit("--pitch-model requires --models to include pitch")
        if arguments.resume:
            raise SystemExit("--pitch-model initializes a new run; omit it when resuming a saved run")
        try:
            copied = initialize_pitch_from_file(voice, arguments.pitch_model)
        except (OSError, ValueError) as error:
            raise SystemExit(f"cannot initialize Pitch from {arguments.pitch_model}: {error}") from error
        print(f"initialized Pitch from compatible DNNI ({copied} parameter tensors); target speaker embedding preserved")
    f0_floor, f0_ceiling, f0_threshold = resolve_f0_profile(
        arguments.f0_profile,
        floor=arguments.f0_floor,
        ceiling=arguments.f0_ceiling,
        voicing_threshold=arguments.f0_voicing_threshold,
    )
    utterances = load_dataset(
        arguments.data,
        arguments.language,
        read_phoneme_map(arguments.phoneme_map),
        f0_range=(f0_floor, f0_ceiling),
        f0_voicing_threshold=f0_threshold,
        f0_smoothing=arguments.f0_smoothing,
        f0_octave_correction=arguments.f0_octave_correction,
        f0_shift_semitones=arguments.f0_shift_semitones,
        f0_reference_cents=arguments.f0_reference_cents,
    )
    phrases = [phrase for utterance in utterances for phrase in make_phrases(utterance)]
    problems = [phrase.problem for phrase in phrases if phrase.problem]
    no_f0 = [utterance.name for utterance in utterances if not np.any(np.isfinite(utterance.f0_values) & (utterance.f0_values > 0))]
    if no_f0:
        print(
            f"F0 note: {len(no_f0)} utterance(s) have no measurable voiced frames; "
            "they are retained for acoustic training and are not counted as damaged WAVs. "
            f"Examples: {', '.join(no_f0[:5])}"
        )
    print(f"{len(utterances)} utterances, {len(phrases)} phrases, {sum(p.frames for p in phrases) * 0.005 / 60:.1f} min")
    total_f0_frames = sum(len(utterance.f0_values) for utterance in utterances)
    voiced_f0 = np.concatenate(
        [utterance.f0_values[utterance.f0_values > 0] for utterance in utterances if np.any(utterance.f0_values > 0)]
    ) if any(np.any(utterance.f0_values > 0) for utterance in utterances) else np.empty(0, dtype=np.float32)
    voiced_percent = 100.0 * sum(int(np.count_nonzero(utterance.f0_values > 0)) for utterance in utterances) / max(total_f0_frames, 1)
    median_f0 = float(np.median(voiced_f0)) if voiced_f0.size else 0.0
    print(
        f"F0 tracking: {arguments.f0_profile} preset, {f0_floor:g}-{f0_ceiling:g} Hz, "
        f"voicing threshold {f0_threshold:g}; voiced {voiced_percent:.1f}% of frames"
        + (f", median voiced F0 {median_f0:.1f} Hz" if median_f0 else ", no voiced F0 frames detected")
    )
    print(
        f"F0 processing: median smoothing={arguments.f0_smoothing} frames, "
        f"isolated-octave correction={'on' if arguments.f0_octave_correction else 'off'}, "
        f"target shift={arguments.f0_shift_semitones:+g} semitones"
    )
    if "pitch" in models and not any(phrase.pitch_example() is not None for phrase in phrases):
        raise SystemExit(
            "Pitch training requires note_seq/note_dur (and usable ph_num word boundaries); "
            "phoneme-only LAB labels cannot train a Pitch model. Train acoustic only, or import note-labelled data."
        )
    if problems and set(models) & {"duration", "pitch"}:
        print(f"{len(problems)} phrases lack usable notes and train the acoustic model only; first: {problems[0]}")
    _check_phonemes(voice, phrases)
    if arguments.vocal_modes is not None:
        voice.set_vocal_modes([name for name in arguments.vocal_modes.split(",") if name])
    styles = {utterance.style for utterance in utterances} - {None}
    if unknown := styles - set(voice.acoustic.vocal_modes):
        raise SystemExit(f"dataset styles that are not vocal modes of the voice (add them with --vocal-modes): {', '.join(sorted(unknown))}")
    train, validation = split(phrases, arguments.validation, arguments.seed)
    device = torch.device(arguments.device)
    run = Path(arguments.run) if arguments.run else None
    resume = None
    if arguments.resume:
        if run is None:
            raise SystemExit("--resume continues the state.pt of a --run directory")
        if (run / "state.pt").exists():
            resume = torch.load(run / "state.pt", map_location=device, weights_only=True)
            if (resume["models"], resume["params"]) != (list(models), arguments.params):
                raise SystemExit(f"{run / 'state.pt'} trained --models {','.join(resume['models'])} --params {resume['params']}; resume with the same")
    voice.to(device)
    if not arguments.keep_level:
        if arguments.gain is not None:
            gain, source = arguments.gain, "given"
        elif arguments.level_reference:
            gain, source = level_gain(Voice.load(arguments.level_reference).to(device), phrases, device), "reference voice"
        else:
            gain, source = level_gain(voice, phrases, device), "starting voice"
        apply_gain(phrases, gain)
        print(f"recordings scaled by {gain:.3f} ({source})")
    config = TrainConfig(
        models=models, params=arguments.params, steps=arguments.steps, learning_rate=arguments.lr, anchor=arguments.anchor,
        evaluate_every=arguments.evaluate_every, seed=arguments.seed, device=arguments.device, rap_languages=voice.files.rap_languages, batch=arguments.batch,
        consonant_weight=arguments.consonant_weight, warmup=arguments.warmup, lr_end=arguments.lr_end, log_every=arguments.log_every,
        pitch_melody_weight=arguments.pitch_melody_weight,
        pitch_motion_weight=arguments.pitch_motion_weight,
        pitch_smoothing_hz=arguments.pitch_smoothing_hz,
        transition_weight=arguments.transition_weight,
        transition_frames=arguments.transition_frames,
        brightness_weight=arguments.brightness_weight,
    )
    calibration = random.Random(arguments.seed).sample(phrases, min(40, len(phrases)))
    models_root = Path(arguments.output).expanduser().parent.parent / "models"
    preexisting = set(models_root.glob("*/*.dnni"))
    previous: set[Path] = set()

    def save(step=None, version=None):
        nonlocal previous
        for report in fit_input_ranges(voice, calibration, device):
            print(f"widened a {report.role} {report.node_type} input range to ±{report.limit:g} (peak {report.peak:.3g})")
        if styles and voice.files.info is not None:
            voice.files.info.mode_extras = mode_levels(voice, calibration, device)
        saved = voice.save(arguments.output, name=arguments.name, version=version or arguments.version)
        if voice.files.info is not None:
            document = json.loads(saved.read_text("utf-8"))
            current = {models_root / kind / f"{document[kind]}.dnni" for kind in ("duration", "f0", "acoustic", "vocoder")}
            in_use = _referenced_models(models_root)
            for path in previous - current - preexisting - in_use:
                path.unlink(missing_ok=True)
            previous = current
        print(f"saved {saved}" + ("" if step is None else f" (step {step})"), flush=True)

    on_final = (lambda step: save(step, arguments.final_version)) if arguments.final_version else None
    journal = on_checkpoint = None
    if run is not None:
        run.mkdir(parents=True, exist_ok=True)
        log_file = (run / "log.jsonl").open("a")

        def journal(record):
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

        def on_checkpoint(state):
            torch.save({"models": list(models), "params": arguments.params, **state}, run / "state.pt.tmp")
            os.replace(run / "state.pt.tmp", run / "state.pt")

    trainer = Trainer(voice, train, validation, config, on_best=save if arguments.save_best else None, on_final=on_final, journal=journal, on_checkpoint=on_checkpoint)
    history = trainer.run(resume)
    save()
    if arguments.history:
        Path(arguments.history).write_text(json.dumps(history, indent=2))


def _referenced_models(models_root: Path) -> set[Path]:
    referenced = set()
    for info in (models_root.parent / "databases").glob("*/info.*.nofs"):
        try:
            document = json.loads(info.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for kind in ("duration", "f0", "acoustic", "vocoder"):
            if isinstance(document.get(kind), str):
                referenced.add(models_root / kind / f"{document[kind]}.dnni")
    return referenced


def _check_phonemes(voice: Voice, phrases) -> None:
    missing = set()
    for phrase in phrases:
        for phone in phrase.phones:
            try:
                voice.acoustic.phoneme_indices([TimedPhoneme(phone.language, phone.symbol, 1)])
            except ValueError:
                missing.add(f"{phone.language}/{phone.symbol}")
    if missing:
        raise SystemExit(f"phonemes not in the voice's phone sets (map them with --phoneme-map): {', '.join(sorted(missing))}")


def import_lab(arguments) -> None:
    from pathlib import Path

    from .data import corpus_recordings, import_labels, lab_recordings

    if arguments.from_filenames:
        from .filename_labels import (
            filename_recordings,
            is_quality_analysis_cache_wav,
            read_phone_table,
            write_lab_files,
        )

        if not arguments.wavs:
            raise SystemExit("--from-filenames requires --wavs")
        voice_symbols = []
        if arguments.voice and not arguments.phone_table:
            from .common import find_language

            phone_sets = Voice.load(arguments.voice).acoustic.phone_sets
            phone_set = phone_sets[find_language(phone_sets, arguments.language)]
            voice_symbols = phone_set.symbols
        phones = read_phone_table(arguments.phone_table, voice_symbols)
        source_system = arguments.source_phoneme_system
        if source_system == "auto":
            source_system = "vocaloid-mandarin" if arguments.language == "mandarin" else None
        elif source_system == "none":
            source_system = None
        unmatched: list[dict[str, str]] = []
        recordings = filename_recordings(arguments.wavs, phones, source_system=source_system, unmatched=unmatched)
        report_path = Path(arguments.output).expanduser() / "unmatched_phones.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8-sig") as report_file:
            writer = csv.DictWriter(
                report_file,
                fieldnames=("filename", "category", "source_token", "error"),
            )
            writer.writeheader()
            writer.writerows(unmatched)
        wav_directory = Path(arguments.wavs).expanduser()
        discovered_wavs = [
            path for path in wav_directory.iterdir()
            if path.is_file() and path.suffix.casefold() == ".wav"
            and not is_quality_analysis_cache_wav(path)
        ] if wav_directory.is_dir() else []
        print(
            f"WAV label generation: retained {len(recordings)} of {len(discovered_wavs)} "
            f"files; skipped {len(unmatched)}"
        )
        if unmatched:
            category_counts = {
                category: sum(item.get("category") == category for item in unmatched)
                for category in sorted({item.get("category", "other") for item in unmatched})
            }
            summary = ", ".join(f"{category}={count}" for category, count in category_counts.items())
            print(
                f"Skipped {len(unmatched)} WAV file(s) ({summary}); "
                f"see {report_path} for each filename and reason"
            )
            blockers = Counter(
                item["source_token"] for item in unmatched
                if item.get("category") == "phone_mapping" and item.get("source_token")
            )
            if blockers:
                top = ", ".join(
                    f"{token}×{count}" for token, count in blockers.most_common(20)
                )
                print(f"Most frequent unmapped source tokens (first blocker per file): {top}")
        if not recordings:
            raise SystemExit(f"no WAV files could be labelled; see {report_path} for phone-mapping/audio errors")
    elif arguments.corpus:
        recordings = corpus_recordings(arguments.corpus)
    elif arguments.labs and arguments.wavs:
        recordings = lab_recordings(arguments.labs, arguments.wavs)
    else:
        raise SystemExit("give --corpus, or --labs and --wavs; use --from-filenames with --wavs to generate labels")

    if arguments.songs:
        by_name = {recording[0]: recording for recording in recordings}
        names = arguments.songs.split(",")
        if missing := [name for name in names if name not in by_name]:
            raise SystemExit(f"no matching recording for: {', '.join(missing)}")
        recordings = [by_name[name] for name in names]
    if arguments.from_filenames:
        from .filename_labels import write_lab_files

        # Keep the generated HTS/vLabeler labels alongside the imported dataset.
        write_lab_files(recordings, Path(arguments.output) / "lab")
        print(f"generated {len(recordings)} LAB files from WAV filenames")
    mapping = dict(pair.split("=", 1) for pair in arguments.map)
    vowels = None
    if arguments.voice:
        from .common import find_language

        phone_sets = Voice.load(arguments.voice).acoustic.phone_sets
        phone_set = phone_sets[find_language(phone_sets, arguments.language)]
        vowels = {s for s, c in zip(phone_set.symbols, phone_set.categories) if c in ("vowel", "diphthong")}
    for line in import_labels(recordings, arguments.output, mapping, vowels):
        print(line)


def convert_dict(arguments) -> None:
    from .common import find_language
    from .data import convert_dictionary, read_dictionary

    phone_sets = Voice.load(arguments.voice).acoustic.phone_sets
    phone_set = phone_sets[find_language(phone_sets, arguments.language)]
    nuclei = {s for s, c in zip(phone_set.symbols, phone_set.categories) if c in ("vowel", "diphthong")}
    count, unknown = convert_dictionary(arguments.data, arguments.output, read_dictionary(arguments.source_dict), read_dictionary(arguments.target_dict), nuclei, arguments.coda_share)
    print(f"converted {count} utterances")
    if unknown:
        print("left unconverted:", ", ".join(f"{phone}×{n}" for phone, n in sorted(unknown.items(), key=lambda item: -item[1])))


def evaluate(arguments) -> None:
    _validate_pitch_options(arguments)
    device = torch.device(arguments.device)
    output = Path(arguments.out) if arguments.out else None
    report = {}
    timbre_diagnostics = {}
    f0_floor, f0_ceiling, f0_threshold = resolve_f0_profile(
        arguments.f0_profile,
        floor=arguments.f0_floor,
        ceiling=arguments.f0_ceiling,
        voicing_threshold=arguments.f0_voicing_threshold,
    )
    for index, path in enumerate(arguments.voice):
        voice = Voice.load(path).to(device)
        phrases = [
            phrase
            for utterance in load_dataset(
                arguments.data,
                arguments.language,
                read_phoneme_map(arguments.phoneme_map),
                f0_range=(f0_floor, f0_ceiling),
                f0_voicing_threshold=f0_threshold,
                f0_smoothing=arguments.f0_smoothing,
                f0_octave_correction=arguments.f0_octave_correction,
                f0_shift_semitones=arguments.f0_shift_semitones,
                f0_reference_cents=arguments.f0_reference_cents,
            )
            for phrase in make_phrases(utterance)
        ]
        _, validation = split(phrases, arguments.validation, arguments.seed)
        gain = level_gain(voice, phrases, device)
        apply_gain(phrases, gain)
        config = TrainConfig(
            models=tuple(arguments.models.split(",")),
            device=arguments.device,
            rap_languages=voice.files.rap_languages,
            pitch_melody_weight=arguments.pitch_melody_weight,
            pitch_motion_weight=arguments.pitch_motion_weight,
            pitch_smoothing_hz=arguments.pitch_smoothing_hz,
            brightness_weight=arguments.brightness_weight,
        )
        metrics = Trainer(voice, [], validation, config, log=lambda message: None).evaluate()
        name = voice.files.metadata.get(".name", path)
        print(f"{name} (gain {gain:.3f}): " + " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items())))
        if output is not None:
            label = f"{index}_{name}"
            report[label] = {"voice": str(path), "gain": gain, **metrics}
            reference_audio, rendered_audio, phrase_names = [], [], []
            for phrase in validation[:arguments.max_phrases]:
                directory = output / f"{phrase.utterance.name}_{phrase.start:.2f}"
                directory.mkdir(parents=True, exist_ok=True)
                recording = np.asarray(phrase.audio() / gain, dtype=np.float32)
                rendered = np.asarray(_render(voice, phrase, device) / gain, dtype=np.float32)
                soundfile.write(str(directory / "recording.wav"), recording, SAMPLE_RATE)
                soundfile.write(str(directory / f"{label}.wav"), rendered, SAMPLE_RATE)
                reference_audio.append(recording)
                rendered_audio.append(rendered)
                phrase_names.append(f"{phrase.utterance.name}_{phrase.start:.2f}")
            if reference_audio:
                timbre_diagnostics[label] = summarize_timbre(
                    reference_audio, rendered_audio, SAMPLE_RATE, phrase_names=phrase_names
                )
                delta = timbre_diagnostics[label]["delta"]
                print(
                    f"timbre {name}: centroid Δ={delta['spectral_centroid_hz']:.1f} Hz, "
                    f"low/mid Δ={delta['low_mid_ratio_db']:.2f} dB, "
                    f"high/mid Δ={delta['high_mid_ratio_db']:.2f} dB"
                )
    if output is not None:
        (output / "metrics.json").write_text(json.dumps(report, indent=2))
        (output / "timbre_diagnostics.json").write_text(json.dumps(timbre_diagnostics, indent=2, ensure_ascii=False))
        print(f"wrote {output}")


def _render(voice: Voice, phrase, device: torch.device):
    hz, _ = phrase.frame_f0()
    with torch.no_grad():
        log_f0 = torch.from_numpy(np.log(hz).astype(np.float32)).to(device)
        frames = voice.acoustic(phrase.timed_phonemes(), log_f0, mode_weights=mode_weights(voice.acoustic, phrase.utterance.style)).frames
        return voice.vocoder(frames).samples.cpu().numpy()


def synthesize_dataset(arguments) -> None:
    from .synthetic import render_dataset

    render_dataset(Voice.load(arguments.voice), arguments.output, arguments.phrases, arguments.language, arguments.seed, arguments.notes)


def inspect(arguments) -> None:
    voice = Voice.load(arguments.voice)
    files = voice.files
    print(f"{files.metadata.get('.name', '?')} ({'info voice' if files.info else 'voice.nofs'}) vocal modes: {', '.join(files.vocal_modes) or '-'}")
    for role in ROLES:
        model = voice.models[role]
        count = sum(parameter.numel() for parameter in model.parameters())
        print(f"  {role:9s} DNNI v{files.models[role].version} {count / 1e6:6.2f}M parameters")
    for phone_set in voice.acoustic.phone_sets:
        print(f"  {phone_set.name}: {' '.join(phone_set.symbols)}")


def package_svpk_command(arguments) -> None:
    output = build_svpk(
        voice=arguments.voice,
        license_file=arguments.license_file,
        profile=arguments.profile,
        background=arguments.background,
        output=arguments.output,
        product_name=arguments.name,
        vendor_name=arguments.vendor,
        version_string=arguments.version_string,
        version_number=arguments.version_number,
        depending_version_number=arguments.depending_version_number,
        header_preset=arguments.header_preset,
    )
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")
    print(f"SVPK diagnostic header preset: {arguments.header_preset}")
    print("Warning: synthetic header fields are not official signatures. This file remains diagnostic-only, is not installable, and is not suitable for distribution.")

def package_sfpk_command(arguments) -> None:
    output = build_sfpk(
        voice=arguments.voice,
        models_root=arguments.models_root,
        output=arguments.output,
        name=arguments.name,
        vendor=arguments.vendor,
        version=arguments.version,
        profile=arguments.profile,
        background=arguments.background,
    )
    print(f"wrote {output} ({output.stat().st_size:,} bytes)")
    print("Warning: this SFPK references DNNI files in the selected local models directory; it does not embed model dependencies.")
    print("Test importing, selecting, and rendering this voice in Synthesizer V Studio Flat before relying on it.")


def audit_data_command(arguments) -> None:
    report = audit_dataset(
        arguments.data,
        min_phone_recordings=arguments.min_phone_recordings,
        min_transition_recordings=arguments.min_transition_recordings,
        short_phone_ms=arguments.short_phone_ms,
        check_audio=arguments.check_audio,
    )
    summary = report["summary"]
    print(f"Dataset audit: {report['dataset']}")
    print(
        f"{summary['rows']} CSV rows; {summary['trainable_rows']} with usable labels and matching WAVs; "
        f"{summary['missing_wavs']} missing WAVs; {summary['invalid_rows']} invalid rows; "
        f"{summary['duplicate_names']} duplicate recording names"
    )
    if arguments.check_audio:
        print(
            f"Audio checks: {summary['unreadable_wavs']} unreadable WAVs; "
            f"{len(report['duration_mismatches'])} label/WAV duration mismatches"
        )
    print(
        f"Coverage: {summary['phone_types']} phone types; {summary['transition_types']} non-silence transitions; "
        f"{summary['short_phone_occurrences']} phone occurrences shorter than {arguments.short_phone_ms:g} ms"
    )
    if report["fatal_errors"]:
        for error in report["fatal_errors"]:
            print(f"ERROR: {error}")
    for label, key in (
        ("Missing WAVs", "missing_wavs"),
        ("Invalid rows", "invalid_rows"),
        ("Unreadable WAVs", "unreadable_wavs"),
        ("Duplicate names", "duplicate_names"),
    ):
        values = report[key]
        if values:
            print(f"{label} (showing up to 10):")
            for item in values[:10]:
                print(f"  {item}")
    if report["rare_phones"]:
        print(f"Underrepresented phones (fewer than {arguments.min_phone_recordings} recordings; showing up to 20):")
        for item in report["rare_phones"][:20]:
            print(f"  {item['phone']}: {item['recordings']} recording(s), {item['occurrences']} occurrence(s)")
    if report["rare_transitions"]:
        print(f"Underrepresented adjacent-phone transitions (fewer than {arguments.min_transition_recordings} recordings; showing up to 20):")
        for item in report["rare_transitions"][:20]:
            print(
                f"  {item['from']} -> {item['to']}: {item['recordings']} recording(s), "
                f"{item['occurrences']} occurrence(s)"
            )
    if report["short_phone_examples"]:
        print("Short phone examples (showing up to 10):")
        for item in report["short_phone_examples"][:10]:
            print(f"  {item['name']}: {item['phone']} at {item['duration_ms']:g} ms")
    if report["duration_mismatches"]:
        print("Label/WAV duration mismatches (showing up to 10):")
        for item in report["duration_mismatches"][:10]:
            print(
                f"  {item['name']}: labels {item['label_seconds']:.3f}s, "
                f"WAV {item['wav_seconds']:.3f}s"
            )
    if arguments.json_out:
        output = Path(arguments.json_out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote audit report: {output}")
    failed = bool(report["fatal_errors"]) or (
        arguments.strict and any(
            summary[key] for key in ("missing_wavs", "unreadable_wavs", "invalid_rows", "duplicate_names")
        )
    )
    if failed:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainsv")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("finetune", help="finetune a voice on a DiffSinger-style dataset")
    command.add_argument("--voice", required=True, help="voice.nofs or info.<version>.nofs")
    command.add_argument("--data", required=True, help="directory with wavs/ and transcriptions.csv")
    command.add_argument("--output", required=True, help="voice.nofs path, or databases/<name>/ directory for info voices")
    command.add_argument("--language", required=True, type=language_name, help="default phoneme language, e.g. japanese or ja")
    command.add_argument("--models", default="acoustic", help=f"comma-separated subset of {','.join(MODELS)}")
    command.add_argument("--params", choices=("all", "embeddings"), default="all", help="train every weight, or only the per-voice speaker/style/vocal-mode vectors")
    command.add_argument("--vocal-modes", help="comma-separated vocal modes of the finetuned voice (info voices): existing ones are kept, new ones start from the default style, others are dropped")
    command.add_argument("--steps", type=int, default=2000)
    command.add_argument(
        "--f0-profile",
        choices=tuple(F0_PROFILES),
        default="standard",
        help="WAV pitch tracking preset: standard, vocaloid, low, high, or wide; explicit F0 arguments override preset values",
    )
    command.add_argument("--f0-floor", type=float, help="override the selected F0 preset's minimum frequency in Hz")
    command.add_argument("--f0-ceiling", type=float, help="override the selected F0 preset's maximum frequency in Hz")
    command.add_argument("--f0-voicing-threshold", type=float, help="override the selected F0 preset's voicing threshold (0, 1]")
    command.add_argument("--f0-smoothing", type=int, choices=F0_SMOOTHING_OPTIONS, default=None, help="override the selected profile's median smoothing width: 0/off, 3/light, 5/medium, 7/strong (5 ms frames)")
    command.add_argument("--f0-octave-correction", action=argparse.BooleanOptionalAction, default=None, help="override the profile's isolated 1-2 octave tracker-spike correction")
    command.add_argument("--f0-shift-semitones", type=float, default=0.0, help="add this semitone offset to measured F0 targets, range -24 to +24")
    command.add_argument("--f0-reference-cents", type=float, default=0.0, help="fine-tune measured F0 reference by cents after cache loading, range -1200 to +1200")
    command.add_argument("--pitch-smoothing-hz", type=float, default=None, help="override the default low-pass cutoff for the Pitch model's smooth melody contour, 0 < cutoff < 100 Hz")
    command.add_argument("--pitch-melody-weight", type=float, default=None, help="override the default Pitch model smooth-melody loss weight")
    command.add_argument("--pitch-motion-weight", type=float, default=None, help="override the default Pitch model fast-motion/vibrato loss weight")
    command.add_argument("--pitch-model", "--pitch-base-model", dest="pitch_model", help="initialize Pitch from a compatible pretrained .dnni; target speaker embedding is preserved")
    command.add_argument("--lr", type=float, help="learning rate (default 2e-5 for all, 3e-3 for embeddings)")
    command.add_argument("--warmup", type=int, default=0, help="steps of linear learning-rate warmup")
    command.add_argument("--lr-end", type=float, help="cosine-decay the learning rate to this by the last step; constant without it")
    command.add_argument("--anchor", type=float, default=1.0, help="L2 pull towards the original weights")
    command.add_argument("--batch", type=int, default=1, help="phrases (audio: windows) averaged per step")
    command.add_argument("--consonant-weight", type=float, default=1.0, help="weight of consonant frames in the training audio loss (held-out scores stay unweighted)")
    command.add_argument("--transition-weight", type=float, default=1.0, help="extra weight for non-silence phoneme handoffs in the acoustic training loss; 1 disables it")
    command.add_argument("--transition-frames", type=int, default=2, help="frames on either side of each speech-phone boundary to emphasize (5 ms per frame)")
    command.add_argument("--brightness-weight", type=float, default=0.0, help="optional loss weight for matching target high/mid spectral balance; 0 disables it")
    command.add_argument("--validation", type=float, default=0.1, help="phrases held out: a fraction, or a count when 1 or more")
    command.add_argument("--evaluate-every", "--val-every", type=int, default=250)
    command.add_argument("--log-every", type=int, default=25)
    command.add_argument("--phoneme-map", help="JSON object mapping dataset phonemes to the voice's phonemes")
    command.add_argument("--keep-level", action="store_true", help="do not rescale the recordings to the base voice's level")
    command.add_argument("--level-reference", help="match the recordings' level to this voice instead of the starting one; use the stock base when continuing from a finetune, so the level cannot drift")
    command.add_argument("--gain", type=float, help="scale the recordings by this factor instead of measuring it")
    command.add_argument("--device", default=DEFAULT_DEVICE, help=f"default {DEFAULT_DEVICE}")
    command.add_argument("--name", help="singer name for the finetuned voice")
    command.add_argument("--version", help="version for info voices (the info.<version>.nofs suffix)")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--history", help="write validation metrics to this JSON file")
    command.add_argument("--run", help="run directory: log.jsonl of every logged step and validation, and state.pt saved at each validation for --resume")
    command.add_argument("--resume", action="store_true", help="continue from the --run directory's state.pt (starts afresh without one)")
    command.add_argument("--save-best", action="store_true", help="also save the voice whenever the held-out score improves")
    command.add_argument("--final-version", help="also save the last step's weights, unselected, as this version")
    command.set_defaults(handler=finetune)

    command = commands.add_parser("audit-data", help="audit WAV completeness, phone coverage, short phones, and adjacent-phone transitions")
    command.add_argument("--data", required=True, help="dataset directory with wavs/ and transcriptions.csv")
    command.add_argument("--min-phone-recordings", type=int, default=3, help="warn when a phone appears in fewer distinct WAVs")
    command.add_argument("--min-transition-recordings", type=int, default=2, help="warn when an adjacent non-silence phone pair appears in fewer distinct WAVs")
    command.add_argument("--short-phone-ms", type=float, default=20.0, help="flag non-silence phone labels shorter than this duration in milliseconds; 0 disables this check")
    command.add_argument("--check-audio", action="store_true", help="inspect WAV headers and compare label duration with WAV duration")
    command.add_argument("--json-out", help="optional path for a detailed JSON report; source dataset is never modified")
    command.add_argument("--strict", action="store_true", help="exit with status 2 when missing WAVs, invalid rows, duplicate names, or unreadable WAVs are found")
    command.set_defaults(handler=audit_data_command)

    command = commands.add_parser("import-lab", help="convert HTS/vLabeler .lab alignments and their wavs into a dataset")
    command.add_argument("--corpus", help="TrainV6 corpus: a directory of <name>.wav and <name>.lab (or wav/ and lab/), or a manifest.json")
    command.add_argument("--labs", help="directory of <name>.lab files (start end phoneme, 100 ns units)")
    command.add_argument("--wavs", help="directory of the matching .wav files, or source WAVs with --from-filenames")
    command.add_argument("--songs", help="comma-separated recording names to import; default every matching recording")
    command.add_argument("--output", required=True)
    command.add_argument("--map", nargs="*", default=[], help="phoneme replacements such as vf=cl")
    command.add_argument("--voice", help="voice phone set for word spans, or as a fallback phone table")
    command.add_argument("--language", default="japanese", type=language_name, help="language phone set for --voice")
    command.add_argument("--from-filenames", action="store_true", help="generate approximate LAB labels from space-separated phones in WAV filenames")
    command.add_argument("--phone-table", help="JSON phone table used to validate and split custom phoneme names")
    command.add_argument("--source-phoneme-system", choices=("auto", "vocaloid-mandarin", "xsampa", "synthv-xsampa", "sv1112-chinese", "sv1112-cantonese", "none"), default="auto", help="source filename phone convention: auto uses VOCALOID Mandarin for Mandarin; xsampa uses standard X-SAMPA; synthv-xsampa uses legacy SynthV Mandarin X-SAMPA labels; sv1112-chinese/cantonese map documented Chinese or Cantonese X-SAMPA/source aliases to SynthV 1.11.2 Old labels")
    command.set_defaults(handler=import_lab)

    command = commands.add_parser("convert-dict", help="re-spell a DiffSinger dataset with SV's dictionary for its language")
    command.add_argument("--data", required=True, help="source dataset (read only)")
    command.add_argument("--output", required=True, help="new dataset directory; its wavs/ links to the source")
    command.add_argument("--source-dict", required=True, help="the dataset's syllable dictionary, e.g. DiffSinger's opencpop-extension.txt")
    command.add_argument("--target-dict", required=True, help="SV's dictionary, e.g. clf-data/mandarin-xsampa-dict.txt")
    command.add_argument("--voice", required=True, help="voice whose phone categories mark the target nuclei")
    command.add_argument("--language", required=True, type=language_name)
    command.add_argument("--coda-share", type=float, default=0.3, help="fraction of a rhyme given to its coda (SV's duration models average about 0.3)")
    command.set_defaults(handler=convert_dict)

    command = commands.add_parser("evaluate", help="held-out metrics of one or more voices on a dataset, with renders to listen to")
    command.add_argument("--voice", required=True, action="append", help="repeat to compare several voices, e.g. the base and its finetune")
    command.add_argument("--data", required=True)
    command.add_argument("--language", required=True, type=language_name)
    command.add_argument("--models", default="acoustic")
    command.add_argument(
        "--f0-profile",
        choices=tuple(F0_PROFILES),
        default="standard",
        help="WAV pitch tracking preset; should match the profile used during finetuning",
    )
    command.add_argument("--f0-floor", type=float, help="override the selected F0 preset's minimum frequency in Hz")
    command.add_argument("--f0-ceiling", type=float, help="override the selected F0 preset's maximum frequency in Hz")
    command.add_argument("--f0-voicing-threshold", type=float, help="override the selected F0 preset's voicing threshold (0, 1]")
    command.add_argument("--f0-smoothing", type=int, choices=F0_SMOOTHING_OPTIONS, default=None, help="override the selected profile's median smoothing width in 5 ms frames")
    command.add_argument("--f0-octave-correction", action=argparse.BooleanOptionalAction, default=None, help="override the selected profile's isolated octave-spike correction")
    command.add_argument("--f0-shift-semitones", type=float, default=0.0, help="semitone offset added to measured F0 targets")
    command.add_argument("--f0-reference-cents", type=float, default=0.0, help="fine-tune measured F0 reference in cents")
    command.add_argument("--pitch-smoothing-hz", type=float, default=None, help="override the default low-pass cutoff for the Pitch model's smooth melody contour")
    command.add_argument("--pitch-melody-weight", type=float, default=None, help="override the default Pitch model smooth-melody loss weight")
    command.add_argument("--pitch-motion-weight", type=float, default=None, help="override the default Pitch model fast-motion/vibrato loss weight")
    command.add_argument("--brightness-weight", type=float, default=0.0, help="brightness-balance loss weight (0 disables)")
    command.add_argument("--validation", type=float, default=0.1)
    command.add_argument("--phoneme-map")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--device", default=DEFAULT_DEVICE, help=f"default {DEFAULT_DEVICE}")
    command.add_argument("--out", help="write metrics.json and, per held-out phrase, the recording next to each voice's render (recorded timing and pitch)")
    command.add_argument("--max-phrases", type=int, default=4, help="held-out phrases rendered to --out")
    command.set_defaults(handler=evaluate)

    command = commands.add_parser("synthesize-dataset", help="render a labelled DiffSinger-style dataset with a voice")
    command.add_argument("--voice", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--phrases", type=int, default=40)
    command.add_argument("--language", default="japanese", type=language_name)
    command.add_argument("--notes", type=int, default=8)
    command.add_argument("--seed", type=int, default=0)
    command.set_defaults(handler=synthesize_dataset)

    command = commands.add_parser("inspect", help="summarise a voice's models and phone sets")
    command.add_argument("--voice", required=True)
    command.set_defaults(handler=inspect)
    command = commands.add_parser("package-svpk", help="build a diagnostic-only SVPK container with a selectable synthetic header field (not installable in Sharp)")
    command.add_argument("--voice", required=True, help="trained voice.nofs file")
    command.add_argument("--license", dest="license_file", help="optional license text file to include")
    command.add_argument("--profile", required=True, help="voicebank avatar PNG")
    command.add_argument("--background", required=True, help="voicebank background PNG")
    command.add_argument("--output", required=True, help="output .svpk path")
    command.add_argument("--name", required=True, help="product / singer name")
    command.add_argument("--vendor", required=True, help="publisher or author name")
    command.add_argument("--version-string", default="101")
    command.add_argument("--version-number", type=int, default=101)
    command.add_argument("--depending-version-number", type=int, default=0)
    command.add_argument(
        "--header-preset",
        choices=("zero", "random", "sha256-chain", "aes256-cbc-test"),
        default="zero",
        help="diagnostic-only 256-byte field generator; none of these presets recreates the official signing/authentication data",
    )
    command.set_defaults(handler=package_svpk_command)

    command = commands.add_parser("package-sfpk", help="build an experimental Flat SFPK that references local DNNI dependencies")
    command.add_argument("--voice", required=True, help="trained binary voice.nofs or existing info.nofs JSON")
    command.add_argument("--models-root", required=True, help="existing Synthesizer V models directory containing/written with <kind>/<model-id>.dnni")
    command.add_argument("--output", required=True, help="output .sfpk path")
    command.add_argument("--name", help="optional singer name override")
    command.add_argument("--vendor", help="optional vendor override")
    command.add_argument("--version", help="optional version override, for example 101 or 106b1")
    command.add_argument("--profile", help="optional avatar PNG to include as profile.png in the SFPK")
    command.add_argument("--background", help="optional background PNG to include as background.png in the SFPK")
    command.set_defaults(handler=package_sfpk_command)

    for command in commands.choices.values():
        command.add_argument("--config", action="append", default=[], metavar="FILE", help="JSON file of option values; repeat for several, later files win and command-line options override them")
    return parser


def language_name(value: str) -> str:
    return LANGUAGE_CODES.get(value, value)


def apply_configs(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    reader = argparse.ArgumentParser(add_help=False)
    reader.add_argument("--config", action="append", default=[])
    paths = reader.parse_known_args(argv)[0].config
    if not paths:
        return
    commands = parser._subparsers._group_actions[0].choices
    name = next(argument for argument in argv if argument in commands)
    command = commands[name]
    values = {}
    for path in paths:
        loaded = json.loads(Path(path).read_text("utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit(f"{path}: a config is a JSON object of option values")
        values.update({key.replace("-", "_"): value for key, value in loaded.items()})
    known = {}
    for action in command._actions:
        if action.option_strings and action.dest not in ("help", "config"):
            known[action.dest] = action
            known.update({option.lstrip("-").replace("-", "_"): action for option in action.option_strings})
    if unknown := sorted(set(values) - known.keys()):
        raise SystemExit(f"{name} has no such options: {', '.join(unknown)}")
    for key, value in values.items():
        known[key].required = False
        command.set_defaults(**{known[key].dest: value})


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = parser()
    apply_configs(root, argv)
    arguments = root.parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
