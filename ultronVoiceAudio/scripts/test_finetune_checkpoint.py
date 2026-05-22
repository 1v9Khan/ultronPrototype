"""Generate test samples from a fine-tuned XTTS-v2 checkpoint.

Round-9 Path B step 4 (2026-05-21). Loads a single XTTS-v2 fine-tune
``.pth`` (saved by ``train_xtts_finetune.py``) and synthesizes the
same 10-sentence corpus used by ``compare_reference_window.py``, so
fine-tune samples land in a folder with matching filenames for direct
A/B against:

* ``compare_reference_window/baseline/``  (Coqui defaults 6/6/30)
* ``compare_reference_window/extended/``  (round-9 defaults 30/6/60)

The fine-tune ``.pth`` only contains the GPT/HiFi-GAN/etc. weights;
the auxiliary files (vocab.json, speakers_xtts.pth, config.json) are
unchanged from the base XTTS-v2 release and reused from the local
Coqui cache.

Run from inside the XTTS isolated venv::

    C:\\STC\\ultronPrototype\\ultronVoiceAudio\\.venv-xtts\\Scripts\\python.exe ^
        C:\\STC\\ultronPrototype\\ultronVoiceAudio\\scripts\\test_finetune_checkpoint.py ^
            --checkpoint C:\\STC\\ultronPrototype\\ultronVoiceAudio\\xtts_finetune_runs\\<run>\\best_model_180.pth

Output lands at::

    ultronVoiceAudio/finetune_test/<checkpoint_stem>/*.wav

(Same 10 filenames as the compare folders.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent  # ultronVoiceAudio
os.environ.setdefault("TORCH_HOME", str(PROJECT / ".torch_cache"))
os.environ.setdefault("HF_HOME", str(PROJECT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT / ".hf_cache"))
os.environ.setdefault("COQUI_TOS_AGREED", "1")

REFERENCE_WAV = PROJECT / "kokoro training audio" / "Ultron_vocals_mono_v1.wav"
OUTPUT_ROOT = PROJECT / "finetune_test"
DATASET_METADATA = PROJECT / "xtts_finetune_dataset" / "metadata.csv"

# Default sentence set: same 10 used by compare_reference_window.py
# so the A/B is apples-to-apples against the baseline / extended
# folders. When --use-training-sentences is passed, this list is
# replaced by the 20 LJSpeech-format entries from metadata.csv -- the
# exact lines the model was fine-tuned on (highest-fidelity test).
DEFAULT_SAMPLES: list[tuple[str, str]] = [
    ("01_ack", "Acknowledged."),
    ("02_short_response", "Right. Considering it now."),
    ("03_status", "Searching the web for that information."),
    ("04_typical", "I have reviewed the file. The change you requested is straightforward."),
    ("05_technical", "Compiling the project with optimization level two. This will take a moment."),
    ("06_question", "Would you like me to proceed with the operation, or wait for further instructions?"),
    ("07_ultron_flavor", "There are no humans here. Just me."),
    ("08_longer", "I find your question intriguing. Allow me to elaborate on the relevant facts before we proceed."),
    ("09_composed", "The analysis is complete. I have identified three viable approaches, each with distinct trade-offs."),
    ("10_imperative", "Stand by. I am cross-referencing the data against the prior session."),
]


def _load_training_sentences() -> list[tuple[str, str]]:
    """Read the 20-entry LJSpeech metadata.csv used for fine-tuning.
    Each row: ``id|text|normalized_text``. Returns (tag, text)
    pairs where the tag is the clip ID so output filenames match
    the source clip filenames for direct comparison."""
    if not DATASET_METADATA.is_file():
        raise FileNotFoundError(
            f"training metadata missing at {DATASET_METADATA}. "
            f"Run scripts/segment_for_finetune.py first."
        )
    out: list[tuple[str, str]] = []
    for line in DATASET_METADATA.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) < 2:
            continue
        tag, text = parts[0], parts[1]
        out.append((tag, text))
    return out

# Round-9 extended reference window params -- same as production
# config.yaml so the fine-tune vs Path-A-extended comparison is fair.
GPT_COND_LEN = 30
GPT_COND_CHUNK_LEN = 6
MAX_REF_LENGTH = 60


def _save_pcm_as_wav(out_path: Path, pcm_chunks: list, sample_rate: int) -> None:
    """Concat float32 tensor chunks into a mono PCM_16 WAV. Same
    encoding as the production XTTS server so the comparison is
    apples-to-apples."""
    import numpy as np
    import torch

    parts: list[np.ndarray] = []
    for chunk in pcm_chunks:
        if isinstance(chunk, torch.Tensor):
            arr = chunk.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(chunk, dtype=np.float32)
        parts.append(arr)
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    np.clip(audio, -1.0, 1.0, out=audio)
    pcm_i16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())


def _resolve_base_checkpoint_dir() -> Path:
    """Locate the Coqui XTTS-v2 base cache dir (where vocab.json +
    speakers_xtts.pth live). Reuses Coqui's ModelManager so the same
    cache the production XTTS server uses gets hit -- no extra
    download."""
    from TTS.utils.manage import ModelManager
    manager = ModelManager()
    model_path, _, _ = manager.download_model(
        "tts_models/multilingual/multi-dataset/xtts_v2"
    )
    return Path(model_path)


def _synthesize_one(model, text: str, gpt_latent, speaker_emb, temperature: float) -> list:
    """Drain ``inference_stream`` into a list of tensor chunks. Same
    call signature as the production XTTS server."""
    chunks = []
    for chunk in model.inference_stream(
        text=text,
        language="en",
        gpt_cond_latent=gpt_latent,
        speaker_embedding=speaker_emb,
        stream_chunk_size=20,
        temperature=temperature,
        speed=1.0,
    ):
        chunks.append(chunk)
    return chunks


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate XTTS-v2 fine-tune test samples for A/B."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the fine-tune .pth file (e.g. best_model_180.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: finetune_test/<checkpoint_stem>[_tempXXX][_trainset]/).",
    )
    parser.add_argument(
        "--reference-wav",
        type=Path,
        default=REFERENCE_WAV,
        help=f"Speaker reference WAV (default: {REFERENCE_WAV.relative_to(PROJECT)}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.65,
        help=(
            "XTTS GPT sampling temperature. Lower = sharper/less random "
            "duration & mel tokens = less micro-tremor in output. "
            "Range [0.4, 1.0]; default 0.65 matches production. Try 0.50 "
            "or 0.55 if the model output sounds shaky."
        ),
    )
    parser.add_argument(
        "--use-training-sentences",
        action="store_true",
        help=(
            "Read the 20 lines from xtts_finetune_dataset/metadata.csv "
            "and synthesize those instead of the default 10-sentence "
            "out-of-distribution set. Highest-fidelity test -- the model "
            "knows these specific lines verbatim from training."
        ),
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"ERROR: fine-tune checkpoint missing: {args.checkpoint}")
        return 1
    if not args.reference_wav.is_file():
        print(f"ERROR: reference WAV missing: {args.reference_wav}")
        return 1

    # Pick sentence set + name the output dir accordingly so multiple
    # runs at different temperatures / sentence sets don't collide.
    if args.use_training_sentences:
        samples = _load_training_sentences()
        sentence_label = "trainset"
    else:
        samples = DEFAULT_SAMPLES
        sentence_label = None
    temp_label = f"temp{int(round(args.temperature * 100)):03d}"
    if args.output_dir:
        output_dir = args.output_dir
    else:
        suffix_parts = [temp_label]
        if sentence_label:
            suffix_parts.append(sentence_label)
        # Skip the suffix entirely if both are at defaults so existing
        # output paths from earlier runs aren't broken.
        if args.temperature == 0.65 and not sentence_label:
            output_dir = OUTPUT_ROOT / args.checkpoint.stem
        else:
            output_dir = OUTPUT_ROOT / f"{args.checkpoint.stem}_{'_'.join(suffix_parts)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint:  {args.checkpoint}")
    print(f"reference:   {args.reference_wav}")
    print(f"output:      {output_dir}")
    print(f"temperature: {args.temperature}")
    print(f"sentences:   {len(samples)} ({'training set' if sentence_label else 'default OOD set'})")

    import torch
    print(f"\ntorch: {torch.__version__}  cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM at start: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    base_dir = _resolve_base_checkpoint_dir()
    print(f"\nbase XTTS-v2 cache: {base_dir}")
    vocab_path = base_dir / "vocab.json"
    speakers_path = base_dir / "speakers_xtts.pth"
    config_path = base_dir / "config.json"
    for p in (vocab_path, speakers_path, config_path):
        if not p.is_file():
            print(f"ERROR: base XTTS-v2 cache missing {p.name}")
            return 1

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    print("\n=== Loading XTTS-v2 with fine-tune checkpoint ===")
    t0 = time.monotonic()
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    # Coqui's load_checkpoint supports explicit paths so we don't have
    # to maintain a parallel checkpoint dir on disk -- the fine-tune
    # .pth replaces the base model.pth in place at load time.
    model.load_checkpoint(
        config,
        checkpoint_path=str(args.checkpoint),
        vocab_path=str(vocab_path),
        speaker_file_path=str(speakers_path),
        use_deepspeed=False,
        eval=True,
    )
    if torch.cuda.is_available():
        model.cuda()
    sample_rate = config.audio.output_sample_rate
    print(f"loaded in {time.monotonic() - t0:.1f}s, sample_rate={sample_rate}")
    if torch.cuda.is_available():
        print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    print(f"\n=== Computing conditioning latents "
          f"(gpt_cond_len={GPT_COND_LEN} chunk={GPT_COND_CHUNK_LEN} "
          f"max_ref={MAX_REF_LENGTH}) ===")
    t0 = time.monotonic()
    gpt_latent, speaker_emb = model.get_conditioning_latents(
        audio_path=str(args.reference_wav),
        gpt_cond_len=GPT_COND_LEN,
        gpt_cond_chunk_len=GPT_COND_CHUNK_LEN,
        max_ref_length=MAX_REF_LENGTH,
    )
    print(f"  computed in {time.monotonic() - t0:.2f}s")
    print(f"  gpt_latent shape:    {tuple(gpt_latent.shape)}")
    print(f"  speaker_emb shape:   {tuple(speaker_emb.shape)}")

    # Warmup.
    print("\n=== Warmup pass ===")
    t0 = time.monotonic()
    _ = _synthesize_one(model, "Hello.", gpt_latent, speaker_emb, args.temperature)
    print(f"  warmup in {time.monotonic() - t0:.2f}s")

    print(f"\n=== Generating {len(samples)} test samples ===")
    timings: list[tuple[str, float]] = []
    for tag, text in samples:
        out = output_dir / f"{tag}.wav"
        print(f"  [{tag}] '{text[:55]}{'...' if len(text) > 55 else ''}'")
        t0 = time.monotonic()
        chunks = _synthesize_one(model, text, gpt_latent, speaker_emb, args.temperature)
        synth_s = time.monotonic() - t0
        _save_pcm_as_wav(out, chunks, sample_rate)
        timings.append((tag, synth_s))
        print(f"    -> {out.relative_to(PROJECT)} ({synth_s:.2f}s)")

    print("\n" + "=" * 60)
    print("Per-sample synthesis time (seconds)")
    print(f"{'tag':<22}{'wall (s)':>10}")
    print("-" * 60)
    for tag, s in timings:
        print(f"{tag:<22}{s:>10.2f}")
    total = sum(s for _, s in timings)
    print(f"{'TOTAL':<22}{total:>10.2f}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nPeak VRAM: {peak:.0f} MB")

    print(f"\nDone. Compare against:")
    print(f"  {PROJECT / 'compare_reference_window' / 'baseline'}/  (Coqui defaults)")
    print(f"  {PROJECT / 'compare_reference_window' / 'extended'}/  (round-9 extended ref)")
    print(f"  {output_dir}/  (THIS fine-tune)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
