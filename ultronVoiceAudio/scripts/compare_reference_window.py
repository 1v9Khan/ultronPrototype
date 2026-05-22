"""A/B compare XTTS-v2 with Coqui-default vs extended reference window.

Round-9 (2026-05-20) investigation. The production XTTS server has
been calling ``model.get_conditioning_latents(audio_path=...)`` with
no overrides -- meaning Coqui's library defaults
``gpt_cond_len=6``, ``gpt_cond_chunk_len=6``, ``max_ref_length=30``
apply. Even though we hand it the full 3-minute
``Ultron_vocals_mono_v1.wav``, only ~6 s of audio reaches the GPT
prosody encoder and ~30 s reaches the HiFi-GAN speaker encoder.

This script loads XTTS once, computes TWO sets of conditioning
latents from the same reference WAV:

* ``baseline`` -- ``gpt_cond_len=6``, ``gpt_cond_chunk_len=6``,
  ``max_ref_length=30`` (Coqui library defaults).
* ``extended`` -- ``gpt_cond_len=30``, ``gpt_cond_chunk_len=6``,
  ``max_ref_length=60`` (new round-9 production defaults).

Then synthesises the same fixed set of utterances through each set
of latents and writes to two sibling output folders so the user can
A/B listen. No v3 filter is applied -- the comparison should be on
the raw XTTS cloning quality; the user can apply the v3 filter
afterwards if they like the baseline cloning improvement.

Run from inside the XTTS isolated venv:

    C:\\STC\\ultronPrototype\\ultronVoiceAudio\\.venv-xtts\\Scripts\\python.exe ^
        C:\\STC\\ultronPrototype\\ultronVoiceAudio\\scripts\\compare_reference_window.py

Outputs land at::

    ultronVoiceAudio/compare_reference_window/baseline/*.wav
    ultronVoiceAudio/compare_reference_window/extended/*.wav

Both folders contain the same set of filenames so it's easy to
A/B in any audio player.
"""

from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path

# Workaround for Windows env vars pointing at non-existent D:\ caches.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent  # C:/STC/ultronPrototype/ultronVoiceAudio
os.environ["TORCH_HOME"] = str(PROJECT / ".torch_cache")
os.environ["HF_HOME"] = str(PROJECT / ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(PROJECT / ".hf_cache")
os.environ["COQUI_TOS_AGREED"] = "1"
(PROJECT / ".torch_cache").mkdir(exist_ok=True)
(PROJECT / ".hf_cache").mkdir(exist_ok=True)

REFERENCE_WAV = PROJECT / "kokoro training audio" / "Ultron_vocals_mono_v1.wav"
OUTPUT_ROOT = PROJECT / "compare_reference_window"
BASELINE_DIR = OUTPUT_ROOT / "baseline"
EXTENDED_DIR = OUTPUT_ROOT / "extended"

# Fixed sample list. Chosen to exercise:
#   - short interjections (the most-common production utterance shape)
#   - typical statements
#   - longer composed sentences with multiple clauses
#   - sentences with technical vocabulary (where XTTS prosody often breaks)
#   - lines that benefit from authoritative cadence (Ultron flavor)
SAMPLES: list[tuple[str, str]] = [
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

# Conditioning-latent presets. The whole point of this script.
BASELINE_PARAMS = dict(gpt_cond_len=6, gpt_cond_chunk_len=6, max_ref_length=30)
EXTENDED_PARAMS = dict(gpt_cond_len=30, gpt_cond_chunk_len=6, max_ref_length=60)


def _save_pcm_as_wav(out_path: Path, pcm_chunks: list, sample_rate: int) -> None:
    """Concatenate the streaming float32 tensors and write a mono
    16-bit PCM WAV. Matches the production server's PCM-i16
    encoding so the comparison is apples-to-apples."""
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


def _synthesize_one(model, text: str, gpt_latent, speaker_emb) -> list:
    """Drain the inference_stream generator into a list of tensor
    chunks. The same code path the production server uses."""
    chunks = []
    for chunk in model.inference_stream(
        text=text,
        language="en",
        gpt_cond_latent=gpt_latent,
        speaker_embedding=speaker_emb,
        stream_chunk_size=20,
        temperature=0.65,  # round-9 production default
        speed=1.0,         # native; user evaluates raw XTTS prosody before v3 filter speed adjust
    ):
        chunks.append(chunk)
    return chunks


def main() -> int:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    if not REFERENCE_WAV.is_file():
        print(f"ERROR: reference audio missing: {REFERENCE_WAV}")
        return 1
    print(f"reference: {REFERENCE_WAV} ({REFERENCE_WAV.stat().st_size / 1e6:.1f} MB)")

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)

    # Late imports so env vars are set first.
    from TTS.utils.manage import ModelManager
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    print("\n=== Loading XTTS v2 ===")
    t0 = time.monotonic()
    manager = ModelManager()
    model_path, _, _ = manager.download_model(
        "tts_models/multilingual/multi-dataset/xtts_v2"
    )
    config = XttsConfig()
    config.load_json(str(Path(model_path) / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(model_path),
        eval=True,
    )
    if torch.cuda.is_available():
        model.cuda()
    sample_rate = config.audio.output_sample_rate
    print(f"XTTS loaded in {time.monotonic() - t0:.1f}s, sample_rate={sample_rate}")
    if torch.cuda.is_available():
        print(f"VRAM after model load: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    print("\n=== Computing conditioning latents (baseline 6/6/30) ===")
    t0 = time.monotonic()
    gpt_latent_b, speaker_emb_b = model.get_conditioning_latents(
        audio_path=str(REFERENCE_WAV), **BASELINE_PARAMS
    )
    print(f"  baseline latents computed in {time.monotonic() - t0:.2f}s")
    print(f"  gpt_latent shape:    {tuple(gpt_latent_b.shape)}")
    print(f"  speaker_emb shape:   {tuple(speaker_emb_b.shape)}")

    print("\n=== Computing conditioning latents (extended 30/6/60) ===")
    t0 = time.monotonic()
    gpt_latent_e, speaker_emb_e = model.get_conditioning_latents(
        audio_path=str(REFERENCE_WAV), **EXTENDED_PARAMS
    )
    print(f"  extended latents computed in {time.monotonic() - t0:.2f}s")
    print(f"  gpt_latent shape:    {tuple(gpt_latent_e.shape)}")
    print(f"  speaker_emb shape:   {tuple(speaker_emb_e.shape)}")

    # Warmup pass so the first real sample isn't slowed by JIT.
    print("\n=== Warmup pass ===")
    t0 = time.monotonic()
    _ = _synthesize_one(model, "Hello.", gpt_latent_b, speaker_emb_b)
    print(f"  warmup in {time.monotonic() - t0:.2f}s")

    print(f"\n=== Generating {len(SAMPLES)} samples through each preset ===")
    print(f"  baseline -> {BASELINE_DIR}")
    print(f"  extended -> {EXTENDED_DIR}")
    timings: list[tuple[str, str, float]] = []
    for tag, text in SAMPLES:
        for preset_name, out_dir, gpt_l, spk_e in [
            ("baseline", BASELINE_DIR, gpt_latent_b, speaker_emb_b),
            ("extended", EXTENDED_DIR, gpt_latent_e, speaker_emb_e),
        ]:
            out = out_dir / f"{tag}.wav"
            print(f"  [{preset_name}] [{tag}] '{text[:55]}...'" if len(text) > 55
                  else f"  [{preset_name}] [{tag}] '{text}'")
            t0 = time.monotonic()
            chunks = _synthesize_one(model, text, gpt_l, spk_e)
            synth_s = time.monotonic() - t0
            _save_pcm_as_wav(out, chunks, sample_rate)
            timings.append((preset_name, tag, synth_s))
            print(f"    -> {out.relative_to(PROJECT)} ({synth_s:.2f}s)")

    # Summary table.
    print("\n" + "=" * 60)
    print("Per-sample synthesis time (seconds)")
    print(f"{'tag':<22}{'baseline':>12}{'extended':>12}")
    print("-" * 60)
    by_tag: dict[str, dict[str, float]] = {}
    for preset_name, tag, s in timings:
        by_tag.setdefault(tag, {})[preset_name] = s
    for tag, d in by_tag.items():
        print(f"{tag:<22}{d.get('baseline', 0):>12.2f}{d.get('extended', 0):>12.2f}")
    total_b = sum(s for n, _, s in timings if n == "baseline")
    total_e = sum(s for n, _, s in timings if n == "extended")
    print("-" * 60)
    print(f"{'TOTAL':<22}{total_b:>12.2f}{total_e:>12.2f}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nPeak VRAM: {peak:.0f} MB")

    print("\nDone. A/B compare:")
    print(f"  {BASELINE_DIR}")
    print(f"  {EXTENDED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
