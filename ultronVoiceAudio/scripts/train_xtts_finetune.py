"""XTTS-v2 GPT fine-tune driver for the Ultron voice corpus.

Round-9 Path B step 3 (2026-05-20). Loads the LJSpeech-format
dataset built by ``segment_for_finetune.py`` and fine-tunes the
GPT module of XTTS-v2 against it. The DVAE and speaker encoder
checkpoints stay frozen at the published Coqui weights -- we are
adapting the prosody/timbre model that maps text + speaker
embedding to audio tokens.

Runs inside the isolated ``.venv-xtts`` venv (Coqui TTS only lives
there)::

    C:\\STC\\ultronPrototype\\ultronVoiceAudio\\.venv-xtts\\Scripts\\python.exe ^
        C:\\STC\\ultronPrototype\\ultronVoiceAudio\\scripts\\train_xtts_finetune.py

Output runs land under ``ultronVoiceAudio/xtts_finetune_runs/<run_name>/``
with checkpoints every ``--save-every-step`` steps. The best
checkpoint can later be loaded by the XTTS server in place of the
stock checkpoint.

Note on corpus size: a 3-minute reference clip yields roughly
25-35 training segments of 5-15 s each. This is well below the
ideal corpus size for XTTS-v2 fine-tuning (the Coqui community
sweet spot is ~30 min - 2 hr of clean speech). Expect prosodic
overfitting and limited generalisation to text far outside the
training distribution. A good "first pass" check is whether the
fine-tuned model preserves the Ultron timbre while the base
model's prosody bleeds through enough to handle unseen text.

The script is intentionally a thin wrapper -- the real heavy
lifting lives in Coqui's ``Trainer`` + ``GPTTrainer`` machinery.
First-run failures are expected: the Coqui XTTS-v2 fine-tune
recipe is finicky about disk paths, CUDA versions, and the exact
shape of constituent files in the cached model dir. Surface any
``FileNotFoundError`` clearly and ASK the user before patching.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent           # ultronVoiceAudio
ROOT = PROJECT.parent           # ultronPrototype

# Pin Coqui caches to the workshop dir so we never write into the
# user's roaming profile or D:\ (which doesn't exist on this machine).
os.environ.setdefault("TORCH_HOME", str(PROJECT / ".torch_cache"))
os.environ.setdefault("HF_HOME", str(PROJECT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT / ".hf_cache"))
os.environ.setdefault("COQUI_TOS_AGREED", "1")
# CUDA allocator: reduces fragmentation OOM on long training runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
(PROJECT / ".torch_cache").mkdir(exist_ok=True)
(PROJECT / ".hf_cache").mkdir(exist_ok=True)

DEFAULT_DATASET_DIR = PROJECT / "xtts_finetune_dataset"
DEFAULT_OUTPUT_DIR = PROJECT / "xtts_finetune_runs"
DEFAULT_REFERENCE = PROJECT / "kokoro training audio" / "Ultron_vocals_mono_v1.wav"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XTTS-v2 GPT fine-tune driver.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing metadata.csv + wavs/ (LJSpeech format).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write training runs / checkpoints.",
    )
    parser.add_argument(
        "--run-name",
        default=time.strftime("ultron_finetune_%Y%m%d_%H%M%S"),
        help="Subdirectory name under --out-path for this training run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs. With a small corpus the model "
             "overfits quickly; start with 10 and listen to the test "
             "samples at each checkpoint to find the sweet spot.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Per-device batch size. XTTS-v2 GPT trainer is memory-"
             "hungry; 3 fits on a 12 GB 4070 Ti with the rest of "
             "the orchestrator paused.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=84,
        help="Gradient accumulation steps. effective_batch = batch_size "
             "* grad_accum (default 3*84=252). Coqui's recipe uses 252 "
             "for XTTS-v2 to stabilise short-segment training.",
    )
    parser.add_argument(
        "--save-every-step",
        type=int,
        default=500,
        help="Checkpoint every N steps. With 25-35 segments and "
             "effective batch 252, an epoch is well under 1 step, so "
             "for small corpora bump this LOWER (e.g. 50) to actually "
             "see checkpoints during a run.",
    )
    parser.add_argument(
        "--save-n-checkpoints",
        type=int,
        default=5,
        help="Max periodic checkpoints to retain. Older ones get "
             "rotated out. Each XTTS-v2 checkpoint is ~5.6 GB so "
             "watch disk space; on a 20-clip corpus 5 is usually "
             "plenty.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="AdamW base learning rate. Coqui recipe default 5e-6.",
    )
    parser.add_argument(
        "--reference-wav",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Speaker reference WAV used for inference at "
             "checkpoint-time test sentences (NOT training).",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=200,
        help="Cap on token length per training sample; mirrors Coqui default.",
    )
    parser.add_argument(
        "--max-wav-length",
        type=int,
        default=255995,
        help="Max audio length in samples (~11.6s at 22050 Hz); Coqui default.",
    )
    parser.add_argument(
        "--restore-path",
        type=Path,
        default=None,
        help="Resume from a prior checkpoint (best.pth / latest.pth) "
             "instead of starting fresh.",
    )
    return parser


def _resolve_base_checkpoint_dir() -> Path:
    """Locate the cached Coqui XTTS-v2 base checkpoint directory.

    The first call downloads (~2 GB) via Coqui's ``ModelManager``;
    later calls are instant.
    """
    from TTS.utils.manage import ModelManager
    manager = ModelManager()
    model_path, _, _ = manager.download_model(
        "tts_models/multilingual/multi-dataset/xtts_v2"
    )
    return Path(model_path)


def _verify_constituent_files(model_dir: Path) -> dict:
    """Return a dict of the constituent files the GPT trainer needs.
    Raises FileNotFoundError early with a clear message if any are
    missing -- the failure mode is otherwise an opaque KeyError deep
    in the trainer."""
    candidates = {
        "DVAE_CHECKPOINT": model_dir / "dvae.pth",
        "MEL_NORM_FILE": model_dir / "mel_stats.pth",
        "TOKENIZER_FILE": model_dir / "vocab.json",
        "XTTS_CHECKPOINT": model_dir / "model.pth",
    }
    missing = [k for k, p in candidates.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "XTTS-v2 base checkpoint dir is missing required files: "
            + ", ".join(missing)
            + f". Look under {model_dir}; you may need to re-run "
            "scripts/download_models.py or download the XTTS-v2 base "
            "model via Coqui's ModelManager."
        )
    return {k: str(p) for k, p in candidates.items()}


def main(argv=None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        print(f"ERROR: dataset dir missing: {args.dataset}. "
              f"Run scripts/segment_for_finetune.py first.")
        return 1
    if not (args.dataset / "metadata.csv").is_file():
        print(f"ERROR: dataset metadata.csv missing under {args.dataset}.")
        return 1
    if not args.reference_wav.is_file():
        print(f"ERROR: reference WAV missing: {args.reference_wav}.")
        return 1

    args.out_path.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_path / args.run_name
    print(f"dataset:     {args.dataset}")
    print(f"out_path:    {args.out_path}")
    print(f"run_name:    {args.run_name}")
    print(f"run_dir:     {run_dir}")

    print("\nlocating Coqui XTTS-v2 base checkpoint...")
    base_dir = _resolve_base_checkpoint_dir()
    print(f"  base dir: {base_dir}")
    files = _verify_constituent_files(base_dir)
    for k, v in files.items():
        print(f"  {k}: {v}")

    # Late imports so missing constituent file errors above surface
    # cleanly before paying the import cost of trainer + GPTTrainer.
    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.xtts_config import XttsConfig  # noqa: F401  (round-trip sanity)
    # Coqui 0.27.5 keeps the trainer-side classes in gpt_trainer, but
    # ``XttsAudioConfig`` lives with the model definition itself.
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs,
        GPTTrainer,
        GPTTrainerConfig,
    )
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples

    dataset_cfg = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="ultron_reference",
        path=str(args.dataset),
        meta_file_train="metadata.csv",
        language="en",
    )

    audio_cfg = XttsAudioConfig(
        sample_rate=22050,        # matches what segment_for_finetune.py writes
        dvae_sample_rate=22050,
        output_sample_rate=24000, # XTTS-v2 native output SR
    )

    # 2026-05-20: the published XTTS-v2 base checkpoint uses the
    # LEGACY audio-token vocab:
    #   gpt_num_audio_tokens=1026 (mel_embedding/mel_head shape [1026, 1024])
    #   gpt_start_audio_token=1024
    #   gpt_stop_audio_token=1025
    #   gpt_use_perceiver_resampler=True (the original XTTS-v2 architecture)
    # Coqui 0.27.5 defaults all four to the NEW vocab (8194/8192/8193/
    # False) which is incompatible with the checkpoint. If even one is
    # wrong the result is either (a) a size_mismatch on load_state_dict
    # or (b) a CUDA device-side assert when the data pipeline emits
    # token IDs outside the embedding's [0, 1026) range. All four must
    # be set explicitly. These are XttsArgs fields, inherited by
    # GPTArgs -- they accept the kwargs through inheritance.
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=args.max_wav_length,
        max_text_length=args.max_text_length,
        mel_norm_file=files["MEL_NORM_FILE"],
        dvae_checkpoint=files["DVAE_CHECKPOINT"],
        xtts_checkpoint=files["XTTS_CHECKPOINT"],
        tokenizer_file=files["TOKENIZER_FILE"],
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_perceiver_resampler=True,
    )

    test_sentences = [
        {
            "text": "Acknowledged. Initiating the requested operation.",
            "speaker_wav": str(args.reference_wav),
            "language": "en",
        },
        {
            "text": "There are no humans here. Just me.",
            "speaker_wav": str(args.reference_wav),
            "language": "en",
        },
        {
            "text": "I have completed the analysis. The optimal solution requires three steps.",
            "speaker_wav": str(args.reference_wav),
            "language": "en",
        },
    ]

    config = GPTTrainerConfig(
        output_path=str(run_dir),
        model_args=model_args,
        run_name=args.run_name,
        project_name="UltronXTTS",
        run_description="XTTS-v2 fine-tune on 3-min Ultron reference (round-9, 2026-05-20)",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_cfg,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=2,
        eval_split_max_size=256,
        print_step=10,
        plot_step=50,
        log_model_step=100,
        save_step=args.save_every_step,
        save_n_checkpoints=args.save_n_checkpoints,
        save_checkpoints=True,
        print_eval=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.learning_rate,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=test_sentences,
        epochs=args.epochs,
    )

    print("\nloading dataset samples...")
    # 2026-05-20: Coqui's split_dataset() asserts eval_split_size >= 0.05.
    # With 20-clip corpus the smallest legal split is 0.05 (1 sample).
    # Bump to 0.1 (2 eval / 18 train) so eval isn't degenerate.
    train_samples, eval_samples = load_tts_samples(
        [dataset_cfg],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.1,
    )
    print(f"  train samples: {len(train_samples)}")
    print(f"  eval  samples: {len(eval_samples)}")
    if not train_samples:
        print("ERROR: zero training samples loaded. Check metadata.csv format "
              "(expected `id|text|normalized_text` per line).")
        return 1

    print("\nconstructing GPTTrainer + Trainer...")
    model = GPTTrainer.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(
            restore_path=str(args.restore_path) if args.restore_path else None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=args.grad_accum,
        ),
        config,
        output_path=str(run_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print(f"\n=== STARTING TRAINING (epochs={args.epochs}, save_every={args.save_every_step}) ===")
    trainer.fit()
    print("\n=== TRAINING COMPLETE ===")
    print(f"checkpoints under {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
