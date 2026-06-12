"""Kokoro TTS engine (2026-05-19, Track 5).

Wrapper around the Kokoro StyleTTS2 + ISTFTNet inference model. Same
public surface as :class:`ultron.tts.xtts_v3.XttsV3Speech` (and the
legacy :class:`ultron.tts.speech.TextToSpeech`) so the orchestrator
can switch engines via ``tts.engine`` without touching the playback
path.

Module ships unconditionally; the actual Kokoro weights load lazily
on first ``warmup()`` / ``speak()`` call. When the weights aren't on
disk (the typical state on a fresh checkout), the engine surfaces a
clear :class:`KokoroEngineLoadError` rather than silently producing
silence -- callers can fall back to a different engine via config.

Three things deliberately omitted vs the XTTS engine to keep the
scope of this change small:

1. **No automatic v3 pedalboard filter chain.** Kokoro is intended to
   be fine-tuned on POST-filter audio (so the filter character is
   baked into the model weights and chunk streaming becomes
   tractable -- see the 2026-05-19 design conversation). The runtime
   filter pass exists as an opt-in ``apply_runtime_filter`` flag for
   pre-fine-tune use while the corpus is being prepared.
2. **No isolated venv subprocess.** Kokoro's dep tree (transformers,
   phonemizer, scipy) overlaps cleanly with the main Ultron venv.
   In-process loading saves a CUDA context + ~50 ms IPC overhead per
   synth.
3. **No fine-tune training code.** Training pipelines live in
   ``ultronVoiceAudio/`` per the existing voice-prep convention.
   This module is inference-only.

Default ``tts.engine`` is unchanged. To use Kokoro: place weights at
``models/kokoro/`` and set ``tts.engine: kokoro`` in ``config.yaml``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, ClassVar, Iterable, NamedTuple, Optional, Tuple

import numpy as np

from ultron.utils.logging import get_logger

logger = get_logger("tts.kokoro")


# Mirror of ultron.tts.xtts_v3._QUEUE_GET_TIMEOUT_SECONDS / the legacy
# speech.py constant. Long enough to absorb a slow first-clip synth
# (Kokoro lazy-load on first call) without false-killing the playback
# loop.
_QUEUE_GET_TIMEOUT_SECONDS = 60.0


# ----------------------------------------------------------------------
# Public exceptions + types
# ----------------------------------------------------------------------


class KokoroEngineLoadError(RuntimeError):
    """Raised when Kokoro weights / dependencies are unavailable."""


class KokoroSynthError(RuntimeError):
    """Raised when an inference call fails."""


Clip = Tuple[np.ndarray, int]


class ClipItem(NamedTuple):
    """Mirror of the XTTS / legacy ClipItem shape for queue uniformity."""

    audio: np.ndarray
    sample_rate: int
    is_known_last: bool = False


# Kokoro models the SAME native sample rate as XTTS (24 kHz) so the
# orchestrator's output-stream pre-open machinery can hand the
# device handle between engines without re-opening.
_KOKORO_DEFAULT_SAMPLE_RATE: int = 24000


# ----------------------------------------------------------------------
# Fine-tune state-dict compatibility shim
# ----------------------------------------------------------------------


def _make_kokoro_finetune_compat(src: Path) -> Path:
    """Convert a parametrizations-API state dict to old-weight_norm
    naming so the pip-installed ``kokoro`` package's KModel can load
    the fine-tuned weights without falling through to ``strict=False``.

    The training submodule (``ultronVoiceAudio/kokoro_finetune/kokoro``)
    uses ``torch.nn.utils.parametrizations.weight_norm`` (new API),
    storing weight_norm as ``<layer>.parametrizations.weight.original0``
    (magnitude g) and ``...original1`` (direction v). The pip package
    uses ``torch.nn.utils.weight_norm`` (old API), expecting
    ``<layer>.weight_g`` and ``<layer>.weight_v``.

    Loading the fine-tune directly into the pip KModel silently
    falls through to ``strict=False`` and leaves every weight_norm-
    parametrized layer at random init -- producing loud static
    instead of speech.

    The conversion is cached on disk so repeated startups are fast.
    Cache invalidates when the source ``.pth`` is newer than the
    cached compat file.

    Args:
        src: Path to the original fine-tune ``.pth`` (parametrizations
            API).

    Returns:
        Path to the compat ``.pth`` (old weight_norm API) -- safe to
        pass to ``KModel(model=...)``.
    """
    import torch
    compat = src.with_name(src.stem + "__compat.pth")
    if compat.is_file() and compat.stat().st_mtime >= src.stat().st_mtime:
        return compat

    logger.info(
        "Kokoro: converting fine-tune state dict from parametrizations "
        "API to old weight_norm naming (%s -> %s)",
        src.name, compat.name,
    )
    sd = torch.load(str(src), map_location="cpu", weights_only=True)
    converted: dict = {}
    n_renamed = 0
    for top, inner in sd.items():
        if not isinstance(inner, dict):
            converted[top] = inner
            continue
        new_inner = {}
        for k, v in inner.items():
            if k.endswith(".parametrizations.weight.original0"):
                new_k = k.replace(
                    ".parametrizations.weight.original0", ".weight_g",
                )
                n_renamed += 1
            elif k.endswith(".parametrizations.weight.original1"):
                new_k = k.replace(
                    ".parametrizations.weight.original1", ".weight_v",
                )
                n_renamed += 1
            else:
                new_k = k
            new_inner[new_k] = v
        converted[top] = new_inner
    torch.save(converted, str(compat))
    logger.info(
        "Kokoro: renamed %d keys, saved compat state dict at %s",
        n_renamed, compat,
    )
    return compat


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------


class KokoroSpeech:
    """Kokoro StyleTTS2 inference engine.

    Drop-in for :class:`XttsV3Speech` and :class:`TextToSpeech` --
    exposes ``speak`` / ``speak_stream`` / ``warmup`` /
    ``prepare_output_stream`` / ``stop`` so the playback path doesn't
    change when the orchestrator swaps engines.

    Args:
        model_path: directory containing the Kokoro weights + voices.
            Defaults to ``models/kokoro/``. The directory must exist
            for the engine to load; missing weights produce a
            :class:`KokoroEngineLoadError` on first inference.
        voice: name of the voice to render. Production-tuned Ultron
            voice is loaded from ``model_path/voices/{voice}.pt`` once
            the fine-tune lands; pre-fine-tune we fall back to one of
            Kokoro's stock voices (typically ``af_alloy`` or
            ``am_michael``) so the engine boots even before the
            corpus is prepared.
        device: ``"cpu"`` or ``"cuda"``. Kokoro is genuinely fast on
            CPU (StyleTTS2 + ISTFTNet is feed-forward; near-realtime
            on modern CPUs). Default ``"cpu"`` keeps the GPU free for
            LLM + Whisper. Set to ``"cuda"`` to push synthesis on the
            GPU for ~3x faster inference.
        speed: speech-rate multiplier (1.0 = native). Mirrors the
            XTTS speed knob -- the orchestrator can hot-swap engines
            without re-tuning cadence.
        apply_runtime_filter: when True, the v3 Ultron pedalboard
            filter runs on Kokoro's output (CPU; ~10-30 ms /
            sentence). Useful pre-fine-tune so the voice character
            matches the XTTS pipeline. Default False since the
            target end-state is Kokoro fine-tuned on already-
            filtered audio (filter baked into weights).
        filter_preset: pedalboard preset name when
            ``apply_runtime_filter`` is True.
        apply_spectral_smooth: when True, run the lightweight
            spectral magnitude smoothing pass (STFT median-filter
            ISTFT) on every synth output. Designed to mask the
            pitch wobble produced by an under-trained fine-tune
            checkpoint (Stage 1 only, or Stage 2 pre-SLM-joint).
            Cost is ~10 ms per second of audio; hidden by the
            round-8c producer-consumer pipeline on clips 2+ and
            pre-applied at cache-build time for cached acks.
            Default True since shipping with the partial fine-tune
            is the current state.
        spectral_smooth_window: width of the STFT magnitude median
            filter in frames. Default 5 frames at hop=512,
            sr=24 kHz = ~107 ms smoothing window -- A/B sweet spot
            on the partial-fine-tune corpus (2026-05-22). 3 frames
            (~64 ms) leaves audible wobble; 7+ frames (~150 ms+)
            starts softening fricatives. Pass 1 to no-op without
            removing the call site.
    """

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        voice: str = "af_alloy",
        device: str = "cpu",
        speed: float = 1.0,
        apply_runtime_filter: bool = False,
        filter_preset: str = "v3_heavy",
        apply_spectral_smooth: bool = True,
        spectral_smooth_window: int = 5,
        apply_trim_fade: bool = True,
        trim_fade_threshold_db: float = -40.0,
        flush_chars: str = ".!?\n",
        sample_rate: int = _KOKORO_DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.model_path = Path(model_path) if model_path else Path("models/kokoro")
        # 2026-05-22: when the configured voice name resolves to a local
        # ``.pt`` voicepack on disk (e.g. the fine-tuned ``ultron`` voice
        # at ``models/kokoro/voices/ultron.pt``), pass the FULL path to
        # KPipeline so it loads from disk via the
        # ``voice.endswith('.pt')`` branch in
        # ``kokoro.pipeline.KPipeline.load_single_voice`` instead of
        # trying to download from HF. Stock voice names (``am_michael``,
        # ``af_alloy``, etc.) pass through unchanged so the HF download
        # path still works.
        local_voicepack = self.model_path / "voices" / f"{voice}.pt"
        if local_voicepack.is_file():
            self.voice = str(local_voicepack)
            self._voice_display = voice
            logger.info(
                "Kokoro: using local voicepack %s for voice %r",
                local_voicepack, voice,
            )
        else:
            self.voice = voice
            self._voice_display = voice
        self.device = device
        self.speed = float(speed)
        self.apply_runtime_filter = bool(apply_runtime_filter)
        self.filter_preset = filter_preset
        self.apply_spectral_smooth = bool(apply_spectral_smooth)
        self.spectral_smooth_window = int(spectral_smooth_window)
        self.apply_trim_fade = bool(apply_trim_fade)
        self.trim_fade_threshold_db = float(trim_fade_threshold_db)
        self.flush_chars = set(flush_chars)
        self._sample_rate = int(sample_rate)
        self._model = None
        self._model_lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None
        self._stop_event = threading.Event()
        self._playback_lock = threading.Lock()
        # 2026-05-15 latency parity: pre-open output stream slot.
        self._preopened_stream = None
        self._preopened_lock = threading.Lock()
        # 2026-05-20 round 8d: pre-computed ack clip cache slot. The
        # orchestrator's ``_kick_off_ack_clip_prewarm`` calls
        # ``set_ack_cache`` after warmup; ``_synthesize`` checks here
        # before running the live KPipeline call so cached "Mm." /
        # "Right." / "Querying external sources." return in ~5 ms
        # instead of ~200-400 ms of CPU synth.
        self._ack_cache = None
        # Lazy import inside the synth path so a missing kokoro
        # install doesn't crash at module import time -- callers can
        # construct the engine and discover the load failure at the
        # first inference call (matches the XTTS pattern).

    # ------------------------------------------------------------------
    # Lifecycle / lazy load
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def is_available(self) -> bool:
        """True iff the engine has successfully loaded (or hasn't tried).

        Returns False after a prior load attempt failed; the engine
        won't retry the load until :meth:`reset_load_error` clears
        the cached failure. Used by the orchestrator to decide
        whether to fall back to XTTS / legacy.
        """
        if self._load_error is not None:
            return False
        return True

    def reset_load_error(self) -> None:
        """Clear the cached load-failure state so the next inference
        retries the load. Useful after the operator drops the
        weights into ``model_path`` mid-session."""
        with self._model_lock:
            self._load_error = None

    def move_to_device(self, device: str) -> None:
        """Hot-swap the Kokoro model between ``"cpu"`` and ``"cuda"``.

        Tries an in-place ``.to(device)`` on the loaded KModel (fast
        path; ~50-200 ms). Falls back to tearing down the model so the
        next ``_synthesize`` lazy-reloads on the new device (~1-5 s
        first call after flip; subsequent calls fast). Fail-open: any
        failure leaves the engine at its previous device with a WARN
        log -- callers can retry.

        Used by GamingModeManager to free ~330 MB of VRAM during
        gaming sessions: engage flips cuda -> cpu, disengage flips
        cpu -> cuda. Cached ack clips are device-agnostic (already
        rendered to int16) so they keep playing without reload.

        Args:
            device: ``"cpu"`` or ``"cuda"``. No-op when already there.
        """
        if device not in ("cpu", "cuda"):
            raise ValueError(f"Kokoro move_to_device: unknown device {device!r}")
        with self._model_lock:
            if device == self.device and self._loaded:
                return
            prior_device = self.device
            self.device = device

            if self._loaded and self._model is not None:
                inner = getattr(self._model, "model", None)
                if inner is not None and hasattr(inner, "to"):
                    try:
                        self._model.model = inner.to(device).eval()
                        logger.info(
                            "Kokoro: moved model %s -> %s in place",
                            prior_device, device,
                        )
                        if prior_device == "cuda" and device == "cpu":
                            self._try_empty_cuda_cache()
                        return
                    except Exception as e:                        # noqa: BLE001
                        logger.warning(
                            "Kokoro: in-place .to(%s) failed (%s); "
                            "tearing down for lazy reload.", device, e,
                        )

            # Tear down so next inference rebuilds on the new device.
            self._model = None
            self._loaded = False
            self._load_error = None
            if prior_device == "cuda":
                self._try_empty_cuda_cache()
            logger.info(
                "Kokoro: torn down (was %s); next synth lazy-loads on %s",
                prior_device, device,
            )

    @staticmethod
    def _try_empty_cuda_cache() -> None:
        """Best-effort ``torch.cuda.empty_cache()`` (silent on failure)."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure_loaded(self) -> None:
        """Lazy-load Kokoro on first use.

        Raises :class:`KokoroEngineLoadError` on failure (missing
        directory, missing package, etc.). The failure is cached --
        subsequent calls fail fast without retrying the import.
        """
        if self._loaded:
            return
        if self._load_error is not None:
            raise KokoroEngineLoadError(self._load_error)
        with self._model_lock:
            if self._loaded:
                return
            if self._load_error is not None:
                raise KokoroEngineLoadError(self._load_error)
            try:
                self._do_load()
                self._loaded = True
            except Exception as e:                            # noqa: BLE001
                msg = f"Kokoro load failed: {e}"
                self._load_error = msg
                logger.warning(msg)
                raise KokoroEngineLoadError(msg) from e

    def _do_load(self) -> None:
        """Construct the Kokoro pipeline object.

        Tries the ``kokoro`` package's high-level API first (preferred
        for production); falls back to a manual StyleTTS2 + ISTFTNet
        load if the package isn't installed. Both paths assume the
        weights are in ``self.model_path``.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Kokoro model directory not found: {self.model_path}. "
                f"Download weights with scripts/download_models.py or "
                f"point tts.kokoro.model_path at the correct location."
            )
        try:
            # Preferred path: hexgrad/kokoro PyPI package.
            from kokoro import KModel, KPipeline                # type: ignore
        except ImportError as e:
            raise KokoroEngineLoadError(
                "The ``kokoro`` package is not installed. Add it to "
                "the venv via ``uv pip install kokoro`` (or ``pip "
                "install kokoro``) and re-run."
            ) from e
        # ``lang_code='a'`` selects American English. The pipeline
        # internally loads ISTFTNet vocoder + StyleTTS2 acoustic model.
        #
        # 2026-05-22 fine-tune integration: if a converted Kokoro-
        # format fine-tune weights file is present at
        # ``model_path/ultron_finetune.pth``, construct an explicit
        # KModel pointing at those weights and hand it to KPipeline.
        # Without this, voicepack alone only provides the style
        # vectors; the underlying decoder / predictor / text_encoder
        # remain stock Kokoro, which dilutes the trained voice
        # character. The converted file is produced from a StyleTTS2
        # Stage-2 checkpoint via
        # ``ultronVoiceAudio/kokoro_finetune/scripts/test_inference.py``
        # ``convert_checkpoint`` helper (bert + bert_encoder +
        # predictor + text_encoder + decoder, ~330 MB).
        finetune_path = self.model_path / "ultron_finetune.pth"
        if finetune_path.is_file():
            try:
                # 2026-05-22: the fine-tune was trained with PyTorch's
                # NEW parametrization API
                # (``torch.nn.utils.parametrizations.weight_norm``)
                # which stores LSTM/Conv weight_norm as
                # ``<layer>.parametrizations.weight.original0`` and
                # ``...original1``. The pip-installed kokoro package
                # uses the OLD API (``torch.nn.utils.weight_norm``)
                # which expects ``<layer>.weight_g`` and ``weight_v``.
                # Without conversion, KModel.load_state_dict's
                # strict=False fallback silently leaves the affected
                # modules at random init -> loud static output instead
                # of speech. Convert the keys on first load (cached
                # on disk for subsequent restarts).
                compat_path = _make_kokoro_finetune_compat(finetune_path)
                kmodel = KModel(
                    repo_id="hexgrad/Kokoro-82M",
                    model=str(compat_path),
                ).to(self.device).eval()
                self._model = KPipeline(
                    lang_code="a",
                    repo_id="hexgrad/Kokoro-82M",
                    model=kmodel,
                )
                logger.info(
                    "Kokoro: loaded fine-tuned model weights from %s "
                    "(decoder + predictor + text_encoder + bert)",
                    finetune_path,
                )
            except Exception as e:                              # noqa: BLE001
                logger.warning(
                    "Kokoro: fine-tuned model load failed (%s); "
                    "falling back to stock KPipeline with voicepack "
                    "only.", e,
                )
                self._model = KPipeline(
                    lang_code="a",
                    device=self.device,
                )
        else:
            self._model = KPipeline(
                lang_code="a",
                device=self.device,
            )
        logger.info(
            "Kokoro ready (voice=%s, device=%s, sample_rate=%d)",
            self._voice_display, self.device, self._sample_rate,
        )

    def warmup(self, text: str = "Online.") -> None:
        """Touch the inference pipeline with a tiny request.

        Fail-open: load failures are logged WARN and the warmup is a
        no-op. The first real ``speak`` call will surface the same
        error if it persists.
        """
        if not text.strip():
            return
        try:
            t0 = time.monotonic()
            self._synthesize(text)
            logger.info(
                "Kokoro warmup complete in %.0f ms",
                (time.monotonic() - t0) * 1000,
            )
        except KokoroEngineLoadError as e:
            logger.warning("Kokoro warmup skipped: %s", e)
        except Exception as e:                                # noqa: BLE001
            logger.warning("Kokoro warmup failed (%s); engine may be unhealthy", e)

    # ------------------------------------------------------------------
    # Public synth + playback API (mirrors XttsV3Speech)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal playback interrupt -- mirrors XTTS stop()."""
        self._stop_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        with self._preopened_lock:
            s = self._preopened_stream
            self._preopened_stream = None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def speak(self, text: str) -> None:
        """Synth + play synchronously. Mirrors XttsV3Speech.speak()."""
        if not text.strip():
            return
        self._stop_event.clear()
        clip = self._synthesize(text)
        if clip[0].size > 0 and not self._stop_event.is_set():
            self._play(clip)

    def prepare_output_stream(self) -> None:
        """Pre-open the PortAudio output device.

        Mirrors the 2026-05-15 latency-pass pattern on XTTS. The
        orchestrator calls this on a daemon thread during STT so the
        ~50 ms device-open cost overlaps with transcription. Fails
        open -- live ``speak_stream`` opens fresh if pre-open
        couldn't complete.
        """
        with self._preopened_lock:
            if self._preopened_stream is not None:
                return
            try:
                import sounddevice as sd
                stream = sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=2,
                    dtype="int16",
                )
                stream.start()
                # 50 ms silence write wakes the device clock.
                silence = np.zeros((self._sample_rate // 20, 2), dtype=np.int16)
                stream.write(silence)
                self._preopened_stream = stream
            except Exception as e:                            # noqa: BLE001
                logger.warning("Kokoro pre-open failed: %s", e)

    def speak_stream(self, fragments: Iterable[str]) -> None:
        """Consume token fragments + play sentence-by-sentence.

        2026-05-20 round 8c rewrite: producer-consumer pipeline (mirror
        of :meth:`XttsV3Speech.speak_stream`). Synth runs on a worker
        thread that pushes :class:`ClipItem` onto a bounded queue;
        playback consumes on the main thread holding a single open
        :class:`sounddevice.OutputStream` for the whole call. Synth of
        sentence N+1 overlaps playback of sentence N, eliminating the
        multi-second pauses the prior sequential loop produced (each
        inter-sentence gap was the full ~200-600 ms CPU synth cost).
        Also adds safe-sentence-boundary detection so ellipses /
        decimals / mid-domain dots / known abbreviations don't
        fragment the audio.
        """
        self._stop_event.clear()

        try:
            from ultron.config import get_config
            tts_cfg = get_config().tts
            spec_open = tts_cfg.speculative_stream_open_enabled
            low_latency = tts_cfg.output_low_latency_mode
        except Exception:
            spec_open = False
            low_latency = False

        audio_q: queue.Queue[Optional[ClipItem]] = queue.Queue(maxsize=8)

        def synth_worker() -> None:
            try:
                self._run_synth_loop(
                    fragments=fragments,
                    push=lambda item: audio_q.put(item),
                )
            except Exception as e:                                # noqa: BLE001
                logger.error("Kokoro synth worker error: %s", e)
            finally:
                audio_q.put(None)

        worker = threading.Thread(
            target=synth_worker, daemon=True, name="kokoro-synth",
        )
        worker.start()

        try:
            import sounddevice as sd
        except Exception as e:                                    # noqa: BLE001
            logger.warning(
                "sounddevice unavailable -- skipping Kokoro playback: %s", e,
            )
            worker.join(timeout=2.0)
            return

        sr = self._sample_rate
        block_frames = max(1, int(sr * 0.05))
        stream = None
        first_item: Optional[ClipItem] = None

        try:
            with self._playback_lock:
                if self._stop_event.is_set():
                    return

                # Prefer the pre-opened stream (opened during STT on a
                # daemon thread per Orchestrator._kick_off_tts_preopen).
                stream = self._consume_preopened_stream(sr)
                if stream is None:
                    stream = self._open_output_stream(sr, low_latency)
                    stream.start()
                    # 50 ms silence write wakes the device clock.
                    self._write_silence(stream, sr, 0.05)

                try:
                    first_item = audio_q.get(
                        timeout=_QUEUE_GET_TIMEOUT_SECONDS,
                    )
                except queue.Empty:
                    logger.warning(
                        "Kokoro playback queue starved before first clip",
                    )
                    return
                if first_item is None:
                    return

                # Inter-sentence pause (matches XTTS / legacy parity).
                try:
                    from config import settings as _legacy_settings
                    pause_ms = _legacy_settings.TTS_PAUSE_MS
                except Exception:
                    pause_ms = 180

                item = first_item
                while True:
                    if self._stop_event.is_set():
                        return
                    audio = self._stereo_pcm(item.audio)
                    for start in range(0, audio.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(audio[start : start + block_frames])

                    if item.is_known_last:
                        self._write_silence(stream, sr, 0.05)
                        break

                    if pause_ms > 0 and not self._stop_event.is_set():
                        self._write_silence(stream, sr, pause_ms / 1000.0)

                    # 2026-05-22: poll with silence chunks while the
                    # CPU synth catches up. A blocking ``Queue.get``
                    # here would drain the PortAudio output buffer to
                    # empty -- producing an audible click at the
                    # underflow point. Continuous silence keeps the
                    # device clock fed without adding deliberate delay
                    # beyond what synth latency already costs.
                    nxt, timed_out = self._drain_queue_with_silence(
                        audio_q, stream, sr,
                    )
                    if timed_out:
                        logger.warning(
                            "Kokoro playback waited %.0fs without next "
                            "clip; ending", _QUEUE_GET_TIMEOUT_SECONDS,
                        )
                        self._write_silence(stream, sr, 0.05)
                        break
                    if nxt is None:
                        # Synth worker's finally sentinel: normal end.
                        if self._stop_event.is_set():
                            return
                        self._write_silence(stream, sr, 0.05)
                        break
                    item = nxt
        except Exception as e:                                    # noqa: BLE001
            logger.warning("Kokoro streaming playback error: %s", e)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            worker.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Ack-cache wiring (2026-05-20 round 8d)
    # ------------------------------------------------------------------

    def set_ack_cache(self, cache) -> None:
        """Wire a pre-computed ack clip cache.

        Mirror of :meth:`XttsV3Speech.set_ack_cache` -- once installed,
        :meth:`_synthesize` checks the cache before invoking the live
        KPipeline call. Cache hits return the stored ``(pcm, sr)``
        clip directly (~5 ms lookup), skipping ~200-400 ms of CPU
        synth + optional v3 filter. Pass ``None`` to detach.

        The orchestrator wires this from ``_kick_off_ack_clip_prewarm``
        after Kokoro warmup so cached clips are byte-identical to the
        live path (same engine settings, same voice, same runtime
        filter state).
        """
        self._ack_cache = cache
        if cache is not None:
            logger.info(
                "Kokoro: ack clip cache attached (%d phrases enrolled)",
                len(cache.phrases),
            )

    # ------------------------------------------------------------------
    # Internal: synth + playback
    # ------------------------------------------------------------------

    def _synthesize(self, text: str) -> Clip:
        """Run Kokoro inference on a sentence and return int16 PCM.

        2026-05-20 round 8d: ack-cache check happens BEFORE the
        KPipeline call so cached phrases (Mm. / Right. / Considering. /
        Querying external sources. / etc.) return their pre-rendered
        clip in ~5 ms. Cache miss falls through to the live path
        unchanged.
        """
        # Pre-synthesis hygiene (2026-06-11): stage directions, control
        # tokens, and punctuation-only fragments never reach the voice
        # (observed live: "*repositions window...*" spoken aloud by the
        # 3B preset). Pure regex on a short string -- microseconds; an
        # empty result returns a zero clip the playback paths skip.
        try:
            from ultron.tts.text_hygiene import sanitize_spoken_text

            text = sanitize_spoken_text(text)
        except Exception:                                     # noqa: BLE001
            pass
        if not text:
            return np.zeros(0, dtype=np.int16), self._sample_rate

        # Cache-hit fast path. ``getattr`` keeps the engine
        # instantiable in unit-test fixtures that bypass __init__.
        ack_cache = getattr(self, "_ack_cache", None)
        if ack_cache is not None:
            cached = ack_cache.get(text)
            if cached is not None:
                logger.debug(
                    "Kokoro: ack-cache hit for %r", text[:40],
                )
                return cached

        self._ensure_loaded()
        if self._model is None:
            raise KokoroEngineLoadError("Kokoro model is None after load")
        try:
            # KPipeline returns a generator of (graphemes, phonemes,
            # audio_tensor) tuples per sentence; we concatenate.
            audio_chunks: list[np.ndarray] = []
            generator = self._model(text, voice=self.voice, speed=self.speed)
            try:
                for _gs, _ps, audio in generator:
                    if audio is None:
                        continue
                    # ``audio`` is a torch Tensor (cpu or cuda). Convert
                    # to numpy float32 [-1, 1].
                    try:
                        arr = audio.detach().cpu().numpy().astype(np.float32)
                    except AttributeError:
                        # Already a numpy array.
                        arr = np.asarray(audio, dtype=np.float32)
                    audio_chunks.append(arr)
            finally:
                # 2026-06-11 VRAM hygiene: explicitly close the KPipeline
                # generator so its retained intermediate-tensor refs are
                # dropped NOW rather than whenever the GC happens to run.
                # On the CUDA Kokoro path (the user's config) those refs
                # otherwise linger on the GPU between turns, which is the
                # per-response VRAM creep. close() just triggers
                # GeneratorExit on an already-exhausted generator -- no
                # CUDA sync, zero hot-path latency. Fail-open.
                try:
                    generator.close()
                except Exception:                             # noqa: BLE001
                    pass
        except Exception as e:                                # noqa: BLE001
            raise KokoroSynthError(f"Kokoro inference failed: {e}") from e

        if not audio_chunks:
            return np.zeros(0, dtype=np.int16), self._sample_rate

        pcm_f32 = np.concatenate(audio_chunks)

        # Spectral magnitude smoothing for under-trained fine-tunes.
        # Lightweight (~10 ms/sec audio); masks pitch wobble without
        # smearing consonants. Fail-open: any error degrades
        # silently to the raw output rather than dropping the clip.
        if self.apply_spectral_smooth:
            try:
                from ultron.tts.spectral_smooth import spectral_smooth
                pcm_f32 = spectral_smooth(
                    pcm_f32, sr=self._sample_rate,
                    median_window_frames=self.spectral_smooth_window,
                )
            except Exception as e:                            # noqa: BLE001
                logger.warning(
                    "Kokoro spectral smoothing failed (passing through): %s",
                    e,
                )

        # Boundary noise trimmer + fade-in/fade-out for the partial
        # fine-tune. Strips the brief noise bursts the undertrained
        # model generates before and after speech, then applies short
        # linear fades to prevent clicks. Fail-open: errors pass
        # through without dropping the clip.
        if self.apply_trim_fade:
            try:
                from ultron.tts.spectral_smooth import trim_and_fade
                pcm_f32 = trim_and_fade(
                    pcm_f32, sr=self._sample_rate,
                    threshold_db=self.trim_fade_threshold_db,
                )
            except Exception as e:                            # noqa: BLE001
                logger.warning(
                    "Kokoro trim/fade failed (passing through): %s", e,
                )

        # Optional pre-fine-tune runtime filter pass.
        if self.apply_runtime_filter:
            try:
                from ultron.tts.ultron_filter import apply_filter
                pcm_f32 = apply_filter(
                    pcm_f32, self._sample_rate,
                    preset=self.filter_preset,
                    tail_silence_ms=200.0,
                )
            except Exception as e:                            # noqa: BLE001
                logger.warning("Ultron filter on Kokoro output failed: %s", e)

        # Clip + convert to int16 (mirrors the XTTS engine's tail).
        np.clip(pcm_f32, -1.0, 1.0, out=pcm_f32)
        out_pcm = (pcm_f32 * 32767.0).astype(np.int16)

        # Output-quality watcher: non-blocking enqueue for blip analysis
        # on a daemon thread (hard onsets/tails, boundary noise bursts,
        # join discontinuities, dropouts, clipping). Cost here is a
        # try/except + queue put -- the locked synth hot path is
        # otherwise untouched, and ack-cache hits above (static,
        # pre-rendered clips) deliberately skip analysis.
        try:
            from ultron.audio.output_quality import get_output_watcher

            watcher = get_output_watcher()
            if watcher is not None:
                watcher.submit(out_pcm, self._sample_rate, label=text[:60])
        except Exception:                                     # noqa: BLE001
            pass

        return out_pcm, self._sample_rate

    def _play(self, clip: Clip) -> None:
        """Single-shot playback."""
        pcm, sr = clip
        try:
            import sounddevice as sd
        except Exception as e:                                # noqa: BLE001
            logger.warning("sounddevice unavailable -- skipping Kokoro playback: %s", e)
            return
        if pcm.size == 0:
            return
        with self._playback_lock:
            if self._stop_event.is_set():
                return
            try:
                # Stereo expand for the output stream.
                stereo = np.column_stack((pcm, pcm)).astype(np.int16, copy=False)
                stream = self._consume_preopened_stream(sr)
                opened_here = False
                if stream is None:
                    stream = sd.OutputStream(
                        samplerate=sr, channels=2, dtype="int16",
                    )
                    stream.start()
                    opened_here = True
                try:
                    block_frames = max(1, int(sr * 0.05))
                    for start in range(0, stereo.shape[0], block_frames):
                        if self._stop_event.is_set():
                            return
                        stream.write(stereo[start: start + block_frames])
                finally:
                    if opened_here:
                        try:
                            stream.stop()
                            stream.close()
                        except Exception:
                            pass
            except Exception as e:                            # noqa: BLE001
                logger.warning("Kokoro playback error: %s", e)

    def _consume_preopened_stream(self, sr: int):
        """Take ownership of any pre-opened output stream."""
        with self._preopened_lock:
            s = self._preopened_stream
            self._preopened_stream = None
        if s is None:
            return None
        if sr != self._sample_rate:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
            return None
        return s

    # ------------------------------------------------------------------
    # Producer/consumer helpers (2026-05-20 round 8c)
    # ------------------------------------------------------------------

    # Common English abbreviations that end with `.` but do NOT mark a
    # sentence boundary. Lower-cased; the boundary check normalises
    # before lookup. Mirror of :data:`XttsV3Speech._ABBREVIATIONS` --
    # both engines benefit identically from rejecting these mid-token
    # `.` flushes. Duplicated rather than imported to keep the engines
    # independently testable.
    _ABBREVIATIONS: ClassVar[frozenset[str]] = frozenset({
        "mr", "mrs", "ms", "dr", "st", "jr", "sr", "fr",
        "vs", "etc", "eg", "ie", "cf", "al", "esp",
        "inc", "co", "ltd", "corp", "llc",
        "ave", "blvd", "rd", "pkwy", "hwy",
        "no", "nos",
        "approx", "vol", "ed", "eds", "rev", "ref",
    })

    @classmethod
    def _is_safe_sentence_boundary(
        cls, text: str, pos: int, *, buffer_complete: bool,
    ) -> bool:
        """Return True if ``text[pos]`` is a flushable sentence end.

        Pure function -- mirror of
        :meth:`XttsV3Speech._is_safe_sentence_boundary`. Rejects mid-
        token periods that would otherwise fragment the audio
        (ellipsis, decimals, domains, abbreviation chains).
        """
        ch = text[pos]
        n = len(text)
        if ch == "\n":
            return True
        if ch in "!?":
            return True
        if ch != ".":
            return False
        # Ellipsis suppression.
        if pos + 1 < n and text[pos + 1] == ".":
            return False
        if pos > 0 and text[pos - 1] == ".":
            return False
        # Acronym continuation: "L.L." where each L is a letter.
        if (
            pos >= 2
            and text[pos - 2] == "."
            and text[pos - 1].isalpha()
        ):
            return False
        # Decimal: digit.digit (e.g. "3.14").
        if (
            pos > 0
            and text[pos - 1].isdigit()
            and pos + 1 < n
            and text[pos + 1].isdigit()
        ):
            return False
        # Mid-domain: letter.letter ("Dictionary.com").
        if (
            pos > 0
            and text[pos - 1].isalpha()
            and pos + 1 < n
            and text[pos + 1].isalpha()
        ):
            return False
        # Trailing `.` with no next char: wait for more unless we're
        # at end of stream.
        if pos + 1 >= n:
            return buffer_complete
        next_ch = text[pos + 1]
        if next_ch.isspace():
            start = pos
            while start > 0 and text[start - 1].isalpha():
                start -= 1
            token = text[start:pos].lower()
            if token and token in cls._ABBREVIATIONS:
                return False
            return True
        return True

    def _find_next_sentence_boundary(
        self, text: str, *, buffer_complete: bool,
    ) -> int:
        """Return position+1 of the next safe boundary, or 0 if none."""
        for i, ch in enumerate(text):
            if ch in self.flush_chars:
                if self._is_safe_sentence_boundary(
                    text, i, buffer_complete=buffer_complete,
                ):
                    return i + 1
        return 0

    def _run_synth_loop(
        self,
        *,
        fragments: Iterable[str],
        push: Callable[[ClipItem], None],
    ) -> None:
        """Walk fragments, synth on safe sentence boundaries, push ClipItems.

        Cumulative pending-buffer pattern: collect fragments, scan for
        the next safe sentence boundary, synth that slice, push,
        repeat. End-of-stream flushes whatever remains via
        ``buffer_complete=True``.

        Unlike :meth:`XttsV3Speech._run_synth_loop` we do NOT sub-split
        long sentences -- Kokoro is a single feed-forward inference
        per call and has no 4096-token GPT context to overflow.
        """
        pending = ""
        for frag in fragments:
            if self._stop_event.is_set():
                return
            if not frag:
                continue
            pending += frag
            while True:
                cut = self._find_next_sentence_boundary(
                    pending, buffer_complete=False,
                )
                if cut == 0:
                    break
                sentence = pending[:cut].strip()
                pending = pending[cut:]
                if not sentence:
                    continue
                try:
                    clip = self._synthesize(sentence)
                except KokoroEngineLoadError as e:
                    logger.warning(
                        "Kokoro load error mid-stream (%s); skipping "
                        "sentence: %r", e, sentence[:40],
                    )
                    continue
                except Exception as e:                            # noqa: BLE001
                    logger.warning(
                        "Kokoro synth failure (%s); skipping sentence: %r",
                        e, sentence[:40],
                    )
                    continue
                if clip[0].size > 0 and not self._stop_event.is_set():
                    push(ClipItem(
                        audio=clip[0],
                        sample_rate=clip[1],
                        is_known_last=False,
                    ))
        tail = pending.strip()
        if tail and not self._stop_event.is_set():
            try:
                clip = self._synthesize(tail)
            except KokoroEngineLoadError as e:
                logger.warning("Kokoro load error on tail (%s)", e)
                return
            except Exception as e:                                # noqa: BLE001
                logger.warning("Kokoro tail synth failure (%s)", e)
                return
            if clip[0].size > 0 and not self._stop_event.is_set():
                push(ClipItem(
                    audio=clip[0],
                    sample_rate=clip[1],
                    is_known_last=False,
                ))

    @staticmethod
    def _stereo_pcm(pcm: np.ndarray) -> np.ndarray:
        """Expand mono int16 PCM to interleaved stereo. No-op for stereo."""
        if pcm.ndim == 2 and pcm.shape[1] == 2:
            return pcm.astype(np.int16, copy=False)
        return np.column_stack((pcm, pcm)).astype(np.int16, copy=False)

    def _open_output_stream(self, sr: int, low_latency: bool):
        """Open a fresh sounddevice OutputStream at ``sr``.

        Honors ``tts.output_low_latency_mode`` for the PortAudio
        ``latency='low'`` hint (saves 30-100 ms OS-buffering on most
        Windows hosts; falls back gracefully if the host ignores it).
        """
        import sounddevice as sd
        kwargs = dict(samplerate=sr, channels=2, dtype="int16")
        if low_latency:
            kwargs["latency"] = "low"
        return sd.OutputStream(**kwargs)

    @staticmethod
    def _write_silence(stream, sr: int, seconds: float) -> None:
        """Write ``seconds`` of stereo silence to ``stream``. Best-effort."""
        if seconds <= 0:
            return
        frames = max(1, int(sr * seconds))
        silence = np.zeros((frames, 2), dtype=np.int16)
        try:
            stream.write(silence)
        except Exception:
            pass

    def _drain_queue_with_silence(
        self,
        audio_q: "queue.Queue[Optional[ClipItem]]",
        stream,
        sr: int,
        *,
        poll_seconds: float = 0.020,
        deadline_seconds: float = _QUEUE_GET_TIMEOUT_SECONDS,
    ) -> Tuple[Optional[ClipItem], bool]:
        """Poll ``audio_q`` while keeping the output stream fed.

        Kokoro CPU synth often runs slower than playback of the
        previous clip on multi-sentence replies. A blocking
        ``Queue.get`` would let the PortAudio output buffer drain to
        empty -- producing an audible click at the underflow point.
        This polls in short intervals and writes matching silence
        blocks between polls, so the device clock stays fed without
        adding deliberate extra delay beyond what synth already costs.

        Returns a tuple ``(item, timed_out)`` where:
        - ``item`` is the next :class:`ClipItem`, the sentinel
          ``None`` from the synth worker's finally, or ``None`` after
          a timeout / stop signal.
        - ``timed_out`` is True ONLY when the deadline expired without
          seeing any value (the genuine starve case). False on
          sentinel and stop event so callers can distinguish "normal
          end of stream" from "synth worker hung".
        """
        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                return None, False
            try:
                item = audio_q.get(timeout=poll_seconds)
                return item, False
            except queue.Empty:
                self._write_silence(stream, sr, poll_seconds)
        return None, True


__all__ = [
    "ClipItem",
    "KokoroEngineLoadError",
    "KokoroSpeech",
    "KokoroSynthError",
]
