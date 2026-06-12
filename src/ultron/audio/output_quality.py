"""TTS output-quality watcher: catch audible blips in synthesized clips.

The Kokoro fine-tune occasionally produces boundary artifacts -- brief
noise bursts before/after speech, hard un-faded onsets that pop on
stream start, discontinuity clicks at sentence-concatenation joins, and
(more rarely) internal dropouts. ``trim_and_fade`` mitigates most of
them at synth time; this module DETECTS whatever still slips through,
live, so regressions in the voice output are observable instead of
anecdotal.

2026-06-12 detector adjudication (174 live records analysed): natural
prosody routinely produces 60-430 ms near-silent gaps -- stop-consonant
closures, clause pauses, inter-sentence pauses -- always entered via a
gradual energy decay, so they must NOT flag. The internal-dropout check
is two-tier (>= 600 ms dead air always flags; 100-600 ms gaps flag only
with speech-level energy on BOTH edges = a digital hard cut), and short
quiet loud-runs isolated at the clip edges are stripped into the
leading/trailing burst classes (the measured real-artifact shape, which
the legacy 25%-of-peak burst gate used to reject -- misclassifying the
dead air before the burst as an internal dropout).

Design constraints (voice-baseline contract):

* The synth hot path only pays a try/except + a non-blocking queue put
  (microseconds). All waveform analysis runs on a single daemon thread.
* Fully fail-open: a watcher bug can never break synthesis or playback.
* Findings are logged at WARNING and appended as JSONL records to
  ``logs/audio_quality.jsonl`` for offline review; per-session counters
  are available via :meth:`OutputQualityWatcher.stats`.

Analysis is per synthesized CLIP (the pre-playback PCM). A clip that
starts at high amplitude WILL pop when the output stream opens, so the
hard-onset/tail checks act as the proxy for playback-side pops; device
or driver artifacts that never appear in the PCM are out of scope.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("ultron.audio.output_quality")

__all__ = [
    "BlipFinding",
    "ClipQualityReport",
    "analyze_clip",
    "OutputQualityWatcher",
    "get_output_watcher",
    "reset_output_watcher",
]

# Analysis frame size for the RMS envelope (10 ms).
_FRAME_MS = 10.0
# How far into each clip edge we look for isolated noise bursts.
_EDGE_WINDOW_MS = 250.0
# Minimum clip duration worth analyzing.
_MIN_CLIP_S = 0.05


@dataclass(frozen=True)
class BlipFinding:
    """One detected audio artifact.

    Attributes:
        kind: one of ``hard_onset`` / ``hard_tail`` / ``leading_burst`` /
            ``trailing_burst`` / ``discontinuity`` / ``internal_dropout`` /
            ``clipping`` / ``dc_offset``.
        position_ms: approximate artifact position from clip start.
        magnitude: kind-specific severity (amplitude, fraction, or dB).
        detail: short human-readable description.
    """

    kind: str
    position_ms: float
    magnitude: float
    detail: str


@dataclass(frozen=True)
class ClipQualityReport:
    """Analysis result for one synthesized clip.

    Attributes:
        duration_s: clip duration in seconds.
        peak: peak |amplitude| in [0, 1].
        rms: full-clip RMS in [0, 1].
        findings: detected artifacts (empty -> clean clip).
        label: caller-provided context (e.g. leading words of the text).
    """

    duration_s: float
    peak: float
    rms: float
    findings: tuple[BlipFinding, ...] = ()
    label: str = ""

    @property
    def clean(self) -> bool:
        """True iff no artifact was detected."""
        return not self.findings


def _to_float(pcm: np.ndarray) -> np.ndarray:
    """Normalise int16/float input to float32 in [-1, 1]."""
    arr = np.asarray(pcm)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    return np.clip(arr.astype(np.float32), -1.0, 1.0)


def _loud_runs(loud: np.ndarray) -> list:
    """Consecutive True spans of ``loud`` as inclusive (start, end) runs."""
    runs: list = []
    start = None
    for i, flag in enumerate(loud):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(loud) - 1))
    return runs


def analyze_clip(
    pcm: np.ndarray,
    sample_rate: int,
    *,
    label: str = "",
    edge_abs_threshold: float = 0.05,
    burst_silence_rms: float = 0.004,
    burst_min_ratio: float = 0.25,
    discontinuity_jump: float = 0.5,
    discontinuity_outlier_ratio: float = 8.0,
    discontinuity_window_ms: float = 5.0,
    dropout_ms: float = 100.0,
    dead_air_ms: float = 600.0,
    dropout_adjacent_ratio: float = 0.25,
    edge_burst_max_ms: float = 130.0,
    edge_burst_gap_ms: float = 200.0,
    clipping_fraction: float = 0.001,
    dc_offset_threshold: float = 0.02,
) -> ClipQualityReport:
    """Inspect one synthesized clip for audible artifacts.

    2026-06-12 adjudication (from 174 live audio_quality.jsonl
    records): natural prosody renders 60-430 ms near-silent gaps --
    stop-consonant closures (60-70 ms), clause pauses (100-260 ms),
    inter-sentence pauses (290-400 ms) -- always entered via a GRADUAL
    energy decay. A genuine digital dropout is either a hard cut
    (speech-level energy on BOTH gap edges) or grossly long dead air.
    The detector therefore uses a two-tier internal-dropout rule and
    strips short isolated edge runs into the burst classes (the same
    run-grouping the trim_and_fade fix uses), so quiet terminal bursts
    report under their TRUE kind instead of polluting the dropout body.

    Args:
        pcm: mono int16 or float PCM.
        sample_rate: sample rate of ``pcm``.
        label: context recorded on the report (e.g. text prefix).
        edge_abs_threshold: |amplitude| within the first/last ~5 ms that
            counts as a hard (un-faded) onset/tail -- these pop on
            stream open/close.
        burst_silence_rms: frame RMS at/below this is "silence" for the
            isolated-burst and dropout checks.
        burst_min_ratio: an edge burst inside the legacy edge-window
            check must reach this fraction of the clip's peak frame RMS
            to be reported (the run-strip path has NO such gate -- the
            measured real bursts sat at only 3-12% of peak).
        discontinuity_jump: minimum adjacent-sample jump (in [-1,1]
            units) considered audible as a concatenation click -- the
            absolute floor; the jump must ALSO be an outlier (below).
        discontinuity_outlier_ratio: the jump must exceed this multiple
            of the median adjacent-sample diff in the window around it
            to count as a click. Calibration (2026-06-12, live data +
            synthetic populations): pure tones cap at ~1.41x their own
            median diff at ANY frequency; hot broadband fricative
            noise sits at ~4.6-5.9x; a join offset inside a loud 1 kHz
            vowel measures ~9x; production-plausible joins/clicks (at
            quiet concatenation boundaries) measure 35-170x. The
            default 8.0 separates those populations with margin on
            both sides.
        discontinuity_window_ms: half-width of the local-diff window
            around the candidate jump.
        dropout_ms: minimum hard-cut gap INSIDE speech reported as an
            internal dropout when BOTH gap edges carry speech-level
            energy (>= ``dropout_adjacent_ratio`` of the envelope
            peak). Below this, gaps are natural closures/pauses.
        dead_air_ms: gap length that is reported as an internal dropout
            REGARDLESS of edge energy (max observed natural pause was
            430 ms; 600 gives ~40% margin).
        dropout_adjacent_ratio: fraction of the envelope peak both gap
            edges must reach for the hard-cut tier (natural pauses
            decay to ~3-9% of peak at the gap edge).
        edge_burst_max_ms: loud runs at the clip edges no longer than
            this are stripped from the speech body (mirrors the
            trim_and_fade discard cap; measured artifact tails span
            46-126 ms).
        edge_burst_gap_ms: minimum silence between an edge run and the
            speech body for the strip (measured isolation 210-490 ms).
        clipping_fraction: fraction of samples at full scale that counts
            as clipping.
        dc_offset_threshold: |mean| that counts as DC offset.

    Returns:
        A :class:`ClipQualityReport`; ``report.clean`` iff no artifact.
    """
    x = _to_float(pcm)
    n = int(x.size)
    if n == 0 or sample_rate <= 0:
        return ClipQualityReport(0.0, 0.0, 0.0, (), label)
    duration_s = n / float(sample_rate)
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(np.square(x))))
    if duration_s < _MIN_CLIP_S:
        return ClipQualityReport(duration_s, peak, rms, (), label)

    findings: list[BlipFinding] = []

    # --- Hard onset / tail (un-faded edges pop on stream start/stop).
    edge = max(1, int(sample_rate * 0.005))
    onset_peak = float(np.abs(x[:edge]).max())
    if onset_peak > edge_abs_threshold:
        findings.append(BlipFinding(
            "hard_onset", 0.0, onset_peak,
            f"clip starts at |amp|={onset_peak:.3f} with no fade-in",
        ))
    tail_peak = float(np.abs(x[-edge:]).max())
    if tail_peak > edge_abs_threshold:
        findings.append(BlipFinding(
            "hard_tail", duration_s * 1000.0, tail_peak,
            f"clip ends at |amp|={tail_peak:.3f} with no fade-out",
        ))

    # --- Frame RMS envelope.
    frame = max(1, int(sample_rate * _FRAME_MS / 1000.0))
    usable = (n // frame) * frame
    env = np.sqrt(np.mean(
        np.square(x[:usable].reshape(-1, frame)), axis=1,
    )) if usable >= frame else np.array([rms], dtype=np.float32)
    env_peak = float(env.max()) if env.size else 0.0
    loud = env > burst_silence_rms
    edge_frames = max(1, int(_EDGE_WINDOW_MS / _FRAME_MS))

    if env_peak > 0.0 and loud.any():
        # --- Run-aware body endpoints (2026-06-12). Short loud runs
        # at the clip edges isolated from the body by real silence are
        # the measured artifact class -- strip them from the body and
        # report them under their TRUE kind, with no peak-ratio gate
        # (the real bursts measured only 3-12% of clip peak, which the
        # legacy 25% gate rejected, misclassifying them as the gap
        # BEFORE the burst = "internal_dropout"). Mirrors the
        # trim_and_fade run-grouping so detector and trimmer agree.
        runs = _loud_runs(loud)
        bm = max(1, int(np.ceil(edge_burst_max_ms / _FRAME_MS)))
        bg = max(1, int(np.ceil(edge_burst_gap_ms / _FRAME_MS)))
        trailing_stripped: list = []
        leading_stripped: list = []
        while len(runs) > 1:
            s, e = runs[-1]
            gap = s - runs[-2][1] - 1
            if (e - s + 1) <= bm and gap >= bg:
                trailing_stripped.append(runs.pop())
            else:
                break
        while len(runs) > 1:
            s, e = runs[0]
            gap = runs[1][0] - e - 1
            if (e - s + 1) <= bm and gap >= bg:
                leading_stripped.append(runs.pop(0))
            else:
                break
        for s, e in trailing_stripped:
            findings.append(BlipFinding(
                "trailing_burst", s * _FRAME_MS,
                float(env[s:e + 1].max()),
                f"isolated {((e - s + 1) * _FRAME_MS):.0f}ms burst at "
                f"{s * _FRAME_MS:.0f}ms after speech ends",
            ))
        for s, e in leading_stripped:
            findings.append(BlipFinding(
                "leading_burst", s * _FRAME_MS,
                float(env[s:e + 1].max()),
                f"isolated {((e - s + 1) * _FRAME_MS):.0f}ms burst at "
                f"{s * _FRAME_MS:.0f}ms before speech onset",
            ))
        first_loud = runs[0][0]
        last_loud = runs[-1][1]

        # --- Isolated leading burst: noise spike near the clip start
        # separated from the speech body by silence. (Legacy
        # edge-window check; covers sub-200ms-gap cases the run strip
        # above doesn't. Skipped when the strip already reported.)
        head = env[:min(edge_frames, len(env))]
        head_loud = np.flatnonzero(head > burst_silence_rms)
        if not leading_stripped and head_loud.size:
            burst_start = int(head_loud[0])
            burst_end = burst_start
            while burst_end + 1 < len(head) and head[burst_end + 1] > burst_silence_rms:
                burst_end += 1
            after = env[burst_end + 1:]
            gap = 0
            for v in after:
                if v <= burst_silence_rms:
                    gap += 1
                else:
                    break
            burst_peak = float(head[burst_start:burst_end + 1].max())
            if (
                gap >= 3
                and (burst_end + 1 + gap) <= last_loud
                and burst_peak >= burst_min_ratio * env_peak
            ):
                pos = burst_start * _FRAME_MS
                findings.append(BlipFinding(
                    "leading_burst", pos, burst_peak,
                    f"isolated {((burst_end - burst_start + 1) * _FRAME_MS):.0f}ms "
                    f"burst at {pos:.0f}ms before speech onset",
                ))

        # --- Isolated trailing burst (mirror).
        tail_env = env[::-1][:min(edge_frames, len(env))]
        tail_loud = np.flatnonzero(tail_env > burst_silence_rms)
        if not trailing_stripped and tail_loud.size:
            burst_start = int(tail_loud[0])
            burst_end = burst_start
            while burst_end + 1 < len(tail_env) and tail_env[burst_end + 1] > burst_silence_rms:
                burst_end += 1
            after = env[::-1][burst_end + 1:]
            gap = 0
            for v in after:
                if v <= burst_silence_rms:
                    gap += 1
                else:
                    break
            burst_peak = float(tail_env[burst_start:burst_end + 1].max())
            if (
                gap >= 3
                and (len(env) - 1 - (burst_end + gap + 1)) >= first_loud
                and burst_peak >= burst_min_ratio * env_peak
            ):
                pos = (len(env) - 1 - burst_end) * _FRAME_MS
                findings.append(BlipFinding(
                    "trailing_burst", pos, burst_peak,
                    f"isolated {((burst_end - burst_start + 1) * _FRAME_MS):.0f}ms "
                    f"burst at {pos:.0f}ms after speech ends",
                ))

        # --- Internal dropout: two-tier rule (2026-06-12).
        # Tier 1 (dead air): a gap >= dead_air_ms is reported
        # regardless of edge energy -- no natural pause is that long.
        # Tier 2 (hard cut): a gap in [dropout_ms, dead_air_ms) is
        # reported ONLY when BOTH gap edges carry speech-level energy
        # (>= dropout_adjacent_ratio of the envelope peak) -- the
        # signature of a digital cut. Natural pauses (stop closures,
        # clause/sentence pauses) decay gradually into the gap, so
        # their edges sit at a few percent of peak and never trip it.
        dropout_frames = max(1, int(dropout_ms / _FRAME_MS))
        dead_air_frames = max(1, int(dead_air_ms / _FRAME_MS))
        body = loud[first_loud:last_loud + 1]
        run = 0
        for i, is_loud in enumerate(body):
            if not is_loud:
                run += 1
                continue
            if run >= dead_air_frames:
                pos = (first_loud + i - run) * _FRAME_MS
                findings.append(BlipFinding(
                    "internal_dropout", pos, run * _FRAME_MS,
                    f"{run * _FRAME_MS:.0f}ms dead air inside speech "
                    f"at {pos:.0f}ms",
                ))
            elif run >= dropout_frames:
                # body starts on a loud frame, so the first quiet run
                # begins at env index >= first_loud + 1 -- the pre
                # index below is always valid.
                pre = float(env[first_loud + i - run - 1])
                post = float(env[first_loud + i])
                if (
                    pre >= dropout_adjacent_ratio * env_peak
                    and post >= dropout_adjacent_ratio * env_peak
                ):
                    pos = (first_loud + i - run) * _FRAME_MS
                    findings.append(BlipFinding(
                        "internal_dropout", pos, run * _FRAME_MS,
                        f"{run * _FRAME_MS:.0f}ms hard cut inside "
                        f"speech at {pos:.0f}ms",
                    ))
            run = 0

    # --- Discontinuity click: instantaneous sample jump (bad join).
    # 2026-06-12 adjudication: an absolute jump threshold alone is the
    # wrong test -- ALL 112 live discontinuity findings (incl. every
    # clean post-trimmer-fix ack clip) measured jumps of 0.50-0.67 at
    # 0.82-1.33x the LOCAL envelope: legitimate loud high-frequency
    # speech, whose smooth waveform produces large adjacent-sample
    # diffs by construction (a pure tone's max diff is only ~1.41x its
    # own median diff at ANY frequency; broadband fricative noise caps
    # ~4-5x). A genuine click/offset join is an OUTLIER against its
    # neighbourhood: 10-50x the local median diff. So the jump must be
    # both audible (absolute floor) AND an outlier vs the median
    # adjacent-sample diff in a small window around it.
    if n > 1:
        diffs = np.abs(np.diff(x))
        max_jump = float(diffs.max())
        if max_jump > discontinuity_jump:
            j = int(np.argmax(diffs))
            w = max(1, int(sample_rate * discontinuity_window_ms / 1000.0))
            lo = max(0, j - w)
            hi = min(int(diffs.size), j + w + 1)
            local_med = float(np.median(diffs[lo:hi]))
            ratio = max_jump / max(local_med, 1e-6)
            if ratio >= discontinuity_outlier_ratio:
                pos = j / sample_rate * 1000.0
                findings.append(BlipFinding(
                    "discontinuity", pos, max_jump,
                    f"adjacent-sample jump {max_jump:.2f} "
                    f"({ratio:.0f}x the local diff median) at {pos:.0f}ms",
                ))

    # --- Clipping.
    clipped = float(np.mean(np.abs(x) >= 0.999))
    if clipped > clipping_fraction:
        findings.append(BlipFinding(
            "clipping", 0.0, clipped,
            f"{clipped * 100.0:.2f}% of samples at full scale",
        ))

    # --- DC offset.
    dc = float(np.mean(x))
    if abs(dc) > dc_offset_threshold:
        findings.append(BlipFinding(
            "dc_offset", 0.0, dc, f"mean amplitude {dc:+.3f}",
        ))

    return ClipQualityReport(duration_s, peak, rms, tuple(findings), label)


class OutputQualityWatcher:
    """Background analyzer for synthesized clips.

    ``submit`` is the hot-path entry: a non-blocking queue put (clips
    are DROPPED, never waited on, when the analyzer is behind). The
    daemon thread runs :func:`analyze_clip` and, for any finding, logs a
    WARNING and appends a JSONL record to ``jsonl_path``.
    """

    def __init__(
        self,
        *,
        jsonl_path: Optional[Path] = None,
        waveform_path: Optional[Path] = None,
        max_queue: int = 16,
        analyze_kwargs: Optional[dict] = None,
    ) -> None:
        """Create the watcher and start its daemon thread.

        Args:
            jsonl_path: findings log; None disables the JSONL sink.
            waveform_path: per-clip envelope stream (EVERY clip, clean or
                flagged) for the control panel's waveform pane; None
                disables it. Size-bounded (rewritten keeping the newest
                records when it grows past ~768 KB).
            max_queue: bounded queue size (overflow drops clips).
            analyze_kwargs: threshold overrides for :func:`analyze_clip`.
        """
        self._jsonl_path = jsonl_path
        self._waveform_path = waveform_path
        self._analyze_kwargs = dict(analyze_kwargs or {})
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._clips_seen = 0
        self._clips_flagged = 0
        self._dropped = 0
        self._findings_by_kind: dict[str, int] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="tts-output-quality", daemon=True,
        )
        self._thread.start()

    def submit(self, pcm: np.ndarray, sample_rate: int, label: str = "") -> None:
        """Enqueue a clip for analysis (hot path; never blocks/raises)."""
        try:
            self._queue.put_nowait((pcm, int(sample_rate), str(label)[:60]))
        except queue.Full:
            with self._lock:
                self._dropped += 1
        except Exception:  # noqa: BLE001 - hot path must never raise
            pass

    def stats(self) -> dict:
        """Session counters: clips seen/flagged/dropped + findings by kind."""
        with self._lock:
            return {
                "clips_seen": self._clips_seen,
                "clips_flagged": self._clips_flagged,
                "clips_dropped": self._dropped,
                "findings_by_kind": dict(self._findings_by_kind),
            }

    def close(self, timeout_s: float = 2.0) -> None:
        """Stop the analyzer thread (best-effort, used by tests/shutdown)."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        self._thread.join(timeout=timeout_s)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                pcm, sr, label = item
                report = analyze_clip(pcm, sr, label=label, **self._analyze_kwargs)
                with self._lock:
                    self._clips_seen += 1
                    if report.findings:
                        self._clips_flagged += 1
                        for f in report.findings:
                            self._findings_by_kind[f.kind] = (
                                self._findings_by_kind.get(f.kind, 0) + 1
                            )
                if report.findings:
                    kinds = ", ".join(
                        f"{f.kind}@{f.position_ms:.0f}ms" for f in report.findings
                    )
                    logger.warning(
                        "audio blip detected (%s) in clip %r (%.2fs, peak=%.2f): %s",
                        kinds, label, report.duration_s, report.peak,
                        "; ".join(f.detail for f in report.findings),
                    )
                    self._append_jsonl(report)
                self._append_waveform(pcm, sr, report)
            except Exception as e:  # noqa: BLE001 - analyzer must survive
                logger.debug("output-quality analysis failed: %s", e)

    # Envelope resolution for the control panel's waveform pane.
    _ENVELOPE_POINTS = 120
    _WAVEFORM_MAX_BYTES = 768 * 1024
    _WAVEFORM_KEEP_LINES = 80

    def _append_waveform(
        self, pcm: np.ndarray, sample_rate: int, report: ClipQualityReport,
    ) -> None:
        """Write one compact per-clip envelope record (EVERY clip).

        The control panel tails this stream and renders each clip's
        waveform with red markers at the analyzer's finding positions.
        Size-bounded: when the file grows past ``_WAVEFORM_MAX_BYTES``
        it is rewritten keeping the newest ``_WAVEFORM_KEEP_LINES``.
        """
        if self._waveform_path is None:
            return
        try:
            x = np.abs(_to_float(pcm))
            if x.size == 0:
                return
            n = self._ENVELOPE_POINTS
            usable = (x.size // n) * n
            if usable >= n:
                env = x[:usable].reshape(n, -1).max(axis=1)
            else:
                env = x
            record = {
                "timestamp": time.time(),
                "label": report.label,
                "duration_s": round(report.duration_s, 3),
                "peak": round(report.peak, 4),
                "env": [round(float(v), 3) for v in env],
                "findings": [
                    {"kind": f.kind, "position_ms": round(f.position_ms, 1)}
                    for f in report.findings
                ],
            }
            path = self._waveform_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            if path.stat().st_size > self._WAVEFORM_MAX_BYTES:
                lines = path.read_text(encoding="utf-8").splitlines()
                path.write_text(
                    "\n".join(lines[-self._WAVEFORM_KEEP_LINES:]) + "\n",
                    encoding="utf-8",
                )
        except Exception as e:  # noqa: BLE001 - sink failures are non-fatal
            logger.debug("waveform append failed: %s", e)

    def _append_jsonl(self, report: ClipQualityReport) -> None:
        if self._jsonl_path is None:
            return
        try:
            record = {
                "timestamp": time.time(),
                "label": report.label,
                "duration_s": round(report.duration_s, 3),
                "peak": round(report.peak, 4),
                "rms": round(report.rms, 4),
                "findings": [
                    {
                        "kind": f.kind,
                        "position_ms": round(f.position_ms, 1),
                        "magnitude": round(f.magnitude, 4),
                        "detail": f.detail,
                    }
                    for f in report.findings
                ],
            }
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as e:  # noqa: BLE001 - sink failures are non-fatal
            logger.debug("output-quality jsonl append failed: %s", e)


_watcher: Optional[OutputQualityWatcher] = None
_watcher_lock = threading.Lock()
# Process-wide kill switch (the test sweep disables the singleton so
# stubbed-synth unit tests never spawn analyzer threads or write to the
# live logs dir -- mirrors the observation-writer conftest pattern).
_enabled_override = True


def set_output_watcher_enabled(enabled: bool) -> None:
    """Process-wide override: ``False`` makes :func:`get_output_watcher`
    return None (closing any existing watcher). Used by the test
    sweep's session fixture; tests of the watcher itself opt back in.
    """
    global _enabled_override
    _enabled_override = bool(enabled)
    if not _enabled_override:
        reset_output_watcher()


def get_output_watcher() -> Optional[OutputQualityWatcher]:
    """Return the process-wide watcher, building it from config on first use.

    Returns None (and never raises) when the feature is disabled or
    construction fails -- callers treat None as "no watcher".
    """
    global _watcher
    if not _enabled_override:
        return None
    if _watcher is not None:
        return _watcher
    with _watcher_lock:
        if _watcher is not None:
            return _watcher
        try:
            from ultron.config import LOGS_DIR, get_config

            cfg = getattr(getattr(get_config(), "tts", None), "output_watch", None)
            if cfg is None or not getattr(cfg, "enabled", False):
                return None
            waveform_name = getattr(
                cfg, "waveform_jsonl_filename", "audio_waveform.jsonl",
            )
            _watcher = OutputQualityWatcher(
                jsonl_path=Path(LOGS_DIR) / getattr(
                    cfg, "jsonl_filename", "audio_quality.jsonl",
                ),
                waveform_path=(
                    Path(LOGS_DIR) / waveform_name
                    if getattr(cfg, "waveform_enabled", True) else None
                ),
                max_queue=int(getattr(cfg, "max_queue", 16)),
            )
            logger.info(
                "TTS output-quality watcher active (findings -> %s)",
                _watcher._jsonl_path,
            )
            return _watcher
        except Exception as e:  # noqa: BLE001 - fail-open
            logger.debug("output-quality watcher unavailable: %s", e)
            return None


def reset_output_watcher() -> None:
    """Tear down the singleton (tests + clean shutdown)."""
    global _watcher
    with _watcher_lock:
        if _watcher is not None:
            try:
                _watcher.close()
            except Exception:  # noqa: BLE001
                pass
            _watcher = None
