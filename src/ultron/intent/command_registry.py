"""Declarative description-based command registry (T18, additive).

Pattern lifted in spirit (not in source) from aider's
``commands.py``'s ``cmd_*`` method discovery (Apache 2.0; see
``THIRD_PARTY_NOTICES.md``).

The catalog rated T18 a 3-5 dev-day "refactor of the routing
surface" that "touches a hot path" — high risk if done as a
replacement. So this module ships as a PURELY ADDITIVE registry:

  * Nothing in the existing intent classifier consults it.
  * Nothing in the orchestrator dispatches via it.
  * It exists so future surfaces — a ``/help`` voice command, a
    LLM-driven intent classifier with a per-phrase prompt corpus,
    an alternate dispatch layer for advanced users — can opt in
    without first ripping out the existing 22-value
    :class:`ultron.routing.intent_kinds.RoutingIntentKind` enum.

Each command is a frozen dataclass:

  * ``name`` — stable identifier (``"engage_gaming_mode"``,
    ``"open_calendar"``).
  * ``description`` — one-line text shown by ``/help``.
  * ``phrases`` — sample phrases the user might say. Useful as a
    prompt-corpus for embedding-based matching or as a quick-match
    table.
  * ``handler`` — callable invoked when the command fires
    (signature is caller's choice; the registry just hands it back).
  * ``examples`` — optional natural-language examples for richer
    help output.

Registration is via three paths:

  * :meth:`CommandRegistry.register` — programmatic.
  * :func:`command` decorator — ergonomic, attaches metadata to a
    function and registers it on the *module-level* default registry.
    Useful for "drop a new command in a single file" extensions.
  * :meth:`CommandRegistry.register_from_dict` — load from JSON / YAML
    config when the operator wants to script extensions without
    writing Python.

Matching is intentionally NOT yet wired to the intent recognizer.
The registry exposes a :meth:`match` helper that returns the best
phrase match by substring or simple normalized comparison — but the
existing recognizer continues to be the source of truth. Future
work can swap in embedding-based matching using the same
``phrases`` corpus.

Fail-open posture: every registry operation that could fail (file
load, conflicting names, missing handler) is logged and either
silently skipped or returned to the caller as a bool. Never raises
into production.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


logger = logging.getLogger("ultron.intent.command_registry")


# Sentinel handler used when a command is registered for help-text
# purposes only (no action to dispatch). Calling it logs a debug
# message and returns None.
def _noop_handler(*_args: Any, **_kwargs: Any) -> None:
    logger.debug("command_registry: noop handler invoked")


@dataclass(frozen=True)
class Command:
    """One registered command.

    Attributes:
        name: Stable identifier. Lowercase + underscores by
            convention but the registry doesn't enforce a shape.
        description: One-line human-readable summary. Shown in
            ``/help`` output and used as a prompt-corpus phrase.
        phrases: Sample utterances the user might say to trigger
            this command. Future work uses this for embedding-
            based intent matching.
        handler: Callable invoked when the command fires. Signature
            is caller's responsibility — the registry just stores
            the callable. Defaults to the noop handler so registries
            built for help-only can omit handlers.
        examples: Optional natural-language examples for richer
            help output. Empty tuple by default.
        tags: Free-form labels for grouping in ``/help`` and for
            external filtering (e.g. ``{"voice", "coding"}``).
    """

    name: str
    description: str
    phrases: Sequence[str] = field(default_factory=tuple)
    handler: Callable[..., Any] = _noop_handler
    examples: Sequence[str] = field(default_factory=tuple)
    tags: frozenset[str] = field(default_factory=frozenset)


class CommandRegistry:
    """Additive registry of voice / dispatch commands.

    Thread-safe (internal lock). Sole source-of-truth lookups go
    through :meth:`get`; bulk operations use :meth:`list_all`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._commands: Dict[str, Command] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, command: Command, *, overwrite: bool = False) -> bool:
        """Add ``command`` to the registry.

        Args:
            command: The :class:`Command` to register.
            overwrite: When True, replace any existing command with
                the same name. When False (default), conflicting
                names are rejected and a warning is logged.

        Returns:
            True on success, False on rejected conflict.
        """
        with self._lock:
            if not overwrite and command.name in self._commands:
                logger.warning(
                    "command_registry: rejected duplicate registration "
                    "for name=%r (existing: %r)",
                    command.name,
                    self._commands[command.name].description,
                )
                return False
            self._commands[command.name] = command
        return True

    def register_from_dict(
        self,
        entry: Dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> bool:
        """Build + register a :class:`Command` from a plain dict.

        Expected shape::

            {
                "name": "engage_gaming_mode",
                "description": "Switch ultron into gaming-mode VRAM profile.",
                "phrases": ["engage gaming mode", "switch to gaming mode"],
                "examples": ["I'm about to play Valorant"],
                "tags": ["voice", "vram"]
            }

        Optional ``handler`` is omitted (defaults to noop) because
        external configs don't typically ship callables.
        """
        try:
            name = str(entry["name"]).strip()
            description = str(entry.get("description") or "").strip()
        except (KeyError, TypeError) as exc:
            logger.debug(
                "command_registry: register_from_dict missing required key: %s",
                exc,
            )
            return False
        if not name:
            return False
        phrases = tuple(
            str(p) for p in (entry.get("phrases") or [])
            if isinstance(p, str) and p.strip()
        )
        examples = tuple(
            str(e) for e in (entry.get("examples") or [])
            if isinstance(e, str) and e.strip()
        )
        tags = frozenset(
            str(t) for t in (entry.get("tags") or [])
            if isinstance(t, str) and t.strip()
        )
        return self.register(
            Command(
                name=name,
                description=description,
                phrases=phrases,
                examples=examples,
                tags=tags,
            ),
            overwrite=overwrite,
        )

    def register_from_json_file(
        self,
        path: Path | str,
        *,
        overwrite: bool = False,
    ) -> int:
        """Load a JSON file of command dicts and register each.

        The file should contain a single top-level JSON array of
        command dicts (one dict per command). Returns the count of
        commands successfully registered.

        Fail-open: missing or malformed files log a warning and
        return 0.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "command_registry: could not load %s: %s", path, exc,
            )
            return 0
        if not isinstance(data, list):
            logger.warning(
                "command_registry: expected JSON array in %s, got %s",
                path, type(data).__name__,
            )
            return 0
        n = 0
        for entry in data:
            if isinstance(entry, dict) and self.register_from_dict(
                entry, overwrite=overwrite,
            ):
                n += 1
        return n

    def unregister(self, name: str) -> bool:
        """Remove the command by name. Returns True iff one was removed."""
        with self._lock:
            return self._commands.pop(name, None) is not None

    def clear(self) -> None:
        """Drop every registered command. Test escape hatch."""
        with self._lock:
            self._commands.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Command]:
        """Look up a command by name; None when unknown."""
        with self._lock:
            return self._commands.get(name)

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._commands

    def list_all(self) -> List[Command]:
        """Snapshot of every registered command, sorted by name."""
        with self._lock:
            return sorted(self._commands.values(), key=lambda c: c.name)

    def list_by_tag(self, tag: str) -> List[Command]:
        """Subset of commands carrying ``tag``."""
        with self._lock:
            return sorted(
                (c for c in self._commands.values() if tag in c.tags),
                key=lambda c: c.name,
            )

    def match(self, utterance: str) -> Optional[Command]:
        """Best-effort case-insensitive substring match against phrases.

        This is intentionally simple — embedding-based matching is
        future work. Returns the first command whose any phrase is
        a substring of the (lowercased) utterance, or None.

        Useful for quick-match tests + as a fallback when the
        embedding-based recognizer is unavailable.
        """
        if not utterance:
            return None
        needle = utterance.lower().strip()
        if not needle:
            return None
        with self._lock:
            for cmd in self._commands.values():
                for phrase in cmd.phrases:
                    if phrase and phrase.lower() in needle:
                        return cmd
        return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._commands)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.has(name)

    # ------------------------------------------------------------------
    # Help rendering
    # ------------------------------------------------------------------

    def format_help(
        self,
        *,
        tag_filter: Optional[str] = None,
        include_examples: bool = False,
    ) -> str:
        """Render a Markdown-style help listing for narration.

        Useful for a future ``/help`` voice command — "Ultron, what
        can you do?" → narrate this.
        """
        commands = (
            self.list_by_tag(tag_filter) if tag_filter else self.list_all()
        )
        if not commands:
            return "No commands registered."
        lines: List[str] = []
        for cmd in commands:
            lines.append(f"- **{cmd.name}**: {cmd.description}")
            if cmd.phrases:
                lines.append("    - Try: " + ", ".join(f'"{p}"' for p in cmd.phrases[:3]))
            if include_examples and cmd.examples:
                for ex in cmd.examples[:2]:
                    lines.append(f"    - Example: {ex}")
        return "\n".join(lines)


# Module-level default registry. The ``@command`` decorator targets
# this instance. Callers that want isolation construct their own
# :class:`CommandRegistry`.
DEFAULT_REGISTRY = CommandRegistry()


def command(
    name: str,
    *,
    description: str = "",
    phrases: Sequence[str] = (),
    examples: Sequence[str] = (),
    tags: Iterable[str] = (),
    registry: Optional[CommandRegistry] = None,
    overwrite: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as a :class:`Command`.

    Usage::

        @command(
            name="engage_gaming_mode",
            description="Switch to gaming-mode VRAM profile.",
            phrases=["engage gaming mode", "switch to gaming mode"],
            tags=["voice", "vram"],
        )
        def handle_engage_gaming_mode(...):
            ...

    The function's docstring is used as the description when the
    explicit ``description=`` argument is empty.

    Args:
        name: Command identifier (required).
        description: One-line summary. Falls back to the function's
            docstring's first line when empty.
        phrases / examples / tags: Forwarded to :class:`Command`.
        registry: Override the target registry. Default is the
            module-level :data:`DEFAULT_REGISTRY`.
        overwrite: When True, replace an existing command of the
            same name. Default False (logs + returns the original
            function unchanged on conflict).
    """
    # Explicit None-check instead of ``registry or DEFAULT_REGISTRY``:
    # CommandRegistry defines ``__len__``, so an empty registry is
    # falsy under boolean coercion and the ``or`` short-circuit would
    # silently fall back to the default — a footgun caught in tests.
    target = DEFAULT_REGISTRY if registry is None else registry

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_description = description.strip()
        if not resolved_description:
            doc = (fn.__doc__ or "").strip().splitlines()
            resolved_description = doc[0] if doc else ""
        target.register(
            Command(
                name=name,
                description=resolved_description,
                phrases=tuple(phrases),
                handler=fn,
                examples=tuple(examples),
                tags=frozenset(tags),
            ),
            overwrite=overwrite,
        )
        return fn

    return _wrap


__all__ = [
    "Command",
    "CommandRegistry",
    "DEFAULT_REGISTRY",
    "command",
]
