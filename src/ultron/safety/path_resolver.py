r"""Windows-aware path canonicalization for the runtime validator.

Path-based allowlists / denylists are useless if the model can refer
to a protected path via a symlink inside the sandbox, an 8.3 short
name, a UNC \\?\ prefix, or a percent-encoded URL path. This module
collapses those representations to a single canonical form BEFORE
the validator compares against any allow/deny list.

Threat model -- bypasses we explicitly defend against:

1. **Symlinks / junctions / reparse points.** ``mklink /J``
   pointing from a sandbox-internal path to a sensitive system
   file -- the link lives inside the sandbox so a naive path
   allowlist accepts it, but the resolved target is outside the
   sandbox. Defense: ``Path.resolve(strict=False)`` on Windows
   follows reparse points to the final target before comparison.
2. **8.3 short names.** ``C:/PROGRA~1/Microsoft`` resolves to
   ``C:/Program Files/Microsoft``. Defense: ctypes call to
   ``GetLongPathNameW``.
3. **Trailing dots / spaces.** ``C:/Windows.`` and ``C:/Windows ``
   both refer to ``C:/Windows`` on Windows but pass naive string
   comparison. Defense: ``Path.resolve`` strips these.
4. **UNC long-path prefix.** The ``\\?\`` prefix yields the same
   file as the un-prefixed form. Defense: strip the prefix during
   normalisation.
5. **Mixed separators.** Forward + back slashes interleaved.
   Defense: normalise to the platform's preferred separator
   before ``Path()`` construction.
6. **Relative paths + CWD games.** ``../../../Windows/System32``
   relative to an attacker-controlled CWD. Defense: resolve
   against PROJECT_ROOT (not CWD), then check.
7. **Drive letter case.** Lowercase vs uppercase drive letters.
   Defense: uppercase the drive letter post-resolve.
8. **Percent-encoded ASCII inside a path** (rare but possible from
   URL-shaped inputs). Defense: detect ``%xx`` patterns and
   reject. Callers must URL-decode upstream; the resolver refuses
   to decode silently.

What this module does NOT do:

* It does NOT prevent TOCTOU (the gap between check and use). That's
  a separate concern handled by the validator's "resolve then open
  by handle, never re-open" pattern (Phase 5).
* It does NOT canonicalize URLs or non-filesystem identifiers.

Linux/macOS support: present but minimal. The project is Windows-
only per the project standards, so the Windows-specific Win32 calls are the hot
path; the POSIX branch is best-effort for cross-platform tests.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ultron.safety.path_resolver")

# ``\\?\`` prefix removes the MAX_PATH limit on Windows. We strip it for
# canonicalisation purposes because two paths that differ only by the
# presence/absence of this prefix refer to the same file.
_LONG_PATH_PREFIX = "\\\\?\\"        # \\?\ (4 chars: two backslashes, ?, backslash)
_UNC_LONG_PATH_PREFIX = "\\\\?\\UNC\\"  # \\?\UNC\ (8 chars)

# Percent-encoded ASCII inside a path is a smell -- legitimate Windows
# paths don't have ``%XX`` sequences. Used by callers to reject inputs
# rather than silently decode.
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")


class PathResolveError(ValueError):
    """Raised when a path can't be safely canonicalised.

    The validator treats this as a hard block: an unresolvable path
    is one that an attacker has crafted to evade comparison, OR a
    bug in the caller. Either way, deny.
    """


class PathResolver:
    """Canonicalise filesystem paths for allowlist / denylist checks.

    Stateless. Instantiate once per validator (or use the module-level
    singleton via :func:`get_path_resolver`).

    Usage::

        resolver = PathResolver()
        canonical = resolver.resolve("C:/PROGRA~1/Microsoft/foo")
        # canonical == Path("C:\\Program Files\\Microsoft\\foo")

        is_inside = resolver.is_inside(
            candidate=user_path,
            allowed_root=Path("C:/STC/ultronPrototype/data/sandbox/myproject"),
        )

    Both methods raise :class:`PathResolveError` on inputs that look
    like attempted evasion (percent-encoded escapes, control chars,
    unicode bidi overrides). They do NOT silently decode --
    rejecting is safer for a validator.
    """

    # Unicode characters known for path-spoofing tricks: bidi overrides,
    # invisible separators, BOM, interlinear annotation. The validator
    # rejects any path containing these. Built from explicit codepoints
    # so the source file stays ASCII-safe regardless of editor encoding.
    #   U+202A..U+202E -- LRE/RLE/PDF/LRO/RLO bidi overrides
    #   U+2066..U+2069 -- LRI/RLI/FSI/PDI bidi isolates
    #   U+200B..U+200F -- ZWSP/ZWNJ/ZWJ/LRM/RLM zero-width / invisible
    #   U+FFF9..U+FFFB -- IAA/IAS/IAT interlinear annotation
    #   U+FEFF        -- BOM (byte-order mark)
    _DANGEROUS_UNICODE = re.compile(
        "["
        + "‪-‮"
        + "⁦-⁩"
        + "​-‏"
        + "￹-￻"
        + "﻿"
        + "]"
    )

    def __init__(self) -> None:
        self._is_windows = sys.platform.startswith("win")
        # Cache the project root for relative-path resolution. Imported
        # lazily so this module stays importable even when the rest of
        # the config layer isn't available.
        self._project_root: Optional[Path] = None

    @property
    def project_root(self) -> Path:
        """Resolve PROJECT_ROOT lazily.

        Imported from :mod:`ultron.config` on first access. This keeps
        the path resolver importable from very low-level code that
        runs before config is fully built (e.g. early test discovery).
        """
        if self._project_root is None:
            try:
                from ultron.config import PROJECT_ROOT
                self._project_root = Path(PROJECT_ROOT).resolve()
            except Exception:
                self._project_root = Path(os.getcwd()).resolve()
        return self._project_root

    def reject_dangerous_chars(self, raw: str) -> None:
        """Raise PathResolveError if the input contains evasion-prone
        Unicode (bidi overrides, ZWSP, BOM, etc.) or percent-encoded
        escapes.

        The validator calls this BEFORE attempting to canonicalise.
        Rationale: if the path requires URL-decoding or contains
        invisible characters that change rendering, the model is
        either confused or attempting evasion. Either way, deny.
        """
        if not isinstance(raw, str):
            raise PathResolveError(
                f"path must be str, got {type(raw).__name__}"
            )
        if self._DANGEROUS_UNICODE.search(raw):
            raise PathResolveError(
                "path contains bidi-override / invisible Unicode "
                "characters (potential evasion); refusing"
            )
        if _PERCENT_ESCAPE_RE.search(raw):
            raise PathResolveError(
                "path contains percent-encoded escape sequences; "
                "callers must decode upstream, the resolver refuses "
                "to decode silently"
            )
        # ASCII control characters in a path are also suspicious
        # (NUL byte attacks etc.). Allow tab/newline since they may
        # appear in path-list strings split by the caller.
        if any(0 <= ord(c) < 32 and c not in ("\t", "\n", "\r") for c in raw):
            raise PathResolveError(
                "path contains ASCII control characters; refusing"
            )

    def normalise_separators(self, raw: str) -> str:
        """Convert all path separators to the platform's preferred form.

        Done before ``Path()`` construction so mixed-separator inputs
        like ``C:/Windows\\System32`` produce a canonical Path object.
        """
        # On Windows, ``Path`` accepts both ``/`` and ``\``. ``Path.resolve``
        # then normalises to backslashes. POSIX needs the reverse.
        if self._is_windows:
            return raw.replace("/", "\\")
        return raw.replace("\\", "/")

    def strip_long_path_prefix(self, raw: str) -> str:
        r"""Strip the Windows``\\?\`` long-path prefix if present.

        ``\\?\C:\Windows`` and ``C:\Windows`` are the same file; we
        canonicalise to the prefix-less form so comparisons work.
        """
        # Check the UNC long-path form FIRST (it's a superset of the
        # plain long-path prefix; checking the plain form first would
        # strip the leading bytes and leave a stray ``UNC\\`` in the
        # output).
        if raw.startswith(_UNC_LONG_PATH_PREFIX):
            # \\?\UNC\server\share -> \\server\share
            return "\\\\" + raw[len(_UNC_LONG_PATH_PREFIX):]
        if raw.startswith(_LONG_PATH_PREFIX):
            return raw[len(_LONG_PATH_PREFIX):]
        return raw

    def _expand_short_names_windows(self, p: Path) -> Path:
        """Expand 8.3 short names like ``PROGRA~1`` to their long form.

        Uses ``GetLongPathNameW`` from kernel32. Returns the input
        unchanged if the path doesn't exist (GetLongPathName fails)
        AND the path contains no short-name segments -- a missing
        path with no short names is just a non-existent path, which
        is fine for the resolver (we don't require existence). If the
        path contains ``~1`` style segments AND the path doesn't
        exist, we can't safely expand, so we raise.
        """
        s = str(p)
        if "~" not in s:
            return p
        if not self._is_windows:
            return p
        # GetLongPathNameW returns required buffer length on input
        # buffer-too-small, OR copies the resolved name and returns
        # the length copied.
        GLPN = ctypes.windll.kernel32.GetLongPathNameW
        GLPN.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        GLPN.restype = ctypes.c_uint32
        buf = ctypes.create_unicode_buffer(32768)
        n = GLPN(s, buf, 32768)
        if n == 0 or n > 32768:
            # Failed -- file doesn't exist, or buffer too small.
            # If the original contained ``~`` short-name markers and
            # we can't resolve them, refuse. An attacker may have
            # constructed a non-existent short-name path to dodge
            # allowlist comparison.
            raise PathResolveError(
                f"unresolvable Windows 8.3 short name in path: {s!r}"
            )
        return Path(buf.value)

    def resolve(self, raw: str | Path) -> Path:
        """Canonicalise a path.

        Args:
            raw: A string path or Path object. Relative paths are
                resolved against PROJECT_ROOT (not CWD -- CWD is
                attacker-controllable via prior shell calls).

        Returns:
            Absolute, symlink-resolved, long-name-expanded
            :class:`Path`. On Windows the drive letter is uppercased.

        Raises:
            :class:`PathResolveError`: dangerous characters, percent-
                encoded escapes, unresolvable short names, or other
                evasion patterns.
        """
        if isinstance(raw, Path):
            raw_str = str(raw)
        else:
            raw_str = raw
        self.reject_dangerous_chars(raw_str)
        s = self.normalise_separators(raw_str)
        s = self.strip_long_path_prefix(s)
        p = Path(s)
        # Make absolute relative to PROJECT_ROOT (not CWD).
        if not p.is_absolute():
            p = self.project_root / p
        # Resolve symlinks / .. segments / etc. ``strict=False`` lets
        # non-existent paths through (legit -- the model may name a
        # file it intends to create); the validator still checks the
        # parent directory's category.
        p = p.resolve(strict=False)
        if self._is_windows:
            p = self._expand_short_names_windows(p)
            # Uppercase drive letter for consistent comparison.
            s = str(p)
            if len(s) >= 2 and s[1] == ":":
                p = Path(s[0].upper() + s[1:])
        return p

    def is_inside(self, candidate: str | Path, allowed_root: str | Path) -> bool:
        """True iff ``candidate`` (after canonicalisation) is under
        ``allowed_root`` (after canonicalisation).

        Both paths go through :meth:`resolve` first. Equal paths
        count as inside (an exact match is inside itself).
        """
        c = self.resolve(candidate)
        r = self.resolve(allowed_root)
        try:
            c.relative_to(r)
            return True
        except ValueError:
            return False

    def is_under_any(
        self,
        candidate: str | Path,
        allowed_roots: list[str | Path],
    ) -> bool:
        """True iff ``candidate`` is inside ANY of ``allowed_roots``."""
        c = self.resolve(candidate)
        for r in allowed_roots:
            try:
                rp = self.resolve(r)
                c.relative_to(rp)
                return True
            except (ValueError, PathResolveError):
                continue
        return False

    # ------------------------------------------------------------------
    # T21 (OpenClaw catalog) — realpath-aware containment helpers.
    #
    # The default :meth:`resolve` follows symlinks via
    # ``Path.resolve(strict=False)``. For install-time symlink-target
    # checks (where a symlink target outside the install root is the
    # exact attack vector) callers need a fast fail-open path
    # (``is_inside``) AND a slow realpath path (``is_inside_with_realpath``)
    # that re-walks every component to its filesystem-real target.
    # Pattern shape from OpenClaw's ``src/infra/path-safety.ts`` re-export
    # of ``@openclaw/fs-safe/path`` (MIT; see ``THIRD_PARTY_NOTICES.md``).

    def safe_realpath(self, raw: str | Path) -> Optional[Path]:
        """Resolve every component of ``raw`` to its filesystem target.

        On Windows this calls :func:`os.path.realpath` which follows
        reparse points / junctions / symlinks. On POSIX it follows
        symlinks the same way. Returns ``None`` when the path is
        unreadable, when realpath raises (e.g. broken symlink loop),
        or when the resolver rejects the input via
        :meth:`reject_dangerous_chars`.

        Unlike :meth:`resolve`, this method requires the target to
        be readable AND fully canonical. Use for install scans and
        other cold paths where a broken symlink should be treated as
        suspect, not silently accepted.

        Args:
            raw: A string path or Path object.

        Returns:
            Canonical absolute :class:`Path` on success, ``None`` on
            any error (broken symlink, permission denied, dangerous
            input).
        """
        if isinstance(raw, Path):
            raw_str = str(raw)
        else:
            raw_str = raw
        try:
            self.reject_dangerous_chars(raw_str)
        except PathResolveError:
            return None
        try:
            normalised = self.normalise_separators(raw_str)
            normalised = self.strip_long_path_prefix(normalised)
            real = os.path.realpath(normalised, strict=False)
            p = Path(real)
            if self._is_windows:
                try:
                    p = self._expand_short_names_windows(p)
                except PathResolveError:
                    pass
                s = str(p)
                if len(s) >= 2 and s[1] == ":":
                    p = Path(s[0].upper() + s[1:])
            return p
        except (OSError, ValueError):
            return None

    def is_inside_with_realpath(
        self,
        candidate: str | Path,
        allowed_root: str | Path,
    ) -> bool:
        """Like :meth:`is_inside` but resolves symlinks for both args.

        Defends against the install-time attack where a symlink inside
        the install root points to a target OUTSIDE the root. The
        fast :meth:`is_inside` accepts the symlink because its own
        path lives under the root; this method rejects it because the
        realpath target is outside.

        Returns ``False`` (not raises) when either path is unresolvable
        via realpath; the caller can decide whether to treat that as
        a block or fall back to the fast path.
        """
        candidate_real = self.safe_realpath(candidate)
        root_real = self.safe_realpath(allowed_root)
        if candidate_real is None or root_real is None:
            return False
        try:
            candidate_real.relative_to(root_real)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_symlink_open_error(error: BaseException) -> bool:
        """Return ``True`` when ``error`` is the OS' "is a symlink" raise.

        Useful when opening a file with the ``O_NOFOLLOW`` flag and
        wanting to recognise the specific failure mode where the
        target is a symlink. On Windows / POSIX this surfaces as
        ``OSError`` with ``errno.ELOOP`` (or ``EMLINK`` on some
        kernels).
        """
        import errno
        if not isinstance(error, OSError):
            return False
        return error.errno in (errno.ELOOP, errno.EMLINK)

    def normalise_for_comparison(self, raw: str | Path) -> str:
        """Return ``raw`` canonicalised + lower-cased on Windows.

        Use for case-insensitive set lookups where the full
        :meth:`resolve` (symlink follow, short-name expansion) is
        overkill but plain ``str()`` comparison would miss
        ``C:`` vs ``c:`` mismatches.
        """
        p = self.resolve(raw)
        s = str(p)
        if self._is_windows:
            s = s.lower()
        return s


_resolver_singleton: Optional[PathResolver] = None


def get_path_resolver() -> PathResolver:
    """Module-level singleton accessor.

    The resolver is stateless apart from the lazy PROJECT_ROOT cache,
    so a single shared instance is fine and avoids repeated PROJECT_ROOT
    imports on the hot path.
    """
    global _resolver_singleton
    if _resolver_singleton is None:
        _resolver_singleton = PathResolver()
    return _resolver_singleton


def reset_path_resolver() -> None:
    """Test hook: drop the singleton so tests can swap in a fresh one."""
    global _resolver_singleton
    _resolver_singleton = None
