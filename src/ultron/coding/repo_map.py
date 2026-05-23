"""PageRank-weighted repo map with token-budget binary search.

Pattern lifted in spirit (not in source) from aider's ``repomap.py``
(Apache 2.0; see ``THIRD_PARTY_NOTICES.md``). This is the headline
batch-2 deliverable from the external-codebase catalog at
``F:\\reference_repos\\catalog\\01_aider.md``.

Pipeline:

  1. Walk the project tree for source files (``find_source_files``).
     Languages with vendored ``*-tags.scm`` queries
     (:mod:`ultron.coding.tree_sitter_tags`) are eligible.
  2. Extract per-file tags (defs + refs) via
     :func:`tree_sitter_tags.extract_tags`. Results memoized via
     :class:`ultron.utils.mtime_cache.MtimeCache`.
  3. Build a ``networkx.MultiDiGraph``:
       * Nodes = files (relative paths, POSIX).
       * Edges = referencing-file → defining-file, weighted by
         ``mul * sqrt(num_refs)``.
       * ``mul`` modifiers per identifier:
           - ``*10`` when the identifier appears in ``mentioned_idents``
           - ``*10`` when the identifier is 8+ chars AND snake/kebab/
             camelCase (heuristic for "real" names vs. ``i``/``x``)
           - ``*0.1`` when the identifier starts with ``_`` (private)
           - ``*0.1`` when the identifier is defined in 5+ places
             (common name, low signal)
       * ``mul`` per edge: ``*50`` when the referencer is in the
         chat-file set.
       * Self-edge (weight 0.1) for defs with no refs anywhere (works
         around tree-sitter quirks).
  4. Build a personalization vector:
       * Files in ``chat_fnames``: ``personalize``
       * Files in ``mentioned_fnames``: ``personalize``
       * Files whose path components match ``mentioned_idents``:
         ``personalize``
       * Files in the
         :func:`ultron.coding.important_files.is_important` allowlist:
         ``personalize * 0.5`` (creative extension — README et al. get
         a boost so they float to the top without inbound edges).
  5. Run ``nx.pagerank(G, weight="weight", personalization=...,
     dangling=...)``. Fail open to non-personalized PageRank on
     ``ZeroDivisionError``; return empty on a second failure.
  6. Distribute each node's rank across its outgoing edges
     proportionally → per-(file, ident) rank.
  7. Sort ranked tags by score descending; prepend the
     important-file allowlist (so a high-priority file always lands
     in the output even if it has zero inbound edges).
  8. Binary-search the largest prefix that fits in ``max_map_tokens``
     when rendered via the patched ``grep_ast.TreeContext``. 15 %
     tolerance, 30 iterations max.

Cache: optional :class:`MtimeCache` shared with the tag extractor.
Per-call in-memory tree cache keyed by ``(rel_fname, sorted(lois),
mtime)``.

Voice-utterance personalisation (creative extension #1 from the
catalog): :func:`extract_idents_from_text` mines a free-form text
string for snake_case / kebab-case / camelCase / dotted identifiers.
Callers (voice path) can feed the transcript through this and pass
the result as ``mentioned_idents`` — the map self-tunes to the
ongoing conversation.

Fail-open posture: every step degrades to a partial or empty result
rather than raising. Callers receive ``""`` when no map can be built;
they should treat that the same as "no repo map available".
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ultron.coding.important_files import is_important, promoted_score
from ultron.coding.tree_sitter_tags import Tag, extract_tags_for_files
from ultron.utils.mtime_cache import MtimeCache
from ultron.utils.token_budget import char_count_tokens


warnings.simplefilter("ignore", category=FutureWarning)


logger = logging.getLogger("ultron.coding.repo_map")


# Directories the walker skips wholesale. Mirrors
# :data:`ultron.coding.project_introspect.SKIP_DIRECTORIES` plus a few
# entries specific to source-graph use (we don't want vendored deps to
# pollute the PageRank).
SKIP_DIRECTORIES: FrozenSet[str] = frozenset({
    "__pycache__",
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "htmlcov",
    "coverage",
    ".coverage",
    "site-packages",
    "vendor",
    "bower_components",
    "tmp",
    "temp",
    ".turbo",
    ".parcel-cache",
    "models",       # ultron-specific: gitignored model weights
    "logs",
})

DEFAULT_MAX_MAP_TOKENS = 1024
DEFAULT_MAX_MAP_TOKENS_NO_CHAT_FILES = 8192
DEFAULT_TOLERANCE = 0.15
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_BINARY_SEARCH_DIVISOR = 25
LINE_TRUNCATE_LENGTH = 100


# Identifier-mining regex for :func:`extract_idents_from_text`.
# Matches snake_case (>=2 underscore-joined alphanum chunks), kebab-case
# (>=2 hyphen-joined chunks), camelCase / PascalCase (>=2 mixed-case
# words), and dotted notation (``module.attr.subattr``). Single-word
# identifiers are excluded — they generate too much noise from prose.
_IDENT_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b"),
    re.compile(r"\b[a-z][a-z0-9]+(?:-[a-z0-9]+)+\b"),
    re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b"),
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"),
)


# ---------------------------------------------------------------------------
# Source-file discovery
# ---------------------------------------------------------------------------


def find_source_files(directory: Path | str) -> List[Path]:
    """Walk ``directory`` collecting source files for the repo map.

    Skips :data:`SKIP_DIRECTORIES` and hidden directories (except a
    few carve-outs like ``.github``). A file is considered source when
    :func:`grep_ast.filename_to_lang` returns a non-empty language.
    Single-file inputs (when ``directory`` is itself a file) are
    returned unchanged.
    """
    root = Path(directory)
    if not root.is_dir():
        if root.is_file():
            return [root]
        return []

    try:
        from grep_ast import filename_to_lang  # type: ignore[import-not-found]
    except ImportError:
        # Fall back to extension-based heuristic.
        def filename_to_lang(name: str) -> Optional[str]:  # type: ignore[no-redef]
            return Path(name).suffix.lower().lstrip(".") or None

    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRECTORIES
            and (not d.startswith(".") or d in {".github", ".claude"})
        ]
        for fname in filenames:
            full = Path(dirpath) / fname
            try:
                if not full.is_file():
                    continue
                if full.stat().st_size == 0:
                    continue
            except OSError:
                continue
            if filename_to_lang(fname):
                out.append(full)
    return out


# ---------------------------------------------------------------------------
# Identifier mining (creative extension)
# ---------------------------------------------------------------------------


def extract_idents_from_text(text: str) -> Set[str]:
    """Mine a free-form text for snake_case / camelCase / kebab-case
    identifiers.

    Used by the voice path to feed the user's transcript into the
    PageRank personalization vector — when the user says "fix the
    parakeet streaming bug", ``parakeet_streaming`` (if present in the
    repo) gets a 10x weight boost on the next map render.

    Only multi-token identifiers are returned. Plain English words
    pass through unchanged.
    """
    if not text:
        return set()
    out: Set[str] = set()
    for pat in _IDENT_PATTERNS:
        out.update(pat.findall(text))
    return out


# ---------------------------------------------------------------------------
# grep_ast TreeContext compatibility shim
# ---------------------------------------------------------------------------


_GREP_AST_PATCHED = False


def _ensure_grep_ast_patched() -> None:
    """Replace ``grep_ast.grep_ast.get_parser`` with one that returns a
    standard ``tree_sitter.Parser``.

    The default ``grep_ast.tsl.get_parser`` ships a
    ``tree-sitter-language-pack`` Parser whose Node API is incompatible
    with the standard tree-sitter Python bindings (``parser.parse``
    rejects bytes, ``root_node`` is a callable rather than a property,
    etc.). ``TreeContext`` and our own tag extractor both want the
    standard API, so we substitute it at import time.

    Idempotent: subsequent calls are no-ops.
    """
    global _GREP_AST_PATCHED
    if _GREP_AST_PATCHED:
        return
    try:
        import grep_ast.grep_ast as ga_mod  # type: ignore[import-not-found]
        import grep_ast.tsl as ga_tsl  # type: ignore[import-not-found]
        import tree_sitter  # type: ignore[import-not-found]
    except ImportError:
        return

    def _patched_get_parser(lang: str) -> Any:
        return tree_sitter.Parser(ga_tsl.get_language(lang))

    ga_mod.get_parser = _patched_get_parser
    _GREP_AST_PATCHED = True


# ---------------------------------------------------------------------------
# RepoMap
# ---------------------------------------------------------------------------


class RepoMap:
    """PageRank-weighted repo map with token-budget binary search.

    Args:
        root: Project root directory. All file paths returned in the
            map are relative to this.
        max_map_tokens: Token budget when at least one file is in the
            chat set.
        max_map_tokens_no_chat: Token budget when no chat files are
            supplied (the "give a broader view" mode).
        mtime_cache: Optional MtimeCache shared with the tag extractor.
            When set, repeated calls reuse cached tags across map
            generations.
        token_counter: Callable mapping rendered text to token count.
            Defaults to :func:`char_count_tokens` (length // 4). Real
            callers should pass an LLM-specific tokenizer.

    The map is regenerated on every :meth:`get_map` call by default;
    pass an existing instance to share the tag cache across calls.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        max_map_tokens: int = DEFAULT_MAX_MAP_TOKENS,
        max_map_tokens_no_chat: int = DEFAULT_MAX_MAP_TOKENS_NO_CHAT_FILES,
        mtime_cache: Optional[MtimeCache] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.max_map_tokens = int(max_map_tokens)
        self.max_map_tokens_no_chat = int(max_map_tokens_no_chat)
        self._mtime_cache = mtime_cache
        self._token_counter = token_counter or char_count_tokens
        self._tree_cache: Dict[Tuple[str, Tuple[int, ...], float], str] = {}
        self._tree_context_cache: Dict[str, Dict[str, Any]] = {}
        _ensure_grep_ast_patched()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_map(
        self,
        *,
        chat_files: Iterable[Path | str] = (),
        other_files: Optional[Iterable[Path | str]] = None,
        mentioned_fnames: Iterable[str] = (),
        mentioned_idents: Iterable[str] = (),
        force_refresh: bool = False,
    ) -> str:
        """Build and return the rendered repo map.

        Args:
            chat_files: Files currently "in the chat" (i.e., already
                visible to the LLM). These get a 50x edge-weight
                boost as referencers and are EXCLUDED from the
                rendered output (we don't repeat what the model
                already sees).
            other_files: Candidate files to consider for the map.
                When omitted, walks the entire project tree.
            mentioned_fnames: Filenames the user explicitly named
                in the current turn.
            mentioned_idents: Identifiers (function/class names) the
                user mentioned. The voice path obtains these by
                running the transcript through
                :func:`extract_idents_from_text`.
            force_refresh: Ignored in the v1 implementation (always
                rebuilds). Reserved for parity with aider's interface.

        Returns:
            The rendered map string. Empty when no map could be
            generated (no source files, no tags, etc.).
        """
        _ensure_grep_ast_patched()

        chat_paths = [Path(f).resolve() for f in chat_files]
        mentioned_fnames_set = set(mentioned_fnames)
        mentioned_idents_set = set(mentioned_idents)

        if other_files is None:
            other_paths = find_source_files(self.root)
        else:
            other_paths = [Path(f).resolve() for f in other_files]

        # De-duplicate while preserving chat_files exclusion.
        chat_set = {str(p) for p in chat_paths}
        unique_others = [
            p for p in other_paths if str(p) not in chat_set
        ]

        if not unique_others and not chat_paths:
            return ""

        max_tokens = self.max_map_tokens
        if not chat_paths and self.max_map_tokens_no_chat > 0:
            max_tokens = self.max_map_tokens_no_chat
        if max_tokens <= 0:
            return ""

        try:
            ranked_tags = self._get_ranked_tags(
                chat_paths,
                unique_others,
                mentioned_fnames_set,
                mentioned_idents_set,
            )
        except RecursionError:
            logger.warning(
                "repo_map: PageRank hit RecursionError; disabling for this call"
            )
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("repo_map: get_ranked_tags failed: %s", exc)
            return ""

        if not ranked_tags:
            return ""

        # Hoist important files to the top so README et al. survive
        # token-budget pressure.
        ranked_tags = self._hoist_important(ranked_tags, unique_others)

        rendered = self._binary_search_to_budget(
            ranked_tags,
            chat_paths,
            max_tokens,
        )
        return rendered or ""

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def _get_ranked_tags(
        self,
        chat_paths: Sequence[Path],
        other_paths: Sequence[Path],
        mentioned_fnames: Set[str],
        mentioned_idents: Set[str],
    ) -> List[Tuple[Any, ...]]:
        """Build the file→file graph, run PageRank, distribute ranks
        across out-edges, return the sorted ``(rel_fname, ident,
        line)`` tuples (as ``Tag``-like rows) plus orphan files at
        the tail."""
        try:
            import networkx as nx  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("repo_map: networkx not installed; cannot rank")
            return []

        all_files = sorted(set(chat_paths).union(other_paths))
        chat_rel = {self._rel_posix(p) for p in chat_paths}
        if not all_files:
            return []

        personalize = 100.0 / len(all_files)
        personalization: Dict[str, float] = {}

        defines: Dict[str, Set[str]] = defaultdict(set)
        references: Dict[str, List[str]] = defaultdict(list)
        definitions: Dict[Tuple[str, str], Set[Tag]] = defaultdict(set)

        for path in all_files:
            rel = self._rel_posix(path)
            current_pers = 0.0
            if rel in chat_rel:
                current_pers += personalize
            if rel in mentioned_fnames:
                current_pers = max(current_pers, personalize)
            path_components = set(Path(rel).parts)
            path_components.add(Path(rel).stem)
            path_components.add(Path(rel).name)
            if path_components & mentioned_idents:
                current_pers += personalize
            if current_pers > 0:
                personalization[rel] = current_pers

            try:
                tags = extract_tags_for_files(
                    [path], self.root, cache=self._mtime_cache
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "repo_map: tag extraction failed for %s: %s", path, exc
                )
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel)
                    definitions[(rel, tag.name)].add(tag)
                elif tag.kind == "ref":
                    references[tag.name].append(rel)

        # When tree-sitter produces defs without refs across the board
        # (some languages emit only defs), synthesise self-references so
        # PageRank still terminates with non-zero personalization.
        if not references:
            references = {k: list(v) for k, v in defines.items()}

        idents = set(defines.keys()) & set(references.keys())
        G = nx.MultiDiGraph()

        # Self-edges for defs that have no refs anywhere — works around
        # tree-sitter quirks where a def isn't also automatically a ref.
        for ident, definers in defines.items():
            if ident in references:
                continue
            for definer in definers:
                G.add_edge(definer, definer, weight=0.1, ident=ident)

        for ident in idents:
            definers = defines[ident]
            mul = 1.0
            is_snake = ("_" in ident) and any(c.isalpha() for c in ident)
            is_kebab = ("-" in ident) and any(c.isalpha() for c in ident)
            is_camel = (
                any(c.isupper() for c in ident)
                and any(c.islower() for c in ident)
            )
            if ident in mentioned_idents:
                mul *= 10
            if (is_snake or is_kebab or is_camel) and len(ident) >= 8:
                mul *= 10
            if ident.startswith("_"):
                mul *= 0.1
            if len(definers) > 5:
                mul *= 0.1

            ref_counts = Counter(references[ident])
            for referencer, num_refs in ref_counts.items():
                use_mul = mul
                if referencer in chat_rel:
                    use_mul *= 50
                damped = math.sqrt(num_refs)
                for definer in definers:
                    G.add_edge(
                        referencer,
                        definer,
                        weight=use_mul * damped,
                        ident=ident,
                    )

        # Important-file personalization bonus (creative extension): a
        # README with zero inbound edges still floats up. Half-weight
        # so it doesn't dominate true reference-graph signals.
        for path in all_files:
            rel = self._rel_posix(path)
            if is_important(rel):
                personalization[rel] = (
                    personalization.get(rel, 0.0) + personalize * 0.5
                )

        if personalization:
            pers_args = dict(
                personalization=personalization,
                dangling=personalization,
            )
        else:
            pers_args = {}

        if G.number_of_nodes() == 0:
            return []

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            try:
                ranked = nx.pagerank(G, weight="weight")
            except ZeroDivisionError:
                logger.warning(
                    "repo_map: PageRank failed twice (ZeroDivisionError); "
                    "returning empty"
                )
                return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("repo_map: PageRank failed: %s", exc)
            return []

        # Distribute each node's rank across its outgoing edges
        # proportionally → per-(file, ident) rank.
        ranked_definitions: Dict[Tuple[str, str], float] = defaultdict(float)
        for src in G.nodes:
            src_rank = ranked.get(src, 0.0)
            out_edges = list(G.out_edges(src, data=True))
            if not out_edges:
                continue
            total_weight = sum(data.get("weight", 0.0) for _, _, data in out_edges)
            if total_weight <= 0:
                continue
            for _, dst, data in out_edges:
                portion = src_rank * data.get("weight", 0.0) / total_weight
                data["rank"] = portion
                ident = data.get("ident", "")
                if ident:
                    ranked_definitions[(dst, ident)] += portion

        sorted_defs = sorted(
            ranked_definitions.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )

        ranked_tags: List[Tuple[Any, ...]] = []
        for (rel_fname, ident), _rank in sorted_defs:
            if rel_fname in chat_rel:
                continue
            ranked_tags.extend(definitions.get((rel_fname, ident), set()))

        # Tail: files that didn't surface via the ranking get appended
        # as path-only rows so they're available if budget allows.
        seen_fnames = {t.rel_fname if isinstance(t, Tag) else t[0] for t in ranked_tags}
        other_rels = sorted({self._rel_posix(p) for p in other_paths})
        for rel in other_rels:
            if rel in chat_rel or rel in seen_fnames:
                continue
            ranked_tags.append((rel,))
            seen_fnames.add(rel)

        return ranked_tags

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _hoist_important(
        self,
        ranked_tags: List[Tuple[Any, ...]],
        other_paths: Sequence[Path],
    ) -> List[Tuple[Any, ...]]:
        """Push important-file rows ahead of the PageRank order.

        Without this, the README et al. would be ordered by their
        (small) PageRank score even though we know they're load-
        bearing. Aider does this via ``filter_important_files`` +
        prepending; we mirror that here.
        """
        already_in: Set[str] = set()
        for entry in ranked_tags:
            if isinstance(entry, Tag):
                already_in.add(entry.rel_fname)
            elif entry and entry[0]:
                already_in.add(entry[0])

        important_rels: List[str] = []
        for path in other_paths:
            rel = self._rel_posix(path)
            if rel in already_in:
                continue
            if is_important(rel):
                important_rels.append(rel)

        if not important_rels:
            return ranked_tags
        prefix = [(rel,) for rel in important_rels]
        return prefix + ranked_tags

    def _binary_search_to_budget(
        self,
        ranked_tags: List[Tuple[Any, ...]],
        chat_paths: Sequence[Path],
        max_tokens: int,
    ) -> str:
        """Binary-search the largest prefix that fits ``max_tokens``.

        Reuses the per-file ``TreeContext`` across iterations via
        ``_tree_context_cache``.
        """
        n = len(ranked_tags)
        if n == 0:
            return ""

        chat_rel = {self._rel_posix(p) for p in chat_paths}

        lower_bound = 0
        upper_bound = n
        best_tree = ""
        best_tokens = 0
        middle = min(max(max_tokens // DEFAULT_BINARY_SEARCH_DIVISOR, 1), n)
        iterations = 0
        while lower_bound <= upper_bound and iterations < DEFAULT_MAX_ITERATIONS:
            tree = self._to_tree(ranked_tags[:middle], chat_rel)
            tokens = self._token_counter(tree)
            iterations += 1

            within = tokens <= max_tokens
            err = abs(tokens - max_tokens) / max(max_tokens, 1)
            if within and tokens > best_tokens:
                best_tree = tree
                best_tokens = tokens
                if err < DEFAULT_TOLERANCE:
                    return best_tree
            if tokens < max_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1
            if lower_bound > upper_bound:
                break
            middle = (lower_bound + upper_bound) // 2
            if middle <= 0:
                middle = 1
            if middle > n:
                middle = n

        return best_tree

    def _to_tree(
        self,
        tags: Sequence[Tuple[Any, ...]],
        chat_rel: Set[str],
    ) -> str:
        """Render a slice of ranked tags as TreeContext output.

        Tags coming from PageRank are :class:`Tag` namedtuples; tail
        rows from the orphan-file pass are single-element tuples
        ``(rel_fname,)``. The renderer handles both: namedtuples
        contribute their ``line`` to the lines-of-interest set; tuples
        produce a path-only "file is in the repo" row.
        """
        if not tags:
            return ""

        def _sort_key(tag: Tuple[Any, ...]) -> Any:
            if isinstance(tag, Tag):
                return (tag.rel_fname, tag.line, tag.name, tag.kind)
            return (tag[0],)

        sorted_tags = sorted(tags, key=_sort_key)
        sorted_tags = list(sorted_tags) + [(None,)]

        output_parts: List[str] = []
        cur_fname: Optional[str] = None
        cur_abs: Optional[str] = None
        lois: Optional[List[int]] = None

        for tag in sorted_tags:
            this_rel = tag[0] if not isinstance(tag, Tag) else tag.rel_fname
            if this_rel in chat_rel:
                continue
            if this_rel != cur_fname:
                if lois is not None and cur_fname is not None and cur_abs is not None:
                    output_parts.append("")
                    output_parts.append(f"{cur_fname}:")
                    output_parts.append(
                        self._render_file_context(cur_abs, cur_fname, lois)
                    )
                    lois = None
                elif cur_fname is not None:
                    output_parts.append("")
                    output_parts.append(cur_fname)
                if isinstance(tag, Tag):
                    lois = []
                    cur_abs = tag.fname
                cur_fname = this_rel
            if lois is not None and isinstance(tag, Tag) and tag.line >= 0:
                lois.append(tag.line)

        rendered = "\n".join(output_parts).lstrip("\n")
        if not rendered:
            return ""
        # Long lines (minified JS, etc.) ruin token budgets — clip to
        # a fixed width.
        truncated = "\n".join(
            line[:LINE_TRUNCATE_LENGTH] for line in rendered.splitlines()
        )
        if not truncated.endswith("\n"):
            truncated += "\n"
        return truncated

    def _render_file_context(
        self,
        abs_fname: str,
        rel_fname: str,
        lois: Sequence[int],
    ) -> str:
        """Render the lines-of-interest of one file via ``TreeContext``.

        Cached per ``(rel_fname, sorted(lois), mtime)``; the underlying
        ``TreeContext`` object is also cached per file to skip the
        parse on re-render.
        """
        try:
            mtime = Path(abs_fname).stat().st_mtime
        except OSError:
            return ""
        sorted_lois = tuple(sorted(set(lois)))
        cache_key = (rel_fname, sorted_lois, mtime)
        cached = self._tree_cache.get(cache_key)
        if cached is not None:
            return cached

        ctx = self._get_or_build_context(abs_fname, rel_fname, mtime)
        if ctx is None:
            return ""
        try:
            ctx.lines_of_interest = set()
            ctx.add_lines_of_interest(sorted_lois)
            ctx.add_context()
            rendered = ctx.format()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "repo_map: TreeContext render failed for %s: %s",
                rel_fname,
                exc,
            )
            return ""
        self._tree_cache[cache_key] = rendered
        return rendered

    def _get_or_build_context(
        self,
        abs_fname: str,
        rel_fname: str,
        mtime: float,
    ) -> Optional[Any]:
        cached = self._tree_context_cache.get(rel_fname)
        if cached is not None and cached.get("mtime") == mtime:
            return cached.get("context")
        try:
            from grep_ast import TreeContext  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            code = Path(abs_fname).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        if not code.endswith("\n"):
            code += "\n"
        try:
            ctx = TreeContext(
                rel_fname,
                code,
                color=False,
                line_number=False,
                child_context=False,
                last_line=False,
                margin=0,
                mark_lois=False,
                loi_pad=0,
                show_top_of_file_parent_scope=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "repo_map: TreeContext construction failed for %s: %s",
                rel_fname,
                exc,
            )
            return None
        self._tree_context_cache[rel_fname] = {"context": ctx, "mtime": mtime}
        return ctx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rel_posix(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(self.root)
        except (ValueError, OSError):
            rel = path
        return str(rel).replace("\\", "/")


class RepoMapProviderCache:
    """Per-project RepoMap factory + provider for ProjectSupervisor.

    Holds a shared :class:`MtimeCache` and a dict of
    ``project_path -> RepoMap``. The ``__call__`` method is the
    contract ``ProjectSupervisor.repo_map_provider`` expects:
    ``(project_path, user_text) -> Optional[str]``.

    On each call:

      1. Look up or construct the RepoMap for ``project_path``.
      2. Mine ``user_text`` for identifiers via
         :func:`extract_idents_from_text`.
      3. Return ``rm.get_map(mentioned_idents=...)``.

    Errors degrade to ``None`` so the supervisor's decision flow is
    not perturbed.

    The class is thread-safe via an internal lock around the project
    dict; concurrent supervisor calls for the same project will share
    one RepoMap instance.
    """

    def __init__(
        self,
        *,
        max_map_tokens: int = DEFAULT_MAX_MAP_TOKENS,
        max_map_tokens_no_chat: int = DEFAULT_MAX_MAP_TOKENS_NO_CHAT_FILES,
        mtime_cache: Optional[MtimeCache] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.max_map_tokens = int(max_map_tokens)
        self.max_map_tokens_no_chat = int(max_map_tokens_no_chat)
        self._mtime_cache = mtime_cache
        self._token_counter = token_counter
        self._maps: Dict[str, RepoMap] = {}
        import threading

        self._lock = threading.Lock()

    def get_or_create(self, project_path: str) -> Optional[RepoMap]:
        """Return the RepoMap for ``project_path`` (constructing on miss)."""
        root = Path(project_path)
        try:
            root = root.resolve()
        except OSError:
            return None
        if not root.is_dir():
            return None
        key = str(root)
        with self._lock:
            existing = self._maps.get(key)
            if existing is not None:
                return existing
            rm = RepoMap(
                root,
                max_map_tokens=self.max_map_tokens,
                max_map_tokens_no_chat=self.max_map_tokens_no_chat,
                mtime_cache=self._mtime_cache,
                token_counter=self._token_counter,
            )
            self._maps[key] = rm
            return rm

    def __call__(
        self,
        project_path: str,
        user_text: str,
    ) -> Optional[str]:
        """Provider entry point — matches ProjectSupervisor's contract.

        Performs both flavours of utterance mining:

          * :func:`extract_idents_from_text` for identifier mentions
            (snake_case / camelCase / etc.) — passed as
            ``mentioned_idents`` to bias the PageRank graph weights.
          * :func:`ultron.coding.file_mention_resolver.resolve_mentions`
            for implicit file references — passed as
            ``mentioned_fnames`` to bias the personalization vector
            (catalog T16 wiring).

        Both extractions are fail-open: any exception in mining is
        logged at debug and the corresponding hint is omitted.
        """
        try:
            rm = self.get_or_create(project_path)
            if rm is None:
                return None
            idents = extract_idents_from_text(user_text)
            mentioned_fnames = self._resolve_file_mentions(
                project_path, user_text,
            )
            rendered = rm.get_map(
                mentioned_idents=idents,
                mentioned_fnames=mentioned_fnames,
            )
        except Exception as exc:                                    # noqa: BLE001
            logger.warning(
                "repo_map provider failed for project_path=%s: %s",
                project_path, exc,
            )
            return None
        return rendered or None

    def _resolve_file_mentions(
        self,
        project_path: str,
        user_text: str,
    ) -> set:
        """Mine ``user_text`` for implicit file references.

        Walks the project's source files, runs
        :func:`resolve_mentions` against the candidate list, and
        returns the POSIX-form relative paths the user implicitly
        referenced. Empty set when the resolver isn't available or
        finds no matches.
        """
        try:
            from ultron.coding.file_mention_resolver import resolve_mentions
        except ImportError:
            return set()
        try:
            root = Path(project_path).resolve()
            candidates: List[str] = []
            for path in find_source_files(root):
                try:
                    rel = path.resolve().relative_to(root)
                except (ValueError, OSError):
                    continue
                candidates.append(str(rel).replace("\\", "/"))
            if not candidates:
                return set()
            mentions = resolve_mentions(user_text, candidates)
        except Exception as exc:                                    # noqa: BLE001
            logger.debug(
                "repo_map: file_mention resolver failed for %s: %s",
                project_path, exc,
            )
            return set()
        return {m.path for m in mentions}


__all__ = [
    "RepoMap",
    "RepoMapProviderCache",
    "SKIP_DIRECTORIES",
    "extract_idents_from_text",
    "find_source_files",
]
