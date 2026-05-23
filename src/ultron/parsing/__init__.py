"""File-driven config / data parsing helpers with fail-open semantics.

The OpenHands ``_parse_skill_frontmatter`` + ``_load_skills_from_dir`` pair
established the pattern of "log warning, skip, continue" for every per-file
parse. A single malformed YAML file should never crash a subsystem at
startup. This package generalises that contract for ultron.

Pattern lineage attributed in ``THIRD_PARTY_NOTICES.md``.
"""

from ultron.parsing.frontmatter import (
    FrontmatterResult,
    parse_frontmatter,
    parse_frontmatter_text,
    walk_directory_with_frontmatter,
)

__all__ = [
    "FrontmatterResult",
    "parse_frontmatter",
    "parse_frontmatter_text",
    "walk_directory_with_frontmatter",
]
