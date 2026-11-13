"""Markdown files with YAML frontmatter (used by skills and agents)."""

from pathlib import Path

import yaml

from micro_agent.config import ConfigError


def parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Split a markdown file into (YAML frontmatter, body)."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    try:
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
    except (ValueError, yaml.YAMLError) as e:
        raise ConfigError(f"Invalid frontmatter in {path}: {e}") from None
    if not isinstance(meta, dict):
        raise ConfigError(f"Frontmatter in {path} must be a mapping")
    return meta, body.strip()
