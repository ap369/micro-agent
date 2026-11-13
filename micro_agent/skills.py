"""Skills folder: validated here, served to the agent by deepagents' SkillsMiddleware."""

from pathlib import Path

from deepagents import FilesystemPermission

from micro_agent.config import ConfigError
from micro_agent.console import log
from micro_agent.markdown import parse_frontmatter

SKILLS_ROUTE = "/skills/"
# skills are mounted into the agent's filesystem but must never be modified
SKILLS_READ_ONLY = FilesystemPermission(operations=["write"], paths=[f"{SKILLS_ROUTE}**"], mode="deny")


class SkillCatalog:
    """The `skills/<name>/SKILL.md` folder, mounted read-only at /skills/."""

    def __init__(self, skills_dir: Path):
        self.dir = skills_dir
        self.names = self._validate()

    @property
    def enabled(self) -> bool:
        return bool(self.names)

    @property
    def sources(self) -> list[str] | None:
        """Value for deepagents' `skills=` argument."""
        return [SKILLS_ROUTE] if self.enabled else None

    def _validate(self) -> list[str]:
        """Check every SKILL.md up front so mistakes are config errors, not silently skipped skills."""
        if not self.dir.is_dir():
            log(f"[skills] directory not found, skipping: {self.dir}")
            return []
        names: list[str] = []
        for path in sorted(self.dir.glob("*/SKILL.md")):
            meta, _ = parse_frontmatter(path)
            for field in ("name", "description"):
                if not str(meta.get(field) or "").strip():
                    raise ConfigError(f"{path} needs a '{field}' in its frontmatter")
            if meta["name"] in names:
                raise ConfigError(f"Duplicate skill name '{meta['name']}' ({path})")
            names.append(meta["name"])
        log(f"[skills] found {len(names)}: {', '.join(names) or '-'}")
        return names
