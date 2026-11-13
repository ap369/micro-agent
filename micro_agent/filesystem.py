"""The agents' virtual filesystem: `/` = workspace, `/skills/` = skills folder (read-only)."""

from pathlib import Path

from deepagents.backends import CompositeBackend, FilesystemBackend

from micro_agent.skills import SKILLS_ROUTE, SkillCatalog


def build_backend(workspace: Path, skills: SkillCatalog) -> CompositeBackend:
    routes = {}
    if skills.enabled:
        routes[SKILLS_ROUTE] = FilesystemBackend(root_dir=skills.dir, virtual_mode=True)
    return CompositeBackend(default=FilesystemBackend(root_dir=workspace, virtual_mode=True), routes=routes)


def filesystem_prompt(skills: SkillCatalog) -> str:
    """Explain the layout above to the model; deepagents' own prompt only covers the file tools."""
    lines = ["# Filesystem", "", "Your filesystem root `/` is the workspace: save every deliverable there."]
    if skills.enabled:
        lines.append(f"`{SKILLS_ROUTE}` contains the skills and is read-only.")
    return "\n".join(lines)
