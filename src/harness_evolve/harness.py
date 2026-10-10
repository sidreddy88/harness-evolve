"""
A harness as a directory of editable files.

A harness is everything around a frozen model that shapes how an agent works:
its prompt text, tool descriptions and settings. Keeping that surface in files
(not string literals in code) makes it explicit and bounded: the optimizer
edits files in a copy of the directory, diffs stay readable, and your agent
can be pointed at a candidate harness without code changes.

Layout:
    settings.json            numeric/string knobs (keys starting with "_" are docs;
                             "_doc": {key: what it does} is shown to the proposer)
    tool_descriptions.json   {tool_name: description shown to the model}
    *.prompt                 prompt templates, rendered with str.format(**values);
                             literal braces are written {{ }}

Your agent loads its harness with `load_harness(path)`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Harness:
    directory: Path
    settings: dict
    tool_descriptions: dict[str, str]
    templates: dict[str, str]

    def render(self, template: str, **values) -> str:
        try:
            text = self.templates[template]
        except KeyError:
            raise KeyError(f"harness {self.directory} has no template {template!r}.prompt") from None
        return text.format(**values)

    def tool_description(self, tool: str) -> str:
        try:
            return self.tool_descriptions[tool]
        except KeyError:
            raise KeyError(f"harness {self.directory} has no description for tool {tool!r}") from None

    def setting(self, key: str):
        try:
            return self.settings[key]
        except KeyError:
            raise KeyError(f"harness {self.directory} has no setting {key!r}") from None


def load_harness(directory: str | Path) -> Harness:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"harness directory {root} does not exist")
    settings_path, tools_path = root / "settings.json", root / "tool_descriptions.json"
    settings = {k: v for k, v in (json.loads(settings_path.read_text()) if settings_path.exists() else {}).items()
                if not k.startswith("_")}
    tools = json.loads(tools_path.read_text()) if tools_path.exists() else {}
    templates = {p.stem: p.read_text() for p in sorted(root.glob("*.prompt"))}
    return Harness(directory=root, settings=settings, tool_descriptions=tools, templates=templates)
