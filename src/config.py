"""Configuration loading.

Every stage takes its parameters from a single YAML file so that the settings
reported in the paper can be read off one document rather than recovered from
the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/pipeline.yaml")


class Config:
    """Read-only view over the parsed YAML with dotted-path access."""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        # Relative paths in the config resolve against the repository root,
        # which is the parent of config/.
        return cls(data, root=path.resolve().parent.parent)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        """Resolve one entry of the `paths` block against the repository root."""
        value = Path(self._data["paths"][key])
        return value if value.is_absolute() else self.root / value

    @property
    def domains(self) -> list[str]:
        return list(self._data["domains"])
