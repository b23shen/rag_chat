"""Prompts are Markdown files with {{variable}} slots, loaded by path.

Each workflow keeps its prompts in its own `prompts/` folder. A prompt's version is a hash of
its text, so every trace and extraction log records exactly which prompt produced an output.
"""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SLOT = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    path: Path
    text: str

    @property
    def variables(self) -> frozenset[str]:
        return frozenset(SLOT.findall(self.text))

    @property
    def version(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    def render(self, variables: Mapping[str, object] | None = None) -> str:
        """Fill every slot. Missing or unexpected variables are an error, never a silent blank."""
        given = dict(variables or {})
        missing = self.variables - given.keys()
        extra = given.keys() - self.variables
        if missing or extra:
            raise PromptError(
                f"{self.path.name}: missing {sorted(missing)}, unexpected {sorted(extra)}"
            )
        return SLOT.sub(lambda m: str(given[m[1]]), self.text)


@lru_cache
def load_prompt(path: Path) -> Prompt:
    return Prompt(path=path, text=path.read_text(encoding="utf-8").strip())
