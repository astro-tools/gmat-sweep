"""Syntax check for the Python code blocks in docs/recipes/*.md.

`py_compile` parses each ```python``` block without executing it, so missing
runtime dependencies (`dask_jobqueue`, `dask_kubernetes`, `ray`) don't fail
the test. The goal is to catch typos and stale API names in the recipe
snippets — anything that would render but not parse.
"""

from __future__ import annotations

import py_compile
import re
import tempfile
from pathlib import Path

import pytest

RECIPES_DIR = Path(__file__).resolve().parent.parent / "docs" / "recipes"

_FENCE = re.compile(r"^```python\s*$\n(.*?)^```\s*$", re.DOTALL | re.MULTILINE)


def _python_blocks(md_path: Path) -> list[tuple[int, str]]:
    """Return (1-based block index, source) for every ```python``` fence in the file."""
    text = md_path.read_text()
    return [(i + 1, m.group(1)) for i, m in enumerate(_FENCE.finditer(text))]


def _all_blocks() -> list[tuple[Path, int, str]]:
    out: list[tuple[Path, int, str]] = []
    for md in sorted(RECIPES_DIR.glob("*.md")):
        for idx, src in _python_blocks(md):
            out.append((md, idx, src))
    return out


@pytest.mark.parametrize(
    ("md_path", "block_idx", "source"),
    _all_blocks(),
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_recipe_python_block_compiles(md_path: Path, block_idx: int, source: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source)
        tmp_path = Path(fh.name)
    try:
        py_compile.compile(
            str(tmp_path),
            dfile=f"{md_path.name}#python-block-{block_idx}",
            doraise=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def test_at_least_one_python_block_per_recipe() -> None:
    """Index page is allowed to have no python; the three orchestrator pages must each have ≥1."""
    required = {"slurm.md", "kubernetes.md", "ray-autoscaling.md"}
    found = {md.name for md in RECIPES_DIR.glob("*.md") if _python_blocks(md)}
    missing = required - found
    assert not missing, f"recipe pages missing python snippets: {sorted(missing)}"
