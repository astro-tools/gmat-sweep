# Contributing to gmat-sweep

Thanks for your interest. This page is the one place to learn the workflow.

## Getting set up

```bash
git clone https://github.com/astro-tools/gmat-sweep.git
cd gmat-sweep
uv sync --all-groups
```

You also need a local GMAT install to run integration tests. gmat-sweep does not
ship GMAT binaries and depends on `gmat-run` for the single-run primitive.
On Linux and Windows you can grab a build from
[SourceForge](https://sourceforge.net/projects/gmat/files/GMAT/); R2026a is the
primary development target.

## Branches and PRs

- One issue per branch. Branch names use a short prefix for type:
  - `feat/<slug>` — new capability, tied to a `type:feature` issue.
  - `fix/<slug>` — bug fix, tied to a `type:bug` issue.
  - `chore/<slug>` — infra / tooling / hygiene.
  - `docs/<slug>` — docs-only change.
- Open a PR against `main`. Put `Closes #<N>` in the PR description so the issue
  auto-closes on merge and the project board advances the card to Done.
- Squash-merge is the only merge method. The PR title becomes the squash commit
  subject — write it as a complete imperative sentence.

## Local checks before pushing

```bash
uv run pytest               # unit tests (integration tests are gated behind a marker)
uv run ruff check           # lint
uv run ruff format --check  # formatting
uv run mypy                 # types
```

CI re-runs all four on Ubuntu, Windows, and macOS. Integration tests run in
CI against a cached GMAT install; you do not need to run them locally unless
you are touching the worker, pool, or aggregation paths.

### Coverage thresholds

CI enforces coverage gates on the Ubuntu / Python 3.12 cell:

- Overall coverage must be ≥ 90%.
- Each of `src/gmat_sweep/grids.py`, `src/gmat_sweep/distributions.py`,
  `src/gmat_sweep/manifest.py`, and `src/gmat_sweep/aggregate.py` must be ≥ 95%.

To reproduce locally, sync the same extras the gate cell installs (`--extra
k8s` is needed alongside `dask`/`ray`/`plot` so the mock-based kubernetes
backend tests run instead of skipping at `pytest.importorskip`):

```bash
uv sync --all-groups --extra dask --extra ray --extra plot --extra k8s
uv run pytest -m "integration or not integration" --cov
uv run coverage report --fail-under=90
uv run coverage report --include='src/gmat_sweep/grids.py' --fail-under=95
uv run coverage report --include='src/gmat_sweep/distributions.py' --fail-under=95
uv run coverage report --include='src/gmat_sweep/manifest.py' --fail-under=95
uv run coverage report --include='src/gmat_sweep/aggregate.py' --fail-under=95
```

## Commit messages

Keep them short and imperative. One subject line, optional body.

- "Add full-factorial grid expansion"
- "Fix manifest truncation handling on partial writes"

Do not include AI or tool attribution trailers in commits, PR titles, PR descriptions,
or comments — see the repo-level convention.

## Scope discipline

gmat-sweep's scope is deliberately narrow: run an existing `.script` N times
under N different overrides via `gmat-run`, in parallel, and aggregate the
results into a multi-indexed pandas DataFrame. Before opening a feature issue,
check the existing issues and the roadmap in the README to make sure the work
belongs here.

- **Running a single mission →** [`gmat-run`](https://github.com/astro-tools/gmat-run).
- **Building missions in Python →** [`gmatpyplus`](https://github.com/weasdown/gmatpyplus).
- **Generating `.script` text from Python →** [`pygmat`](https://pypi.org/project/pygmat/).
- **Optimisation (gradient / Bayesian / population) →** out of scope; gmat-sweep
  is a parallel evaluator, not an optimiser.

## Questions

Open a [discussion](https://github.com/orgs/astro-tools/discussions) rather
than an issue for open-ended questions, usage help, or brainstorming. The
astro-tools org runs a single shared discussions space — there is no
per-repo discussions board.
