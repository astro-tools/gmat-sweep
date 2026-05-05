# Supported versions

The CI matrix below is the authoritative supported set. `gmat-sweep`
runs on every cell on every PR, with both unit and integration suites enabled.

## CI matrix

| Axis      | Versions                                                       |
|-----------|----------------------------------------------------------------|
| GMAT      | R2025a, R2026a (R2026a is the primary development target)      |
| Python    | 3.10, 3.11, 3.12                                               |
| Operating system | Ubuntu (`ubuntu-latest`), Windows (`windows-latest`), macOS (`macos-latest`) |

That gives 2 × 3 × 3 = 18 cells covered on every PR.

### Notes per axis

#### GMAT

GMAT installs are provisioned in CI via
[`astro-tools/setup-gmat`](https://github.com/astro-tools/setup-gmat). Locally,
download a build from the
[GMAT SourceForge release page](https://sourceforge.net/projects/gmat/files/GMAT/);
[`gmat-run`'s install guide](https://astro-tools.github.io/gmat-run/install-gmat/)
walks through unpacking and pointing Python at it.

Older GMAT releases (R2022a and earlier) are not in the matrix. The GMAT
project skipped public R2023a and R2024a releases, so R2025a and R2026a are
the only releases supported.

#### Python

3.10 is the floor (`requires-python = ">=3.10"` in `pyproject.toml`). 3.13
is not in the v0.1 matrix; it will be added once `pyarrow` and `joblib`'s
loky backend ship stable wheels for it.

#### Operating system

`macos-latest` is Apple Silicon (arm64); R2025a and R2026a both ship
arm64-compatible `gmatpy` bindings, so both versions run natively on the
runner. GMAT installs are provisioned by `setup-gmat` on macOS the same way
they are on Linux and Windows.

## Runtime dependencies

`gmat-sweep` does **not** ship GMAT. It depends at runtime on:

- A working **GMAT install** discoverable on the host (see the
  [getting-started page](getting-started.md#install)).
- [**`gmat-run`**](https://github.com/astro-tools/gmat-run) ≥ 0.3 — the
  single-run primitive every worker calls into. Installed as a transitive
  dependency from PyPI.

`gmat-run` is in turn responsible for finding, importing, and bootstrapping
`gmatpy`. `gmat-sweep` itself never imports `gmatpy` directly — the import
happens inside each worker subprocess on first call. See the
[FAQ](faq.md#why-does-each-run-go-in-its-own-subprocess) for why.

## Beyond the support matrix

If your environment is outside the supported matrix, the package may still
work — it is pure Python and the only platform-specific bits are inherited
from `gmat-run` and the GMAT install itself. Just do not expect CI to catch
regressions for you.
