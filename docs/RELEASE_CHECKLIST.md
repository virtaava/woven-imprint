# Release Checklist

Use this checklist before publishing a new `woven-imprint` release to PyPI.

## Trigger Model

PyPI publishing is driven by GitHub Releases, not by ordinary pushes.

- Workflow: [.github/workflows/publish.yml](../.github/workflows/publish.yml)
- Trigger: `release.published`

That means a release requires:

1. a version bump committed to `master`
2. a pushed git tag or GitHub Release using that version
3. the GitHub Release to be published

## Pre-Release Checks

Before cutting a release:

1. CI is green on `master`
2. the intended version is set in [pyproject.toml](../pyproject.toml)
3. documentation reflects the new capability or behavior
4. any new provider/runtime path has at least one documented usage path

Recommended local checks:

```bash
ruff check .
ruff format --check .
pytest
uvx pyright --project pyrightconfig.json
python -m build
```

## Version Bump

Update the version in [pyproject.toml](../pyproject.toml):

- patch release for fixes and small backward-compatible additions
- minor release for larger backward-compatible feature additions
- major release for intentional breaking changes

Current release planning guidance:

- CI-only fixes do not require an immediate PyPI release
- release when downstream users need the new provider/runtime surface from PyPI

## Release Flow

1. bump version in `pyproject.toml`
2. commit and push the version bump
3. create a GitHub Release for that version
4. publish the release
5. confirm the `Publish to PyPI` workflow succeeds

## Post-Release Checks

After publishing:

1. verify the release run passed in GitHub Actions
2. verify the new version appears on PyPI
3. verify installation works:

```bash
pip install --upgrade woven-imprint==<version>
```

## Near-Term Release Guidance

The next release should happen when the first usable `gemma_edge` integration milestone is ready, not just because CI was repaired.

Expected next release shape:

- `0.5.3`
- includes:
  - `gemma_edge` provider seam
  - related CI/type fixes
  - any minimal docs needed for downstream use
