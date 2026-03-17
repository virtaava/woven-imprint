# Contributing to Woven Imprint

Thank you for your interest in contributing.

## Getting Started

```bash
git clone https://github.com/virtaava/woven-imprint.git
cd woven-imprint
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
python eval/run_eval.py
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Submitting Changes

1. Fork the repo and create a branch from `master`
2. Make your changes
3. Ensure `ruff check`, `ruff format --check`, and `pytest` all pass
4. Submit a pull request with a clear description of the change

## Reporting Bugs

Use the [bug report template](https://github.com/virtaava/woven-imprint/issues/new?template=bug_report.md) to report issues.

## Feature Requests

Open an issue describing the use case and proposed solution.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
