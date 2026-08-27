"""Tests for the CLI's argument handling."""

import argparse
import json

from tests.helpers import make_test_engine
from woven_imprint import cli


class _FakeEngine:
    """Records the jobs list cmd_maintain resolves and passes through."""

    def __init__(self):
        self.jobs_seen = "not called"

    def run_maintenance(self, character_id=None, jobs=None, budget=None):
        self.jobs_seen = jobs
        return []

    def list_characters(self):
        return []

    def close(self):
        pass


def _run_cmd_maintain(monkeypatch, jobs_arg):
    fake_engine = _FakeEngine()
    monkeypatch.setattr(cli, "_get_engine", lambda db, model: fake_engine)
    args = argparse.Namespace(
        db=None, model=None, character=None, jobs=jobs_arg, budget=None, json=True
    )
    monkeypatch.setattr(
        "builtins.print", lambda *a, **kw: None
    )  # cmd_maintain prints the JSON report
    cli.cmd_maintain(args)
    return fake_engine.jobs_seen


def test_maintain_jobs_strips_whitespace_around_commas(monkeypatch):
    """Regression for M3: `--jobs "buffer_hygiene, dedup"` used to fail on
    the un-stripped " dedup" token (leading space breaks the job-name
    lookup). Matches the MCP maintain tool's existing strip-and-drop-empty
    behavior."""
    jobs = _run_cmd_maintain(monkeypatch, "buffer_hygiene, dedup")
    assert jobs == ["buffer_hygiene", "dedup"]


def test_maintain_jobs_drops_empty_tokens(monkeypatch):
    jobs = _run_cmd_maintain(monkeypatch, "buffer_hygiene,, dedup,")
    assert jobs == ["buffer_hygiene", "dedup"]


def test_maintain_jobs_none_when_omitted(monkeypatch):
    jobs = _run_cmd_maintain(monkeypatch, "")
    assert jobs is None


def test_cmd_export_card_writes_v2_card(monkeypatch, tmp_path):
    """cmd_export_card mirrors cmd_export: match by name prefix, write JSON to -o."""
    engine = make_test_engine()
    engine.create_character("Ada", persona={"backstory": "A detective."})
    monkeypatch.setattr(cli, "_get_engine", lambda db: engine)
    out = tmp_path / "ada_card.json"
    args = argparse.Namespace(character="ada", db=None, output=str(out))
    cli.cmd_export_card(args)
    card = json.loads(out.read_text())
    assert card["spec"] == "chara_card_v2" and card["data"]["name"] == "Ada"
