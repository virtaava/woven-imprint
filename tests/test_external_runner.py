import json
from pathlib import Path

import pytest

from eval.external.runner import RunConfig, run
from tests.helpers import FakeEmbedder, FakeLLM
from woven_imprint import clock

FIXTURES = Path(__file__).resolve().parent.parent / "eval" / "external" / "fixtures"


@pytest.fixture(autouse=True)
def _reset_clock():
    """The runner drives `clock.override()` as a plain (non-restoring) call throughout a
    benchmark run, by design (the harness owns simulated time for its whole run). Reset the
    global clock after each test here so it doesn't leak a stale override into later tests."""
    yield
    clock.override(None)


class ScriptedLLM(FakeLLM):
    """Deterministic LLM for runner tests — never touches the network.

    - QA calls (``generate``) that include retrieved memories ("Relevant memories" in the
      prompt) answer "blue" when the question is about color, else "Not mentioned".
    - Judge calls (``generate_json`` with the grader system prompt) return CORRECT iff the gold
      answer text appears in the model response, else WRONG.
    - Everything else (bookkeeping, session summaries) falls back to FakeLLM's defaults.
    """

    def generate(self, messages, **kw):
        default = super().generate(messages, **kw)
        content = messages[-1].get("content", "") if messages else ""
        if "Relevant memories" in content:
            return "blue" if "color" in content.lower() else "Not mentioned"
        return default

    def generate_json(self, messages, **kw):
        system = messages[0].get("content", "") if messages else ""
        if "expert grader" in system.lower():
            self.call_count += 1
            user = messages[-1].get("content", "") if len(messages) > 1 else ""
            gold = ""
            response = ""
            for line in user.splitlines():
                if line.startswith("Gold answer:"):
                    gold = line[len("Gold answer:") :].strip()
                elif line.startswith("Model response:"):
                    response = line[len("Model response:") :].strip()
            if gold and gold.lower() in response.lower():
                return {"label": "CORRECT"}
            return {"label": "WRONG"}
        return super().generate_json(messages, **kw)


def _cfg(run_id: str, mode: str = "memory") -> RunConfig:
    return RunConfig(
        bench="locomo",
        mode=mode,
        run_id=run_id,
        dataset_path=FIXTURES / "locomo_mini.json",
        limit_conversations=1,
    )


def test_run_memory_mode_checkpoints_and_summary(tmp_path):
    results = run(_cfg("t1"), llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    run_root = tmp_path / "t1"
    conv_id = "conv-mini-1"
    assert (run_root / f"{conv_id}.db").exists()
    assert (run_root / f"{conv_id}.ingest.json").exists()
    assert (run_root / f"{conv_id}.answers.json").exists()

    summary = results["summary"]
    assert 0.0 <= summary["overall_j"] <= 1.0
    assert set(summary["per_category"])  # non-empty
    assert "adversarial_accuracy" in summary

    answers = json.loads((run_root / f"{conv_id}.answers.json").read_text())
    qids = [a["qid"] for a in answers]
    assert len(answers) == 6  # every question in the fixture answered exactly once
    assert len(qids) == len(set(qids))


def test_run_memory_mode_resumes_without_reingesting(tmp_path):
    cfg = _cfg("t1")
    first = run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)
    second = run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    # Resume must not re-run ingestion: the persisted stats (and their wall-clock) are identical.
    assert first["ingest_totals"]["seconds"] == second["ingest_totals"]["seconds"]
    assert first["ingest_totals"]["llm_calls"] == second["ingest_totals"]["llm_calls"]

    answers = json.loads((tmp_path / "t1" / "conv-mini-1.answers.json").read_text())
    qids = [a["qid"] for a in answers]
    assert len(answers) == 6
    assert len(qids) == len(set(qids))


def test_run_fullcontext_mode_creates_no_db(tmp_path):
    results = run(
        _cfg("t2", mode="fullcontext"), llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path
    )

    run_root = tmp_path / "t2"
    assert not (run_root / "conv-mini-1.db").exists()
    assert (run_root / "conv-mini-1.answers.json").exists()
    assert results["summary"]["n_questions"] == 6
    assert results["ingest_totals"]["llm_calls"] == 0


def test_run_writes_results_and_judge_sample_under_out_dir(tmp_path):
    run(_cfg("t3"), llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    results_path = tmp_path / "external_t3.json"
    latest_path = tmp_path / "external_latest.json"
    sample_path = tmp_path / "external_judge_sample.json"
    assert results_path.exists() and latest_path.exists() and sample_path.exists()

    latest = json.loads(latest_path.read_text())
    assert "locomo:memory" in latest


def test_run_id_starting_with_smoke_skips_results_writing(tmp_path):
    run(_cfg("smoke-test"), llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    assert not (tmp_path / "external_smoke-test.json").exists()
    assert not (tmp_path / "external_latest.json").exists()


def test_shard_selects_subset_and_skips_results(tmp_path):
    from eval.external.__main__ import _parse_shard

    assert _parse_shard("1/3") == (1, 3) and _parse_shard(None) is None
    cfg = _cfg("shardtest", mode="fullcontext")
    cfg.shard = (1, 2)  # fixture has 1 conversation at index 0 → shard 1 gets nothing
    cfg.write_results = False
    res = run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)
    assert res["conversations"] == [] and not list(tmp_path.glob("external_*.json"))
    cfg.shard = (0, 2)
    res = run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)
    assert len(res["conversations"]) > 0 and not list(tmp_path.glob("external_*.json"))
