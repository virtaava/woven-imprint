import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from eval.external.locomo import load_locomo, load_locomo_plus
from eval.external.runner import (
    RunConfig,
    _assert_safe_to_aggregate,
    _checkpoint_and_verify_wal,
    _cleanup_mid_ingest_db,
    _load_json,
    _process_probe_memory,
    _save_json,
    judge,
    judge_plus,
    rejudge,
    run,
    run_plus,
)
from tests.helpers import FakeEmbedder, FakeLLM
from woven_imprint import clock
from woven_imprint.config import get_config

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
        if "memory awareness judge" in system.lower():
            # LoCoMo-Plus Cognitive judge: single user-role message, no separate system prompt
            # (see prompts.plus_judge_messages) — deterministic "correct" is enough for these
            # tests, which check record shape/checkpointing, not judge accuracy.
            self.call_count += 1
            return {"label": "correct", "reason": "scripted: links to evidence"}
        return super().generate_json(messages, **kw)


def _cfg(run_id: str, mode: str = "memory") -> RunConfig:
    return RunConfig(
        bench="locomo",
        mode=mode,
        run_id=run_id,
        dataset_path=FIXTURES / "locomo_mini.json",
        limit_conversations=1,
    )


def _plus_cfg(run_id: str, mode: str = "memory", reuse_run: str | None = None) -> RunConfig:
    return RunConfig(
        bench="locomo_plus",
        mode=mode,
        run_id=run_id,
        dataset_path=FIXTURES / "locomo_plus_mini.json",
        base_dataset_path=FIXTURES / "locomo_mini.json",
        reuse_run=reuse_run,
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


# --- LoCoMo-Plus (locomo_plus bench) -----------------------------------------------------


def _make_reuse_run(tmp_path, run_id: str = "plus-base") -> str:
    """Ingest the locomo_mini fixture (memory mode) so its DB/checkpoint can be reused."""
    run(_cfg(run_id), llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)
    return run_id


def test_run_plus_memory_mode_copies_db_and_judges_each_probe(tmp_path):
    reuse_run = _make_reuse_run(tmp_path)

    results = run_plus(
        _plus_cfg("plus-mem", reuse_run=reuse_run),
        llm=ScriptedLLM(),
        embedder=FakeEmbedder(),
        out_dir=tmp_path,
    )

    run_root = tmp_path / "plus-mem"
    probes = results["probes"]
    assert len(probes) == 2  # locomo_plus_mini.json has 2 probes
    for p in probes:
        assert (run_root / f"{p['probe_id']}.json").exists()
        assert not (run_root / f"{p['probe_id']}.db").exists()  # deleted after judging
        assert p["label"] in ("correct", "wrong")
        assert p["base_conv_id"] == "conv-mini-1"
    assert results["summary"]["n_probes"] == 2
    assert 0.0 <= results["summary"]["cognitive_accuracy"] <= 1.0


def test_run_plus_memory_mode_missing_reuse_run_raises(tmp_path):
    with pytest.raises(ValueError):
        run_plus(
            _plus_cfg("plus-noreuse"), llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path
        )

    with pytest.raises(ValueError):
        run_plus(
            _plus_cfg("plus-badreuse", reuse_run="does-not-exist"),
            llm=ScriptedLLM(),
            embedder=FakeEmbedder(),
            out_dir=tmp_path,
        )


def test_run_plus_memory_mode_resumes_without_recopying(tmp_path, monkeypatch):
    reuse_run = _make_reuse_run(tmp_path, "plus-base2")
    cfg = _plus_cfg("plus-resume", reuse_run=reuse_run)
    run_plus(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    run_root = tmp_path / "plus-resume"
    checkpoints = sorted(run_root.glob("*.json"))
    assert len(checkpoints) == 2
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in checkpoints}

    copy_calls: list[tuple] = []
    orig_copy2 = shutil.copy2

    def _counting_copy2(src, dst, *a, **kw):
        copy_calls.append((src, dst))
        return orig_copy2(src, dst, *a, **kw)

    monkeypatch.setattr("eval.external.runner.shutil.copy2", _counting_copy2)

    second = run_plus(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    assert copy_calls == []  # resumed run never re-copied a DB
    mtimes_after = {p.name: p.stat().st_mtime_ns for p in run_root.glob("*.json")}
    assert mtimes_before == mtimes_after
    assert len(second["probes"]) == 2


def test_run_plus_fullcontext_mode_creates_no_db(tmp_path):
    results = run_plus(
        _plus_cfg("plus-full", mode="fullcontext"),
        llm=ScriptedLLM(),
        embedder=FakeEmbedder(),
        out_dir=tmp_path,
    )

    run_root = tmp_path / "plus-full"
    assert not list(run_root.glob("*.db"))
    assert len(results["probes"]) == 2
    assert results["summary"]["n_probes"] == 2


def test_run_plus_writes_results_keyed_locomo_plus(tmp_path):
    run_plus(
        _plus_cfg("plus-results", mode="fullcontext"),
        llm=ScriptedLLM(),
        embedder=FakeEmbedder(),
        out_dir=tmp_path,
    )
    latest = json.loads((tmp_path / "external_latest.json").read_text())
    assert "locomo_plus:fullcontext" in latest


def test_run_plus_smoke_run_id_skips_results(tmp_path):
    run_plus(
        _plus_cfg("smoke-plus", mode="fullcontext"),
        llm=ScriptedLLM(),
        embedder=FakeEmbedder(),
        out_dir=tmp_path,
    )
    assert not (tmp_path / "external_smoke-plus.json").exists()
    assert not (tmp_path / "external_latest.json").exists()


def test_run_plus_limit_caps_probes(tmp_path):
    cfg = _plus_cfg("plus-limit", mode="fullcontext")
    cfg.limit_conversations = 1
    results = run_plus(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)
    assert len(results["probes"]) == 1


# --- Atomic checkpoints ------------------------------------------------------------------


def test_save_json_is_atomic_and_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "x.json"
    _save_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_load_json_treats_corrupt_file_as_missing(tmp_path, capsys):
    path = tmp_path / "x.json"
    path.write_text("{not valid json")

    result = _load_json(path, "default")

    assert result == "default"
    assert "warning" in capsys.readouterr().out.lower()


def test_load_json_missing_file_returns_default(tmp_path):
    assert _load_json(tmp_path / "nope.json", []) == []


def test_cleanup_mid_ingest_db_removes_db_and_wal_shm_without_ingest_checkpoint(tmp_path):
    db = tmp_path / "conv-x.db"
    wal = tmp_path / "conv-x.db-wal"
    shm = tmp_path / "conv-x.db-shm"
    db.write_text("stale")
    wal.write_text("stale-wal")
    shm.write_text("stale-shm")
    paths = {"db": db, "ingest": tmp_path / "conv-x.ingest.json", "answers": tmp_path / "x"}

    _cleanup_mid_ingest_db(paths)

    assert not db.exists() and not wal.exists() and not shm.exists()


def test_cleanup_mid_ingest_db_leaves_db_alone_when_ingest_checkpoint_present(tmp_path):
    db = tmp_path / "conv-x.db"
    ingest = tmp_path / "conv-x.ingest.json"
    db.write_text("real db")
    ingest.write_text("{}")
    paths = {"db": db, "ingest": ingest, "answers": tmp_path / "x"}

    _cleanup_mid_ingest_db(paths)

    assert db.exists()


# --- Judge label exact-match ------------------------------------------------------------


class _LabelLLM(FakeLLM):
    """Returns a fixed judge label verbatim, to test judge()'s exact-match parsing."""

    def __init__(self, label: str):
        super().__init__()
        self._label = label

    def generate_json(self, messages, **kw):
        return {"label": self._label, "reason": "scripted"}


def _qa_question():
    from datetime import datetime, timezone

    from eval.external.common import Question

    return Question(
        qid="q1",
        question="What pet did Alice adopt?",
        answer="a cat",
        category="1",
        evidence=[],
        asked_at=datetime(2023, 1, 2, tzinfo=timezone.utc),
        kind="qa",
    )


def test_judge_incorrect_label_is_not_treated_as_correct():
    result = judge(_qa_question(), "a dog", _LabelLLM("INCORRECT"))
    assert result["label"] == "WRONG"
    assert result["correct"] is False


def test_judge_exact_correct_label_with_whitespace_and_case():
    result = judge(_qa_question(), "a cat", _LabelLLM("  correct \n"))
    assert result["label"] == "CORRECT"
    assert result["correct"] is True


def test_judge_plus_incorrect_label_is_not_treated_as_correct():
    class _PlusLabelLLM(FakeLLM):
        def generate_json(self, messages, **kw):
            return {"label": "incorrectly linked", "reason": "scripted"}

    result = judge_plus("some evidence", "some response", _PlusLabelLLM())
    assert result["label"] == "wrong"
    assert result["correct"] is False


# --- Config restore on run() -------------------------------------------------------------


def test_run_restores_mutated_config_fields(tmp_path):
    woven_cfg = get_config()
    original_interval = woven_cfg.memory.fact_extraction_interval
    original_refresh = woven_cfg.maintenance.callbacks_refresh_on_session_end

    cfg = _cfg("cfgrestore")
    cfg.fact_extraction_interval = original_interval + 5
    run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    assert woven_cfg.memory.fact_extraction_interval == original_interval
    assert woven_cfg.maintenance.callbacks_refresh_on_session_end == original_refresh


# --- Unsharded-aggregation guard ----------------------------------------------------------


def test_run_refuses_unsharded_aggregation_when_answers_incomplete(tmp_path):
    cfg = _cfg("agg1")
    run_root = tmp_path / cfg.run_id
    run_root.mkdir(parents=True)
    # Fewer answer records than the fixture's 6 questions -> looks like a shard still working.
    (run_root / "conv-mini-1.answers.json").write_text(json.dumps([{"qid": "conv-mini-1-q0"}]))

    with pytest.raises(RuntimeError, match="Refusing unsharded aggregation"):
        run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)


def test_run_refuses_unsharded_aggregation_when_db_without_ingest_checkpoint(tmp_path):
    cfg = _cfg("agg2")
    run_root = tmp_path / cfg.run_id
    run_root.mkdir(parents=True)
    (run_root / "conv-mini-1.db").write_text("stale")

    with pytest.raises(RuntimeError, match="Refusing unsharded aggregation"):
        run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)


def test_force_aggregate_bypasses_the_guard(tmp_path):
    cfg = _cfg("agg3")
    cfg.force_aggregate = True
    run_root = tmp_path / cfg.run_id
    run_root.mkdir(parents=True)
    (run_root / "conv-mini-1.answers.json").write_text(json.dumps([{"qid": "conv-mini-1-q0"}]))

    # Doesn't raise; proceeds to (re-)ingest and answer normally.
    results = run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)
    assert results["summary"]["n_questions"] == 6


def test_assert_safe_to_aggregate_passes_on_fresh_run_dir(tmp_path):
    cfg = _cfg("agg4")
    conversations = run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)[
        "conversations"
    ]
    assert conversations  # sanity: fixture produced records
    # A freshly-finished run always looks safe to re-aggregate.
    from eval.external.locomo import load_locomo

    FIXTURES_LOCOMO = load_locomo(FIXTURES / "locomo_mini.json")
    _assert_safe_to_aggregate(tmp_path / cfg.run_id, FIXTURES_LOCOMO, cfg)  # no raise


# --- rejudge -------------------------------------------------------------------------------


def test_rejudge_restores_a_tampered_label(tmp_path):
    cfg = _cfg("rj1")
    run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    answers_path = tmp_path / "rj1" / "conv-mini-1.answers.json"
    answers = json.loads(answers_path.read_text())
    # ScriptedLLM's memory-mode answers are deterministically "Not mentioned" for every
    # non-color question, so every "qa" record here is genuinely WRONG (gold text like "a cat"
    # never appears in "Not mentioned") — tamper one to a fabricated CORRECT and confirm
    # rejudge (re-running the same deterministic judge) flips it back.
    qa_record = next(r for r in answers if r["kind"] == "qa")
    tampered_qid = qa_record["qid"]
    assert qa_record["label"] == "WRONG"  # sanity: this is genuinely wrong pre-tamper
    for r in answers:
        if r["qid"] == tampered_qid:
            r["label"] = "CORRECT"
            r["correct"] = True
    answers_path.write_text(json.dumps(answers))

    rejudge_cfg = RunConfig(
        bench="locomo",
        mode="memory",
        run_id="rj1",
        dataset_path=FIXTURES / "locomo_mini.json",
    )
    results = rejudge(rejudge_cfg, llm=ScriptedLLM(), out_dir=tmp_path)

    # rejudge re-derives the label from the (deterministic) judge rather than trusting the
    # tampered value — it comes back WRONG, same as the original untampered run.
    restored = json.loads(answers_path.read_text())
    fixed = next(r for r in restored if r["qid"] == tampered_qid)
    assert fixed["label"] == "WRONG"
    assert fixed["correct"] is False
    assert fixed["judge_version"] == 2

    assert (tmp_path / "external_rj1.json").exists()
    assert results["summary"]["n_questions"] == 6


def test_rejudge_only_unjudged_skips_already_rejudged_records(tmp_path):
    cfg = _cfg("rj2")
    run(cfg, llm=ScriptedLLM(), embedder=FakeEmbedder(), out_dir=tmp_path)

    answers_path = tmp_path / "rj2" / "conv-mini-1.answers.json"
    answers = json.loads(answers_path.read_text())
    for r in answers:
        r["judge_version"] = 2
        r["label"] = "TAMPERED"
    answers_path.write_text(json.dumps(answers))

    rejudge_cfg = RunConfig(
        bench="locomo",
        mode="memory",
        run_id="rj2",
        dataset_path=FIXTURES / "locomo_mini.json",
    )
    rejudge(rejudge_cfg, llm=ScriptedLLM(), out_dir=tmp_path, only_unjudged=True)

    unchanged = json.loads(answers_path.read_text())
    assert all(r["label"] == "TAMPERED" for r in unchanged)


# --- LoCoMo-Plus reuse-run WAL checkpoint (item 13) -----------------------------------------


def test_checkpoint_and_verify_wal_raises_while_a_reader_holds_the_wal_open(tmp_path):
    db_path = tmp_path / "base.db"
    writer = sqlite3.connect(str(db_path))
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE t(x)")
    writer.execute("INSERT INTO t VALUES (1)")
    writer.commit()

    # A second connection with an open read transaction blocks wal_checkpoint(TRUNCATE) from
    # actually truncating the WAL, even though the insert above is already committed.
    reader = sqlite3.connect(str(db_path))
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM t").fetchall()

    wal_path = db_path.with_name(db_path.name + "-wal")
    assert wal_path.exists() and wal_path.stat().st_size > 0

    try:
        with pytest.raises(RuntimeError, match="live WAL"):
            _checkpoint_and_verify_wal(db_path)
    finally:
        reader.close()
        writer.close()

    # No reader left holding it open -> checkpoint now succeeds and the data survives.
    _checkpoint_and_verify_wal(db_path)
    check = sqlite3.connect(str(db_path))
    try:
        assert check.execute("SELECT x FROM t").fetchall() == [(1,)]
    finally:
        check.close()


def test_process_probe_memory_refuses_a_source_db_with_a_live_wal(tmp_path):
    reuse_run = _make_reuse_run(tmp_path, "wal-base")
    reuse_root = tmp_path / reuse_run
    base_convs = load_locomo(FIXTURES / "locomo_mini.json")
    probes = load_locomo_plus(FIXTURES / "locomo_plus_mini.json", base_convs)
    probe = probes[0]
    conv = base_convs[0]

    source_db = reuse_root / f"{probe.base_conv_id}.db"
    wal_path = source_db.with_name(source_db.name + "-wal")

    # Write an extra row directly (bypassing the product path — just needs the WAL non-empty)
    # and hold a read transaction open on a second connection so it can't be checkpointed away.
    writer = sqlite3.connect(str(source_db))
    writer.execute("CREATE TABLE wal_probe_test(x)")
    writer.execute("INSERT INTO wal_probe_test VALUES (1)")
    writer.commit()

    reader = sqlite3.connect(str(source_db))
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM wal_probe_test").fetchall()

    assert wal_path.exists() and wal_path.stat().st_size > 0

    run_root = tmp_path / "wal-run"
    run_root.mkdir()
    cfg = RunConfig(bench="locomo_plus", mode="memory", run_id="wal-run", reuse_run=reuse_run)

    try:
        with pytest.raises(RuntimeError, match="live WAL"):
            _process_probe_memory(
                probe, conv, cfg, ScriptedLLM(), FakeEmbedder(), reuse_root, run_root
            )
        assert not (run_root / f"{probe.probe_id}.db").exists()  # never copied
    finally:
        reader.close()
        writer.close()

    # Closing the reader lets the checkpoint go through; the probe now processes normally.
    # (The DB copy is deleted by _process_probe_memory itself once judging is done — that's
    # existing, documented behavior — so we verify the row survived the checkpoint on the
    # *source* DB, which is what actually gets copied.)
    record = _process_probe_memory(
        probe, conv, cfg, ScriptedLLM(), FakeEmbedder(), reuse_root, run_root
    )
    assert record["probe_id"] == probe.probe_id
    assert not wal_path.exists() or wal_path.stat().st_size == 0
    check = sqlite3.connect(str(source_db))
    try:
        assert check.execute("SELECT x FROM wal_probe_test").fetchall() == [(1,)]
    finally:
        check.close()
