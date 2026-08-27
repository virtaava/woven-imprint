import json

from eval.external import report


def _item(qid, kind, label="CORRECT", category="1"):
    return {
        "qid": qid,
        "kind": kind,
        "category": category,
        "label": label,
        "bench": "locomo",
        "mode": "memory",
    }


def test_write_judge_sample_excludes_non_qa_kinds(tmp_path):
    items = [
        _item("q1", "qa"),
        _item("q2", "adversarial"),
        _item("q3", "abstain"),
    ]
    report.write_judge_sample(items, n=10, seed=1, results_dir=tmp_path)

    data = json.loads((tmp_path / "external_judge_sample.json").read_text())
    sample = data["locomo:memory"]
    assert [it["qid"] for it in sample] == ["q1"]


def test_write_judge_sample_writes_nothing_when_no_qa_items(tmp_path):
    items = [_item("q2", "adversarial"), _item("q3", "abstain")]
    report.write_judge_sample(items, n=10, seed=1, results_dir=tmp_path)

    assert not (tmp_path / "external_judge_sample.json").exists()


def test_write_is_atomic_and_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "x.json"
    report._write(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_write_results_leaves_no_tmp_files(tmp_path):
    results = {"run_id": "r1", "bench": "locomo", "mode": "memory", "summary": {}}
    report.write_results(results, results_dir=tmp_path)

    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "external_r1.json").exists()
    assert (tmp_path / "external_latest.json").exists()
