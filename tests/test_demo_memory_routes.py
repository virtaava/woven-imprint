"""Tests for the editable-memory/fact HTTP routes on the FastAPI demo server."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_demo_server import app_client, _create_test_character  # noqa: F401,E402


def _setup(app_client):
    client, _t, engine = app_client
    cid = _create_test_character(client, "EditChar")["id"]
    char = engine.get_character(cid)
    m = char.memory.add("The visitor likes tea.", tier="core", importance=0.6)
    return client, engine, cid, m


def test_recall_strips_embeddings_and_has_ids(app_client):
    client, engine, cid, m = _setup(app_client)
    r = client.get("/api/memory", params={"character_id": cid, "query": "tea"})
    assert r.status_code == 200
    mem = r.json()["memories"][0]
    assert "embedding" not in mem and mem["id"] == m["id"] and "created_at" in mem


def test_patch_memory_content_importance_tier_pin(app_client):
    client, engine, cid, m = _setup(app_client)
    r = client.patch(
        f"/api/memory/{m['id']}",
        json={
            "character_id": cid,
            "content": "The visitor likes coffee.",
            "importance": 0.9,
            "tier": "bedrock",
            "pinned": True,
        },
    )
    assert r.status_code == 200
    mem = r.json()["memory"]
    assert mem["content"] == "The visitor likes coffee." and mem["tier"] == "bedrock"
    assert mem["metadata"]["pinned"] is True
    assert "embedding" not in mem
    r = client.get("/api/memory/pinned", params={"character_id": cid})
    assert [x["id"] for x in r.json()["memories"]] == [m["id"]]


def test_patch_memory_validation_and_404(app_client):
    client, engine, cid, m = _setup(app_client)
    assert (
        client.patch(
            f"/api/memory/{m['id']}", json={"character_id": cid, "tier": "granite"}
        ).status_code
        == 400
    )
    assert (
        client.patch(
            f"/api/memory/{m['id']}", json={"character_id": cid, "importance": 2}
        ).status_code
        == 400
    )
    assert (
        client.patch(
            "/api/memory/mem-nope", json={"character_id": cid, "importance": 0.5}
        ).status_code
        == 404
    )
    other = _create_test_character(client, "Other")["id"]
    assert (
        client.patch(
            f"/api/memory/{m['id']}", json={"character_id": other, "importance": 0.5}
        ).status_code
        == 404
    )


def test_delete_memory(app_client):
    client, engine, cid, m = _setup(app_client)
    assert client.delete(f"/api/memory/{m['id']}", params={"character_id": cid}).json() == {
        "deleted": True
    }
    assert client.delete(f"/api/memory/{m['id']}", params={"character_id": cid}).status_code == 404


def test_facts_patch_retract_delete(app_client):
    client, engine, cid, m = _setup(app_client)
    char = engine.get_character(cid)
    f = char.facts.add(
        subject="user", predicate="likes", object="tea", statement=m["content"], memory_id=m["id"]
    )
    r = client.get(f"/api/facts/{cid}")
    assert r.json()["facts"][0]["id"] == f["id"]
    r = client.patch(
        f"/api/facts/{f['id']}",
        json={"character_id": cid, "object": "coffee", "statement": "The visitor likes coffee."},
    )
    assert r.status_code == 200 and r.json()["fact"]["object"] == "coffee"
    r = client.delete(f"/api/facts/{f['id']}", params={"character_id": cid})  # default mode=retract
    assert r.status_code == 200 and r.json()["fact"]["metadata"]["retracted"] is True
    assert client.delete(
        f"/api/facts/{f['id']}", params={"character_id": cid, "mode": "delete"}
    ).json() == {"deleted": True}
    assert (
        client.patch(f"/api/facts/{f['id']}", json={"character_id": cid, "object": "x"}).status_code
        == 404
    )


def test_unauthed_mutations_rejected(app_client):
    client, engine, cid, m = _setup(app_client)
    from starlette.testclient import TestClient

    un = TestClient(client.app, base_url="http://127.0.0.1:7860")
    assert (
        un.patch(
            f"/api/memory/{m['id']}", json={"character_id": cid, "importance": 0.5}
        ).status_code
        == 401
    )
    assert un.delete(f"/api/memory/{m['id']}", params={"character_id": cid}).status_code == 401
