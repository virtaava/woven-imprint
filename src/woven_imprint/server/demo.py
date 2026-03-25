"""FastAPI demo server for woven-imprint."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles

from woven_imprint import __version__
from woven_imprint.config import get_config, WovenConfig, LLMConfig
from woven_imprint.engine import Engine
from woven_imprint.server.models import (
    ChatCompletionRequest,
    CreateCharacterRequest,
    ProviderConfigRequest,
    RecordMessageRequest,
)
from woven_imprint.server.services import (
    create_character_service,
    delete_character_service,
    end_session_service,
    export_character_service,
    extract_last_user_message,
    extract_user_id_from_messages,
    find_character_by_name_or_id,
    get_character_state_service,
    get_relationship_service,
    import_character_service,
    list_characters_service,
    migrate_character_service,
    recall_memories_service,
    record_message_service,
    reflect_character_service,
    start_session_service,
)

logger = logging.getLogger(__name__)

# Module-level state
_engine: Engine | None = None
_auth_token: str = ""
_mutation_lock: asyncio.Lock = asyncio.Lock()

STATIC_DIR = Path(__file__).parent.parent / "demo_static"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> None:
    """Validate Bearer token on protected routes."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != _auth_token:
        raise HTTPException(401, "Invalid or missing auth token")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    yield
    # Shutdown: flush any open sessions, close engine
    if _engine is not None:
        try:
            for char_info in _engine.list_characters():
                try:
                    char = _engine.get_character(char_info["id"])
                    if getattr(char, "_session_id", None):
                        char.end_session()
                except Exception:
                    pass
        except Exception:
            pass
        _engine.close()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    engine: Engine | None = None,
    port: int = 7860,
    host: str = "127.0.0.1",
    token: str | None = None,
) -> tuple[FastAPI, str]:
    """Create the FastAPI demo application.

    Args:
        engine: Pre-built Engine (uses module-level default if None).
        port: Port number (for CORS origin list).
        host: Host to bind to (affects CORS policy).
        token: Fixed auth token (random if None).

    Returns:
        (app, token) tuple.
    """
    global _engine, _auth_token

    if engine is not None:
        _engine = engine
    _auth_token = token or secrets.token_urlsafe(32)

    app = FastAPI(
        title="woven-imprint demo",
        version=__version__,
        lifespan=_lifespan,
    )

    # CORS — locked to localhost by default; if host is 0.0.0.0, allow all origins
    # (needed for Tailscale / remote access)
    if host in ("0.0.0.0", "::"):
        origins = ["*"]
    else:
        origins = [
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    # --- Health (no auth) ---
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": __version__}

    # --- Token-injected index ---
    @app.get("/")
    async def index_with_token():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return JSONResponse({"error": "Demo UI not built"}, 503)
        html = index_path.read_text()
        inject = f'<script>window.__WOVEN_TOKEN__="{_auth_token}";</script>'
        html = html.replace("</head>", f"{inject}</head>", 1)
        return HTMLResponse(html)

    # --- Characters ---
    @app.get("/api/characters", dependencies=[Depends(_check_auth)])
    async def list_characters():
        return {"characters": list_characters_service(_engine)}

    @app.post("/api/characters", dependencies=[Depends(_check_auth)])
    async def create_character(body: CreateCharacterRequest):
        async with _mutation_lock:
            persona = body.persona
            if isinstance(persona, str):
                persona = {"personality": persona}
            result = create_character_service(
                _engine, body.name, persona, body.birthdate,
            )
        status = 201 if result["created"] else 200
        return JSONResponse(result, status_code=status)

    # --- Character management (import/migrate before {character_id} routes) ---
    @app.post("/api/characters/import", dependencies=[Depends(_check_auth)])
    async def import_character(request: Request):
        body = await request.json()
        async with _mutation_lock:
            try:
                result = import_character_service(_engine, body)
                return result
            except Exception as exc:
                raise HTTPException(400, str(exc))

    @app.post("/api/characters/import-file", dependencies=[Depends(_check_auth)])
    async def import_character_file(request: Request):
        """Import character from an uploaded file.

        Supports: JSON exports, SillyTavern PNG cards, ChatGPT exports,
        markdown/text persona files. Uses CharacterImporter.from_file().
        """
        from fastapi import UploadFile, File, Form
        import tempfile

        # Parse multipart form data
        form = await request.form()
        file = form.get("file")
        name = form.get("name", "")

        if not file or not hasattr(file, "read"):
            raise HTTPException(400, "No file uploaded")

        # Save to temp file preserving extension
        original_name = getattr(file, "filename", "upload.bin") or "upload.bin"
        suffix = Path(original_name).suffix or ".bin"
        content = await file.read()

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        async with _mutation_lock:
            try:
                result = migrate_character_service(
                    _engine, name=str(name) if name else None, file_path=tmp_path
                )
                return result
            except Exception as exc:
                raise HTTPException(400, str(exc))
            finally:
                import os
                os.unlink(tmp_path)

    @app.post("/api/characters/migrate", dependencies=[Depends(_check_auth)])
    async def migrate_character(request: Request):
        body = await request.json()
        name = body.get("name", "")
        text = body.get("text")
        if not name:
            raise HTTPException(400, "Name is required")
        async with _mutation_lock:
            try:
                result = migrate_character_service(_engine, name, text=text)
                return result
            except Exception as exc:
                raise HTTPException(400, str(exc))

    @app.get("/api/characters/{character_id}", dependencies=[Depends(_check_auth)])
    async def get_character(character_id: str):
        try:
            return get_character_state_service(_engine, character_id)
        except KeyError:
            raise HTTPException(404, f"Character '{character_id}' not found")

    @app.delete("/api/characters/{character_id}", dependencies=[Depends(_check_auth)])
    async def delete_character(character_id: str):
        async with _mutation_lock:
            try:
                delete_character_service(_engine, character_id)
                return {"ok": True}
            except KeyError:
                raise HTTPException(404, f"Character '{character_id}' not found")

    @app.get("/api/characters/{character_id}/export", dependencies=[Depends(_check_auth)])
    async def export_character(character_id: str):
        try:
            data = export_character_service(_engine, character_id)
            return data
        except KeyError:
            raise HTTPException(404, f"Character '{character_id}' not found")

    @app.post("/api/characters/{character_id}/reflect", dependencies=[Depends(_check_auth)])
    async def reflect_character(character_id: str):
        async with _mutation_lock:
            try:
                result = reflect_character_service(_engine, character_id)
                return result
            except KeyError:
                raise HTTPException(404, f"Character '{character_id}' not found")

    # --- Sessions ---
    @app.post("/api/characters/{character_id}/session", dependencies=[Depends(_check_auth)])
    async def start_session(character_id: str):
        async with _mutation_lock:
            try:
                return start_session_service(_engine, character_id)
            except KeyError:
                raise HTTPException(404, f"Character '{character_id}' not found")

    @app.delete("/api/characters/{character_id}/session", dependencies=[Depends(_check_auth)])
    async def end_session(character_id: str):
        async with _mutation_lock:
            try:
                return end_session_service(_engine, character_id)
            except KeyError:
                raise HTTPException(404, f"Character '{character_id}' not found")

    # --- Record ---
    @app.post("/api/record", dependencies=[Depends(_check_auth)])
    async def record_message(body: RecordMessageRequest):
        async with _mutation_lock:
            try:
                record_message_service(
                    _engine,
                    body.character_id,
                    body.role,
                    body.content,
                    body.user_id,
                    strict_roles=False,
                )
            except KeyError:
                raise HTTPException(404, f"Character '{body.character_id}' not found")
            except ValueError as exc:
                raise HTTPException(400, str(exc))
        return {"ok": True}

    # --- Memory ---
    @app.get("/api/memory", dependencies=[Depends(_check_auth)])
    async def recall_memories(
        character_id: str,
        query: str,
        limit: int = 10,
        user_id: str | None = None,
    ):
        try:
            return recall_memories_service(
                _engine, character_id, query, limit=limit, user_id=user_id,
            )
        except KeyError:
            raise HTTPException(404, f"Character '{character_id}' not found")

    # --- Relationships ---
    @app.get("/api/relationships/{character_id}/{target_id}", dependencies=[Depends(_check_auth)])
    async def get_relationship(character_id: str, target_id: str):
        try:
            rel = get_relationship_service(_engine, character_id, target_id)
            return {"relationship": rel}
        except KeyError:
            raise HTTPException(404, f"Character '{character_id}' not found")

    # --- OpenAI-compatible chat completions ---
    @app.post("/v1/chat/completions", dependencies=[Depends(_check_auth)])
    async def chat_completions(body: ChatCompletionRequest):
        async with _mutation_lock:
            # Find character by model name
            match = find_character_by_name_or_id(_engine, body.model)
            if not match:
                raise HTTPException(404, f"No character matches model '{body.model}'")

            char = _engine.get_character(match["id"])
            user_msg = extract_last_user_message(body.messages)
            user_id = extract_user_id_from_messages(body.messages)

            if not user_msg:
                raise HTTPException(400, "No user message found in messages")

            # Ensure session
            if not getattr(char, "_session_id", None):
                char.start_session()

            response = char.chat(user_msg, user_id=user_id)

            # Read emotion/arc for metadata
            emotion = ""
            arc_phase = ""
            arc_tension = 0.0
            if hasattr(char, "emotion") and char.emotion:
                emotion = getattr(char.emotion, "mood", "")
            if hasattr(char, "arc") and char.arc:
                arc_phase = (
                    char.arc.current_phase.value
                    if hasattr(char.arc.current_phase, "value")
                    else str(char.arc.current_phase)
                )
                arc_tension = getattr(char.arc, "tension", 0.0)

            # Aggressive persistence (P2)
            try:
                char._save_state()
            except Exception:
                pass

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_msg) // 4,
                "completion_tokens": len(response) // 4,
                "total_tokens": (len(user_msg) + len(response)) // 4,
            },
            "woven_imprint": {
                "character": match["name"],
                "emotion": emotion,
                "arc": {"phase": arc_phase, "tension": arc_tension},
            },
        }

    # --- Provider config ---
    @app.get("/api/config/provider", dependencies=[Depends(_check_auth)])
    async def get_provider_config():
        cfg = get_config()
        return {
            "provider": cfg.llm.llm_provider,
            "model": cfg.llm.model,
            "base_url": cfg.llm.base_url,
            "api_key_configured": bool(cfg.llm.api_key),
        }

    @app.post("/api/config/provider", dependencies=[Depends(_check_auth)])
    async def set_provider_config(body: ProviderConfigRequest):
        from woven_imprint.providers import create_llm

        cfg = get_config()
        cfg.llm.llm_provider = body.provider
        cfg.llm.model = body.model
        if body.api_key is not None:
            cfg.llm.api_key = body.api_key
        if body.base_url is not None:
            cfg.llm.base_url = body.base_url

        # Rebuild engine LLM
        try:
            _engine.llm = create_llm(cfg=cfg)
        except Exception as exc:
            raise HTTPException(500, f"Failed to create LLM: {exc}")

        return {
            "provider": body.provider,
            "model": body.model,
            "base_url": body.base_url,
            "api_key_configured": bool(body.api_key or cfg.llm.api_key),
        }

    @app.post("/api/config/provider/test", dependencies=[Depends(_check_auth)])
    async def test_provider(body: ProviderConfigRequest):
        from woven_imprint.providers import create_llm

        test_cfg = WovenConfig()
        test_cfg.llm = LLMConfig(
            llm_provider=body.provider,
            model=body.model,
            api_key=body.api_key or "",
            base_url=body.base_url or "",
        )
        try:
            llm = create_llm(cfg=test_cfg)
            result = llm.generate(
                [{"role": "user", "content": "Say hello in exactly 3 words."}],
                temperature=0.0,
            )
            return {"success": True, "message": f"Connected. Response: {result[:100]}"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    @app.get("/api/config/models", dependencies=[Depends(_check_auth)])
    async def list_available_models(
        provider: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """Fetch available models from a provider by querying its API.

        All providers are queried live — no hardcoded lists. Falls back
        to the configured API key if none is passed explicitly.
        """
        import httpx

        models: list[str] = []
        cfg = get_config()
        key = api_key or cfg.llm.api_key or ""

        async def _try_ollama_tags(url: str) -> list[str]:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{url}/api/tags")
                    if resp.status_code == 200:
                        return [m["name"] for m in resp.json().get("models", [])]
            except Exception:
                pass
            return []

        async def _try_openai_models(url: str, auth_key: str = "") -> list[str]:
            try:
                hdrs: dict[str, str] = {}
                if auth_key:
                    hdrs["Authorization"] = f"Bearer {auth_key}"
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{url}/v1/models", headers=hdrs)
                    if resp.status_code == 200:
                        return [m["id"] for m in resp.json().get("data", [])]
            except Exception:
                pass
            return []

        async def _try_anthropic_models(auth_key: str) -> list[str]:
            try:
                hdrs = {
                    "x-api-key": auth_key,
                    "anthropic-version": "2023-06-01",
                }
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        "https://api.anthropic.com/v1/models", headers=hdrs
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", [])
                        return [m["id"] for m in data]
            except Exception:
                pass
            return []

        if provider == "ollama":
            url = (base_url or "http://localhost:11434").rstrip("/")
            models = await _try_ollama_tags(url)
            if not models:
                models = await _try_openai_models(url)

        elif provider == "anthropic":
            if key:
                models = await _try_anthropic_models(key)

        elif provider == "openai":
            if base_url:
                url = base_url.rstrip("/")
                bare = url.removesuffix("/v1")
                # Try Ollama tags first (for local servers)
                models = await _try_ollama_tags(bare)
                if not models:
                    models = await _try_openai_models(bare, key)
            else:
                # Default OpenAI API
                models = await _try_openai_models(
                    "https://api.openai.com", key
                )

        return {"models": models}

    # --- Static files (after all API routes) ---
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")

    return app, _auth_token


# ---------------------------------------------------------------------------
# Meridian seed helper
# ---------------------------------------------------------------------------

def _import_seed_db(engine: Engine, seed_path: Path) -> None:
    """Import Meridian character data from pre-built seed database."""
    import sqlite3

    seed_conn = sqlite3.connect(str(seed_path))
    seed_conn.row_factory = sqlite3.Row

    row = seed_conn.execute(
        "SELECT * FROM characters WHERE name = 'Meridian' LIMIT 1"
    ).fetchone()
    if not row:
        seed_conn.close()
        return

    char_id = row["id"]

    from woven_imprint.data.meridian_persona import MERIDIAN_PERSONA, MERIDIAN_BIRTHDATE
    char = engine.create_character(
        name="Meridian",
        persona=MERIDIAN_PERSONA,
        birthdate=MERIDIAN_BIRTHDATE,
        character_id=char_id,
    )

    memories = seed_conn.execute(
        "SELECT content, tier, importance FROM memories WHERE character_id = ?",
        (char_id,),
    ).fetchall()

    for mem in memories:
        char.memory.add(
            content=mem["content"],
            tier=mem["tier"],
            importance=mem["importance"],
        )

    seed_conn.close()
    logger.info(f"Imported {len(memories)} seed memories for Meridian")


def _seed_meridian_if_needed(engine: Engine) -> None:
    """Seed the Meridian demo character if not already present."""
    chars = engine.list_characters()
    for c in chars:
        if c["name"].lower() == "meridian":
            return

    # Try seed DB first
    seed_db = Path(__file__).parent.parent / "data" / "meridian_seed.db"
    if seed_db.exists():
        _import_seed_db(engine, seed_db)
        return

    # Fallback: create with persona only (no pre-built knowledge)
    from woven_imprint.data.meridian_persona import MERIDIAN_PERSONA, MERIDIAN_BIRTHDATE
    engine.create_character(
        name="Meridian",
        persona=MERIDIAN_PERSONA,
        birthdate=MERIDIAN_BIRTHDATE,
    )
    logger.info("Seeded Meridian character for demo")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_demo_server(
    port: int = 7860,
    host: str = "127.0.0.1",
    db_path: str | None = None,
    model: str | None = None,
    no_browser: bool = False,
) -> None:
    """Start the demo server (CLI entry point).

    Args:
        port: Port to bind to.
        host: Host to bind to. Default 127.0.0.1 (localhost only).
              Use 0.0.0.0 to expose on all interfaces (e.g. Tailscale).
        db_path: SQLite database path.
        model: LLM model name override.
        no_browser: Skip opening browser.
    """
    import uvicorn
    from woven_imprint.providers import create_llm, create_embedding

    cfg = get_config()
    if db_path:
        cfg.storage.db_path = db_path
    if model:
        cfg.llm.model = model

    resolved_db = cfg.storage.db_path
    Path(resolved_db).parent.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        db_path=resolved_db,
        llm=create_llm(cfg=cfg),
        embedding=create_embedding(cfg=cfg),
    )

    _seed_meridian_if_needed(engine)

    app, token = create_app(engine=engine, port=port, host=host)

    url = f"http://{host}:{port}"
    print(f"\n  woven-imprint demo v{__version__}")
    print(f"  URL:   {url}")
    print(f"  Token: {token}")
    if host != "127.0.0.1":
        print(f"  ⚠ Bound to {host} — accessible from network")
    print()

    if not no_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}")

    uvicorn.run(app, host=host, port=port, log_level="warning")
