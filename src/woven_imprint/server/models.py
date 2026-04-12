"""Pydantic models for demo server request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Requests ---


class CreateCharacterRequest(BaseModel):
    name: str = Field(..., max_length=200)
    persona: dict | str | None = None
    birthdate: str | None = None


class RecordMessageRequest(BaseModel):
    character_id: str
    role: str = Field(..., pattern="^(user|assistant|character|system)$")
    content: str = Field(..., max_length=50_000)
    user_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict] = Field(..., max_length=200)
    temperature: float | None = None
    max_tokens: int | None = None


class ProviderConfigRequest(BaseModel):
    # Extend this list as new providers are added
    provider: str = Field(..., pattern="^(openai|anthropic|ollama|deepseek|gemma_edge)$")
    model: str = Field(..., max_length=200)
    api_key: str | None = Field(None, max_length=500)
    base_url: str | None = Field(None, max_length=500)


# --- Responses ---


class HealthResponse(BaseModel):
    status: str
    version: str


class CharacterResponse(BaseModel):
    id: str
    name: str
    created: bool


class CharacterStateResponse(BaseModel):
    id: str
    name: str
    emotion: dict
    arc: dict


class SessionResponse(BaseModel):
    session_id: str


class EndSessionResponse(BaseModel):
    summary: str | None


class MemoryResponse(BaseModel):
    memories: list[dict]
    context: str


class ProviderConfigResponse(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key_configured: bool


class ConnectionTestResponse(BaseModel):
    success: bool
    message: str
