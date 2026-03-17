"""Centralized configuration — one file governs all defaults.

Priority (highest wins):
1. CLI flags / function arguments
2. Environment variables (WOVEN_IMPRINT_*, OLLAMA_HOST)
3. Config file (~/.woven_imprint/config.yaml)
4. Built-in defaults (this file)

Usage:
    from woven_imprint.config import get_config
    cfg = get_config()
    model = cfg.llm.model          # "llama3.2" or whatever user configured
    threshold = cfg.memory.consolidation_threshold  # 100 or user override
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://127.0.0.1:11434"
    num_ctx: int = 8192
    temperature: float = 0.7
    temperature_json: float = 0.3
    max_tokens: int = 2048
    timeout: int = 120


@dataclass
class MemoryConfig:
    consolidation_threshold: int = 100
    consolidation_interval: int = 20  # turns between auto-consolidation checks
    state_save_interval: int = 10  # turns between state saves
    fact_extraction_interval: int = 3  # extract facts every N turns
    max_message_length: int = 50_000
    fact_importance: float = 0.75
    session_summary_importance: float = 0.85
    clustering_similarity: float = 0.75
    decay_bedrock: float = 0.9999
    decay_core: float = 0.999
    decay_buffer: float = 0.995
    tier_boost_bedrock: float = 0.35
    tier_boost_core: float = 0.2
    tier_boost_buffer: float = 0.0


@dataclass
class ContextConfig:
    total_tokens: int = 6000
    system_prompt_tokens: int = 1000
    memory_tokens: int = 1500
    conversation_tokens: int = 3000
    reserve_tokens: int = 500
    max_turns: int = 20


@dataclass
class RelationshipConfig:
    max_delta: float = 0.15
    key_moments_limit: int = 20


@dataclass
class PersonaConfig:
    growth_threshold: float = 0.6
    growth_min_memories: int = 20
    emotion_decay_rate: float = 0.15
    emotion_neutral_intensity: float = 0.3
    belief_reinforce_delta: float = 0.15


@dataclass
class CharacterConfig:
    parallel: bool = False
    lightweight: bool = False
    enforce_consistency: bool = True


@dataclass
class ServerConfig:
    api_port: int = 8650
    api_key: str | None = None
    cors_origin: str = "http://localhost"
    ui_port: int = 7860
    ui_browser: str = "auto"


@dataclass
class StorageConfig:
    db_path: str = ""
    busy_timeout: int = 5000

    def __post_init__(self):
        if not self.db_path:
            self.db_path = str(Path.home() / ".woven_imprint" / "characters.db")


@dataclass
class WovenConfig:
    """Top-level configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    relationship: RelationshipConfig = field(default_factory=RelationshipConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    character: CharacterConfig = field(default_factory=CharacterConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


# Global singleton
_config: WovenConfig | None = None
_config_path: str = str(Path.home() / ".woven_imprint" / "config.yaml")


def _load_yaml(path: str) -> dict:
    """Load YAML file if it exists."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml

        with open(p) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        # No PyYAML — try simple key:value parsing
        data = {}
        current_section = None
        with open(p) as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    current_section = line[:-1].strip()
                    data[current_section] = {}
                elif current_section and ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    # Parse value types
                    if val.lower() in ("true", "yes"):
                        val = True
                    elif val.lower() in ("false", "no"):
                        val = False
                    elif val.lower() in ("null", "none", "~"):
                        val = None
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                    data[current_section][key] = val
        return data


def _apply_dict(target, data: dict) -> None:
    """Apply dict values to a dataclass instance."""
    for key, val in data.items():
        if hasattr(target, key):
            current = getattr(target, key)
            if isinstance(current, bool):
                setattr(target, key, bool(val))
            elif isinstance(current, int) and not isinstance(val, bool):
                setattr(target, key, int(val))
            elif isinstance(current, float):
                setattr(target, key, float(val))
            elif isinstance(current, str) or current is None:
                setattr(target, key, str(val) if val is not None else None)


def _apply_env(cfg: WovenConfig) -> None:
    """Override config from environment variables."""
    env_map = {
        "WOVEN_IMPRINT_MODEL": ("llm", "model"),
        "WOVEN_IMPRINT_EMBEDDING_MODEL": ("llm", "embedding_model"),
        "OLLAMA_HOST": ("llm", "ollama_host"),
        "WOVEN_IMPRINT_NUM_CTX": ("llm", "num_ctx"),
        "WOVEN_IMPRINT_DB": ("storage", "db_path"),
        "WOVEN_IMPRINT_API_KEY": ("server", "api_key"),
        "WOVEN_IMPRINT_API_PORT": ("server", "api_port"),
        "WOVEN_IMPRINT_UI_PORT": ("server", "ui_port"),
        "WOVEN_IMPRINT_PARALLEL": ("character", "parallel"),
        "WOVEN_IMPRINT_LIGHTWEIGHT": ("character", "lightweight"),
    }

    for env_var, (section, key) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            target = getattr(cfg, section)
            current = getattr(target, key)
            if isinstance(current, bool):
                setattr(target, key, val.lower() in ("true", "1", "yes"))
            elif isinstance(current, int) and not isinstance(current, bool):
                try:
                    setattr(target, key, int(val))
                except ValueError:
                    pass
            elif isinstance(current, float):
                try:
                    setattr(target, key, float(val))
                except ValueError:
                    pass
            else:
                setattr(target, key, val)


def get_config(config_path: str | None = None) -> WovenConfig:
    """Get the global configuration.

    Loads from:
    1. Built-in defaults
    2. Config file (~/.woven_imprint/config.yaml)
    3. Environment variables

    Call `reload_config()` to force re-read.
    """
    global _config

    if _config is not None:
        return _config

    cfg = WovenConfig()

    # Load config file
    path = config_path or _config_path
    file_data = _load_yaml(path)
    for section_name, section_data in file_data.items():
        if hasattr(cfg, section_name) and isinstance(section_data, dict):
            _apply_dict(getattr(cfg, section_name), section_data)

    # Apply environment overrides
    _apply_env(cfg)

    _config = cfg
    return cfg


def reload_config(config_path: str | None = None) -> WovenConfig:
    """Force reload configuration from file + env."""
    global _config
    _config = None
    return get_config(config_path)


def save_default_config(path: str | None = None) -> Path:
    """Write a default config.yaml with all options documented."""
    p = Path(path or _config_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    content = """# Woven Imprint Configuration
# All values shown are defaults. Uncomment and modify to override.

llm:
  model: llama3.2
  embedding_model: nomic-embed-text
  ollama_host: http://127.0.0.1:11434
  num_ctx: 8192
  temperature: 0.7
  temperature_json: 0.3
  max_tokens: 2048
  timeout: 120

memory:
  consolidation_threshold: 100
  consolidation_interval: 20
  state_save_interval: 10
  fact_extraction_interval: 3
  max_message_length: 50000
  fact_importance: 0.75
  session_summary_importance: 0.85
  clustering_similarity: 0.75
  decay_bedrock: 0.9999
  decay_core: 0.999
  decay_buffer: 0.995
  tier_boost_bedrock: 0.35
  tier_boost_core: 0.2
  tier_boost_buffer: 0.0

context:
  total_tokens: 6000
  system_prompt_tokens: 1000
  memory_tokens: 1500
  conversation_tokens: 3000
  reserve_tokens: 500
  max_turns: 20

relationship:
  max_delta: 0.15
  key_moments_limit: 20

persona:
  growth_threshold: 0.6
  growth_min_memories: 20
  emotion_decay_rate: 0.15
  emotion_neutral_intensity: 0.3
  belief_reinforce_delta: 0.15

character:
  parallel: false
  lightweight: false
  enforce_consistency: true

server:
  api_port: 8650
  api_key: null
  cors_origin: http://localhost
  ui_port: 7860
  ui_browser: auto

storage:
  db_path: ~/.woven_imprint/characters.db
  busy_timeout: 5000
"""
    p.write_text(content)
    return p
