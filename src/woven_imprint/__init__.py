"""Woven Imprint — Persistent Character Infrastructure."""

__version__ = "0.5.0"

from .engine import Engine
from .character import Character
from .interaction import interact, group_interaction

__all__ = ["Engine", "Character", "interact", "group_interaction"]
