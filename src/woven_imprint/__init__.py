"""Woven Imprint — Persistent Character Infrastructure."""

__version__ = "0.2.1"

from .engine import Engine
from .character import Character
from .interaction import interact, group_interaction

__all__ = ["Engine", "Character", "interact", "group_interaction"]
