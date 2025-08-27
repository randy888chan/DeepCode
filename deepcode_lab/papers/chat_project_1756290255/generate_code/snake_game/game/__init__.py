# Game module initialization
"""
Game module for Snake Game

This module contains the core game logic including:
- Snake class for player character
- Food generation and management
- Game engine and state management
- Collision detection system
"""

__version__ = "1.0.0"
__author__ = "Snake Game Development Team"

# Import main game classes for easy access
from .snake import Snake
from .food import Food
from .game_engine import GameEngine
from .collision import CollisionDetector

__all__ = [
    'Snake',
    'Food', 
    'GameEngine',
    'CollisionDetector'
]