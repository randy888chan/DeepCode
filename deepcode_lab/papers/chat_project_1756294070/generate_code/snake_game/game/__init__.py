# Game module initialization
"""
游戏核心模块 (Game Core Module)
包含贪吃蛇游戏的核心逻辑组件
"""

from .snake import Snake
from .food import Food
from .game_field import GameField
from .collision import CollisionDetector
from .game_state import GameState

__all__ = [
    'Snake',
    'Food', 
    'GameField',
    'CollisionDetector',
    'GameState'
]