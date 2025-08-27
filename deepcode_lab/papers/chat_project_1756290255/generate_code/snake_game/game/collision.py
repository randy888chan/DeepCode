"""
Collision Detection System for Snake Game

This module provides comprehensive collision detection functionality for the Snake game,
including wall collisions, self-collisions, food collisions, and advanced collision
detection algorithms for game objects.

Author: Snake Game Development Team
Version: 1.0.0
"""

import pygame
import math
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum

# Import game constants
from ..utils.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, GRID_SIZE,
    COLLISION_TOLERANCE, COLLISION_BUFFER,
    DEBUG_MODE
)


class CollisionType(Enum):
    """Enumeration of different collision types in the game."""
    NONE = "none"
    WALL = "wall"
    SELF = "self"
    FOOD = "food"
    OBSTACLE = "obstacle"
    BOUNDARY = "boundary"


class CollisionResult:
    """
    Represents the result of a collision detection operation.
    
    Attributes:
        collision_type (CollisionType): Type of collision detected
        collision_point (Tuple[int, int]): Point where collision occurred
        collision_object (Any): Object involved in collision
        collision_data (Dict): Additional collision data
    """
    
    def __init__(self, collision_type: CollisionType = CollisionType.NONE,
                 collision_point: Optional[Tuple[int, int]] = None,
                 collision_object: Any = None,
                 collision_data: Optional[Dict[str, Any]] = None):
        self.collision_type = collision_type
        self.collision_point = collision_point
        self.collision_object = collision_object
        self.collision_data = collision_data or {}
        self.timestamp = pygame.time.get_ticks()
    
    def has_collision(self) -> bool:
        """Check if any collision was detected."""
        return self.collision_type != CollisionType.NONE
    
    def is_fatal(self) -> bool:
        """Check if the collision is fatal (ends the game)."""
        return self.collision_type in [CollisionType.WALL, CollisionType.SELF, CollisionType.OBSTACLE]
    
    def __str__(self) -> str:
        return f"CollisionResult(type={self.collision_type.value}, point={self.collision_point})"
    
    def __repr__(self) -> str:
        return self.__str__()


class CollisionDetector:
    """
    Advanced collision detection system for the Snake game.
    
    This class provides various collision detection methods including:
    - Wall/boundary collision detection
    - Self-collision detection for the snake
    - Food collision detection
    - General object collision detection
    - Spatial optimization for performance
    """
    
    def __init__(self, game_width: int = WINDOW_WIDTH, game_height: int = WINDOW_HEIGHT,
                 grid_size: int = GRID_SIZE):
        """
        Initialize the collision detector.
        
        Args:
            game_width (int): Width of the game area
            game_height (int): Height of the game area
            grid_size (int): Size of the game grid
        """
        self.game_width = game_width
        self.game_height = game_height
        self.grid_size = grid_size
        
        # Calculate grid dimensions
        self.grid_width = game_width // grid_size
        self.grid_height = game_height // grid_size
        
        # Collision tolerance and buffer settings
        self.tolerance = getattr(globals().get('COLLISION_TOLERANCE', None), 'COLLISION_TOLERANCE', 0)
        self.buffer = getattr(globals().get('COLLISION_BUFFER', None), 'COLLISION_BUFFER', 0)
        
        # Debug mode
        self.debug_mode = getattr(globals().get('DEBUG_MODE', None), 'DEBUG_MODE', False)
        
        # Collision history for debugging
        self.collision_history: List[CollisionResult] = []
        self.max_history_size = 100
        
        # Spatial hash for optimization
        self.spatial_hash: Dict[Tuple[int, int], Set[Any]] = {}
        
    def clear_spatial_hash(self):
        """Clear the spatial hash for the next frame."""
        self.spatial_hash.clear()
    
    def add_to_spatial_hash(self, position: Tuple[int, int], obj: Any):
        """Add an object to the spatial hash at the given position."""
        grid_pos = self._world_to_grid(position)
        if grid_pos not in self.spatial_hash:
            self.spatial_hash[grid_pos] = set()
        self.spatial_hash[grid_pos].add(obj)
    
    def _world_to_grid(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        x, y = position
        grid_x = x // self.grid_size
        grid_y = y // self.grid_size
        return (grid_x, grid_y)
    
    def _grid_to_world(self, grid_position: Tuple[int, int]) -> Tuple[int, int]:
        """Convert grid coordinates to world coordinates."""
        grid_x, grid_y = grid_position
        x = grid_x * self.grid_size
        y = grid_y * self.grid_size
        return (x, y)
    
    def _is_position_valid(self, position: Tuple[int, int]) -> bool:
        """Check if a position is within valid game boundaries."""
        x, y = position
        return 0 <= x < self.game_width and 0 <= y < self.game_height
    
    def check_wall_collision(self, position: Tuple[int, int]) -> CollisionResult:
        """
        Check if a position collides with the game boundaries.
        
        Args:
            position (Tuple[int, int]): Position to check
            
        Returns:
            CollisionResult: Result of the collision check
        """
        x, y = position
        
        # Check boundaries with tolerance
        if (x < self.tolerance or 
            x >= self.game_width - self.tolerance or
            y < self.tolerance or 
            y >= self.game_height - self.tolerance):
            
            collision_result = CollisionResult(
                collision_type=CollisionType.WALL,
                collision_point=position,
                collision_data={
                    'boundary_type': self._get_boundary_type(position),
                    'distance_to_boundary': self._get_distance_to_boundary(position)
                }
            )
            
            self._add_to_history(collision_result)
            return collision_result
        
        return CollisionResult()
    
    def _get_boundary_type(self, position: Tuple[int, int]) -> str:
        """Determine which boundary was hit."""
        x, y = position
        
        if x < self.tolerance:
            return "left"
        elif x >= self.game_width - self.tolerance:
            return "right"
        elif y < self.tolerance:
            return "top"
        elif y >= self.game_height - self.tolerance:
            return "bottom"
        else:
            return "unknown"
    
    def _get_distance_to_boundary(self, position: Tuple[int, int]) -> float:
        """Calculate the minimum distance to any boundary."""
        x, y = position
        
        distances = [
            x,  # Distance to left
            self.game_width - x,  # Distance to right
            y,  # Distance to top
            self.game_height - y  # Distance to bottom
        ]
        
        return min(distances)
    
    def check_self_collision(self, head_position: Tuple[int, int], 
                           body_positions: List[Tuple[int, int]]) -> CollisionResult:
        """
        Check if the snake's head collides with its own body.
        
        Args:
            head_position (Tuple[int, int]): Position of the snake's head
            body_positions (List[Tuple[int, int]]): Positions of the snake's body segments
            
        Returns:
            CollisionResult: Result of the collision check
        """
        # Skip the first few segments to avoid immediate collision
        body_to_check = body_positions[2:] if len(body_positions) > 2 else []
        
        for i, body_pos in enumerate(body_to_check):
            if self._positions_equal(head_position, body_pos):
                collision_result = CollisionResult(
                    collision_type=CollisionType.SELF,
                    collision_point=head_position,
                    collision_object=body_pos,
                    collision_data={
                        'body_segment_index': i + 2,
                        'total_body_length': len(body_positions)
                    }
                )
                
                self._add_to_history(collision_result)
                return collision_result
        
        return CollisionResult()
    
    def check_food_collision(self, head_position: Tuple[int, int], 
                           food_position: Tuple[int, int]) -> CollisionResult:
        """
        Check if the snake's head collides with food.
        
        Args:
            head_position (Tuple[int, int]): Position of the snake's head
            food_position (Tuple[int, int]): Position of the food
            
        Returns:
            CollisionResult: Result of the collision check
        """
        if self._positions_equal(head_position, food_position):
            collision_result = CollisionResult(
                collision_type=CollisionType.FOOD,
                collision_point=head_position,
                collision_object=food_position,
                collision_data={
                    'food_consumed': True,
                    'consumption_time': pygame.time.get_ticks()
                }
            )
            
            self._add_to_history(collision_result)
            return collision_result
        
        return CollisionResult()
    
    def check_multiple_food_collision(self, head_position: Tuple[int, int], 
                                    food_positions: List[Tuple[int, int]]) -> CollisionResult:
        """
        Check collision with multiple food items.
        
        Args:
            head_position (Tuple[int, int]): Position of the snake's head
            food_positions (List[Tuple[int, int]]): List of food positions
            
        Returns:
            CollisionResult: Result of the collision check
        """
        for i, food_pos in enumerate(food_positions):
            result = self.check_food_collision(head_position, food_pos)
            if result.has_collision():
                result.collision_data['food_index'] = i
                return result
        
        return CollisionResult()
    
    def _positions_equal(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """
        Check if two positions are equal within tolerance.
        
        Args:
            pos1 (Tuple[int, int]): First position
            pos2 (Tuple[int, int]): Second position
            
        Returns:
            bool: True if positions are equal within tolerance
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        return abs(x1 - x2) <= self.tolerance and abs(y1 - y2) <= self.tolerance
    
    def check_point_in_rectangle(self, point: Tuple[int, int], 
                               rect_pos: Tuple[int, int], 
                               rect_size: Tuple[int, int]) -> bool:
        """
        Check if a point is inside a rectangle.
        
        Args:
            point (Tuple[int, int]): Point to check
            rect_pos (Tuple[int, int]): Rectangle position (top-left corner)
            rect_size (Tuple[int, int]): Rectangle size (width, height)
            
        Returns:
            bool: True if point is inside rectangle
        """
        px, py = point
        rx, ry = rect_pos
        rw, rh = rect_size
        
        return rx <= px <= rx + rw and ry <= py <= ry + rh
    
    def check_circle_collision(self, pos1: Tuple[int, int], radius1: float,
                             pos2: Tuple[int, int], radius2: float) -> bool:
        """
        Check collision between two circles.
        
        Args:
            pos1 (Tuple[int, int]): Position of first circle
            radius1 (float): Radius of first circle
            pos2 (Tuple[int, int]): Position of second circle
            radius2 (float): Radius of second circle
            
        Returns:
            bool: True if circles collide
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance <= (radius1 + radius2)
    
    def get_safe_spawn_position(self, occupied_positions: List[Tuple[int, int]], 
                              attempts: int = 100) -> Optional[Tuple[int, int]]:
        """
        Find a safe position that doesn't collide with occupied positions.
        
        Args:
            occupied_positions (List[Tuple[int, int]]): List of occupied positions
            attempts (int): Maximum number of attempts to find a safe position
            
        Returns:
            Optional[Tuple[int, int]]: Safe position or None if not found
        """
        import random
        
        for _ in range(attempts):
            # Generate random grid position
            grid_x = random.randint(0, self.grid_width - 1)
            grid_y = random.randint(0, self.grid_height - 1)
            
            # Convert to world coordinates
            world_pos = self._grid_to_world((grid_x, grid_y))
            
            # Check if position is safe
            is_safe = True
            for occupied_pos in occupied_positions:
                if self._positions_equal(world_pos, occupied_pos):
                    is_safe = False
                    break
            
            # Check wall collision
            if is_safe and not self.check_wall_collision(world_pos).has_collision():
                return world_pos
        
        return None
    
    def check_comprehensive_collision(self, head_position: Tuple[int, int],
                                    body_positions: List[Tuple[int, int]],
                                    food_positions: List[Tuple[int, int]] = None) -> CollisionResult:
        """
        Perform comprehensive collision detection for all collision types.
        
        Args:
            head_position (Tuple[int, int]): Position of the snake's head
            body_positions (List[Tuple[int, int]]): Positions of the snake's body
            food_positions (List[Tuple[int, int]], optional): Positions of food items
            
        Returns:
            CollisionResult: Result of the collision check
        """
        # Check wall collision first (highest priority)
        wall_result = self.check_wall_collision(head_position)
        if wall_result.has_collision():
            return wall_result
        
        # Check self collision
        self_result = self.check_self_collision(head_position, body_positions)
        if self_result.has_collision():
            return self_result
        
        # Check food collision
        if food_positions:
            food_result = self.check_multiple_food_collision(head_position, food_positions)
            if food_result.has_collision():
                return food_result
        
        return CollisionResult()
    
    def _add_to_history(self, collision_result: CollisionResult):
        """Add collision result to history for debugging."""
        self.collision_history.append(collision_result)
        
        # Limit history size
        if len(self.collision_history) > self.max_history_size:
            self.collision_history.pop(0)
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """
        Get collision statistics for debugging and analysis.
        
        Returns:
            Dict[str, Any]: Collision statistics
        """
        if not self.collision_history:
            return {
                'total_collisions': 0,
                'collision_types': {},
                'average_time_between_collisions': 0
            }
        
        collision_types = {}
        for collision in self.collision_history:
            collision_type = collision.collision_type.value
            collision_types[collision_type] = collision_types.get(collision_type, 0) + 1
        
        # Calculate average time between collisions
        if len(self.collision_history) > 1:
            time_diffs = []
            for i in range(1, len(self.collision_history)):
                time_diff = self.collision_history[i].timestamp - self.collision_history[i-1].timestamp
                time_diffs.append(time_diff)
            avg_time = sum(time_diffs) / len(time_diffs)
        else:
            avg_time = 0
        
        return {
            'total_collisions': len(self.collision_history),
            'collision_types': collision_types,
            'average_time_between_collisions': avg_time,
            'most_recent_collision': self.collision_history[-1] if self.collision_history else None
        }
    
    def clear_history(self):
        """Clear collision history."""
        self.collision_history.clear()
    
    def debug_render(self, screen: pygame.Surface):
        """
        Render debug information for collision detection.
        
        Args:
            screen (pygame.Surface): Surface to render on
        """
        if not self.debug_mode:
            return
        
        # Render grid
        grid_color = (50, 50, 50)
        for x in range(0, self.game_width, self.grid_size):
            pygame.draw.line(screen, grid_color, (x, 0), (x, self.game_height))
        for y in range(0, self.game_height, self.grid_size):
            pygame.draw.line(screen, grid_color, (0, y), (self.game_width, y))
        
        # Render collision boundaries
        boundary_color = (255, 0, 0, 100)
        if self.tolerance > 0:
            # Top boundary
            pygame.draw.rect(screen, boundary_color, 
                           (0, 0, self.game_width, self.tolerance))
            # Bottom boundary
            pygame.draw.rect(screen, boundary_color, 
                           (0, self.game_height - self.tolerance, self.game_width, self.tolerance))
            # Left boundary
            pygame.draw.rect(screen, boundary_color, 
                           (0, 0, self.tolerance, self.game_height))
            # Right boundary
            pygame.draw.rect(screen, boundary_color, 
                           (self.game_width - self.tolerance, 0, self.tolerance, self.game_height))
    
    def __str__(self) -> str:
        return f"CollisionDetector(grid={self.grid_width}x{self.grid_height}, tolerance={self.tolerance})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Utility functions for collision detection
def distance_between_points(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        point1 (Tuple[int, int]): First point
        point2 (Tuple[int, int]): Second point
        
    Returns:
        float: Distance between the points
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def manhattan_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
    """
    Calculate the Manhattan distance between two points.
    
    Args:
        point1 (Tuple[int, int]): First point
        point2 (Tuple[int, int]): Second point
        
    Returns:
        int: Manhattan distance between the points
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1) + abs(y2 - y1)


def is_point_in_grid_bounds(point: Tuple[int, int], grid_width: int, grid_height: int) -> bool:
    """
    Check if a point is within grid bounds.
    
    Args:
        point (Tuple[int, int]): Point to check
        grid_width (int): Width of the grid
        grid_height (int): Height of the grid
        
    Returns:
        bool: True if point is within bounds
    """
    x, y = point
    return 0 <= x < grid_width and 0 <= y < grid_height


def get_neighboring_positions(position: Tuple[int, int], 
                            include_diagonals: bool = False) -> List[Tuple[int, int]]:
    """
    Get neighboring positions around a given position.
    
    Args:
        position (Tuple[int, int]): Center position
        include_diagonals (bool): Whether to include diagonal neighbors
        
    Returns:
        List[Tuple[int, int]]: List of neighboring positions
    """
    x, y = position
    neighbors = []
    
    # Cardinal directions
    neighbors.extend([
        (x, y - 1),  # Up
        (x, y + 1),  # Down
        (x - 1, y),  # Left
        (x + 1, y)   # Right
    ])
    
    # Diagonal directions
    if include_diagonals:
        neighbors.extend([
            (x - 1, y - 1),  # Top-left
            (x + 1, y - 1),  # Top-right
            (x - 1, y + 1),  # Bottom-left
            (x + 1, y + 1)   # Bottom-right
        ])
    
    return neighbors


# Global collision detector instance
_global_collision_detector: Optional[CollisionDetector] = None


def get_collision_detector() -> CollisionDetector:
    """
    Get the global collision detector instance.
    
    Returns:
        CollisionDetector: Global collision detector instance
    """
    global _global_collision_detector
    if _global_collision_detector is None:
        _global_collision_detector = CollisionDetector()
    return _global_collision_detector


def reset_collision_detector():
    """Reset the global collision detector instance."""
    global _global_collision_detector
    _global_collision_detector = None