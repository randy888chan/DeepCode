"""
Food System for Snake Game

This module implements the Food class that manages food generation, different food types,
visual effects, and scoring mechanics for the Snake game.

Features:
- Random food placement avoiding snake body
- Multiple food types with different scores and effects
- Visual animations and effects
- Collision detection with snake
- Special food items with time-limited availability
"""

import pygame
import random
import time
import math
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum

# Import game constants
from ..utils.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, CELL_SIZE,
    FOOD_COLOR, SPECIAL_FOOD_COLOR, BONUS_FOOD_COLOR,
    SCORE_NORMAL_FOOD, SCORE_SPECIAL_FOOD, SCORE_BONUS_FOOD,
    SPECIAL_FOOD_DURATION, BONUS_FOOD_DURATION,
    FOOD_SPAWN_MARGIN, ANIMATION_SPEED
)


class FoodType(Enum):
    """Enumeration for different types of food"""
    NORMAL = "normal"
    SPECIAL = "special"
    BONUS = "bonus"


class Food:
    """
    Food class that manages food generation, rendering, and game mechanics.
    
    Features:
    - Random placement avoiding snake body
    - Multiple food types with different properties
    - Visual animations and effects
    - Time-based special food mechanics
    - Collision detection support
    """
    
    def __init__(self):
        """Initialize the food system"""
        self.position: Optional[Tuple[int, int]] = None
        self.food_type: FoodType = FoodType.NORMAL
        self.spawn_time: float = 0.0
        self.animation_offset: float = 0.0
        self.is_active: bool = False
        
        # Food type probabilities (out of 100)
        self.food_probabilities = {
            FoodType.NORMAL: 70,    # 70% chance
            FoodType.SPECIAL: 20,   # 20% chance
            FoodType.BONUS: 10      # 10% chance
        }
        
        # Food type properties
        self.food_properties = {
            FoodType.NORMAL: {
                'color': FOOD_COLOR,
                'score': SCORE_NORMAL_FOOD,
                'duration': None,  # Permanent until eaten
                'size_multiplier': 1.0,
                'glow': False
            },
            FoodType.SPECIAL: {
                'color': SPECIAL_FOOD_COLOR,
                'score': SCORE_SPECIAL_FOOD,
                'duration': SPECIAL_FOOD_DURATION,
                'size_multiplier': 1.2,
                'glow': True
            },
            FoodType.BONUS: {
                'color': BONUS_FOOD_COLOR,
                'score': SCORE_BONUS_FOOD,
                'duration': BONUS_FOOD_DURATION,
                'size_multiplier': 1.5,
                'glow': True
            }
        }
        
        # Calculate grid dimensions
        self.grid_width = WINDOW_WIDTH // CELL_SIZE
        self.grid_height = WINDOW_HEIGHT // CELL_SIZE
        
    def spawn_food(self, snake_positions: List[Tuple[int, int]], force_type: Optional[FoodType] = None) -> bool:
        """
        Spawn a new food item at a random location avoiding snake body.
        
        Args:
            snake_positions: List of snake body segment positions to avoid
            force_type: Optional specific food type to spawn
            
        Returns:
            bool: True if food was successfully spawned, False otherwise
        """
        try:
            # Get available positions
            available_positions = self._get_available_positions(snake_positions)
            
            if not available_positions:
                return False
            
            # Choose random position
            self.position = random.choice(available_positions)
            
            # Determine food type
            if force_type:
                self.food_type = force_type
            else:
                self.food_type = self._determine_food_type()
            
            # Set spawn time and activate
            self.spawn_time = time.time()
            self.animation_offset = 0.0
            self.is_active = True
            
            return True
            
        except Exception as e:
            print(f"Error spawning food: {e}")
            return False
    
    def _get_available_positions(self, snake_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Get all available positions for food placement.
        
        Args:
            snake_positions: List of snake body positions to avoid
            
        Returns:
            List of available grid positions
        """
        available_positions = []
        snake_set = set(snake_positions)
        
        # Generate all possible positions with margin
        for x in range(FOOD_SPAWN_MARGIN, self.grid_width - FOOD_SPAWN_MARGIN):
            for y in range(FOOD_SPAWN_MARGIN, self.grid_height - FOOD_SPAWN_MARGIN):
                position = (x * CELL_SIZE, y * CELL_SIZE)
                if position not in snake_set:
                    available_positions.append(position)
        
        return available_positions
    
    def _determine_food_type(self) -> FoodType:
        """
        Determine food type based on probabilities.
        
        Returns:
            FoodType: The selected food type
        """
        rand_value = random.randint(1, 100)
        cumulative_prob = 0
        
        for food_type, probability in self.food_probabilities.items():
            cumulative_prob += probability
            if rand_value <= cumulative_prob:
                return food_type
        
        # Fallback to normal food
        return FoodType.NORMAL
    
    def update(self, delta_time: float) -> bool:
        """
        Update food state including animations and expiration.
        
        Args:
            delta_time: Time elapsed since last update in seconds
            
        Returns:
            bool: True if food is still active, False if expired
        """
        if not self.is_active:
            return False
        
        # Update animation
        self.animation_offset += ANIMATION_SPEED * delta_time
        
        # Check for expiration
        if self._is_expired():
            self.is_active = False
            return False
        
        return True
    
    def _is_expired(self) -> bool:
        """
        Check if the current food has expired.
        
        Returns:
            bool: True if food has expired, False otherwise
        """
        if not self.is_active:
            return True
        
        duration = self.food_properties[self.food_type]['duration']
        if duration is None:
            return False  # Normal food doesn't expire
        
        return time.time() - self.spawn_time > duration
    
    def get_position(self) -> Optional[Tuple[int, int]]:
        """
        Get the current food position.
        
        Returns:
            Tuple of (x, y) coordinates or None if no active food
        """
        return self.position if self.is_active else None
    
    def get_score_value(self) -> int:
        """
        Get the score value for the current food.
        
        Returns:
            int: Score value for eating this food
        """
        if not self.is_active:
            return 0
        
        return self.food_properties[self.food_type]['score']
    
    def get_food_type(self) -> Optional[FoodType]:
        """
        Get the current food type.
        
        Returns:
            FoodType or None if no active food
        """
        return self.food_type if self.is_active else None
    
    def consume_food(self) -> Dict[str, Any]:
        """
        Consume the current food and return its properties.
        
        Returns:
            Dictionary containing food properties (score, type, etc.)
        """
        if not self.is_active:
            return {'score': 0, 'type': None}
        
        food_info = {
            'score': self.get_score_value(),
            'type': self.food_type,
            'position': self.position,
            'properties': self.food_properties[self.food_type].copy()
        }
        
        # Deactivate food
        self.is_active = False
        self.position = None
        
        return food_info
    
    def render(self, screen: pygame.Surface) -> None:
        """
        Render the food on the game screen with visual effects.
        
        Args:
            screen: Pygame surface to render on
        """
        if not self.is_active or not self.position:
            return
        
        try:
            properties = self.food_properties[self.food_type]
            base_color = properties['color']
            size_multiplier = properties['size_multiplier']
            has_glow = properties['glow']
            
            # Calculate animated size
            pulse_factor = 1.0 + 0.1 * math.sin(self.animation_offset * 3)
            final_size = int(CELL_SIZE * size_multiplier * pulse_factor)
            
            # Calculate position for centered rendering
            x, y = self.position
            center_x = x + CELL_SIZE // 2
            center_y = y + CELL_SIZE // 2
            
            # Render glow effect for special foods
            if has_glow:
                self._render_glow_effect(screen, center_x, center_y, final_size, base_color)
            
            # Render main food body
            food_rect = pygame.Rect(
                center_x - final_size // 2,
                center_y - final_size // 2,
                final_size,
                final_size
            )
            
            # Draw food with rounded corners for better appearance
            pygame.draw.ellipse(screen, base_color, food_rect)
            
            # Add highlight for 3D effect
            highlight_size = max(1, final_size // 4)
            highlight_rect = pygame.Rect(
                center_x - final_size // 4,
                center_y - final_size // 4,
                highlight_size,
                highlight_size
            )
            
            # Create lighter color for highlight
            highlight_color = tuple(min(255, c + 60) for c in base_color)
            pygame.draw.ellipse(screen, highlight_color, highlight_rect)
            
        except Exception as e:
            print(f"Error rendering food: {e}")
    
    def _render_glow_effect(self, screen: pygame.Surface, center_x: int, center_y: int, 
                           size: int, base_color: Tuple[int, int, int]) -> None:
        """
        Render a glow effect around special food items.
        
        Args:
            screen: Pygame surface to render on
            center_x: X coordinate of food center
            center_y: Y coordinate of food center
            size: Size of the food item
            base_color: Base color of the food
        """
        try:
            # Create multiple glow layers with decreasing opacity
            glow_layers = 3
            max_glow_size = size + 20
            
            for i in range(glow_layers):
                layer_size = max_glow_size - (i * 6)
                if layer_size <= size:
                    continue
                
                # Calculate alpha for this layer
                alpha = max(10, 50 - (i * 15))
                
                # Create glow color with alpha
                glow_color = (*base_color, alpha)
                
                # Create temporary surface for alpha blending
                glow_surface = pygame.Surface((layer_size, layer_size), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, glow_color, 
                                 (layer_size // 2, layer_size // 2), layer_size // 2)
                
                # Blit glow layer
                screen.blit(glow_surface, 
                           (center_x - layer_size // 2, center_y - layer_size // 2))
                
        except Exception as e:
            print(f"Error rendering glow effect: {e}")
    
    def get_time_remaining(self) -> Optional[float]:
        """
        Get remaining time for time-limited food items.
        
        Returns:
            float: Remaining time in seconds, or None for permanent food
        """
        if not self.is_active:
            return None
        
        duration = self.food_properties[self.food_type]['duration']
        if duration is None:
            return None
        
        elapsed = time.time() - self.spawn_time
        remaining = max(0, duration - elapsed)
        return remaining
    
    def is_food_active(self) -> bool:
        """
        Check if food is currently active.
        
        Returns:
            bool: True if food is active and available
        """
        return self.is_active
    
    def reset(self) -> None:
        """Reset the food system to initial state"""
        self.position = None
        self.food_type = FoodType.NORMAL
        self.spawn_time = 0.0
        self.animation_offset = 0.0
        self.is_active = False
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get current state information for debugging or saving.
        
        Returns:
            Dictionary containing current food state
        """
        return {
            'position': self.position,
            'food_type': self.food_type.value if self.food_type else None,
            'spawn_time': self.spawn_time,
            'is_active': self.is_active,
            'time_remaining': self.get_time_remaining(),
            'score_value': self.get_score_value(),
            'animation_offset': self.animation_offset
        }
    
    def set_food_probabilities(self, probabilities: Dict[FoodType, int]) -> bool:
        """
        Set custom food type probabilities.
        
        Args:
            probabilities: Dictionary mapping food types to probability percentages
            
        Returns:
            bool: True if probabilities were set successfully
        """
        try:
            # Validate probabilities sum to 100
            total = sum(probabilities.values())
            if total != 100:
                print(f"Warning: Food probabilities sum to {total}, not 100")
                return False
            
            self.food_probabilities = probabilities.copy()
            return True
            
        except Exception as e:
            print(f"Error setting food probabilities: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation for debugging"""
        if not self.is_active:
            return "Food(inactive)"
        
        return (f"Food(type={self.food_type.value}, pos={self.position}, "
                f"score={self.get_score_value()}, remaining={self.get_time_remaining()})")
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging"""
        return (f"Food(position={self.position}, type={self.food_type}, "
                f"active={self.is_active}, spawn_time={self.spawn_time}, "
                f"animation_offset={self.animation_offset})")


class FoodManager:
    """
    Manager class for handling multiple food items and advanced food mechanics.
    
    This class can be extended for features like multiple simultaneous foods,
    food chains, or special food events.
    """
    
    def __init__(self):
        """Initialize the food manager"""
        self.primary_food = Food()
        self.food_history: List[Dict[str, Any]] = []
        self.total_food_consumed = 0
        self.food_spawn_count = 0
    
    def spawn_food(self, snake_positions: List[Tuple[int, int]], 
                   force_type: Optional[FoodType] = None) -> bool:
        """
        Spawn food using the primary food instance.
        
        Args:
            snake_positions: Snake body positions to avoid
            force_type: Optional specific food type to spawn
            
        Returns:
            bool: True if food was spawned successfully
        """
        success = self.primary_food.spawn_food(snake_positions, force_type)
        if success:
            self.food_spawn_count += 1
        return success
    
    def update(self, delta_time: float) -> bool:
        """Update the food manager and all food items"""
        return self.primary_food.update(delta_time)
    
    def consume_food(self) -> Dict[str, Any]:
        """Consume food and track statistics"""
        food_info = self.primary_food.consume_food()
        if food_info['score'] > 0:
            self.total_food_consumed += 1
            self.food_history.append({
                'type': food_info['type'].value if food_info['type'] else None,
                'score': food_info['score'],
                'timestamp': time.time()
            })
        return food_info
    
    def render(self, screen: pygame.Surface) -> None:
        """Render all food items"""
        self.primary_food.render(screen)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get food consumption statistics"""
        return {
            'total_consumed': self.total_food_consumed,
            'total_spawned': self.food_spawn_count,
            'history_count': len(self.food_history),
            'current_food': self.primary_food.get_state_info()
        }
    
    def reset(self) -> None:
        """Reset the food manager"""
        self.primary_food.reset()
        self.food_history.clear()
        self.total_food_consumed = 0
        self.food_spawn_count = 0
    
    # Delegate methods to primary food
    def get_position(self) -> Optional[Tuple[int, int]]:
        return self.primary_food.get_position()
    
    def get_score_value(self) -> int:
        return self.primary_food.get_score_value()
    
    def get_food_type(self) -> Optional[FoodType]:
        return self.primary_food.get_food_type()
    
    def is_food_active(self) -> bool:
        return self.primary_food.is_food_active()
    
    def get_time_remaining(self) -> Optional[float]:
        return self.primary_food.get_time_remaining()