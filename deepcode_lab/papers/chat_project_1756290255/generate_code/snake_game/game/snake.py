"""
Snake Game - Snake Class Implementation
Manages the snake's body segments, movement, growth, and collision detection.
"""

import pygame
from typing import List, Tuple, Optional
from utils.constants import (
    DIRECTION_UP, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT,
    DIRECTIONS, GRID_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT,
    SNAKE_HEAD_COLOR, SNAKE_BODY_COLOR, SNAKE_BODY_GRADIENT_COLOR,
    INITIAL_SNAKE_LENGTH, INITIAL_SNAKE_SPEED
)


class Snake:
    """
    Snake class that manages the snake's body, movement, and growth mechanics.
    
    Features:
    - Multi-directional movement with smooth controls
    - Dynamic body growth when eating food
    - Gradient body coloring for visual appeal
    - Collision detection for walls and self-collision
    - Configurable speed and initial length
    """
    
    def __init__(self, initial_position: Tuple[int, int] = None, initial_length: int = INITIAL_SNAKE_LENGTH):
        """
        Initialize the snake with starting position and length.
        
        Args:
            initial_position: Starting position (x, y) in grid coordinates
            initial_length: Initial length of the snake body
        """
        # Set default starting position to center of screen if not provided
        if initial_position is None:
            center_x = (WINDOW_WIDTH // GRID_SIZE) // 2
            center_y = (WINDOW_HEIGHT // GRID_SIZE) // 2
            initial_position = (center_x, center_y)
        
        # Initialize snake body as list of (x, y) grid positions
        self.body: List[Tuple[int, int]] = []
        self.direction = DIRECTION_RIGHT  # Default starting direction
        self.next_direction = DIRECTION_RIGHT  # Buffered direction for smooth input
        self.speed = INITIAL_SNAKE_SPEED
        self.grow_pending = 0  # Number of segments to grow
        
        # Create initial snake body
        start_x, start_y = initial_position
        for i in range(initial_length):
            # Create body segments from head backwards
            self.body.append((start_x - i, start_y))
    
    def get_head_position(self) -> Tuple[int, int]:
        """Get the current position of the snake's head."""
        return self.body[0] if self.body else (0, 0)
    
    def get_body_positions(self) -> List[Tuple[int, int]]:
        """Get all body segment positions."""
        return self.body.copy()
    
    def set_direction(self, new_direction: Tuple[int, int]) -> bool:
        """
        Set the snake's movement direction with validation.
        
        Args:
            new_direction: Direction tuple (dx, dy)
            
        Returns:
            bool: True if direction was set successfully, False if invalid
        """
        # Validate direction is in allowed directions
        if new_direction not in DIRECTIONS:
            return False
        
        # Prevent immediate reversal (snake can't go backwards into itself)
        current_direction = self.direction
        if len(self.body) > 1:
            # Check if new direction is opposite to current direction
            if (new_direction[0] == -current_direction[0] and 
                new_direction[1] == -current_direction[1]):
                return False
        
        # Buffer the direction change for next update
        self.next_direction = new_direction
        return True
    
    def update(self) -> None:
        """
        Update snake position and handle movement logic.
        Called once per game frame.
        """
        # Apply buffered direction change
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.get_head_position()
        new_head_x = head_x + self.direction[0]
        new_head_y = head_y + self.direction[1]
        new_head = (new_head_x, new_head_y)
        
        # Add new head to front of body
        self.body.insert(0, new_head)
        
        # Handle growth or remove tail
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            # Remove tail segment if not growing
            self.body.pop()
    
    def grow(self, segments: int = 1) -> None:
        """
        Make the snake grow by specified number of segments.
        
        Args:
            segments: Number of segments to add to the snake
        """
        self.grow_pending += segments
    
    def check_wall_collision(self) -> bool:
        """
        Check if snake head has collided with game boundaries.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        head_x, head_y = self.get_head_position()
        
        # Check boundaries (grid coordinates)
        max_x = (WINDOW_WIDTH // GRID_SIZE) - 1
        max_y = (WINDOW_HEIGHT // GRID_SIZE) - 1
        
        return (head_x < 0 or head_x > max_x or 
                head_y < 0 or head_y > max_y)
    
    def check_self_collision(self) -> bool:
        """
        Check if snake head has collided with its own body.
        
        Returns:
            bool: True if self-collision detected, False otherwise
        """
        head = self.get_head_position()
        
        # Check if head position matches any body segment (excluding head itself)
        return head in self.body[1:]
    
    def check_food_collision(self, food_position: Tuple[int, int]) -> bool:
        """
        Check if snake head has collided with food.
        
        Args:
            food_position: Position of food in grid coordinates
            
        Returns:
            bool: True if food collision detected, False otherwise
        """
        return self.get_head_position() == food_position
    
    def get_length(self) -> int:
        """Get current length of the snake."""
        return len(self.body)
    
    def set_speed(self, new_speed: float) -> None:
        """
        Set the snake's movement speed.
        
        Args:
            new_speed: New speed value (higher = faster)
        """
        self.speed = max(0.1, new_speed)  # Ensure minimum speed
    
    def reset(self, initial_position: Tuple[int, int] = None, initial_length: int = INITIAL_SNAKE_LENGTH) -> None:
        """
        Reset snake to initial state.
        
        Args:
            initial_position: Starting position (x, y) in grid coordinates
            initial_length: Initial length of the snake body
        """
        self.__init__(initial_position, initial_length)
    
    def render(self, screen: pygame.Surface) -> None:
        """
        Render the snake on the game screen with gradient effect.
        
        Args:
            screen: Pygame surface to draw on
        """
        if not self.body:
            return
        
        # Draw snake body with gradient effect
        for i, (x, y) in enumerate(self.body):
            # Convert grid coordinates to pixel coordinates
            pixel_x = x * GRID_SIZE
            pixel_y = y * GRID_SIZE
            
            # Create rectangle for this segment
            segment_rect = pygame.Rect(pixel_x, pixel_y, GRID_SIZE, GRID_SIZE)
            
            if i == 0:
                # Draw head with special color
                pygame.draw.rect(screen, SNAKE_HEAD_COLOR, segment_rect)
                # Add border for head
                pygame.draw.rect(screen, (255, 255, 255), segment_rect, 2)
            else:
                # Draw body with gradient effect
                # Calculate gradient factor based on position in body
                gradient_factor = 1.0 - (i / len(self.body))
                
                # Interpolate between body colors
                r = int(SNAKE_BODY_COLOR[0] * gradient_factor + 
                       SNAKE_BODY_GRADIENT_COLOR[0] * (1 - gradient_factor))
                g = int(SNAKE_BODY_COLOR[1] * gradient_factor + 
                       SNAKE_BODY_GRADIENT_COLOR[1] * (1 - gradient_factor))
                b = int(SNAKE_BODY_COLOR[2] * gradient_factor + 
                       SNAKE_BODY_GRADIENT_COLOR[2] * (1 - gradient_factor))
                
                body_color = (r, g, b)
                pygame.draw.rect(screen, body_color, segment_rect)
                
                # Add subtle border
                pygame.draw.rect(screen, (200, 200, 200), segment_rect, 1)
    
    def get_state_info(self) -> dict:
        """
        Get current state information for debugging or saving.
        
        Returns:
            dict: Dictionary containing snake state information
        """
        return {
            'head_position': self.get_head_position(),
            'body_length': self.get_length(),
            'direction': self.direction,
            'speed': self.speed,
            'grow_pending': self.grow_pending,
            'body_positions': self.get_body_positions()
        }
    
    def __str__(self) -> str:
        """String representation of the snake for debugging."""
        return f"Snake(head={self.get_head_position()}, length={self.get_length()}, direction={self.direction})"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (f"Snake(head={self.get_head_position()}, length={self.get_length()}, "
                f"direction={self.direction}, speed={self.speed}, body={self.body})")