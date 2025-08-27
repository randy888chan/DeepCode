"""
Snake class implementation for the Snake Desktop Game.
Handles snake behavior, movement logic, growth, and collision detection.
"""

import pygame
from typing import List, Tuple, Optional
from config.constants import (
    DIRECTIONS, GRID_WIDTH, GRID_HEIGHT, SNAKE_INITIAL_LENGTH,
    SNAKE_INITIAL_POSITION, SNAKE_INITIAL_DIRECTION, COLORS
)


class Snake:
    """
    Snake class that manages the snake's position, movement, and behavior.
    """
    
    def __init__(self, initial_pos: Optional[Tuple[int, int]] = None, 
                 initial_direction: str = SNAKE_INITIAL_DIRECTION):
        """
        Initialize the snake with starting position and direction.
        
        Args:
            initial_pos: Starting position (grid_x, grid_y). If None, uses default.
            initial_direction: Initial movement direction.
        """
        # Set initial position
        if initial_pos is None:
            self.head_pos = list(SNAKE_INITIAL_POSITION)
        else:
            self.head_pos = list(initial_pos)
        
        # Initialize snake segments (head + body)
        self.segments = []
        for i in range(SNAKE_INITIAL_LENGTH):
            # Create initial body segments behind the head
            if initial_direction == "RIGHT":
                segment_pos = [self.head_pos[0] - i, self.head_pos[1]]
            elif initial_direction == "LEFT":
                segment_pos = [self.head_pos[0] + i, self.head_pos[1]]
            elif initial_direction == "DOWN":
                segment_pos = [self.head_pos[0], self.head_pos[1] - i]
            else:  # UP
                segment_pos = [self.head_pos[0], self.head_pos[1] + i]
            
            self.segments.append(segment_pos)
        
        # Movement properties
        self.direction = initial_direction
        self.next_direction = initial_direction
        self.grow_pending = 0  # Number of segments to grow
        
        # State tracking
        self.alive = True
        self.last_move_time = 0
        self.move_delay = 200  # milliseconds between moves (will be set by difficulty)
        
        # Statistics
        self.length = len(self.segments)
        self.food_eaten = 0
        
    def set_direction(self, new_direction: str) -> bool:
        """
        Set the next direction for the snake.
        Prevents immediate reversal (180-degree turn).
        
        Args:
            new_direction: New direction to move ("UP", "DOWN", "LEFT", "RIGHT")
            
        Returns:
            bool: True if direction was set successfully, False if invalid
        """
        if not self.alive:
            return False
            
        # Check if direction is valid
        if new_direction not in DIRECTIONS:
            return False
        
        # Prevent immediate reversal (can't go directly opposite)
        opposite_directions = {
            "UP": "DOWN",
            "DOWN": "UP", 
            "LEFT": "RIGHT",
            "RIGHT": "LEFT"
        }
        
        if new_direction == opposite_directions.get(self.direction):
            return False
        
        self.next_direction = new_direction
        return True
    
    def update(self, current_time: int) -> bool:
        """
        Update snake position based on time and movement speed.
        
        Args:
            current_time: Current game time in milliseconds
            
        Returns:
            bool: True if snake moved, False otherwise
        """
        if not self.alive:
            return False
        
        # Check if enough time has passed for next move
        if current_time - self.last_move_time < self.move_delay:
            return False
        
        # Update direction
        self.direction = self.next_direction
        
        # Calculate new head position
        direction_delta = DIRECTIONS[self.direction]
        new_head = [
            self.head_pos[0] + direction_delta[0],
            self.head_pos[1] + direction_delta[1]
        ]
        
        # Check boundary collision
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.alive = False
            return False
        
        # Check self-collision
        if new_head in self.segments:
            self.alive = False
            return False
        
        # Move snake
        self.segments.insert(0, new_head)
        self.head_pos = new_head
        
        # Handle growth
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            # Remove tail if not growing
            self.segments.pop()
        
        # Update statistics
        self.length = len(self.segments)
        self.last_move_time = current_time
        
        return True
    
    def grow(self, segments: int = 1) -> None:
        """
        Make the snake grow by specified number of segments.
        
        Args:
            segments: Number of segments to grow
        """
        self.grow_pending += segments
        self.food_eaten += 1
    
    def get_head_position(self) -> Tuple[int, int]:
        """
        Get the current head position.
        
        Returns:
            Tuple[int, int]: Head position (grid_x, grid_y)
        """
        return tuple(self.head_pos)
    
    def get_segments(self) -> List[Tuple[int, int]]:
        """
        Get all snake segments as a list of positions.
        
        Returns:
            List[Tuple[int, int]]: List of segment positions
        """
        return [tuple(segment) for segment in self.segments]
    
    def get_body_segments(self) -> List[Tuple[int, int]]:
        """
        Get body segments (excluding head).
        
        Returns:
            List[Tuple[int, int]]: List of body segment positions
        """
        return [tuple(segment) for segment in self.segments[1:]]
    
    def is_alive(self) -> bool:
        """
        Check if snake is alive.
        
        Returns:
            bool: True if alive, False if dead
        """
        return self.alive
    
    def get_length(self) -> int:
        """
        Get current snake length.
        
        Returns:
            int: Number of segments
        """
        return self.length
    
    def get_food_eaten(self) -> int:
        """
        Get number of food items eaten.
        
        Returns:
            int: Food count
        """
        return self.food_eaten
    
    def set_speed(self, move_delay: int) -> None:
        """
        Set snake movement speed.
        
        Args:
            move_delay: Delay between moves in milliseconds
        """
        self.move_delay = max(50, move_delay)  # Minimum 50ms delay
    
    def reset(self, initial_pos: Optional[Tuple[int, int]] = None,
              initial_direction: str = SNAKE_INITIAL_DIRECTION) -> None:
        """
        Reset snake to initial state.
        
        Args:
            initial_pos: Starting position. If None, uses default.
            initial_direction: Initial direction.
        """
        self.__init__(initial_pos, initial_direction)
    
    def check_collision_with_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if given position collides with snake body.
        
        Args:
            pos: Position to check (grid_x, grid_y)
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        return list(pos) in self.segments
    
    def get_direction(self) -> str:
        """
        Get current movement direction.
        
        Returns:
            str: Current direction
        """
        return self.direction
    
    def get_next_direction(self) -> str:
        """
        Get next planned direction.
        
        Returns:
            str: Next direction
        """
        return self.next_direction
    
    def can_move_to(self, direction: str) -> bool:
        """
        Check if snake can move in given direction.
        
        Args:
            direction: Direction to check
            
        Returns:
            bool: True if movement is valid, False otherwise
        """
        if not self.alive or direction not in DIRECTIONS:
            return False
        
        # Calculate potential new head position
        direction_delta = DIRECTIONS[direction]
        new_head = [
            self.head_pos[0] + direction_delta[0],
            self.head_pos[1] + direction_delta[1]
        ]
        
        # Check boundaries
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            return False
        
        # Check self-collision
        if new_head in self.segments:
            return False
        
        return True
    
    def get_debug_info(self) -> dict:
        """
        Get debug information about the snake.
        
        Returns:
            dict: Debug information
        """
        return {
            "head_position": self.get_head_position(),
            "length": self.length,
            "direction": self.direction,
            "next_direction": self.next_direction,
            "alive": self.alive,
            "grow_pending": self.grow_pending,
            "food_eaten": self.food_eaten,
            "move_delay": self.move_delay,
            "segments_count": len(self.segments)
        }