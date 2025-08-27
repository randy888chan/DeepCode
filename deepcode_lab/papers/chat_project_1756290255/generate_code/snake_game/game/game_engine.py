"""
Game Engine Module for Snake Game

This module implements the core game engine that manages the main game loop,
state management, scoring system, and coordinates all game components.

Classes:
    GameEngine: Main game engine class that controls game flow
    GameState: Enum for different game states
    ScoreManager: Manages scoring and statistics
    GameTimer: Handles timing and speed progression
"""

import pygame
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from pathlib import Path

# Import game components
from .snake import Snake
from .food import Food, FoodManager, FoodType
from .collision import CollisionDetector, CollisionResult, CollisionType
from ..utils.constants import *


class GameState(Enum):
    """Enumeration for different game states"""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    SETTINGS = "settings"
    HIGH_SCORES = "high_scores"
    LOADING = "loading"
    TRANSITION = "transition"


class ScoreManager:
    """Manages scoring system and game statistics"""
    
    def __init__(self):
        """Initialize the score manager"""
        self.current_score = 0
        self.high_score = 0
        self.food_eaten = 0
        self.time_played = 0.0
        self.level = 1
        self.score_multiplier = 1.0
        self.combo_count = 0
        self.max_combo = 0
        self.statistics = {
            'games_played': 0,
            'total_food_eaten': 0,
            'total_time_played': 0.0,
            'best_score': 0,
            'best_combo': 0,
            'average_score': 0.0
        }
        self.load_statistics()
    
    def add_score(self, points: int, food_type: FoodType = None) -> int:
        """
        Add points to the current score with multipliers
        
        Args:
            points: Base points to add
            food_type: Type of food consumed for bonus calculations
            
        Returns:
            int: Actual points added after multipliers
        """
        # Calculate bonus multipliers
        bonus_multiplier = 1.0
        
        # Food type bonus
        if food_type == FoodType.BONUS:
            bonus_multiplier *= 2.0
        elif food_type == FoodType.SPEED:
            bonus_multiplier *= 1.5
        elif food_type == FoodType.GOLDEN:
            bonus_multiplier *= 3.0
        
        # Combo bonus
        if self.combo_count > 0:
            combo_bonus = min(self.combo_count * 0.1, 2.0)  # Max 200% bonus
            bonus_multiplier *= (1.0 + combo_bonus)
        
        # Level bonus
        level_bonus = (self.level - 1) * 0.1
        bonus_multiplier *= (1.0 + level_bonus)
        
        # Calculate final points
        final_points = int(points * bonus_multiplier * self.score_multiplier)
        self.current_score += final_points
        
        # Update combo
        self.combo_count += 1
        self.max_combo = max(self.max_combo, self.combo_count)
        
        # Check for level up
        self.check_level_up()
        
        return final_points
    
    def reset_combo(self):
        """Reset the combo counter"""
        self.combo_count = 0
    
    def check_level_up(self) -> bool:
        """
        Check if player should level up based on score
        
        Returns:
            bool: True if leveled up
        """
        new_level = (self.current_score // SCORE_LEVEL_THRESHOLD) + 1
        if new_level > self.level:
            self.level = new_level
            return True
        return False
    
    def food_consumed(self, food_type: FoodType):
        """Record food consumption statistics"""
        self.food_eaten += 1
        self.statistics['total_food_eaten'] += 1
    
    def game_over(self):
        """Handle game over scoring and statistics"""
        self.statistics['games_played'] += 1
        self.statistics['total_time_played'] += self.time_played
        
        # Update high score
        if self.current_score > self.high_score:
            self.high_score = self.current_score
            self.statistics['best_score'] = self.current_score
        
        # Update best combo
        if self.max_combo > self.statistics['best_combo']:
            self.statistics['best_combo'] = self.max_combo
        
        # Calculate average score
        if self.statistics['games_played'] > 0:
            total_score = self.statistics.get('total_score', 0) + self.current_score
            self.statistics['total_score'] = total_score
            self.statistics['average_score'] = total_score / self.statistics['games_played']
        
        self.save_statistics()
    
    def reset_game(self):
        """Reset current game statistics"""
        self.current_score = 0
        self.food_eaten = 0
        self.time_played = 0.0
        self.level = 1
        self.combo_count = 0
        self.max_combo = 0
        self.score_multiplier = 1.0
    
    def get_score_info(self) -> Dict[str, Any]:
        """Get current score information"""
        return {
            'current_score': self.current_score,
            'high_score': self.high_score,
            'food_eaten': self.food_eaten,
            'level': self.level,
            'combo': self.combo_count,
            'max_combo': self.max_combo,
            'time_played': self.time_played,
            'multiplier': self.score_multiplier
        }
    
    def load_statistics(self):
        """Load statistics from file"""
        try:
            scores_file = Path(SCORES_FILE)
            if scores_file.exists():
                with open(scores_file, 'r') as f:
                    data = json.load(f)
                    self.high_score = data.get('high_score', 0)
                    self.statistics.update(data.get('statistics', {}))
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    def save_statistics(self):
        """Save statistics to file"""
        try:
            scores_file = Path(SCORES_FILE)
            scores_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'high_score': self.high_score,
                'statistics': self.statistics,
                'last_updated': time.time()
            }
            
            with open(scores_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving statistics: {e}")


class GameTimer:
    """Handles game timing and speed progression"""
    
    def __init__(self):
        """Initialize the game timer"""
        self.start_time = 0.0
        self.pause_time = 0.0
        self.total_pause_time = 0.0
        self.last_update = 0.0
        self.delta_time = 0.0
        self.game_speed = 1.0
        self.target_fps = TARGET_FPS
        self.is_paused = False
    
    def start(self):
        """Start the game timer"""
        current_time = time.time()
        self.start_time = current_time
        self.last_update = current_time
        self.total_pause_time = 0.0
        self.is_paused = False
    
    def update(self) -> float:
        """
        Update the timer and calculate delta time
        
        Returns:
            float: Delta time since last update
        """
        if self.is_paused:
            return 0.0
        
        current_time = time.time()
        self.delta_time = (current_time - self.last_update) * self.game_speed
        self.last_update = current_time
        return self.delta_time
    
    def pause(self):
        """Pause the timer"""
        if not self.is_paused:
            self.pause_time = time.time()
            self.is_paused = True
    
    def resume(self):
        """Resume the timer"""
        if self.is_paused:
            self.total_pause_time += time.time() - self.pause_time
            self.last_update = time.time()
            self.is_paused = False
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed game time (excluding pauses)"""
        if self.is_paused:
            return self.pause_time - self.start_time - self.total_pause_time
        else:
            return time.time() - self.start_time - self.total_pause_time
    
    def set_speed(self, speed: float):
        """Set game speed multiplier"""
        self.game_speed = max(0.1, min(speed, 5.0))  # Clamp between 0.1x and 5.0x


class GameEngine:
    """
    Main game engine that coordinates all game systems
    """
    
    def __init__(self, screen: pygame.Surface):
        """
        Initialize the game engine
        
        Args:
            screen: Pygame surface for rendering
        """
        self.screen = screen
        self.clock = pygame.time.Clock()
        
        # Game state
        self.current_state = GameState.MENU
        self.previous_state = GameState.MENU
        self.state_transition_time = 0.0
        
        # Game components
        self.snake = Snake()
        self.food_manager = FoodManager()
        self.collision_detector = CollisionDetector()
        self.score_manager = ScoreManager()
        self.timer = GameTimer()
        
        # Game settings
        self.difficulty = "medium"
        self.game_speed = DIFFICULTY_LEVELS[self.difficulty]["speed"]
        self.move_timer = 0.0
        self.move_interval = 1.0 / self.game_speed
        
        # Input handling
        self.keys_pressed = set()
        self.last_direction_change = 0.0
        self.direction_change_cooldown = 0.1  # Prevent too rapid direction changes
        
        # Game flags
        self.running = True
        self.game_over_flag = False
        self.paused = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0.0
        self.fps_update_timer = 0.0
        
        # Initialize game
        self.initialize_game()
    
    def initialize_game(self):
        """Initialize game components and settings"""
        try:
            # Load game configuration
            self.load_configuration()
            
            # Initialize components
            self.snake.reset()
            self.food_manager.clear_all_food()
            self.collision_detector.reset()
            self.score_manager.reset_game()
            
            # Spawn initial food
            self.food_manager.spawn_food(self.snake.get_body_positions())
            
            print("Game engine initialized successfully")
            
        except Exception as e:
            print(f"Error initializing game engine: {e}")
            self.running = False
    
    def load_configuration(self):
        """Load game configuration from file"""
        try:
            config_file = Path(CONFIG_FILE)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                    # Load difficulty settings
                    self.difficulty = config.get('difficulty', 'medium')
                    if self.difficulty in DIFFICULTY_LEVELS:
                        difficulty_config = DIFFICULTY_LEVELS[self.difficulty]
                        self.game_speed = difficulty_config["speed"]
                        self.move_interval = 1.0 / self.game_speed
                    
                    # Load other settings
                    self.timer.target_fps = config.get('target_fps', TARGET_FPS)
                    
        except Exception as e:
            print(f"Error loading configuration: {e}")
            # Use default settings
            self.difficulty = "medium"
            self.game_speed = DIFFICULTY_LEVELS[self.difficulty]["speed"]
            self.move_interval = 1.0 / self.game_speed
    
    def handle_events(self, events: List[pygame.event.Event]):
        """
        Handle pygame events
        
        Args:
            events: List of pygame events
        """
        current_time = time.time()
        
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                
                # Handle state-specific input
                if self.current_state == GameState.PLAYING:
                    self.handle_game_input(event.key, current_time)
                elif self.current_state == GameState.PAUSED:
                    self.handle_pause_input(event.key)
                elif self.current_state == GameState.GAME_OVER:
                    self.handle_game_over_input(event.key)
            
            elif event.type == pygame.KEYUP:
                self.keys_pressed.discard(event.key)
    
    def handle_game_input(self, key: int, current_time: float):
        """Handle input during gameplay"""
        # Direction change cooldown
        if current_time - self.last_direction_change < self.direction_change_cooldown:
            return
        
        # Movement controls
        direction_changed = False
        
        if key in [pygame.K_UP, pygame.K_w]:
            direction_changed = self.snake.set_direction(DIRECTION_UP)
        elif key in [pygame.K_DOWN, pygame.K_s]:
            direction_changed = self.snake.set_direction(DIRECTION_DOWN)
        elif key in [pygame.K_LEFT, pygame.K_a]:
            direction_changed = self.snake.set_direction(DIRECTION_LEFT)
        elif key in [pygame.K_RIGHT, pygame.K_d]:
            direction_changed = self.snake.set_direction(DIRECTION_RIGHT)
        elif key == pygame.K_SPACE:
            self.toggle_pause()
        elif key == pygame.K_ESCAPE:
            self.change_state(GameState.MENU)
        
        if direction_changed:
            self.last_direction_change = current_time
    
    def handle_pause_input(self, key: int):
        """Handle input during pause state"""
        if key in [pygame.K_SPACE, pygame.K_p]:
            self.toggle_pause()
        elif key == pygame.K_ESCAPE:
            self.change_state(GameState.MENU)
    
    def handle_game_over_input(self, key: int):
        """Handle input during game over state"""
        if key == pygame.K_SPACE:
            self.restart_game()
        elif key == pygame.K_ESCAPE:
            self.change_state(GameState.MENU)
    
    def update(self, delta_time: float):
        """
        Update game logic
        
        Args:
            delta_time: Time elapsed since last update
        """
        if not self.running:
            return
        
        # Update timer
        self.timer.update()
        
        # Update based on current state
        if self.current_state == GameState.PLAYING:
            self.update_gameplay(delta_time)
        elif self.current_state == GameState.PAUSED:
            pass  # No updates during pause
        elif self.current_state == GameState.GAME_OVER:
            self.update_game_over(delta_time)
        
        # Update performance counters
        self.update_performance_counters(delta_time)
    
    def update_gameplay(self, delta_time: float):
        """Update gameplay logic"""
        # Update move timer
        self.move_timer += delta_time
        
        # Check if it's time to move the snake
        if self.move_timer >= self.move_interval:
            self.move_timer = 0.0
            
            # Move snake
            self.snake.update()
            
            # Check collisions
            self.check_collisions()
            
            # Update food
            self.food_manager.update(delta_time)
            
            # Update score manager time
            self.score_manager.time_played += delta_time
            
            # Check for speed progression
            self.update_speed_progression()
    
    def check_collisions(self):
        """Check all collision types"""
        head_pos = self.snake.get_head_position()
        body_positions = self.snake.get_body_positions()
        
        # Check wall collision
        if self.collision_detector.check_wall_collision(head_pos).collision_occurred:
            self.game_over()
            return
        
        # Check self collision
        if self.collision_detector.check_self_collision(head_pos, body_positions[1:]).collision_occurred:
            self.game_over()
            return
        
        # Check food collision
        food_positions = self.food_manager.get_all_food_positions()
        for food_pos in food_positions:
            if self.collision_detector.check_food_collision(head_pos, food_pos).collision_occurred:
                self.handle_food_collision(food_pos)
    
    def handle_food_collision(self, food_position: Tuple[int, int]):
        """Handle collision with food"""
        # Get food info before consumption
        food_info = self.food_manager.get_food_at_position(food_position)
        if food_info:
            # Consume food
            consumed_food = self.food_manager.consume_food_at_position(food_position)
            
            if consumed_food:
                # Grow snake
                growth_amount = consumed_food.get('growth', 1)
                self.snake.grow(growth_amount)
                
                # Add score
                score_value = consumed_food.get('score', SCORE_NORMAL_FOOD)
                food_type = consumed_food.get('type', FoodType.NORMAL)
                points_added = self.score_manager.add_score(score_value, food_type)
                
                # Record food consumption
                self.score_manager.food_consumed(food_type)
                
                # Handle special food effects
                self.handle_special_food_effects(consumed_food)
                
                # Spawn new food
                self.food_manager.spawn_food(self.snake.get_body_positions())
    
    def handle_special_food_effects(self, food_info: Dict[str, Any]):
        """Handle special effects from different food types"""
        food_type = food_info.get('type', FoodType.NORMAL)
        
        if food_type == FoodType.SPEED:
            # Temporarily increase game speed
            self.game_speed *= 1.2
            self.move_interval = 1.0 / self.game_speed
        elif food_type == FoodType.BONUS:
            # Increase score multiplier temporarily
            self.score_manager.score_multiplier *= 1.5
        elif food_type == FoodType.GOLDEN:
            # Special golden food effects
            self.snake.grow(2)  # Extra growth
            self.score_manager.score_multiplier *= 2.0
    
    def update_speed_progression(self):
        """Update game speed based on score progression"""
        level = self.score_manager.level
        base_speed = DIFFICULTY_LEVELS[self.difficulty]["speed"]
        
        # Increase speed by 10% per level, max 3x base speed
        speed_multiplier = min(1.0 + (level - 1) * 0.1, 3.0)
        self.game_speed = base_speed * speed_multiplier
        self.move_interval = 1.0 / self.game_speed
    
    def update_game_over(self, delta_time: float):
        """Update game over state"""
        # Could add game over animations or effects here
        pass
    
    def update_performance_counters(self, delta_time: float):
        """Update performance monitoring"""
        self.frame_count += 1
        self.fps_update_timer += delta_time
        
        if self.fps_update_timer >= 1.0:  # Update FPS every second
            self.fps_counter = self.frame_count / self.fps_update_timer
            self.frame_count = 0
            self.fps_update_timer = 0.0
    
    def game_over(self):
        """Handle game over"""
        self.game_over_flag = True
        self.score_manager.game_over()
        self.change_state(GameState.GAME_OVER)
    
    def restart_game(self):
        """Restart the game"""
        self.game_over_flag = False
        self.initialize_game()
        self.change_state(GameState.PLAYING)
        self.timer.start()
    
    def toggle_pause(self):
        """Toggle pause state"""
        if self.current_state == GameState.PLAYING:
            self.change_state(GameState.PAUSED)
            self.timer.pause()
        elif self.current_state == GameState.PAUSED:
            self.change_state(GameState.PLAYING)
            self.timer.resume()
    
    def change_state(self, new_state: GameState):
        """
        Change game state
        
        Args:
            new_state: New game state to transition to
        """
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_transition_time = time.time()
            
            # Handle state-specific initialization
            if new_state == GameState.PLAYING:
                if self.previous_state == GameState.MENU:
                    self.restart_game()
                self.timer.start()
            elif new_state == GameState.MENU:
                self.timer.pause()
    
    def set_difficulty(self, difficulty: str):
        """
        Set game difficulty
        
        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard')
        """
        if difficulty in DIFFICULTY_LEVELS:
            self.difficulty = difficulty
            difficulty_config = DIFFICULTY_LEVELS[difficulty]
            self.game_speed = difficulty_config["speed"]
            self.move_interval = 1.0 / self.game_speed
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state information"""
        return {
            'state': self.current_state.value,
            'score_info': self.score_manager.get_score_info(),
            'snake_length': self.snake.get_length(),
            'snake_position': self.snake.get_head_position(),
            'food_count': len(self.food_manager.get_all_food_positions()),
            'game_speed': self.game_speed,
            'elapsed_time': self.timer.get_elapsed_time(),
            'fps': self.fps_counter,
            'difficulty': self.difficulty,
            'paused': self.paused
        }
    
    def render_debug_info(self, screen: pygame.Surface):
        """Render debug information"""
        if DEBUG_MODE:
            debug_info = [
                f"FPS: {self.fps_counter:.1f}",
                f"State: {self.current_state.value}",
                f"Score: {self.score_manager.current_score}",
                f"Level: {self.score_manager.level}",
                f"Speed: {self.game_speed:.1f}",
                f"Snake Length: {self.snake.get_length()}",
                f"Combo: {self.score_manager.combo_count}"
            ]
            
            font = pygame.font.Font(None, 24)
            y_offset = 10
            
            for info in debug_info:
                text_surface = font.render(info, True, WHITE)
                screen.blit(text_surface, (10, y_offset))
                y_offset += 25
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.score_manager.save_statistics()
            print("Game engine cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def is_running(self) -> bool:
        """Check if game engine is running"""
        return self.running
    
    def stop(self):
        """Stop the game engine"""
        self.running = False
        self.cleanup()


# Utility functions for game engine integration
def create_game_engine(screen: pygame.Surface) -> GameEngine:
    """
    Factory function to create a game engine instance
    
    Args:
        screen: Pygame surface for rendering
        
    Returns:
        GameEngine: Configured game engine instance
    """
    return GameEngine(screen)


def get_default_game_config() -> Dict[str, Any]:
    """Get default game configuration"""
    return {
        'difficulty': 'medium',
        'target_fps': TARGET_FPS,
        'sound_enabled': True,
        'music_enabled': True,
        'master_volume': DEFAULT_MASTER_VOLUME,
        'effects_volume': DEFAULT_EFFECTS_VOLUME,
        'music_volume': DEFAULT_MUSIC_VOLUME,
        'fullscreen': False,
        'vsync': True,
        'show_fps': DEBUG_MODE,
        'controls': {
            'up': ['UP', 'w'],
            'down': ['DOWN', 's'],
            'left': ['LEFT', 'a'],
            'right': ['RIGHT', 'd'],
            'pause': ['SPACE', 'p'],
            'menu': ['ESCAPE']
        }
    }


if __name__ == "__main__":
    # Test the game engine
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake Game - Engine Test")
    
    engine = create_game_engine(screen)
    
    print("Game Engine Test")
    print("================")
    print(f"Initial state: {engine.current_state}")
    print(f"Game configuration: {get_default_game_config()}")
    print(f"Score info: {engine.score_manager.get_score_info()}")
    
    # Test state changes
    engine.change_state(GameState.PLAYING)
    print(f"Changed to: {engine.current_state}")
    
    # Test game state info
    game_state = engine.get_game_state()
    print(f"Game state info: {game_state}")
    
    pygame.quit()
    print("Game engine test completed successfully!")