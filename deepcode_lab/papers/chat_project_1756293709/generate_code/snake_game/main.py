#!/usr/bin/env python3
"""
Pixel Art Snake Game - Main Entry Point
A retro-style Snake game with beautiful pixel graphics, scoring system, and difficulty levels.
"""

import pygame
import sys
import os
from typing import Optional

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import GameConfig, Colors
from utils.score_manager import ScoreManager
from game.game_state import GameState
from game.snake import Snake
from game.food import Food
from game.collision import CollisionDetector
from ui.menu import MenuManager
from ui.hud import HUD


class SnakeGame:
    """Main game class that manages the entire Snake game."""
    
    def __init__(self):
        """Initialize the Snake game."""
        pygame.init()
        
        # Initialize display
        self.screen = pygame.display.set_mode((GameConfig.WINDOW_WIDTH, GameConfig.WINDOW_HEIGHT))
        pygame.display.set_caption("Pixel Art Snake Game")
        
        # Initialize clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Initialize game components
        self.game_state = GameState()
        self.score_manager = ScoreManager()
        self.menu_manager = MenuManager(self.screen)
        self.hud = HUD(self.screen)
        
        # Game objects (initialized when game starts)
        self.snake: Optional[Snake] = None
        self.food: Optional[Food] = None
        self.collision_detector: Optional[CollisionDetector] = None
        
        # Game variables
        self.running = True
        self.last_move_time = 0
        
    def initialize_game_objects(self):
        """Initialize game objects for a new game."""
        self.snake = Snake()
        self.food = Food()
        self.collision_detector = CollisionDetector()
        
        # Generate first food
        self.food.generate_food(self.snake.get_body())
        
        # Reset game state
        self.game_state.reset_game()
        self.last_move_time = pygame.time.get_ticks()
        
    def handle_events(self):
        """Handle all pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if self.game_state.current_state == "menu":
                    self.handle_menu_input(event.key)
                elif self.game_state.current_state == "playing":
                    self.handle_game_input(event.key)
                elif self.game_state.current_state == "paused":
                    self.handle_pause_input(event.key)
                elif self.game_state.current_state == "game_over":
                    self.handle_game_over_input(event.key)
                    
    def handle_menu_input(self, key):
        """Handle input in menu state."""
        if key == pygame.K_RETURN or key == pygame.K_SPACE:
            self.start_new_game()
        elif key == pygame.K_1:
            self.game_state.set_difficulty("easy")
        elif key == pygame.K_2:
            self.game_state.set_difficulty("medium")
        elif key == pygame.K_3:
            self.game_state.set_difficulty("hard")
        elif key == pygame.K_ESCAPE:
            self.running = False
            
    def handle_game_input(self, key):
        """Handle input during gameplay."""
        if key == pygame.K_UP and self.snake.direction != "down":
            self.snake.set_direction("up")
        elif key == pygame.K_DOWN and self.snake.direction != "up":
            self.snake.set_direction("down")
        elif key == pygame.K_LEFT and self.snake.direction != "right":
            self.snake.set_direction("left")
        elif key == pygame.K_RIGHT and self.snake.direction != "left":
            self.snake.set_direction("right")
        elif key == pygame.K_SPACE or key == pygame.K_p:
            self.game_state.pause_game()
        elif key == pygame.K_ESCAPE:
            self.game_state.set_state("menu")
            
    def handle_pause_input(self, key):
        """Handle input when game is paused."""
        if key == pygame.K_SPACE or key == pygame.K_p:
            self.game_state.resume_game()
        elif key == pygame.K_ESCAPE:
            self.game_state.set_state("menu")
        elif key == pygame.K_r:
            self.start_new_game()
            
    def handle_game_over_input(self, key):
        """Handle input in game over state."""
        if key == pygame.K_RETURN or key == pygame.K_SPACE or key == pygame.K_r:
            self.start_new_game()
        elif key == pygame.K_ESCAPE:
            self.game_state.set_state("menu")
            
    def start_new_game(self):
        """Start a new game."""
        self.initialize_game_objects()
        self.game_state.start_game()
        
    def update_game(self):
        """Update game logic."""
        if self.game_state.current_state != "playing":
            return
            
        current_time = pygame.time.get_ticks()
        move_delay = self.game_state.get_move_delay()
        
        # Check if it's time to move the snake
        if current_time - self.last_move_time >= move_delay:
            self.last_move_time = current_time
            
            # Move snake
            self.snake.move()
            
            # Check collisions
            if self.collision_detector.check_wall_collision(self.snake.get_head()):
                self.game_over()
                return
                
            if self.collision_detector.check_self_collision(self.snake.get_body()):
                self.game_over()
                return
                
            # Check food collision
            if self.collision_detector.check_food_collision(self.snake.get_head(), self.food.position):
                self.eat_food()
                
    def eat_food(self):
        """Handle food consumption."""
        # Grow snake
        self.snake.grow()
        
        # Update score
        points = self.game_state.calculate_food_points()
        self.game_state.add_score(points)
        
        # Check for level up
        self.game_state.check_level_up()
        
        # Generate new food
        self.food.generate_food(self.snake.get_body())
        
    def game_over(self):
        """Handle game over."""
        # Save high score
        self.score_manager.save_score(self.game_state.score)
        
        # Set game over state
        self.game_state.set_state("game_over")
        
    def render(self):
        """Render the game."""
        # Clear screen
        self.screen.fill(Colors.BACKGROUND)
        
        if self.game_state.current_state == "menu":
            self.menu_manager.draw_main_menu(
                self.game_state.difficulty,
                self.score_manager.get_high_score()
            )
            
        elif self.game_state.current_state in ["playing", "paused"]:
            # Draw game objects
            if self.snake:
                self.snake.draw(self.screen)
            if self.food:
                self.food.draw(self.screen)
                
            # Draw HUD
            self.hud.draw_game_hud(
                self.game_state.score,
                self.game_state.level,
                self.game_state.difficulty
            )
            
            # Draw pause overlay if paused
            if self.game_state.current_state == "paused":
                self.menu_manager.draw_pause_overlay()
                
        elif self.game_state.current_state == "game_over":
            # Draw final game state
            if self.snake:
                self.snake.draw(self.screen)
            if self.food:
                self.food.draw(self.screen)
                
            # Draw game over screen
            self.menu_manager.draw_game_over_screen(
                self.game_state.score,
                self.score_manager.get_high_score(),
                self.score_manager.is_new_high_score(self.game_state.score)
            )
            
        # Update display
        pygame.display.flip()
        
    def run(self):
        """Main game loop."""
        print("🐍 Starting Pixel Art Snake Game...")
        print("Controls:")
        print("  Arrow Keys - Move snake")
        print("  SPACE/P - Pause/Resume")
        print("  R - Restart (when paused/game over)")
        print("  ESC - Back to menu/Quit")
        print("  1/2/3 - Select difficulty (menu)")
        print()
        
        while self.running:
            # Handle events
            self.handle_events()
            
            # Update game logic
            self.update_game()
            
            # Render everything
            self.render()
            
            # Control FPS
            self.clock.tick(GameConfig.FPS)
            
        # Cleanup
        pygame.quit()
        print("Thanks for playing! 🐍")


def main():
    """Main entry point."""
    try:
        game = SnakeGame()
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()