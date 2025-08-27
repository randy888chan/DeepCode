"""
Game Screen Module - Handles the main game display and HUD
"""

import pygame
import math
from typing import Optional, Dict, Any, List, Tuple
from ..utils.constants import *
from ..game.game_engine import GameEngine


class HUD:
    """Heads-Up Display for showing game information"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self._init_fonts()
        
        # HUD positioning
        self.score_pos = (20, 20)
        self.level_pos = (20, 60)
        self.high_score_pos = (WINDOW_WIDTH - 200, 20)
        self.time_pos = (WINDOW_WIDTH - 200, 60)
        self.fps_pos = (WINDOW_WIDTH - 100, WINDOW_HEIGHT - 30)
        
        # Animation properties
        self.score_animation_time = 0
        self.score_scale = 1.0
        self.last_score = 0
        
    def _init_fonts(self):
        """Initialize fonts for HUD elements"""
        try:
            self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
            self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
            self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)
        except pygame.error:
            # Fallback to default font
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 18)
    
    def update(self, delta_time: float, current_score: int):
        """Update HUD animations and effects"""
        # Score animation when score changes
        if current_score != self.last_score:
            self.score_animation_time = 0.5  # Animation duration
            self.last_score = current_score
        
        if self.score_animation_time > 0:
            self.score_animation_time -= delta_time
            # Scale effect for score change
            progress = 1 - (self.score_animation_time / 0.5)
            self.score_scale = 1.0 + 0.3 * math.sin(progress * math.pi)
        else:
            self.score_scale = 1.0
    
    def render(self, game_data: Dict[str, Any]):
        """Render all HUD elements"""
        self._render_score(game_data.get('score', 0))
        self._render_level(game_data.get('level', 1))
        self._render_high_score(game_data.get('high_score', 0))
        self._render_game_time(game_data.get('game_time', 0))
        
        if game_data.get('show_fps', False):
            self._render_fps(game_data.get('fps', 0))
        
        # Render pause indicator if game is paused
        if game_data.get('is_paused', False):
            self._render_pause_indicator()
    
    def _render_score(self, score: int):
        """Render current score with animation"""
        text = f"Score: {score}"
        
        # Apply scale animation
        if self.score_scale != 1.0:
            font = pygame.font.Font(None, int(FONT_SIZE_LARGE * self.score_scale))
        else:
            font = self.font_large
        
        text_surface = font.render(text, True, UI_TEXT_COLOR)
        
        # Center the scaled text
        if self.score_scale != 1.0:
            rect = text_surface.get_rect()
            pos = (self.score_pos[0] - rect.width // 4, self.score_pos[1] - rect.height // 4)
        else:
            pos = self.score_pos
        
        self.screen.blit(text_surface, pos)
    
    def _render_level(self, level: int):
        """Render current level"""
        text = f"Level: {level}"
        text_surface = self.font_medium.render(text, True, UI_TEXT_COLOR)
        self.screen.blit(text_surface, self.level_pos)
    
    def _render_high_score(self, high_score: int):
        """Render high score"""
        text = f"High: {high_score}"
        text_surface = self.font_medium.render(text, True, UI_SECONDARY_COLOR)
        self.screen.blit(text_surface, self.high_score_pos)
    
    def _render_game_time(self, game_time: float):
        """Render game time"""
        minutes = int(game_time // 60)
        seconds = int(game_time % 60)
        text = f"Time: {minutes:02d}:{seconds:02d}"
        text_surface = self.font_small.render(text, True, UI_SECONDARY_COLOR)
        self.screen.blit(text_surface, self.time_pos)
    
    def _render_fps(self, fps: float):
        """Render FPS counter"""
        text = f"FPS: {fps:.1f}"
        text_surface = self.font_small.render(text, True, UI_DEBUG_COLOR)
        self.screen.blit(text_surface, self.fps_pos)
    
    def _render_pause_indicator(self):
        """Render pause indicator"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Pause text
        pause_text = "PAUSED"
        text_surface = self.font_large.render(pause_text, True, WHITE)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(text_surface, text_rect)
        
        # Instructions
        instruction_text = "Press SPACE to resume"
        instruction_surface = self.font_medium.render(instruction_text, True, UI_SECONDARY_COLOR)
        instruction_rect = instruction_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
        self.screen.blit(instruction_surface, instruction_rect)


class GameBoard:
    """Handles the game board rendering and grid display"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.grid_surface = None
        self._create_grid_surface()
        
        # Visual effects
        self.border_pulse_time = 0
        self.border_pulse_speed = 2.0
    
    def _create_grid_surface(self):
        """Create a pre-rendered grid surface for performance"""
        self.grid_surface = pygame.Surface((GAME_AREA_WIDTH, GAME_AREA_HEIGHT))
        self.grid_surface.fill(GAME_BACKGROUND_COLOR)
        
        # Draw grid lines
        for x in range(0, GAME_AREA_WIDTH, CELL_SIZE):
            pygame.draw.line(self.grid_surface, GRID_COLOR, 
                           (x, 0), (x, GAME_AREA_HEIGHT), 1)
        
        for y in range(0, GAME_AREA_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.grid_surface, GRID_COLOR, 
                           (0, y), (GAME_AREA_WIDTH, y), 1)
    
    def update(self, delta_time: float):
        """Update board animations"""
        self.border_pulse_time += delta_time * self.border_pulse_speed
    
    def render(self, game_data: Dict[str, Any]):
        """Render the game board"""
        # Fill background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Render grid
        self.screen.blit(self.grid_surface, GAME_AREA_OFFSET)
        
        # Render animated border
        self._render_border()
    
    def _render_border(self):
        """Render animated border around game area"""
        # Calculate pulse effect
        pulse_intensity = (math.sin(self.border_pulse_time) + 1) / 2
        border_color = [
            int(BORDER_COLOR[0] + (255 - BORDER_COLOR[0]) * pulse_intensity * 0.3),
            int(BORDER_COLOR[1] + (255 - BORDER_COLOR[1]) * pulse_intensity * 0.3),
            int(BORDER_COLOR[2] + (255 - BORDER_COLOR[2]) * pulse_intensity * 0.3)
        ]
        
        # Draw border
        border_rect = pygame.Rect(
            GAME_AREA_OFFSET[0] - BORDER_WIDTH,
            GAME_AREA_OFFSET[1] - BORDER_WIDTH,
            GAME_AREA_WIDTH + 2 * BORDER_WIDTH,
            GAME_AREA_HEIGHT + 2 * BORDER_WIDTH
        )
        
        pygame.draw.rect(self.screen, border_color, border_rect, BORDER_WIDTH)


class GameOverScreen:
    """Handles game over display and animations"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_large = None
        self.font_medium = None
        self._init_fonts()
        
        # Animation properties
        self.fade_alpha = 0
        self.text_scale = 0
        self.animation_time = 0
        self.is_animating = False
    
    def _init_fonts(self):
        """Initialize fonts"""
        try:
            self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
            self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 24)
    
    def start_animation(self):
        """Start game over animation"""
        self.is_animating = True
        self.animation_time = 0
        self.fade_alpha = 0
        self.text_scale = 0
    
    def update(self, delta_time: float):
        """Update game over animations"""
        if not self.is_animating:
            return
        
        self.animation_time += delta_time
        
        # Fade in effect
        if self.animation_time < 1.0:
            self.fade_alpha = int(255 * (self.animation_time / 1.0))
        else:
            self.fade_alpha = 255
        
        # Text scale effect
        if self.animation_time < 0.5:
            self.text_scale = self.animation_time / 0.5
        else:
            self.text_scale = 1.0
        
        # Stop animation after 2 seconds
        if self.animation_time >= 2.0:
            self.is_animating = False
    
    def render(self, game_data: Dict[str, Any]):
        """Render game over screen"""
        if not self.is_animating and self.fade_alpha == 0:
            return
        
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(self.fade_alpha // 2)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game Over text
        if self.text_scale > 0:
            font_size = int(FONT_SIZE_LARGE * self.text_scale)
            font = pygame.font.Font(None, font_size)
            
            game_over_text = "GAME OVER"
            text_surface = font.render(game_over_text, True, GAME_OVER_COLOR)
            text_surface.set_alpha(self.fade_alpha)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
            self.screen.blit(text_surface, text_rect)
            
            # Final score
            score = game_data.get('final_score', 0)
            score_text = f"Final Score: {score}"
            score_surface = self.font_medium.render(score_text, True, UI_TEXT_COLOR)
            score_surface.set_alpha(self.fade_alpha)
            score_rect = score_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(score_surface, score_rect)
            
            # Instructions
            instruction_text = "Press R to restart or ESC to menu"
            instruction_surface = self.font_medium.render(instruction_text, True, UI_SECONDARY_COLOR)
            instruction_surface.set_alpha(self.fade_alpha)
            instruction_rect = instruction_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            self.screen.blit(instruction_surface, instruction_rect)


class GameScreen:
    """Main game screen that coordinates all visual elements"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.hud = HUD(screen)
        self.game_board = GameBoard(screen)
        self.game_over_screen = GameOverScreen(screen)
        
        # Performance tracking
        self.clock = pygame.time.Clock()
        self.fps = 0
        self.frame_count = 0
        self.fps_update_time = 0
        
        # Game state tracking
        self.last_game_state = None
    
    def update(self, delta_time: float, game_engine: GameEngine):
        """Update all screen components"""
        # Update FPS calculation
        self.frame_count += 1
        self.fps_update_time += delta_time
        
        if self.fps_update_time >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / self.fps_update_time
            self.frame_count = 0
            self.fps_update_time = 0
        
        # Get game data
        game_data = self._get_game_data(game_engine)
        
        # Update components
        self.hud.update(delta_time, game_data.get('score', 0))
        self.game_board.update(delta_time)
        
        # Handle game state changes
        current_state = game_data.get('game_state')
        if current_state != self.last_game_state:
            if current_state == 'GAME_OVER':
                self.game_over_screen.start_animation()
            self.last_game_state = current_state
        
        # Update game over screen
        if current_state == 'GAME_OVER':
            self.game_over_screen.update(delta_time)
    
    def render(self, game_engine: GameEngine):
        """Render the complete game screen"""
        # Get game data
        game_data = self._get_game_data(game_engine)
        
        # Render game board
        self.game_board.render(game_data)
        
        # Render game objects (snake and food)
        self._render_game_objects(game_engine)
        
        # Render HUD
        self.hud.render(game_data)
        
        # Render game over screen if needed
        if game_data.get('game_state') == 'GAME_OVER':
            self.game_over_screen.render(game_data)
        
        # Update display
        pygame.display.flip()
    
    def _get_game_data(self, game_engine: GameEngine) -> Dict[str, Any]:
        """Extract game data for rendering"""
        try:
            return {
                'score': game_engine.score_manager.get_current_score(),
                'level': game_engine.score_manager.get_current_level(),
                'high_score': game_engine.score_manager.get_high_score(),
                'game_time': game_engine.timer.get_elapsed_time(),
                'game_state': game_engine.current_state.name,
                'is_paused': game_engine.current_state.name == 'PAUSED',
                'final_score': game_engine.score_manager.get_current_score(),
                'fps': self.fps,
                'show_fps': DEBUG_MODE
            }
        except AttributeError:
            # Fallback for missing attributes
            return {
                'score': 0,
                'level': 1,
                'high_score': 0,
                'game_time': 0,
                'game_state': 'PLAYING',
                'is_paused': False,
                'final_score': 0,
                'fps': self.fps,
                'show_fps': False
            }
    
    def _render_game_objects(self, game_engine: GameEngine):
        """Render snake and food objects"""
        try:
            # Render food first (so snake appears on top)
            if hasattr(game_engine, 'food_manager') and game_engine.food_manager.is_food_active():
                game_engine.food_manager.render(self.screen)
            
            # Render snake
            if hasattr(game_engine, 'snake'):
                game_engine.snake.render(self.screen)
                
        except AttributeError as e:
            # Handle missing game objects gracefully
            if DEBUG_MODE:
                print(f"Warning: Could not render game objects: {e}")
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def reset(self):
        """Reset screen state for new game"""
        self.last_game_state = None
        self.game_over_screen.fade_alpha = 0
        self.game_over_screen.is_animating = False
        self.hud.score_animation_time = 0
        self.hud.last_score = 0


# Factory function for creating game screen
def create_game_screen(screen: pygame.Surface) -> GameScreen:
    """Factory function to create a game screen instance"""
    return GameScreen(screen)


# Utility functions for screen management
def get_screen_center() -> Tuple[int, int]:
    """Get the center point of the screen"""
    return (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)


def get_game_area_center() -> Tuple[int, int]:
    """Get the center point of the game area"""
    return (
        GAME_AREA_OFFSET[0] + GAME_AREA_WIDTH // 2,
        GAME_AREA_OFFSET[1] + GAME_AREA_HEIGHT // 2
    )


def screen_to_grid_position(screen_pos: Tuple[int, int]) -> Tuple[int, int]:
    """Convert screen coordinates to grid coordinates"""
    x = (screen_pos[0] - GAME_AREA_OFFSET[0]) // CELL_SIZE
    y = (screen_pos[1] - GAME_AREA_OFFSET[1]) // CELL_SIZE
    return (x, y)


def grid_to_screen_position(grid_pos: Tuple[int, int]) -> Tuple[int, int]:
    """Convert grid coordinates to screen coordinates"""
    x = grid_pos[0] * CELL_SIZE + GAME_AREA_OFFSET[0]
    y = grid_pos[1] * CELL_SIZE + GAME_AREA_OFFSET[1]
    return (x, y)