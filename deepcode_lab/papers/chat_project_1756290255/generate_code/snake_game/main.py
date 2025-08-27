#!/usr/bin/env python3
"""
贪吃蛇游戏 (Snake Game)
Main entry point for the Snake Game application.

This module initializes the game environment, handles the main game loop,
and coordinates all game systems including UI, audio, and game logic.
"""

import sys
import os
import pygame
import traceback
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import game modules (will be implemented)
try:
    from utils.constants import *
    from utils.helpers import setup_logging, handle_error
    from utils.data_manager import DataManager
    from ui.menu import MainMenu
    from ui.game_screen import GameScreen
    from ui.settings import SettingsScreen
    from game.game_engine import GameEngine
    from audio.sound_manager import SoundManager
except ImportError as e:
    print(f"Warning: Some modules not yet implemented: {e}")
    # Define basic constants for initial setup
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    FPS = 60
    GAME_TITLE = "贪吃蛇游戏 (Snake Game)"


class SnakeGameApp:
    """
    Main application class that manages the entire Snake Game.
    
    This class handles:
    - Pygame initialization and cleanup
    - Game state management (menu, playing, settings, etc.)
    - Main game loop coordination
    - Error handling and logging
    - Resource management
    """
    
    def __init__(self):
        """Initialize the Snake Game application."""
        self.running = False
        self.clock = None
        self.screen = None
        self.game_state = "menu"  # menu, playing, settings, paused, game_over
        
        # Game systems (will be initialized)
        self.data_manager = None
        self.sound_manager = None
        self.game_engine = None
        self.main_menu = None
        self.game_screen = None
        self.settings_screen = None
        
        # Initialize the application
        self._initialize_pygame()
        self._initialize_systems()
    
    def _initialize_pygame(self):
        """Initialize Pygame and create the main window."""
        try:
            # Initialize Pygame
            pygame.init()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Set up the display
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption(GAME_TITLE)
            
            # Set up the game clock for FPS control
            self.clock = pygame.time.Clock()
            
            # Set window icon (if available)
            try:
                icon_path = project_root / "assets" / "images" / "snake_icon.png"
                if icon_path.exists():
                    icon = pygame.image.load(str(icon_path))
                    pygame.display.set_icon(icon)
            except Exception:
                pass  # Icon not critical for functionality
                
            print(f"✅ Pygame initialized successfully")
            print(f"   Window size: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
            print(f"   Target FPS: {FPS}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Pygame: {e}")
            sys.exit(1)
    
    def _initialize_systems(self):
        """Initialize all game systems and components."""
        try:
            # Initialize data manager for configuration and scores
            try:
                self.data_manager = DataManager()
                print("✅ Data manager initialized")
            except Exception as e:
                print(f"⚠️  Data manager initialization failed: {e}")
                self.data_manager = None
            
            # Initialize sound manager
            try:
                self.sound_manager = SoundManager()
                print("✅ Sound manager initialized")
            except Exception as e:
                print(f"⚠️  Sound manager initialization failed: {e}")
                self.sound_manager = None
            
            # Initialize game systems
            try:
                self.game_engine = GameEngine(self.screen, self.sound_manager)
                print("✅ Game engine initialized")
            except Exception as e:
                print(f"⚠️  Game engine initialization failed: {e}")
                self.game_engine = None
            
            # Initialize UI screens
            try:
                self.main_menu = MainMenu(self.screen, self.sound_manager)
                self.game_screen = GameScreen(self.screen, self.game_engine)
                self.settings_screen = SettingsScreen(self.screen, self.data_manager)
                print("✅ UI screens initialized")
            except Exception as e:
                print(f"⚠️  UI screens initialization failed: {e}")
                # Create minimal fallback UI
                self._create_fallback_ui()
                
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            traceback.print_exc()
    
    def _create_fallback_ui(self):
        """Create minimal fallback UI if main UI systems fail."""
        print("🔧 Creating fallback UI...")
        # This will be a simple text-based interface for basic functionality
        self.main_menu = None
        self.game_screen = None
        self.settings_screen = None
    
    def handle_events(self):
        """Handle all pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            # Handle events based on current game state
            if self.game_state == "menu":
                self._handle_menu_events(event)
            elif self.game_state == "playing":
                self._handle_game_events(event)
            elif self.game_state == "settings":
                self._handle_settings_events(event)
            elif self.game_state == "paused":
                self._handle_pause_events(event)
            elif self.game_state == "game_over":
                self._handle_game_over_events(event)
    
    def _handle_menu_events(self, event):
        """Handle events in the main menu state."""
        if self.main_menu:
            result = self.main_menu.handle_event(event)
            if result == "start_game":
                self.game_state = "playing"
                if self.game_engine:
                    self.game_engine.start_new_game()
            elif result == "settings":
                self.game_state = "settings"
            elif result == "quit":
                self.running = False
        else:
            # Fallback menu handling
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.game_state = "playing"
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def _handle_game_events(self, event):
        """Handle events during gameplay."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game_state = "paused"
            elif self.game_engine:
                self.game_engine.handle_input(event)
        
        if self.game_engine and self.game_engine.is_game_over():
            self.game_state = "game_over"
    
    def _handle_settings_events(self, event):
        """Handle events in the settings screen."""
        if self.settings_screen:
            result = self.settings_screen.handle_event(event)
            if result == "back":
                self.game_state = "menu"
        else:
            # Fallback settings handling
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.game_state = "menu"
    
    def _handle_pause_events(self, event):
        """Handle events when the game is paused."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game_state = "playing"
            elif event.key == pygame.K_m:
                self.game_state = "menu"
    
    def _handle_game_over_events(self, event):
        """Handle events in the game over state."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.game_state = "playing"
                if self.game_engine:
                    self.game_engine.start_new_game()
            elif event.key == pygame.K_ESCAPE:
                self.game_state = "menu"
    
    def update(self):
        """Update all game systems based on current state."""
        if self.game_state == "playing":
            if self.game_engine:
                self.game_engine.update()
        elif self.game_state == "menu":
            if self.main_menu:
                self.main_menu.update()
        elif self.game_state == "settings":
            if self.settings_screen:
                self.settings_screen.update()
    
    def render(self):
        """Render the current game state to the screen."""
        # Clear the screen
        self.screen.fill((20, 20, 30))  # Dark blue background
        
        # Render based on current state
        if self.game_state == "menu":
            self._render_menu()
        elif self.game_state == "playing":
            self._render_game()
        elif self.game_state == "settings":
            self._render_settings()
        elif self.game_state == "paused":
            self._render_pause()
        elif self.game_state == "game_over":
            self._render_game_over()
        
        # Update the display
        pygame.display.flip()
    
    def _render_menu(self):
        """Render the main menu."""
        if self.main_menu:
            self.main_menu.render()
        else:
            # Fallback menu rendering
            font = pygame.font.Font(None, 48)
            title = font.render("贪吃蛇游戏", True, (255, 255, 255))
            title_rect = title.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 100))
            self.screen.blit(title, title_rect)
            
            font_small = pygame.font.Font(None, 24)
            instructions = [
                "Press SPACE to start",
                "Press ESC to quit"
            ]
            for i, text in enumerate(instructions):
                rendered = font_small.render(text, True, (200, 200, 200))
                rect = rendered.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + i*30))
                self.screen.blit(rendered, rect)
    
    def _render_game(self):
        """Render the game screen."""
        if self.game_screen and self.game_engine:
            self.game_screen.render()
        else:
            # Fallback game rendering
            font = pygame.font.Font(None, 36)
            text = font.render("Game Running (Basic Mode)", True, (255, 255, 255))
            text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            self.screen.blit(text, text_rect)
            
            font_small = pygame.font.Font(None, 24)
            instruction = font_small.render("Press ESC to pause", True, (200, 200, 200))
            inst_rect = instruction.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50))
            self.screen.blit(instruction, inst_rect)
    
    def _render_settings(self):
        """Render the settings screen."""
        if self.settings_screen:
            self.settings_screen.render()
        else:
            # Fallback settings rendering
            font = pygame.font.Font(None, 36)
            text = font.render("Settings", True, (255, 255, 255))
            text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            self.screen.blit(text, text_rect)
    
    def _render_pause(self):
        """Render the pause screen."""
        # Render the game in the background (dimmed)
        if self.game_screen and self.game_engine:
            self.game_screen.render()
        
        # Add pause overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Pause text
        font = pygame.font.Font(None, 48)
        pause_text = font.render("PAUSED", True, (255, 255, 255))
        pause_rect = pause_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        self.screen.blit(pause_text, pause_rect)
        
        # Instructions
        font_small = pygame.font.Font(None, 24)
        instructions = [
            "Press ESC to resume",
            "Press M for main menu"
        ]
        for i, text in enumerate(instructions):
            rendered = font_small.render(text, True, (200, 200, 200))
            rect = rendered.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 60 + i*30))
            self.screen.blit(rendered, rect)
    
    def _render_game_over(self):
        """Render the game over screen."""
        # Render the game in the background (dimmed)
        if self.game_screen and self.game_engine:
            self.game_screen.render()
        
        # Add game over overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        font = pygame.font.Font(None, 48)
        game_over_text = font.render("GAME OVER", True, (255, 100, 100))
        game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Score display
        if self.game_engine:
            score = self.game_engine.get_score()
            font_medium = pygame.font.Font(None, 36)
            score_text = font_medium.render(f"Score: {score}", True, (255, 255, 255))
            score_rect = score_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 60))
            self.screen.blit(score_text, score_rect)
        
        # Instructions
        font_small = pygame.font.Font(None, 24)
        instructions = [
            "Press SPACE to play again",
            "Press ESC for main menu"
        ]
        for i, text in enumerate(instructions):
            rendered = font_small.render(text, True, (200, 200, 200))
            rect = rendered.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 120 + i*30))
            self.screen.blit(rendered, rect)
    
    def run(self):
        """Main game loop."""
        print(f"🎮 Starting Snake Game...")
        print(f"   Initial state: {self.game_state}")
        
        self.running = True
        
        try:
            while self.running:
                # Handle events
                self.handle_events()
                
                # Update game systems
                self.update()
                
                # Render everything
                self.render()
                
                # Control frame rate
                self.clock.tick(FPS)
                
        except KeyboardInterrupt:
            print("\n🛑 Game interrupted by user")
        except Exception as e:
            print(f"❌ Game crashed: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and quit the game."""
        print("🧹 Cleaning up resources...")
        
        # Save any pending data
        if self.data_manager:
            try:
                self.data_manager.save_all()
            except Exception as e:
                print(f"⚠️  Failed to save data: {e}")
        
        # Stop audio
        if self.sound_manager:
            try:
                self.sound_manager.cleanup()
            except Exception as e:
                print(f"⚠️  Failed to cleanup audio: {e}")
        
        # Quit Pygame
        try:
            pygame.quit()
            print("✅ Pygame cleaned up successfully")
        except Exception as e:
            print(f"⚠️  Pygame cleanup warning: {e}")


def main():
    """Main entry point for the Snake Game application."""
    print("=" * 60)
    print("🐍 贪吃蛇游戏 (Snake Game) - Starting...")
    print("=" * 60)
    
    try:
        # Create and run the game application
        app = SnakeGameApp()
        app.run()
        
    except Exception as e:
        print(f"❌ Failed to start game: {e}")
        traceback.print_exc()
        return 1
    
    print("👋 Thanks for playing!")
    return 0


if __name__ == "__main__":
    sys.exit(main())