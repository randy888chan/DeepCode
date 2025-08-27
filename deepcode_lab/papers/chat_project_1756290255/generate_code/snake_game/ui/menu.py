"""
Main Menu Interface Module

This module handles the main menu interface, including navigation,
menu options, and transitions to different game states.
"""

import pygame
import math
import time
from typing import List, Tuple, Optional, Callable, Dict, Any
from enum import Enum

from ..utils.constants import *


class MenuState(Enum):
    """Menu state enumeration"""
    MAIN = "main"
    SETTINGS = "settings"
    HIGH_SCORES = "high_scores"
    ABOUT = "about"
    DIFFICULTY = "difficulty"


class MenuOption:
    """Represents a menu option with text, action, and visual properties"""
    
    def __init__(self, text: str, action: Callable = None, enabled: bool = True):
        self.text = text
        self.action = action
        self.enabled = enabled
        self.selected = False
        self.hover_time = 0.0
        self.animation_offset = 0.0
        
    def update(self, dt: float, is_selected: bool):
        """Update menu option animation state"""
        self.selected = is_selected
        
        if is_selected:
            self.hover_time += dt
            self.animation_offset = math.sin(self.hover_time * 3) * 5
        else:
            self.hover_time = 0.0
            self.animation_offset = 0.0


class MenuButton:
    """Interactive menu button with hover effects and animations"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 action: Callable = None, font_size: int = FONT_SIZE_MEDIUM):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.font_size = font_size
        self.enabled = True
        self.hovered = False
        self.pressed = False
        self.hover_alpha = 0
        self.press_scale = 1.0
        
        # Animation properties
        self.hover_time = 0.0
        self.glow_intensity = 0.0
        
    def update(self, dt: float, mouse_pos: Tuple[int, int], mouse_pressed: bool):
        """Update button state and animations"""
        self.hovered = self.rect.collidepoint(mouse_pos) and self.enabled
        self.pressed = self.hovered and mouse_pressed
        
        # Update hover animation
        if self.hovered:
            self.hover_time += dt
            self.hover_alpha = min(255, self.hover_alpha + dt * 500)
            self.glow_intensity = (math.sin(self.hover_time * 4) + 1) * 0.5
        else:
            self.hover_time = 0.0
            self.hover_alpha = max(0, self.hover_alpha - dt * 300)
            self.glow_intensity = 0.0
            
        # Update press animation
        if self.pressed:
            self.press_scale = max(0.95, self.press_scale - dt * 10)
        else:
            self.press_scale = min(1.0, self.press_scale + dt * 10)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the button with animations"""
        if not self.enabled:
            return
            
        # Calculate scaled rect
        scaled_width = int(self.rect.width * self.press_scale)
        scaled_height = int(self.rect.height * self.press_scale)
        scaled_x = self.rect.centerx - scaled_width // 2
        scaled_y = self.rect.centery - scaled_height // 2
        scaled_rect = pygame.Rect(scaled_x, scaled_y, scaled_width, scaled_height)
        
        # Draw glow effect
        if self.glow_intensity > 0:
            glow_color = (*MENU_ACCENT_COLOR[:3], int(self.glow_intensity * 100))
            glow_rect = scaled_rect.inflate(10, 10)
            pygame.draw.rect(screen, glow_color, glow_rect, border_radius=8)
        
        # Draw button background
        bg_color = MENU_BUTTON_HOVER_COLOR if self.hovered else MENU_BUTTON_COLOR
        pygame.draw.rect(screen, bg_color, scaled_rect, border_radius=5)
        
        # Draw button border
        border_color = MENU_ACCENT_COLOR if self.hovered else MENU_BORDER_COLOR
        pygame.draw.rect(screen, border_color, scaled_rect, width=2, border_radius=5)
        
        # Draw text
        text_color = WHITE if self.hovered else MENU_TEXT_COLOR
        text_surface = font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=scaled_rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_click(self) -> bool:
        """Handle button click and execute action"""
        if self.enabled and self.action:
            self.action()
            return True
        return False


class AnimatedBackground:
    """Animated background for the menu"""
    
    def __init__(self):
        self.particles = []
        self.time = 0.0
        self.gradient_offset = 0.0
        
        # Initialize particles
        for _ in range(20):
            self.particles.append({
                'x': random.randint(0, WINDOW_WIDTH),
                'y': random.randint(0, WINDOW_HEIGHT),
                'speed': random.uniform(10, 30),
                'size': random.randint(2, 6),
                'alpha': random.randint(50, 150),
                'direction': random.uniform(0, 2 * math.pi)
            })
    
    def update(self, dt: float):
        """Update background animation"""
        self.time += dt
        self.gradient_offset = (self.gradient_offset + dt * 20) % 360
        
        # Update particles
        for particle in self.particles:
            particle['x'] += math.cos(particle['direction']) * particle['speed'] * dt
            particle['y'] += math.sin(particle['direction']) * particle['speed'] * dt
            
            # Wrap around screen
            if particle['x'] < 0:
                particle['x'] = WINDOW_WIDTH
            elif particle['x'] > WINDOW_WIDTH:
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = WINDOW_HEIGHT
            elif particle['y'] > WINDOW_HEIGHT:
                particle['y'] = 0
    
    def draw(self, screen: pygame.Surface):
        """Draw animated background"""
        # Draw gradient background
        for y in range(0, WINDOW_HEIGHT, 4):
            ratio = y / WINDOW_HEIGHT
            color_intensity = int(20 + 10 * math.sin(self.gradient_offset + ratio * 2))
            color = (color_intensity, color_intensity // 2, color_intensity // 3)
            pygame.draw.rect(screen, color, (0, y, WINDOW_WIDTH, 4))
        
        # Draw particles
        for particle in self.particles:
            alpha = int(particle['alpha'] * (0.5 + 0.5 * math.sin(self.time + particle['x'] * 0.01)))
            color = (*MENU_ACCENT_COLOR[:3], alpha)
            pygame.draw.circle(screen, color, 
                             (int(particle['x']), int(particle['y'])), 
                             particle['size'])


class MainMenu:
    """Main menu interface class"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.state = MenuState.MAIN
        self.selected_option = 0
        self.transition_alpha = 0
        self.transition_direction = 1
        
        # Initialize fonts
        try:
            self.title_font = pygame.font.Font(None, FONT_SIZE_TITLE)
            self.menu_font = pygame.font.Font(None, FONT_SIZE_LARGE)
            self.small_font = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        except:
            self.title_font = pygame.font.Font(None, 48)
            self.menu_font = pygame.font.Font(None, 32)
            self.small_font = pygame.font.Font(None, 24)
        
        # Initialize background
        self.background = AnimatedBackground()
        
        # Initialize menu options
        self.main_options = [
            MenuOption("开始游戏", self.start_game),
            MenuOption("难度设置", self.show_difficulty),
            MenuOption("游戏设置", self.show_settings),
            MenuOption("最高分", self.show_high_scores),
            MenuOption("关于游戏", self.show_about),
            MenuOption("退出游戏", self.quit_game)
        ]
        
        self.difficulty_options = [
            MenuOption("简单", lambda: self.set_difficulty("easy")),
            MenuOption("普通", lambda: self.set_difficulty("normal")),
            MenuOption("困难", lambda: self.set_difficulty("hard")),
            MenuOption("极限", lambda: self.set_difficulty("extreme")),
            MenuOption("返回", self.show_main)
        ]
        
        # Initialize buttons
        self.buttons = []
        self.create_buttons()
        
        # Game state callbacks
        self.on_start_game = None
        self.on_quit_game = None
        self.on_show_settings = None
        
        # Animation properties
        self.logo_bounce = 0.0
        self.menu_slide_offset = 0.0
        
    def create_buttons(self):
        """Create menu buttons"""
        self.buttons.clear()
        
        if self.state == MenuState.MAIN:
            options = self.main_options
        elif self.state == MenuState.DIFFICULTY:
            options = self.difficulty_options
        else:
            options = []
        
        button_width = 200
        button_height = 50
        button_spacing = 60
        start_y = WINDOW_HEIGHT // 2 - (len(options) * button_spacing) // 2
        
        for i, option in enumerate(options):
            x = WINDOW_WIDTH // 2 - button_width // 2
            y = start_y + i * button_spacing
            button = MenuButton(x, y, button_width, button_height, 
                              option.text, option.action, FONT_SIZE_MEDIUM)
            self.buttons.append(button)
    
    def update(self, dt: float, events: List[pygame.event.Event]):
        """Update menu state and animations"""
        # Update background
        self.background.update(dt)
        
        # Update animations
        self.logo_bounce += dt * 2
        
        # Handle transition
        if self.transition_direction != 0:
            self.transition_alpha += self.transition_direction * dt * 500
            if self.transition_alpha >= 255:
                self.transition_alpha = 255
                self.transition_direction = 0
            elif self.transition_alpha <= 0:
                self.transition_alpha = 0
                self.transition_direction = 0
        
        # Get mouse state
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        
        # Update buttons
        for button in self.buttons:
            button.update(dt, mouse_pos, mouse_pressed)
        
        # Handle events
        for event in events:
            if event.type == pygame.KEYDOWN:
                self.handle_keyboard_input(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_mouse_click(mouse_pos)
    
    def handle_keyboard_input(self, key: int):
        """Handle keyboard input for menu navigation"""
        if key == pygame.K_UP or key == pygame.K_w:
            self.selected_option = (self.selected_option - 1) % len(self.buttons)
        elif key == pygame.K_DOWN or key == pygame.K_s:
            self.selected_option = (self.selected_option + 1) % len(self.buttons)
        elif key == pygame.K_RETURN or key == pygame.K_SPACE:
            if self.buttons:
                self.buttons[self.selected_option].handle_click()
        elif key == pygame.K_ESCAPE:
            if self.state != MenuState.MAIN:
                self.show_main()
            else:
                self.quit_game()
    
    def handle_mouse_click(self, mouse_pos: Tuple[int, int]):
        """Handle mouse click on menu buttons"""
        for button in self.buttons:
            if button.rect.collidepoint(mouse_pos):
                button.handle_click()
                break
    
    def draw(self):
        """Draw the menu interface"""
        # Draw background
        self.background.draw(self.screen)
        
        # Draw title
        self.draw_title()
        
        # Draw menu options
        self.draw_menu_options()
        
        # Draw transition overlay
        if self.transition_alpha > 0:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(self.transition_alpha)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
    
    def draw_title(self):
        """Draw the game title with animation"""
        title_text = "贪吃蛇游戏"
        bounce_offset = math.sin(self.logo_bounce) * 10
        
        # Draw title shadow
        shadow_surface = self.title_font.render(title_text, True, BLACK)
        shadow_rect = shadow_surface.get_rect(center=(WINDOW_WIDTH // 2 + 3, 
                                                     150 + bounce_offset + 3))
        self.screen.blit(shadow_surface, shadow_rect)
        
        # Draw title
        title_surface = self.title_font.render(title_text, True, MENU_TITLE_COLOR)
        title_rect = title_surface.get_rect(center=(WINDOW_WIDTH // 2, 
                                                   150 + bounce_offset))
        self.screen.blit(title_surface, title_rect)
        
        # Draw subtitle
        if self.state == MenuState.MAIN:
            subtitle_text = "Snake Game"
            subtitle_surface = self.small_font.render(subtitle_text, True, MENU_TEXT_COLOR)
            subtitle_rect = subtitle_surface.get_rect(center=(WINDOW_WIDTH // 2, 
                                                             190 + bounce_offset))
            self.screen.blit(subtitle_surface, subtitle_rect)
    
    def draw_menu_options(self):
        """Draw menu options and buttons"""
        # Draw buttons
        for i, button in enumerate(self.buttons):
            # Highlight selected button
            if i == self.selected_option:
                button.hovered = True
            button.draw(self.screen, self.menu_font)
        
        # Draw state-specific content
        if self.state == MenuState.DIFFICULTY:
            self.draw_difficulty_info()
        elif self.state == MenuState.HIGH_SCORES:
            self.draw_high_scores()
        elif self.state == MenuState.ABOUT:
            self.draw_about_info()
    
    def draw_difficulty_info(self):
        """Draw difficulty selection information"""
        info_y = WINDOW_HEIGHT - 150
        
        difficulty_info = {
            "简单": "速度较慢，适合新手",
            "普通": "标准速度，经典体验", 
            "困难": "速度较快，挑战性强",
            "极限": "极高速度，专家级别"
        }
        
        # Draw current difficulty info
        if self.selected_option < len(self.difficulty_options) - 1:
            option_text = self.difficulty_options[self.selected_option].text
            if option_text in difficulty_info:
                info_text = difficulty_info[option_text]
                info_surface = self.small_font.render(info_text, True, MENU_TEXT_COLOR)
                info_rect = info_surface.get_rect(center=(WINDOW_WIDTH // 2, info_y))
                self.screen.blit(info_surface, info_rect)
    
    def draw_high_scores(self):
        """Draw high scores display"""
        # This would typically load from a file
        scores = [
            ("玩家1", 1500),
            ("玩家2", 1200),
            ("玩家3", 1000),
            ("玩家4", 800),
            ("玩家5", 600)
        ]
        
        start_y = 300
        for i, (name, score) in enumerate(scores):
            score_text = f"{i+1}. {name}: {score}"
            score_surface = self.small_font.render(score_text, True, MENU_TEXT_COLOR)
            score_rect = score_surface.get_rect(center=(WINDOW_WIDTH // 2, 
                                                       start_y + i * 30))
            self.screen.blit(score_surface, score_rect)
    
    def draw_about_info(self):
        """Draw about information"""
        about_lines = [
            "贪吃蛇游戏 v1.0",
            "使用 Pygame 开发",
            "支持多种难度级别",
            "包含音效和动画效果",
            "",
            "控制方式:",
            "方向键或 WASD 移动",
            "空格键暂停游戏",
            "ESC 键返回菜单"
        ]
        
        start_y = 300
        for i, line in enumerate(about_lines):
            if line:
                text_surface = self.small_font.render(line, True, MENU_TEXT_COLOR)
                text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, 
                                                         start_y + i * 25))
                self.screen.blit(text_surface, text_rect)
    
    def start_transition(self):
        """Start menu transition animation"""
        self.transition_direction = 1
        self.transition_alpha = 0
    
    def change_state(self, new_state: MenuState):
        """Change menu state with transition"""
        self.state = new_state
        self.selected_option = 0
        self.create_buttons()
        self.start_transition()
    
    # Menu action methods
    def start_game(self):
        """Start the game"""
        if self.on_start_game:
            self.on_start_game()
    
    def show_difficulty(self):
        """Show difficulty selection"""
        self.change_state(MenuState.DIFFICULTY)
    
    def show_settings(self):
        """Show settings menu"""
        if self.on_show_settings:
            self.on_show_settings()
        else:
            self.change_state(MenuState.MAIN)
    
    def show_high_scores(self):
        """Show high scores"""
        self.change_state(MenuState.HIGH_SCORES)
    
    def show_about(self):
        """Show about information"""
        self.change_state(MenuState.ABOUT)
    
    def show_main(self):
        """Return to main menu"""
        self.change_state(MenuState.MAIN)
    
    def set_difficulty(self, difficulty: str):
        """Set game difficulty"""
        # This would typically save to configuration
        print(f"Difficulty set to: {difficulty}")
        self.show_main()
    
    def quit_game(self):
        """Quit the game"""
        if self.on_quit_game:
            self.on_quit_game()
        else:
            pygame.quit()
            exit()
    
    def set_callbacks(self, on_start_game: Callable = None, 
                     on_quit_game: Callable = None,
                     on_show_settings: Callable = None):
        """Set callback functions for menu actions"""
        self.on_start_game = on_start_game
        self.on_quit_game = on_quit_game
        self.on_show_settings = on_show_settings


# Factory function
def create_main_menu(screen: pygame.Surface) -> MainMenu:
    """Create a main menu instance"""
    return MainMenu(screen)


# Utility functions
def get_menu_center() -> Tuple[int, int]:
    """Get the center point of the menu"""
    return (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)


def create_menu_button(x: int, y: int, width: int, height: int, 
                      text: str, action: Callable = None) -> MenuButton:
    """Create a menu button with standard styling"""
    return MenuButton(x, y, width, height, text, action)


def interpolate_color(color1: Tuple[int, int, int], 
                     color2: Tuple[int, int, int], 
                     factor: float) -> Tuple[int, int, int]:
    """Interpolate between two colors"""
    factor = max(0, min(1, factor))
    return (
        int(color1[0] + (color2[0] - color1[0]) * factor),
        int(color1[1] + (color2[1] - color1[1]) * factor),
        int(color1[2] + (color2[2] - color1[2]) * factor)
    )