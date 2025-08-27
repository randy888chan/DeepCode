"""
Settings UI Module for Snake Game

This module implements the settings configuration interface, allowing players to:
- Adjust game difficulty levels
- Configure audio settings (master volume, sound effects, music)
- Customize control schemes
- Modify display settings
- Save and load configuration preferences

The settings interface integrates with the main menu system and provides
real-time feedback for configuration changes.
"""

import pygame
import json
import math
from typing import Dict, Any, Callable, Optional, List, Tuple
from enum import Enum
from pathlib import Path

# Import game constants and utilities
from ..utils.constants import *


class SettingType(Enum):
    """Types of settings that can be configured"""
    SLIDER = "slider"
    TOGGLE = "toggle"
    DROPDOWN = "dropdown"
    BUTTON = "button"


class SettingItem:
    """Individual setting item with value management and UI rendering"""
    
    def __init__(self, key: str, label: str, setting_type: SettingType, 
                 value: Any, min_val: float = 0, max_val: float = 1, 
                 options: List[str] = None, callback: Callable = None):
        self.key = key
        self.label = label
        self.setting_type = setting_type
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.options = options or []
        self.callback = callback
        
        # UI state
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.is_hovered = False
        self.is_active = False
        self.slider_dragging = False
        
    def update(self, mouse_pos: Tuple[int, int], mouse_pressed: bool, events: List[pygame.event.Event]):
        """Update setting item state based on user input"""
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
        if self.setting_type == SettingType.SLIDER:
            self._update_slider(mouse_pos, mouse_pressed, events)
        elif self.setting_type == SettingType.TOGGLE:
            self._update_toggle(events)
        elif self.setting_type == SettingType.DROPDOWN:
            self._update_dropdown(events)
        elif self.setting_type == SettingType.BUTTON:
            self._update_button(events)
    
    def _update_slider(self, mouse_pos: Tuple[int, int], mouse_pressed: bool, events: List[pygame.event.Event]):
        """Update slider value based on mouse interaction"""
        slider_rect = pygame.Rect(self.rect.x + 200, self.rect.y + 10, 200, 20)
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if slider_rect.collidepoint(event.pos):
                    self.slider_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.slider_dragging = False
        
        if self.slider_dragging and mouse_pressed:
            relative_x = mouse_pos[0] - slider_rect.x
            relative_x = max(0, min(relative_x, slider_rect.width))
            ratio = relative_x / slider_rect.width
            self.value = self.min_val + ratio * (self.max_val - self.min_val)
            
            if self.callback:
                self.callback(self.key, self.value)
    
    def _update_toggle(self, events: List[pygame.event.Event]):
        """Update toggle state on click"""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.is_hovered:
                    self.value = not self.value
                    if self.callback:
                        self.callback(self.key, self.value)
    
    def _update_dropdown(self, events: List[pygame.event.Event]):
        """Update dropdown selection"""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.is_hovered:
                    # Cycle through options
                    current_index = self.options.index(self.value) if self.value in self.options else 0
                    next_index = (current_index + 1) % len(self.options)
                    self.value = self.options[next_index]
                    if self.callback:
                        self.callback(self.key, self.value)
    
    def _update_button(self, events: List[pygame.event.Event]):
        """Update button press state"""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.is_hovered and self.callback:
                    self.callback(self.key, self.value)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font, y_pos: int):
        """Draw the setting item on screen"""
        self.rect = pygame.Rect(50, y_pos, WINDOW_WIDTH - 100, 50)
        
        # Draw background
        bg_color = MENU_BUTTON_HOVER_COLOR if self.is_hovered else MENU_BUTTON_COLOR
        pygame.draw.rect(screen, bg_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=5)
        
        # Draw label
        label_surface = font.render(self.label, True, WHITE)
        screen.blit(label_surface, (self.rect.x + 10, self.rect.y + 15))
        
        # Draw setting-specific UI
        if self.setting_type == SettingType.SLIDER:
            self._draw_slider(screen, font)
        elif self.setting_type == SettingType.TOGGLE:
            self._draw_toggle(screen, font)
        elif self.setting_type == SettingType.DROPDOWN:
            self._draw_dropdown(screen, font)
        elif self.setting_type == SettingType.BUTTON:
            self._draw_button(screen, font)
    
    def _draw_slider(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw slider control"""
        slider_rect = pygame.Rect(self.rect.x + 200, self.rect.y + 10, 200, 20)
        
        # Draw slider track
        pygame.draw.rect(screen, GRAY, slider_rect, border_radius=10)
        
        # Draw slider fill
        fill_width = int((self.value - self.min_val) / (self.max_val - self.min_val) * slider_rect.width)
        fill_rect = pygame.Rect(slider_rect.x, slider_rect.y, fill_width, slider_rect.height)
        pygame.draw.rect(screen, SNAKE_HEAD_COLOR, fill_rect, border_radius=10)
        
        # Draw slider handle
        handle_x = slider_rect.x + fill_width - 5
        handle_rect = pygame.Rect(handle_x, slider_rect.y - 5, 10, 30)
        pygame.draw.rect(screen, WHITE, handle_rect, border_radius=5)
        
        # Draw value text
        value_text = f"{self.value:.2f}" if isinstance(self.value, float) else str(self.value)
        value_surface = font.render(value_text, True, WHITE)
        screen.blit(value_surface, (self.rect.x + 420, self.rect.y + 15))
    
    def _draw_toggle(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw toggle switch"""
        toggle_rect = pygame.Rect(self.rect.x + 300, self.rect.y + 15, 60, 20)
        
        # Draw toggle background
        bg_color = SNAKE_HEAD_COLOR if self.value else GRAY
        pygame.draw.rect(screen, bg_color, toggle_rect, border_radius=10)
        
        # Draw toggle handle
        handle_x = toggle_rect.x + 35 if self.value else toggle_rect.x + 5
        handle_rect = pygame.Rect(handle_x, toggle_rect.y + 2, 16, 16)
        pygame.draw.circle(screen, WHITE, handle_rect.center, 8)
        
        # Draw status text
        status_text = "ON" if self.value else "OFF"
        status_surface = font.render(status_text, True, WHITE)
        screen.blit(status_surface, (self.rect.x + 380, self.rect.y + 15))
    
    def _draw_dropdown(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw dropdown selection"""
        dropdown_rect = pygame.Rect(self.rect.x + 250, self.rect.y + 10, 150, 30)
        
        # Draw dropdown background
        pygame.draw.rect(screen, GRAY, dropdown_rect, border_radius=5)
        pygame.draw.rect(screen, WHITE, dropdown_rect, 2, border_radius=5)
        
        # Draw current value
        value_surface = font.render(str(self.value), True, WHITE)
        screen.blit(value_surface, (dropdown_rect.x + 10, dropdown_rect.y + 5))
        
        # Draw dropdown arrow
        arrow_points = [
            (dropdown_rect.right - 20, dropdown_rect.y + 10),
            (dropdown_rect.right - 10, dropdown_rect.y + 20),
            (dropdown_rect.right - 30, dropdown_rect.y + 20)
        ]
        pygame.draw.polygon(screen, WHITE, arrow_points)
    
    def _draw_button(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw action button"""
        button_rect = pygame.Rect(self.rect.x + 250, self.rect.y + 10, 100, 30)
        
        # Draw button
        button_color = MENU_BUTTON_HOVER_COLOR if self.is_hovered else MENU_BUTTON_COLOR
        pygame.draw.rect(screen, button_color, button_rect, border_radius=5)
        pygame.draw.rect(screen, WHITE, button_rect, 2, border_radius=5)
        
        # Draw button text
        button_surface = font.render(str(self.value), True, WHITE)
        text_rect = button_surface.get_rect(center=button_rect.center)
        screen.blit(button_surface, text_rect)


class SettingsMenu:
    """Main settings menu interface with configuration management"""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
        self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        
        # Settings data
        self.settings = self._load_settings()
        self.setting_items = self._create_setting_items()
        
        # UI state
        self.scroll_offset = 0
        self.max_scroll = 0
        self.back_button_rect = pygame.Rect(50, WINDOW_HEIGHT - 80, 100, 40)
        self.save_button_rect = pygame.Rect(WINDOW_WIDTH - 150, WINDOW_HEIGHT - 80, 100, 40)
        
        # Callbacks
        self.on_back = None
        self.on_settings_changed = None
        
        # Animation
        self.fade_alpha = 0
        self.target_alpha = 255
        
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from configuration file"""
        try:
            config_path = Path(CONFIG_FILE)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        # Return default settings
        return {
            'master_volume': DEFAULT_MASTER_VOLUME,
            'sound_effects': True,
            'background_music': True,
            'difficulty': 'Medium',
            'controls': 'Arrow Keys',
            'fullscreen': False,
            'show_grid': True,
            'particle_effects': True,
            'auto_save': True
        }
    
    def _save_settings(self):
        """Save current settings to configuration file"""
        try:
            config_path = Path(CONFIG_FILE)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
                
            if self.on_settings_changed:
                self.on_settings_changed(self.settings)
                
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def _create_setting_items(self) -> List[SettingItem]:
        """Create setting items from configuration"""
        items = []
        
        # Audio Settings
        items.append(SettingItem(
            'master_volume', 'Master Volume', SettingType.SLIDER,
            self.settings.get('master_volume', 0.7), 0.0, 1.0,
            callback=self._on_setting_changed
        ))
        
        items.append(SettingItem(
            'sound_effects', 'Sound Effects', SettingType.TOGGLE,
            self.settings.get('sound_effects', True),
            callback=self._on_setting_changed
        ))
        
        items.append(SettingItem(
            'background_music', 'Background Music', SettingType.TOGGLE,
            self.settings.get('background_music', True),
            callback=self._on_setting_changed
        ))
        
        # Game Settings
        items.append(SettingItem(
            'difficulty', 'Difficulty', SettingType.DROPDOWN,
            self.settings.get('difficulty', 'Medium'),
            options=['Easy', 'Medium', 'Hard', 'Expert'],
            callback=self._on_setting_changed
        ))
        
        items.append(SettingItem(
            'controls', 'Control Scheme', SettingType.DROPDOWN,
            self.settings.get('controls', 'Arrow Keys'),
            options=['Arrow Keys', 'WASD', 'Custom'],
            callback=self._on_setting_changed
        ))
        
        # Display Settings
        items.append(SettingItem(
            'fullscreen', 'Fullscreen Mode', SettingType.TOGGLE,
            self.settings.get('fullscreen', False),
            callback=self._on_setting_changed
        ))
        
        items.append(SettingItem(
            'show_grid', 'Show Grid', SettingType.TOGGLE,
            self.settings.get('show_grid', True),
            callback=self._on_setting_changed
        ))
        
        items.append(SettingItem(
            'particle_effects', 'Particle Effects', SettingType.TOGGLE,
            self.settings.get('particle_effects', True),
            callback=self._on_setting_changed
        ))
        
        # System Settings
        items.append(SettingItem(
            'auto_save', 'Auto Save', SettingType.TOGGLE,
            self.settings.get('auto_save', True),
            callback=self._on_setting_changed
        ))
        
        # Action Buttons
        items.append(SettingItem(
            'reset_scores', 'Reset High Scores', SettingType.BUTTON,
            'Reset', callback=self._on_reset_scores
        ))
        
        items.append(SettingItem(
            'reset_settings', 'Reset to Defaults', SettingType.BUTTON,
            'Reset', callback=self._on_reset_settings
        ))
        
        return items
    
    def _on_setting_changed(self, key: str, value: Any):
        """Handle setting value changes"""
        self.settings[key] = value
        
        # Apply immediate effects for certain settings
        if key == 'fullscreen':
            self._apply_fullscreen_setting(value)
        elif key in ['master_volume', 'sound_effects', 'background_music']:
            self._apply_audio_settings()
    
    def _on_reset_scores(self, key: str, value: Any):
        """Reset high scores"""
        try:
            scores_path = Path(SCORES_FILE)
            if scores_path.exists():
                scores_path.unlink()
            print("High scores reset successfully")
        except Exception as e:
            print(f"Error resetting scores: {e}")
    
    def _on_reset_settings(self, key: str, value: Any):
        """Reset all settings to defaults"""
        self.settings = {
            'master_volume': DEFAULT_MASTER_VOLUME,
            'sound_effects': True,
            'background_music': True,
            'difficulty': 'Medium',
            'controls': 'Arrow Keys',
            'fullscreen': False,
            'show_grid': True,
            'particle_effects': True,
            'auto_save': True
        }
        
        # Update setting items
        for item in self.setting_items:
            if item.key in self.settings:
                item.value = self.settings[item.key]
        
        print("Settings reset to defaults")
    
    def _apply_fullscreen_setting(self, fullscreen: bool):
        """Apply fullscreen display setting"""
        # This would be handled by the main application
        pass
    
    def _apply_audio_settings(self):
        """Apply audio settings changes"""
        # This would be handled by the audio manager
        pass
    
    def set_callbacks(self, on_back: Callable = None, on_settings_changed: Callable = None):
        """Set callback functions for menu actions"""
        self.on_back = on_back
        self.on_settings_changed = on_settings_changed
    
    def update(self, dt: float, events: List[pygame.event.Event]):
        """Update settings menu state"""
        # Update fade animation
        if self.fade_alpha < self.target_alpha:
            self.fade_alpha = min(self.target_alpha, self.fade_alpha + dt * 500)
        
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        
        # Handle scrolling
        for event in events:
            if event.type == pygame.MOUSEWHEEL:
                self.scroll_offset -= event.y * 30
                self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset))
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check back button
                if self.back_button_rect.collidepoint(event.pos):
                    if self.on_back:
                        self.on_back()
                
                # Check save button
                elif self.save_button_rect.collidepoint(event.pos):
                    self._save_settings()
        
        # Update setting items
        for i, item in enumerate(self.setting_items):
            y_pos = 150 + i * 70 - self.scroll_offset
            if -50 <= y_pos <= WINDOW_HEIGHT + 50:  # Only update visible items
                item.update(mouse_pos, mouse_pressed, events)
        
        # Calculate max scroll
        total_height = len(self.setting_items) * 70
        visible_height = WINDOW_HEIGHT - 250
        self.max_scroll = max(0, total_height - visible_height)
    
    def draw(self):
        """Draw the settings menu"""
        # Draw background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw animated background pattern
        self._draw_background_pattern()
        
        # Draw title
        title_surface = self.font_large.render("SETTINGS", True, WHITE)
        title_rect = title_surface.get_rect(center=(WINDOW_WIDTH // 2, 80))
        self.screen.blit(title_surface, title_rect)
        
        # Draw settings items
        for i, item in enumerate(self.setting_items):
            y_pos = 150 + i * 70 - self.scroll_offset
            if -50 <= y_pos <= WINDOW_HEIGHT + 50:  # Only draw visible items
                item.draw(self.screen, self.font_medium, y_pos)
        
        # Draw scroll indicator
        if self.max_scroll > 0:
            self._draw_scroll_indicator()
        
        # Draw buttons
        self._draw_buttons()
        
        # Apply fade effect
        if self.fade_alpha < 255:
            fade_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            fade_surface.fill(BACKGROUND_COLOR)
            fade_surface.set_alpha(255 - self.fade_alpha)
            self.screen.blit(fade_surface, (0, 0))
    
    def _draw_background_pattern(self):
        """Draw animated background pattern"""
        time_offset = pygame.time.get_ticks() * 0.001
        
        for x in range(0, WINDOW_WIDTH, 40):
            for y in range(0, WINDOW_HEIGHT, 40):
                alpha = int(20 + 10 * math.sin(time_offset + x * 0.01 + y * 0.01))
                color = (*SNAKE_BODY_COLOR, alpha)
                
                # Create surface with per-pixel alpha
                dot_surface = pygame.Surface((4, 4), pygame.SRCALPHA)
                dot_surface.fill(color)
                self.screen.blit(dot_surface, (x, y))
    
    def _draw_scroll_indicator(self):
        """Draw scroll indicator on the right side"""
        indicator_height = 200
        indicator_y = 150
        
        # Draw scroll track
        track_rect = pygame.Rect(WINDOW_WIDTH - 20, indicator_y, 10, indicator_height)
        pygame.draw.rect(self.screen, GRAY, track_rect, border_radius=5)
        
        # Draw scroll thumb
        thumb_ratio = min(1.0, indicator_height / (len(self.setting_items) * 70))
        thumb_height = max(20, int(indicator_height * thumb_ratio))
        
        scroll_ratio = self.scroll_offset / self.max_scroll if self.max_scroll > 0 else 0
        thumb_y = indicator_y + int((indicator_height - thumb_height) * scroll_ratio)
        
        thumb_rect = pygame.Rect(WINDOW_WIDTH - 20, thumb_y, 10, thumb_height)
        pygame.draw.rect(self.screen, WHITE, thumb_rect, border_radius=5)
    
    def _draw_buttons(self):
        """Draw back and save buttons"""
        # Draw back button
        back_color = MENU_BUTTON_HOVER_COLOR if self.back_button_rect.collidepoint(pygame.mouse.get_pos()) else MENU_BUTTON_COLOR
        pygame.draw.rect(self.screen, back_color, self.back_button_rect, border_radius=5)
        pygame.draw.rect(self.screen, WHITE, self.back_button_rect, 2, border_radius=5)
        
        back_text = self.font_medium.render("BACK", True, WHITE)
        back_text_rect = back_text.get_rect(center=self.back_button_rect.center)
        self.screen.blit(back_text, back_text_rect)
        
        # Draw save button
        save_color = MENU_BUTTON_HOVER_COLOR if self.save_button_rect.collidepoint(pygame.mouse.get_pos()) else MENU_BUTTON_COLOR
        pygame.draw.rect(self.screen, save_color, self.save_button_rect, border_radius=5)
        pygame.draw.rect(self.screen, WHITE, self.save_button_rect, 2, border_radius=5)
        
        save_text = self.font_medium.render("SAVE", True, WHITE)
        save_text_rect = save_text.get_rect(center=self.save_button_rect.center)
        self.screen.blit(save_text, save_text_rect)


def create_settings_menu(screen: pygame.Surface) -> SettingsMenu:
    """Factory function to create a settings menu instance"""
    return SettingsMenu(screen)


def get_default_settings() -> Dict[str, Any]:
    """Get default settings configuration"""
    return {
        'master_volume': DEFAULT_MASTER_VOLUME,
        'sound_effects': True,
        'background_music': True,
        'difficulty': 'Medium',
        'controls': 'Arrow Keys',
        'fullscreen': False,
        'show_grid': True,
        'particle_effects': True,
        'auto_save': True
    }