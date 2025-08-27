"""
Utility helper functions for the Snake Game.
Provides common functionality used across the application.
"""

import os
import sys
import logging
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import pygame

from .constants import (
    LOG_FILE, DEBUG_MODE, WINDOW_WIDTH, WINDOW_HEIGHT,
    DEFAULT_MASTER_VOLUME, GAME_TITLE
)


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (default: logging.INFO)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('SnakeGame')
    logger.info(f"Logging initialized - Level: {logging.getLevelName(log_level)}")
    
    return logger


def handle_error(error: Exception, context: str = "Unknown") -> None:
    """
    Handle and log errors with context information.
    
    Args:
        error: The exception that occurred
        context: Context where the error occurred
    """
    logger = logging.getLogger('SnakeGame')
    error_msg = f"Error in {context}: {str(error)}"
    logger.error(error_msg)
    
    if DEBUG_MODE:
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Print to console for immediate feedback
    print(f"ERROR: {error_msg}")


def clamp(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Union[int, float]:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def lerp(start: float, end: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Args:
        start: Starting value
        end: Ending value
        t: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated value
    """
    t = clamp(t, 0.0, 1.0)
    return start + (end - start) * t


def distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)
        
    Returns:
        Distance between points
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return (dx * dx + dy * dy) ** 0.5


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to range [0, 360).
    
    Args:
        angle: Angle in degrees
        
    Returns:
        Normalized angle
    """
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360
    return angle


def format_time(seconds: float) -> str:
    """
    Format time in seconds to MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def format_score(score: int) -> str:
    """
    Format score with thousands separators.
    
    Args:
        score: Score value
        
    Returns:
        Formatted score string
    """
    return f"{score:,}"


def get_center_position(container_width: int, container_height: int, 
                       object_width: int, object_height: int) -> Tuple[int, int]:
    """
    Calculate the center position for an object within a container.
    
    Args:
        container_width: Width of the container
        container_height: Height of the container
        object_width: Width of the object to center
        object_height: Height of the object to center
        
    Returns:
        Tuple of (x, y) coordinates for centered position
    """
    x = (container_width - object_width) // 2
    y = (container_height - object_height) // 2
    return (x, y)


def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON data from file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if loading fails
        
    Returns:
        Loaded data or default value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load JSON from {file_path}: {e}")
        return default


def safe_json_save(data: Any, file_path: Union[str, Path]) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except (IOError, TypeError) as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def get_resource_path(relative_path: str) -> Path:
    """
    Get absolute path to resource, works for dev and PyInstaller bundle.
    
    Args:
        relative_path: Relative path to resource
        
    Returns:
        Absolute path to resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = Path(__file__).parent.parent
    
    return Path(base_path) / relative_path


def create_gradient_surface(width: int, height: int, 
                          start_color: Tuple[int, int, int], 
                          end_color: Tuple[int, int, int],
                          vertical: bool = True) -> pygame.Surface:
    """
    Create a gradient surface.
    
    Args:
        width: Surface width
        height: Surface height
        start_color: Starting color (R, G, B)
        end_color: Ending color (R, G, B)
        vertical: True for vertical gradient, False for horizontal
        
    Returns:
        Gradient surface
    """
    surface = pygame.Surface((width, height))
    
    if vertical:
        for y in range(height):
            t = y / height if height > 0 else 0
            color = [
                int(lerp(start_color[i], end_color[i], t))
                for i in range(3)
            ]
            pygame.draw.line(surface, color, (0, y), (width, y))
    else:
        for x in range(width):
            t = x / width if width > 0 else 0
            color = [
                int(lerp(start_color[i], end_color[i], t))
                for i in range(3)
            ]
            pygame.draw.line(surface, color, (x, 0), (x, height))
    
    return surface


def scale_surface(surface: pygame.Surface, scale_factor: float) -> pygame.Surface:
    """
    Scale a surface by a given factor.
    
    Args:
        surface: Surface to scale
        scale_factor: Scale factor
        
    Returns:
        Scaled surface
    """
    if scale_factor == 1.0:
        return surface
    
    new_width = int(surface.get_width() * scale_factor)
    new_height = int(surface.get_height() * scale_factor)
    
    return pygame.transform.scale(surface, (new_width, new_height))


def center_text_rect(text_rect: pygame.Rect, container_rect: pygame.Rect) -> pygame.Rect:
    """
    Center a text rectangle within a container rectangle.
    
    Args:
        text_rect: Text rectangle to center
        container_rect: Container rectangle
        
    Returns:
        Centered rectangle
    """
    centered_rect = text_rect.copy()
    centered_rect.centerx = container_rect.centerx
    centered_rect.centery = container_rect.centery
    return centered_rect


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list:
    """
    Wrap text to fit within a maximum width.
    
    Args:
        text: Text to wrap
        font: Font to use for measuring
        max_width: Maximum width in pixels
        
    Returns:
        List of wrapped text lines
    """
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        if font.size(test_line)[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # Word is too long, force it on its own line
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def get_system_info() -> Dict[str, str]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    return {
        'platform': sys.platform,
        'python_version': sys.version,
        'pygame_version': pygame.version.ver,
        'sdl_version': pygame.version.SDL,
        'working_directory': str(Path.cwd()),
        'executable': sys.executable
    }


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for logging unhandled exceptions.
    
    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow KeyboardInterrupt to exit normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = logging.getLogger('SnakeGame')
    logger.critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
    # Print to stderr as well
    traceback.print_exception(exc_type, exc_value, exc_traceback)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize configuration data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    validated = {}
    
    # Validate volume settings
    validated['master_volume'] = clamp(
        config.get('master_volume', DEFAULT_MASTER_VOLUME), 0.0, 1.0
    )
    validated['sfx_volume'] = clamp(
        config.get('sfx_volume', DEFAULT_MASTER_VOLUME), 0.0, 1.0
    )
    validated['music_volume'] = clamp(
        config.get('music_volume', DEFAULT_MASTER_VOLUME), 0.0, 1.0
    )
    
    # Validate difficulty
    valid_difficulties = ['easy', 'medium', 'hard', 'expert']
    validated['difficulty'] = config.get('difficulty', 'medium')
    if validated['difficulty'] not in valid_difficulties:
        validated['difficulty'] = 'medium'
    
    # Validate controls
    validated['controls'] = config.get('controls', 'arrows')
    if validated['controls'] not in ['arrows', 'wasd']:
        validated['controls'] = 'arrows'
    
    # Validate display settings
    validated['fullscreen'] = bool(config.get('fullscreen', False))
    validated['vsync'] = bool(config.get('vsync', True))
    
    return validated


def create_error_surface(width: int, height: int, message: str) -> pygame.Surface:
    """
    Create an error display surface.
    
    Args:
        width: Surface width
        height: Surface height
        message: Error message
        
    Returns:
        Error surface
    """
    surface = pygame.Surface((width, height))
    surface.fill((64, 0, 0))  # Dark red background
    
    try:
        font = pygame.font.Font(None, 36)
        text_surface = font.render(message, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(width // 2, height // 2))
        surface.blit(text_surface, text_rect)
    except:
        # Fallback if font loading fails
        pygame.draw.rect(surface, (255, 0, 0), (10, 10, width-20, height-20), 2)
    
    return surface


# Performance monitoring utilities
class PerformanceMonitor:
    """Simple performance monitoring utility."""
    
    def __init__(self):
        self.frame_times = []
        self.max_samples = 60
        
    def add_frame_time(self, frame_time: float):
        """Add a frame time sample."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
    
    def get_average_fps(self) -> float:
        """Get average FPS over recent frames."""
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_min_fps(self) -> float:
        """Get minimum FPS over recent frames."""
        if not self.frame_times:
            return 0.0
        
        max_frame_time = max(self.frame_times)
        return 1.0 / max_frame_time if max_frame_time > 0 else 0.0