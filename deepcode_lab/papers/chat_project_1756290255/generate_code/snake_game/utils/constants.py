"""
Game constants and configuration settings for the Snake Game.

This module contains all the constant values used throughout the game,
including display settings, colors, game mechanics, and file paths.
"""

import pygame
from pathlib import Path

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Game grid settings
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Frame rate
FPS = 60
GAME_SPEED_BASE = 10  # Base game speed (lower = faster)

# ============================================================================
# COLORS (RGB values)
# ============================================================================

# Basic colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

# Game-specific colors
BACKGROUND_COLOR = (20, 25, 40)  # Dark blue-gray
GRID_COLOR = (40, 45, 60)        # Slightly lighter for grid lines

# Snake colors (gradient effect)
SNAKE_HEAD_COLOR = (50, 205, 50)     # Lime green
SNAKE_BODY_COLOR = (34, 139, 34)     # Forest green
SNAKE_BODY_GRADIENT_COLOR = (0, 128, 0)  # Medium green for gradient
SNAKE_TAIL_COLOR = (0, 100, 0)       # Dark green

# Food colors
FOOD_NORMAL_COLOR = (255, 69, 0)     # Red-orange
FOOD_SPECIAL_COLOR = (255, 215, 0)   # Gold
FOOD_BONUS_COLOR = (255, 20, 147)    # Deep pink

# UI colors
UI_BACKGROUND = (30, 35, 50)         # Dark background
UI_BORDER = (70, 75, 90)             # Border color
UI_TEXT_PRIMARY = (255, 255, 255)    # White text
UI_TEXT_SECONDARY = (180, 180, 180)  # Gray text
UI_BUTTON_NORMAL = (60, 70, 90)      # Button background
UI_BUTTON_HOVER = (80, 90, 110)      # Button hover
UI_BUTTON_ACTIVE = (100, 110, 130)   # Button active

# Status colors
COLOR_SUCCESS = (46, 204, 113)       # Green
COLOR_WARNING = (241, 196, 15)       # Yellow
COLOR_ERROR = (231, 76, 60)          # Red
COLOR_INFO = (52, 152, 219)          # Blue

# ============================================================================
# GAME MECHANICS
# ============================================================================

# Movement directions
DIRECTION_UP = (0, -1)
DIRECTION_DOWN = (0, 1)
DIRECTION_LEFT = (-1, 0)
DIRECTION_RIGHT = (1, 0)

# Direction mappings
DIRECTIONS = {
    'UP': DIRECTION_UP,
    'DOWN': DIRECTION_DOWN,
    'LEFT': DIRECTION_LEFT,
    'RIGHT': DIRECTION_RIGHT
}

# Opposite directions (for preventing reverse movement)
OPPOSITE_DIRECTIONS = {
    DIRECTION_UP: DIRECTION_DOWN,
    DIRECTION_DOWN: DIRECTION_UP,
    DIRECTION_LEFT: DIRECTION_RIGHT,
    DIRECTION_RIGHT: DIRECTION_LEFT
}

# Snake initial settings
INITIAL_SNAKE_LENGTH = 3
INITIAL_SNAKE_SPEED = 10

# Key mappings for controls
KEY_MAPPINGS = {
    # Arrow keys
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    
    # WASD keys
    pygame.K_w: 'UP',
    pygame.K_s: 'DOWN',
    pygame.K_a: 'LEFT',
    pygame.K_d: 'RIGHT',
    
    # Game controls
    pygame.K_SPACE: 'PAUSE',
    pygame.K_ESCAPE: 'MENU',
    pygame.K_r: 'RESTART',
    pygame.K_q: 'QUIT'
}

# ============================================================================
# SCORING SYSTEM
# ============================================================================

# Score values
SCORE_NORMAL_FOOD = 10
SCORE_SPECIAL_FOOD = 25
SCORE_BONUS_FOOD = 50

# Score multipliers
SCORE_MULTIPLIER_BASE = 1.0
SCORE_MULTIPLIER_INCREMENT = 0.1

# ============================================================================
# DIFFICULTY LEVELS
# ============================================================================

DIFFICULTY_LEVELS = {
    'EASY': {
        'speed': 15,
        'speed_increment': 0.5,
        'max_speed': 8,
        'special_food_chance': 0.1,
        'bonus_food_chance': 0.05
    },
    'MEDIUM': {
        'speed': 10,
        'speed_increment': 0.8,
        'max_speed': 5,
        'special_food_chance': 0.15,
        'bonus_food_chance': 0.08
    },
    'HARD': {
        'speed': 6,
        'speed_increment': 1.0,
        'max_speed': 3,
        'special_food_chance': 0.2,
        'bonus_food_chance': 0.1
    },
    'EXPERT': {
        'speed': 4,
        'speed_increment': 1.2,
        'max_speed': 2,
        'special_food_chance': 0.25,
        'bonus_food_chance': 0.12
    }
}

DEFAULT_DIFFICULTY = 'MEDIUM'

# ============================================================================
# FOOD TYPES
# ============================================================================

FOOD_TYPES = {
    'NORMAL': {
        'color': FOOD_NORMAL_COLOR,
        'score': SCORE_NORMAL_FOOD,
        'probability': 0.7,
        'duration': None  # Permanent until eaten
    },
    'SPECIAL': {
        'color': FOOD_SPECIAL_COLOR,
        'score': SCORE_SPECIAL_FOOD,
        'probability': 0.2,
        'duration': 300  # 5 seconds at 60 FPS
    },
    'BONUS': {
        'color': FOOD_BONUS_COLOR,
        'score': SCORE_BONUS_FOOD,
        'probability': 0.1,
        'duration': 180  # 3 seconds at 60 FPS
    }
}

# ============================================================================
# AUDIO SETTINGS
# ============================================================================

# Audio file formats
AUDIO_FORMATS = ['.wav', '.ogg', '.mp3']

# Volume settings (0.0 to 1.0)
DEFAULT_MASTER_VOLUME = 0.7
DEFAULT_MUSIC_VOLUME = 0.5
DEFAULT_SFX_VOLUME = 0.8

# Audio channels
AUDIO_CHANNELS = 8

# ============================================================================
# FILE PATHS
# ============================================================================

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent

# Asset directories
ASSETS_DIR = PROJECT_ROOT / 'assets'
SOUNDS_DIR = ASSETS_DIR / 'sounds'
IMAGES_DIR = ASSETS_DIR / 'images'
FONTS_DIR = ASSETS_DIR / 'fonts'

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
CONFIG_FILE = DATA_DIR / 'config.json'
SCORES_FILE = DATA_DIR / 'scores.json'

# Log file
LOG_FILE = DATA_DIR / 'game.log'

# ============================================================================
# UI SETTINGS
# ============================================================================

# Font sizes
FONT_SIZE_LARGE = 48
FONT_SIZE_MEDIUM = 32
FONT_SIZE_SMALL = 24
FONT_SIZE_TINY = 16

# UI element dimensions
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20

# Menu settings
MENU_TITLE_Y = 100
MENU_BUTTON_START_Y = 250
MENU_BUTTON_SPACING = 80

# HUD settings
HUD_MARGIN = 20
HUD_HEIGHT = 60

# ============================================================================
# ANIMATION SETTINGS
# ============================================================================

# Animation durations (in frames at 60 FPS)
FADE_DURATION = 30
SLIDE_DURATION = 20
PULSE_DURATION = 60

# Animation easing
EASE_IN_OUT_CUBIC = lambda t: 4*t*t*t if t < 0.5 else 1-pow(-2*t+2, 3)/2

# ============================================================================
# GAME STATES
# ============================================================================

GAME_STATES = {
    'MENU': 'menu',
    'PLAYING': 'playing',
    'PAUSED': 'paused',
    'GAME_OVER': 'game_over',
    'SETTINGS': 'settings',
    'HIGH_SCORES': 'high_scores'
}

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'PYGAME_INIT': "Failed to initialize Pygame",
    'AUDIO_INIT': "Failed to initialize audio system",
    'FILE_NOT_FOUND': "Required file not found",
    'INVALID_CONFIG': "Invalid configuration file",
    'SAVE_FAILED': "Failed to save game data"
}

# ============================================================================
# DEBUG SETTINGS
# ============================================================================

DEBUG_MODE = False
SHOW_FPS = False
SHOW_GRID = False
SHOW_COLLISION_BOXES = False

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Maximum number of particles for effects
MAX_PARTICLES = 100

# Maximum number of high scores to keep
MAX_HIGH_SCORES = 10

# Auto-save interval (in seconds)
AUTO_SAVE_INTERVAL = 30

# ============================================================================
# VERSION INFO
# ============================================================================

GAME_VERSION = "1.0.0"
GAME_TITLE = "贪吃蛇游戏 (Snake Game)"
GAME_AUTHOR = "Snake Game Developer"