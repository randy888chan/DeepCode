# 🐍 贪吃蛇游戏 (Snake Game)

A modern, feature-rich Snake game built with Python and Pygame, featuring beautiful UI, sound effects, multiple difficulty levels, and cross-platform compatibility.

![Snake Game](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.5.2-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

## 🎮 Features

### Core Gameplay
- **Multi-directional Control**: Arrow keys + WASD support
- **Progressive Difficulty**: Speed increases as you progress
- **Smart Collision Detection**: Efficient wall and self-collision algorithms
- **Food System**: Multiple food types with varying scores and visual effects
- **Scoring System**: Real-time score tracking with bonus multipliers

### Visual & Audio
- **Modern UI Design**: Clean, flat design with smooth animations
- **Visual Effects**: Gradient snake body, animated food, particle effects
- **Complete Audio System**: Background music and contextual sound effects
- **Responsive Design**: Automatic scaling for different screen resolutions
- **60 FPS Performance**: Optimized rendering pipeline

### Game Features
- **Multiple Difficulty Levels**: Easy, Medium, Hard, Expert
- **Pause/Resume**: In-game menu with pause functionality
- **High Score Tracking**: Persistent leaderboard with statistics
- **Customizable Settings**: Controls, audio, difficulty configuration
- **Game Statistics**: Track playtime, games played, best scores

### Technical Features
- **Cross-Platform**: Windows, macOS, and Linux support
- **Data Persistence**: JSON-based configuration and score storage
- **Error Recovery**: Robust file handling with backup mechanisms
- **Modular Architecture**: Clean, extensible codebase
- **Performance Optimized**: Memory management and efficient algorithms

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or Download the Project**
   ```bash
   # If using git
   git clone <repository-url>
   cd snake_game
   
   # Or download and extract the ZIP file
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   # Create virtual environment
   python -m venv snake_env
   
   # Activate virtual environment
   # On Windows:
   snake_env\Scripts\activate
   
   # On macOS/Linux:
   source snake_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Game**
   ```bash
   python main.py
   ```

### Alternative Installation (System-wide)
```bash
# Install dependencies system-wide
pip install pygame==2.5.2 numpy==1.24.3 json5==0.9.11

# Run the game
python main.py
```

## 🎯 How to Play

### Controls
- **Movement**: Arrow Keys or WASD
  - ↑/W: Move Up
  - ↓/S: Move Down
  - ←/A: Move Left
  - →/D: Move Right
- **Game Controls**:
  - `SPACE`: Pause/Resume
  - `ESC`: Return to Menu
  - `ENTER`: Select Menu Option

### Gameplay
1. **Objective**: Guide the snake to eat food and grow longer
2. **Scoring**: Different food types give different points
3. **Avoid**: Hitting walls or the snake's own body
4. **Progression**: Speed increases as your score grows
5. **Goal**: Achieve the highest score possible!

### Food Types
- **Normal Food** (Green): +10 points
- **Bonus Food** (Gold): +25 points (rare)
- **Speed Food** (Blue): +15 points + temporary speed boost

## ⚙️ Configuration

### Game Settings
The game stores settings in `data/config.json`. You can modify:

```json
{
  "difficulty": "medium",
  "master_volume": 0.7,
  "sfx_volume": 0.8,
  "music_volume": 0.6,
  "controls": {
    "up": ["UP", "w"],
    "down": ["DOWN", "s"],
    "left": ["LEFT", "a"],
    "right": ["RIGHT", "d"]
  },
  "display": {
    "fullscreen": false,
    "vsync": true
  }
}
```

### Difficulty Levels
- **Easy**: Slow speed, forgiving gameplay
- **Medium**: Balanced speed and challenge
- **Hard**: Fast-paced, requires quick reflexes
- **Expert**: Maximum speed, ultimate challenge

## 📁 Project Structure

```
snake_game/
├── main.py                 # Main entry point
├── game/                   # Core game logic
│   ├── __init__.py
│   ├── snake.py           # Snake class and movement
│   ├── food.py            # Food generation and types
│   ├── game_engine.py     # Game loop and state management
│   └── collision.py       # Collision detection
├── ui/                     # User interface
│   ├── __init__.py
│   ├── menu.py            # Main menu interface
│   ├── game_screen.py     # Game display and HUD
│   ├── settings.py        # Settings configuration
│   └── effects.py         # Visual effects
├── audio/                  # Audio system
│   ├── __init__.py
│   └── sound_manager.py   # Audio management
├── utils/                  # Utilities and helpers
│   ├── __init__.py
│   ├── constants.py       # Game constants
│   ├── helpers.py         # Utility functions
│   └── data_manager.py    # Data persistence
├── assets/                 # Game assets
│   ├── sounds/            # Audio files
│   ├── images/            # Sprite images
│   └── fonts/             # Custom fonts
├── data/                   # Game data
│   ├── config.json        # Configuration
│   └── scores.json        # High scores
├── requirements.txt        # Dependencies
├── build_config.py        # Build configuration
└── README.md              # This file
```

## 🔧 Development

### Dependencies
- **pygame==2.5.2**: Game development framework
- **numpy==1.24.3**: Numerical computations
- **json5==0.9.11**: Enhanced JSON parsing
- **PyInstaller>=5.0**: Executable building (optional)
- **pillow>=9.0**: Image processing (optional)

### Key Classes
- **SnakeGameApp**: Main application controller
- **Snake**: Snake entity with movement and growth
- **Food**: Food generation and management
- **GameEngine**: Core game loop and state management
- **SoundManager**: Audio system management
- **DataManager**: Configuration and score persistence

### Adding Features
The modular architecture makes it easy to extend:

1. **New Food Types**: Extend the `Food` class in `game/food.py`
2. **Visual Effects**: Add to `ui/effects.py`
3. **Game Modes**: Modify `game/game_engine.py`
4. **Audio**: Extend `audio/sound_manager.py`

## 🏗️ Building Executables

### Using PyInstaller
```bash
# Install PyInstaller
pip install PyInstaller>=5.0

# Build executable
python build_config.py

# Or manually:
pyinstaller --onefile --windowed --name "SnakeGame" main.py
```

### Cross-Platform Notes
- **Windows**: Creates `.exe` executable
- **macOS**: Creates `.app` bundle
- **Linux**: Creates binary executable

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Audio Issues**
   ```bash
   # Check audio system availability
   python -c "import pygame; pygame.mixer.init()"
   ```

3. **Performance Issues**
   - Lower the FPS in `utils/constants.py`
   - Disable visual effects in settings
   - Close other applications

4. **File Permission Errors**
   - Ensure write permissions for `data/` directory
   - Run with appropriate user permissions

### Debug Mode
Enable debug mode by setting `DEBUG = True` in `utils/constants.py`:
```python
DEBUG = True
DEBUG_SHOW_FPS = True
DEBUG_SHOW_COLLISION = True
```

## 📊 Performance

### System Requirements
- **Minimum**: Python 3.8, 512MB RAM, integrated graphics
- **Recommended**: Python 3.9+, 1GB RAM, dedicated graphics
- **Target Performance**: 60 FPS at 1920x1080

### Optimization Features
- Efficient collision detection algorithms
- Optimized rendering pipeline
- Memory management and cleanup
- Configurable performance settings

## 🤝 Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document functions and classes
- Maintain modular architecture

### Testing
```bash
# Run basic functionality test
python main.py

# Test individual modules
python -m game.snake
python -m audio.sound_manager
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Pygame Community**: For the excellent game development framework
- **Python Software Foundation**: For the Python programming language
- **Contributors**: Thanks to all who have contributed to this project

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the project structure and documentation
3. Ensure all dependencies are properly installed
4. Verify Python version compatibility (3.8+)

## 🎉 Enjoy the Game!

Have fun playing the Snake Game! Try to beat your high score and challenge your friends. The game saves your progress automatically, so you can always come back to improve your skills.

**Happy Gaming! 🐍🎮**

---

*Built with ❤️ using Python and Pygame*