"""
游戏常量配置文件
定义贪吃蛇游戏的所有常量参数
"""

# 游戏窗口设置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "贪吃蛇桌面游戏"

# 游戏场地设置
GRID_SIZE = 20  # 每个格子的像素大小
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE  # 40格
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE  # 30格

# 游戏帧率
FPS = 60

# 蛇的移动速度（毫秒）
SNAKE_SPEED = {
    'easy': 200,    # 简单模式：每200ms移动一次
    'medium': 150,  # 中等模式：每150ms移动一次
    'hard': 100     # 困难模式：每100ms移动一次
}

# 颜色定义（RGB格式）
COLORS = {
    # 背景和界面
    'background': (20, 20, 20),      # 深灰色背景
    'grid_line': (40, 40, 40),       # 网格线颜色
    'ui_text': (255, 255, 255),      # 白色文字
    'ui_highlight': (100, 255, 100), # 高亮绿色
    
    # 蛇的颜色
    'snake_head': (0, 255, 0),       # 蛇头：亮绿色
    'snake_body': (0, 200, 0),       # 蛇身：深绿色
    'snake_tail': (0, 150, 0),       # 蛇尾：更深绿色
    
    # 食物颜色
    'food': (255, 0, 0),             # 食物：红色
    'special_food': (255, 255, 0),   # 特殊食物：黄色
    
    # 菜单和UI
    'menu_bg': (30, 30, 30),         # 菜单背景
    'button_normal': (60, 60, 60),   # 按钮正常状态
    'button_hover': (80, 80, 80),    # 按钮悬停状态
    'button_active': (100, 100, 100) # 按钮激活状态
}

# 方向常量
DIRECTIONS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0)
}

# 游戏状态
GAME_STATES = {
    'MENU': 'menu',
    'PLAYING': 'playing',
    'PAUSED': 'paused',
    'GAME_OVER': 'game_over',
    'SETTINGS': 'settings'
}

# 按键映射
KEY_BINDINGS = {
    'move_up': ['w', 'up'],
    'move_down': ['s', 'down'],
    'move_left': ['a', 'left'],
    'move_right': ['d', 'right'],
    'pause': ['space', 'p'],
    'restart': ['r'],
    'quit': ['escape', 'q']
}

# 分数系统
SCORING = {
    'normal_food': 10,      # 普通食物得分
    'special_food': 50,     # 特殊食物得分
    'speed_bonus': {        # 速度奖励倍数
        'easy': 1.0,
        'medium': 1.5,
        'hard': 2.0
    }
}

# 音效文件路径
SOUND_FILES = {
    'eat': 'assets/sounds/eat.wav',
    'game_over': 'assets/sounds/game_over.wav',
    'menu_select': 'assets/sounds/menu_select.wav',
    'background': 'assets/sounds/background.wav'
}

# 图片资源路径
IMAGE_FILES = {
    'snake_head': 'assets/images/snake_head.png',
    'snake_body': 'assets/images/snake_body.png',
    'food': 'assets/images/food.png',
    'background': 'assets/images/background.png'
}

# 数据文件路径
DATA_FILES = {
    'high_scores': 'data/high_scores.json',
    'settings': 'config/settings.json',
    'game_stats': 'data/game_stats.json'
}

# 默认游戏设置
DEFAULT_SETTINGS = {
    'difficulty': 'medium',
    'sound_enabled': True,
    'music_enabled': True,
    'fullscreen': False,
    'show_grid': True,
    'player_name': 'Player'
}

# 蛇的初始设置
SNAKE_INITIAL = {
    'position': (GRID_WIDTH // 2, GRID_HEIGHT // 2),  # 屏幕中央
    'length': 3,                                       # 初始长度
    'direction': DIRECTIONS['RIGHT']                   # 初始方向
}

# 食物生成设置
FOOD_SETTINGS = {
    'special_food_chance': 0.1,  # 特殊食物出现概率（10%）
    'min_distance_from_snake': 2  # 食物与蛇的最小距离
}

# 性能设置
PERFORMANCE = {
    'max_particles': 50,         # 最大粒子数量
    'particle_lifetime': 1000,   # 粒子生命周期（毫秒）
    'render_distance': 100       # 渲染距离
}