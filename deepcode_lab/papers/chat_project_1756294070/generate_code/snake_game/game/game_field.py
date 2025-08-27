"""
游戏场地逻辑 (Game Field Logic)
管理游戏场地、协调各个游戏组件的交互
"""

import pygame
import random
from typing import Tuple, List, Dict, Optional, Any
from config.constants import *
from .snake import Snake
from .food import Food, FoodManager
from .collision import CollisionDetector, AdvancedCollisionDetector


class GameField:
    """
    游戏场地类 - 管理游戏的核心逻辑和状态
    """
    
    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
        """
        初始化游戏场地
        
        Args:
            width: 场地宽度（格子数）
            height: 场地高度（格子数）
        """
        self.width = width
        self.height = height
        self.grid_size = GRID_SIZE
        
        # 游戏组件
        self.snake = Snake()
        self.food_manager = FoodManager()
        self.collision_detector = CollisionDetector(width, height)
        self.advanced_collision = AdvancedCollisionDetector(width, height)
        
        # 游戏状态
        self.score = 0
        self.level = 1
        self.game_state = GAME_STATE_PLAYING
        self.paused = False
        self.game_over = False
        
        # 时间管理
        self.last_update_time = 0
        self.game_start_time = 0
        self.pause_start_time = 0
        self.total_pause_time = 0
        
        # 难度设置
        self.difficulty = DIFFICULTY_NORMAL
        self.speed_multiplier = 1.0
        
        # 统计数据
        self.stats = {
            'food_eaten': 0,
            'total_time': 0,
            'moves_made': 0,
            'collisions_avoided': 0,
            'special_food_eaten': 0
        }
        
        # 初始化游戏
        self._initialize_game()
    
    def _initialize_game(self):
        """初始化游戏状态"""
        # 重置蛇的位置
        initial_pos = (self.width // 2, self.height // 2)
        self.snake.reset(initial_pos)
        
        # 设置初始速度
        self._update_snake_speed()
        
        # 生成初始食物
        self.food_manager.spawn_food(self.snake.get_segments(), pygame.time.get_ticks())
        
        # 重置时间
        self.game_start_time = pygame.time.get_ticks()
        self.last_update_time = self.game_start_time
        self.total_pause_time = 0
        
        # 重置状态
        self.game_state = GAME_STATE_PLAYING
        self.paused = False
        self.game_over = False
    
    def _update_snake_speed(self):
        """根据难度和等级更新蛇的速度"""
        base_speed = SNAKE_SPEED_NORMAL
        
        if self.difficulty == DIFFICULTY_EASY:
            base_speed = SNAKE_SPEED_EASY
        elif self.difficulty == DIFFICULTY_HARD:
            base_speed = SNAKE_SPEED_HARD
        
        # 根据等级调整速度（每级增加5%的速度）
        level_multiplier = 1.0 + (self.level - 1) * 0.05
        final_speed = int(base_speed / (level_multiplier * self.speed_multiplier))
        
        # 确保速度不会太快
        final_speed = max(final_speed, 50)  # 最小50ms间隔
        
        self.snake.set_speed(final_speed)
    
    def set_difficulty(self, difficulty: str):
        """
        设置游戏难度
        
        Args:
            difficulty: 难度级别 (DIFFICULTY_EASY, DIFFICULTY_NORMAL, DIFFICULTY_HARD)
        """
        if difficulty in [DIFFICULTY_EASY, DIFFICULTY_NORMAL, DIFFICULTY_HARD]:
            self.difficulty = difficulty
            self._update_snake_speed()
    
    def update(self, current_time: int) -> bool:
        """
        更新游戏状态
        
        Args:
            current_time: 当前时间戳
            
        Returns:
            bool: 游戏是否继续运行
        """
        if self.paused or self.game_over:
            return True
        
        # 更新蛇的位置
        snake_moved = self.snake.update(current_time)
        
        if snake_moved:
            self.stats['moves_made'] += 1
            
            # 检查碰撞
            if self._check_collisions():
                self._handle_game_over()
                return False
            
            # 检查食物碰撞
            self._check_food_collision(current_time)
        
        # 更新食物状态
        self.food_manager.update(current_time, self.snake.get_segments())
        
        # 更新统计数据
        self._update_stats(current_time)
        
        # 检查等级提升
        self._check_level_up()
        
        self.last_update_time = current_time
        return True
    
    def _check_collisions(self) -> bool:
        """
        检查所有类型的碰撞
        
        Returns:
            bool: 是否发生碰撞
        """
        head_pos = self.snake.get_head_position()
        snake_segments = self.snake.get_segments()
        
        # 检查边界碰撞
        if self.collision_detector.check_boundary_collision(head_pos):
            return True
        
        # 检查自身碰撞
        if self.collision_detector.check_self_collision(head_pos, snake_segments[1:]):
            return True
        
        return False
    
    def _check_food_collision(self, current_time: int):
        """检查食物碰撞"""
        head_pos = self.snake.get_head_position()
        
        # 检查与所有食物的碰撞
        for food in self.food_manager.get_active_foods():
            if food.check_collision(head_pos):
                # 消费食物
                score_gained = food.consume()
                self.score += score_gained
                
                # 蛇增长
                growth_amount = 1
                if food.food_type == FOOD_TYPE_SPECIAL:
                    growth_amount = 2
                    self.stats['special_food_eaten'] += 1
                
                self.snake.grow(growth_amount)
                self.stats['food_eaten'] += 1
                
                # 移除被吃掉的食物
                self.food_manager.remove_food(food)
                
                # 生成新食物
                self.food_manager.spawn_food(self.snake.get_segments(), current_time)
                
                break
    
    def _update_stats(self, current_time: int):
        """更新统计数据"""
        if not self.paused:
            self.stats['total_time'] = current_time - self.game_start_time - self.total_pause_time
    
    def _check_level_up(self):
        """检查是否需要升级"""
        # 每吃10个食物升一级
        new_level = (self.stats['food_eaten'] // 10) + 1
        
        if new_level > self.level:
            self.level = new_level
            self._update_snake_speed()
    
    def _handle_game_over(self):
        """处理游戏结束"""
        self.game_over = True
        self.game_state = GAME_STATE_GAME_OVER
        self.snake.alive = False
    
    def handle_input(self, direction: str) -> bool:
        """
        处理输入方向
        
        Args:
            direction: 方向 (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            bool: 是否成功设置方向
        """
        if self.paused or self.game_over:
            return False
        
        return self.snake.set_direction(direction)
    
    def pause_game(self):
        """暂停游戏"""
        if not self.game_over and not self.paused:
            self.paused = True
            self.pause_start_time = pygame.time.get_ticks()
            self.game_state = GAME_STATE_PAUSED
    
    def resume_game(self):
        """恢复游戏"""
        if self.paused:
            self.paused = False
            self.total_pause_time += pygame.time.get_ticks() - self.pause_start_time
            self.game_state = GAME_STATE_PLAYING
    
    def restart_game(self):
        """重新开始游戏"""
        self.score = 0
        self.level = 1
        self.stats = {
            'food_eaten': 0,
            'total_time': 0,
            'moves_made': 0,
            'collisions_avoided': 0,
            'special_food_eaten': 0
        }
        self.food_manager.clear_all_foods()
        self._initialize_game()
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        获取当前游戏状态
        
        Returns:
            Dict: 包含所有游戏状态信息的字典
        """
        return {
            'score': self.score,
            'level': self.level,
            'difficulty': self.difficulty,
            'game_state': self.game_state,
            'paused': self.paused,
            'game_over': self.game_over,
            'snake_segments': self.snake.get_segments(),
            'snake_alive': self.snake.is_alive(),
            'foods': [
                {
                    'position': food.position,
                    'type': food.food_type,
                    'value': food.value,
                    'active': food.active
                }
                for food in self.food_manager.get_active_foods()
            ],
            'stats': self.stats.copy(),
            'field_size': (self.width, self.height)
        }
    
    def get_safe_moves(self) -> List[str]:
        """
        获取安全的移动方向
        
        Returns:
            List[str]: 安全的方向列表
        """
        safe_directions = []
        current_pos = self.snake.get_head_position()
        snake_segments = self.snake.get_segments()
        
        for direction in [DIRECTION_UP, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT]:
            # 预测下一个位置
            collision_info = self.collision_detector.predict_collision(
                current_pos, direction, snake_segments
            )
            
            if not collision_info['will_collide']:
                safe_directions.append(direction)
        
        return safe_directions
    
    def get_optimal_move(self) -> Optional[str]:
        """
        获取最优移动方向（简单AI）
        
        Returns:
            Optional[str]: 最优方向，如果没有安全方向则返回None
        """
        safe_moves = self.get_safe_moves()
        
        if not safe_moves:
            return None
        
        if not self.food_manager.get_active_foods():
            return safe_moves[0] if safe_moves else None
        
        # 找到最近的食物
        head_pos = self.snake.get_head_position()
        nearest_food = None
        min_distance = float('inf')
        
        for food in self.food_manager.get_active_foods():
            distance = abs(head_pos[0] - food.position[0]) + abs(head_pos[1] - food.position[1])
            if distance < min_distance:
                min_distance = distance
                nearest_food = food
        
        if nearest_food:
            # 计算朝向食物的方向
            food_pos = nearest_food.position
            dx = food_pos[0] - head_pos[0]
            dy = food_pos[1] - head_pos[1]
            
            # 优先选择距离食物更近的方向
            preferred_directions = []
            
            if abs(dx) > abs(dy):
                if dx > 0:
                    preferred_directions.append(DIRECTION_RIGHT)
                else:
                    preferred_directions.append(DIRECTION_LEFT)
                if dy > 0:
                    preferred_directions.append(DIRECTION_DOWN)
                else:
                    preferred_directions.append(DIRECTION_UP)
            else:
                if dy > 0:
                    preferred_directions.append(DIRECTION_DOWN)
                else:
                    preferred_directions.append(DIRECTION_UP)
                if dx > 0:
                    preferred_directions.append(DIRECTION_RIGHT)
                else:
                    preferred_directions.append(DIRECTION_LEFT)
            
            # 选择第一个安全的优先方向
            for direction in preferred_directions:
                if direction in safe_moves:
                    return direction
        
        # 如果没有找到朝向食物的安全方向，返回任意安全方向
        return safe_moves[0] if safe_moves else None
    
    def get_field_info(self) -> Dict[str, Any]:
        """
        获取场地信息
        
        Returns:
            Dict: 场地信息
        """
        return {
            'width': self.width,
            'height': self.height,
            'grid_size': self.grid_size,
            'total_cells': self.width * self.height,
            'occupied_cells': len(self.snake.get_segments()),
            'free_cells': self.width * self.height - len(self.snake.get_segments()),
            'food_count': len(self.food_manager.get_active_foods())
        }
    
    def set_speed_multiplier(self, multiplier: float):
        """
        设置速度倍数
        
        Args:
            multiplier: 速度倍数
        """
        self.speed_multiplier = max(0.1, min(5.0, multiplier))
        self._update_snake_speed()
    
    def get_collision_prediction(self, steps: int = 3) -> Dict[str, Any]:
        """
        预测未来几步的碰撞情况
        
        Args:
            steps: 预测步数
            
        Returns:
            Dict: 碰撞预测信息
        """
        current_pos = self.snake.get_head_position()
        current_direction = self.snake.direction
        snake_segments = self.snake.get_segments()
        
        predictions = []
        
        for step in range(1, steps + 1):
            # 计算未来位置
            future_pos = current_pos
            for _ in range(step):
                next_pos = self.collision_detector.get_next_position(future_pos, current_direction)
                future_pos = next_pos
            
            # 检查碰撞
            collision_info = self.collision_detector.predict_collision(
                current_pos, current_direction, snake_segments
            )
            
            predictions.append({
                'step': step,
                'position': future_pos,
                'collision_info': collision_info
            })
        
        return {
            'current_position': current_pos,
            'current_direction': current_direction,
            'predictions': predictions
        }


class AdvancedGameField(GameField):
    """
    高级游戏场地类 - 包含更多高级功能
    """
    
    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT):
        super().__init__(width, height)
        
        # 高级功能
        self.power_ups = []
        self.obstacles = []
        self.special_zones = []
        
        # AI辅助
        self.ai_enabled = False
        self.ai_suggestions = []
        
        # 性能监控
        self.performance_stats = {
            'fps': 0,
            'update_time': 0,
            'render_time': 0,
            'memory_usage': 0
        }
    
    def add_obstacle(self, position: Tuple[int, int]):
        """添加障碍物"""
        if position not in self.obstacles and position not in self.snake.get_segments():
            self.obstacles.append(position)
    
    def remove_obstacle(self, position: Tuple[int, int]):
        """移除障碍物"""
        if position in self.obstacles:
            self.obstacles.remove(position)
    
    def add_special_zone(self, zone_type: str, positions: List[Tuple[int, int]]):
        """添加特殊区域"""
        self.special_zones.append({
            'type': zone_type,
            'positions': positions,
            'active': True
        })
    
    def enable_ai_assistance(self, enabled: bool = True):
        """启用AI辅助"""
        self.ai_enabled = enabled
        if enabled:
            self._generate_ai_suggestions()
    
    def _generate_ai_suggestions(self):
        """生成AI建议"""
        if not self.ai_enabled:
            return
        
        safe_moves = self.get_safe_moves()
        optimal_move = self.get_optimal_move()
        
        self.ai_suggestions = {
            'safe_moves': safe_moves,
            'optimal_move': optimal_move,
            'risk_level': self._calculate_risk_level(),
            'suggestions': self._generate_move_suggestions()
        }
    
    def _calculate_risk_level(self) -> str:
        """计算当前风险等级"""
        safe_moves = self.get_safe_moves()
        
        if len(safe_moves) == 0:
            return "CRITICAL"
        elif len(safe_moves) == 1:
            return "HIGH"
        elif len(safe_moves) == 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_move_suggestions(self) -> List[str]:
        """生成移动建议"""
        suggestions = []
        
        safe_moves = self.get_safe_moves()
        if not safe_moves:
            suggestions.append("No safe moves available!")
            return suggestions
        
        optimal_move = self.get_optimal_move()
        if optimal_move:
            suggestions.append(f"Recommended move: {optimal_move}")
        
        if len(safe_moves) == 1:
            suggestions.append("Only one safe direction available!")
        
        return suggestions
    
    def update(self, current_time: int) -> bool:
        """重写更新方法，添加高级功能"""
        # 记录更新开始时间
        update_start = pygame.time.get_ticks()
        
        # 调用父类更新
        result = super().update(current_time)
        
        # 更新AI建议
        if self.ai_enabled:
            self._generate_ai_suggestions()
        
        # 更新性能统计
        self.performance_stats['update_time'] = pygame.time.get_ticks() - update_start
        
        return result
    
    def get_advanced_state(self) -> Dict[str, Any]:
        """获取高级游戏状态"""
        base_state = self.get_game_state()
        
        advanced_state = {
            **base_state,
            'obstacles': self.obstacles.copy(),
            'special_zones': self.special_zones.copy(),
            'ai_enabled': self.ai_enabled,
            'ai_suggestions': self.ai_suggestions.copy() if self.ai_enabled else {},
            'performance_stats': self.performance_stats.copy()
        }
        
        return advanced_state