"""
食物类实现 (Food Class Implementation)
管理游戏中食物的生成、位置和状态
"""

import random
import pygame
from typing import Tuple, List, Optional
from config.constants import (
    GRID_WIDTH, GRID_HEIGHT, GRID_SIZE,
    FOOD_COLOR, FOOD_SPECIAL_COLOR,
    FOOD_SCORE_VALUE, FOOD_SPECIAL_SCORE_VALUE,
    FOOD_SPECIAL_PROBABILITY, FOOD_SPAWN_DELAY,
    FOOD_SPECIAL_DURATION
)


class Food:
    """
    食物类 - 管理游戏中的食物生成和状态
    
    功能:
    - 随机生成食物位置
    - 避免与蛇身体重叠
    - 支持普通食物和特殊食物
    - 管理特殊食物的时间限制
    """
    
    def __init__(self):
        """初始化食物系统"""
        self.position: Optional[Tuple[int, int]] = None
        self.is_special: bool = False
        self.spawn_time: int = 0
        self.last_spawn_time: int = 0
        self.special_duration: int = FOOD_SPECIAL_DURATION
        
        # 食物状态
        self.active: bool = False
        self.score_value: int = FOOD_SCORE_VALUE
        
        # 生成延迟控制
        self.spawn_delay: int = FOOD_SPAWN_DELAY
        
    def spawn(self, snake_segments: List[Tuple[int, int]], current_time: int) -> bool:
        """
        生成新食物
        
        Args:
            snake_segments: 蛇的所有身体段位置
            current_time: 当前游戏时间
            
        Returns:
            bool: 是否成功生成食物
        """
        # 检查生成延迟
        if current_time - self.last_spawn_time < self.spawn_delay:
            return False
            
        # 获取所有可用位置
        available_positions = self._get_available_positions(snake_segments)
        
        if not available_positions:
            return False
            
        # 随机选择位置
        self.position = random.choice(available_positions)
        
        # 决定是否生成特殊食物
        self.is_special = random.random() < FOOD_SPECIAL_PROBABILITY
        
        # 设置食物属性
        if self.is_special:
            self.score_value = FOOD_SPECIAL_SCORE_VALUE
            self.spawn_time = current_time
        else:
            self.score_value = FOOD_SCORE_VALUE
            
        self.active = True
        self.last_spawn_time = current_time
        
        return True
        
    def _get_available_positions(self, snake_segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        获取所有可用的食物生成位置
        
        Args:
            snake_segments: 蛇的所有身体段位置
            
        Returns:
            List[Tuple[int, int]]: 可用位置列表
        """
        available_positions = []
        snake_set = set(snake_segments)
        
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                position = (x, y)
                if position not in snake_set:
                    available_positions.append(position)
                    
        return available_positions
        
    def update(self, current_time: int) -> bool:
        """
        更新食物状态
        
        Args:
            current_time: 当前游戏时间
            
        Returns:
            bool: 食物是否仍然有效
        """
        if not self.active:
            return False
            
        # 检查特殊食物是否过期
        if self.is_special:
            if current_time - self.spawn_time > self.special_duration:
                self.remove()
                return False
                
        return True
        
    def consume(self) -> int:
        """
        消费食物（被蛇吃掉）
        
        Returns:
            int: 食物的分数值
        """
        if not self.active:
            return 0
            
        score = self.score_value
        self.remove()
        return score
        
    def remove(self):
        """移除食物"""
        self.position = None
        self.active = False
        self.is_special = False
        self.score_value = FOOD_SCORE_VALUE
        self.spawn_time = 0
        
    def get_position(self) -> Optional[Tuple[int, int]]:
        """
        获取食物位置
        
        Returns:
            Optional[Tuple[int, int]]: 食物位置，如果没有食物则返回None
        """
        return self.position if self.active else None
        
    def get_color(self) -> Tuple[int, int, int]:
        """
        获取食物颜色
        
        Returns:
            Tuple[int, int, int]: RGB颜色值
        """
        if self.is_special:
            return FOOD_SPECIAL_COLOR
        return FOOD_COLOR
        
    def get_score_value(self) -> int:
        """
        获取食物分数值
        
        Returns:
            int: 分数值
        """
        return self.score_value if self.active else 0
        
    def is_active(self) -> bool:
        """
        检查食物是否激活
        
        Returns:
            bool: 食物是否激活
        """
        return self.active
        
    def is_special_food(self) -> bool:
        """
        检查是否为特殊食物
        
        Returns:
            bool: 是否为特殊食物
        """
        return self.is_special and self.active
        
    def get_remaining_time(self, current_time: int) -> int:
        """
        获取特殊食物剩余时间
        
        Args:
            current_time: 当前游戏时间
            
        Returns:
            int: 剩余时间（毫秒），如果不是特殊食物则返回0
        """
        if not self.is_special or not self.active:
            return 0
            
        elapsed = current_time - self.spawn_time
        remaining = max(0, self.special_duration - elapsed)
        return remaining
        
    def check_collision(self, position: Tuple[int, int]) -> bool:
        """
        检查指定位置是否与食物碰撞
        
        Args:
            position: 要检查的位置
            
        Returns:
            bool: 是否发生碰撞
        """
        if not self.active or not self.position:
            return False
            
        return self.position == position
        
    def force_spawn_at(self, position: Tuple[int, int], is_special: bool = False, current_time: int = 0):
        """
        强制在指定位置生成食物（用于测试或特殊情况）
        
        Args:
            position: 食物位置
            is_special: 是否为特殊食物
            current_time: 当前时间
        """
        self.position = position
        self.is_special = is_special
        self.active = True
        
        if is_special:
            self.score_value = FOOD_SPECIAL_SCORE_VALUE
            self.spawn_time = current_time
        else:
            self.score_value = FOOD_SCORE_VALUE
            
        self.last_spawn_time = current_time
        
    def reset(self):
        """重置食物系统到初始状态"""
        self.position = None
        self.is_special = False
        self.spawn_time = 0
        self.last_spawn_time = 0
        self.active = False
        self.score_value = FOOD_SCORE_VALUE
        
    def get_info(self) -> dict:
        """
        获取食物信息（用于调试和状态显示）
        
        Returns:
            dict: 食物状态信息
        """
        return {
            'position': self.position,
            'active': self.active,
            'is_special': self.is_special,
            'score_value': self.score_value,
            'spawn_time': self.spawn_time,
            'last_spawn_time': self.last_spawn_time
        }
        
    def set_spawn_delay(self, delay: int):
        """
        设置食物生成延迟
        
        Args:
            delay: 延迟时间（毫秒）
        """
        self.spawn_delay = max(0, delay)
        
    def set_special_duration(self, duration: int):
        """
        设置特殊食物持续时间
        
        Args:
            duration: 持续时间（毫秒）
        """
        self.special_duration = max(1000, duration)  # 最少1秒


class FoodManager:
    """
    食物管理器 - 管理多个食物实例和高级食物逻辑
    
    功能:
    - 管理多个食物同时存在
    - 控制食物生成频率
    - 处理特殊食物事件
    """
    
    def __init__(self, max_foods: int = 1):
        """
        初始化食物管理器
        
        Args:
            max_foods: 最大同时存在的食物数量
        """
        self.max_foods = max_foods
        self.foods: List[Food] = []
        self.last_spawn_attempt = 0
        self.spawn_interval = FOOD_SPAWN_DELAY
        
    def update(self, snake_segments: List[Tuple[int, int]], current_time: int):
        """
        更新所有食物状态
        
        Args:
            snake_segments: 蛇的所有身体段位置
            current_time: 当前游戏时间
        """
        # 更新现有食物
        self.foods = [food for food in self.foods if food.update(current_time)]
        
        # 尝试生成新食物
        if (len(self.foods) < self.max_foods and 
            current_time - self.last_spawn_attempt >= self.spawn_interval):
            
            new_food = Food()
            if new_food.spawn(snake_segments, current_time):
                self.foods.append(new_food)
                
            self.last_spawn_attempt = current_time
            
    def check_collisions(self, position: Tuple[int, int]) -> List[Food]:
        """
        检查与指定位置碰撞的所有食物
        
        Args:
            position: 要检查的位置
            
        Returns:
            List[Food]: 发生碰撞的食物列表
        """
        collided_foods = []
        for food in self.foods:
            if food.check_collision(position):
                collided_foods.append(food)
                
        return collided_foods
        
    def consume_food_at(self, position: Tuple[int, int]) -> int:
        """
        消费指定位置的食物
        
        Args:
            position: 食物位置
            
        Returns:
            int: 获得的总分数
        """
        total_score = 0
        foods_to_remove = []
        
        for food in self.foods:
            if food.check_collision(position):
                total_score += food.consume()
                foods_to_remove.append(food)
                
        # 移除被消费的食物
        for food in foods_to_remove:
            if food in self.foods:
                self.foods.remove(food)
                
        return total_score
        
    def get_all_positions(self) -> List[Tuple[int, int]]:
        """
        获取所有活跃食物的位置
        
        Returns:
            List[Tuple[int, int]]: 所有食物位置
        """
        positions = []
        for food in self.foods:
            pos = food.get_position()
            if pos:
                positions.append(pos)
        return positions
        
    def get_foods(self) -> List[Food]:
        """
        获取所有食物实例
        
        Returns:
            List[Food]: 食物列表
        """
        return self.foods.copy()
        
    def clear_all(self):
        """清除所有食物"""
        for food in self.foods:
            food.remove()
        self.foods.clear()
        
    def reset(self):
        """重置食物管理器"""
        self.clear_all()
        self.last_spawn_attempt = 0
        
    def set_max_foods(self, max_foods: int):
        """
        设置最大食物数量
        
        Args:
            max_foods: 最大食物数量
        """
        self.max_foods = max(1, max_foods)
        
        # 如果当前食物数量超过新的最大值，移除多余的食物
        while len(self.foods) > self.max_foods:
            food = self.foods.pop()
            food.remove()
            
    def set_spawn_interval(self, interval: int):
        """
        设置食物生成间隔
        
        Args:
            interval: 生成间隔（毫秒）
        """
        self.spawn_interval = max(100, interval)