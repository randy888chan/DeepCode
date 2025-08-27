"""
碰撞检测器 (Collision Detector)
处理游戏中所有碰撞检测逻辑，包括蛇与边界、蛇与自身、蛇与食物的碰撞
"""

from typing import List, Tuple, Optional, Dict, Any
import pygame
from config.constants import (
    GRID_WIDTH, GRID_HEIGHT, GRID_SIZE,
    DIRECTIONS, DIRECTION_UP, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT
)


class CollisionDetector:
    """
    碰撞检测器类
    负责处理游戏中的所有碰撞检测逻辑
    """
    
    def __init__(self):
        """初始化碰撞检测器"""
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.collision_cache = {}  # 碰撞检测缓存，提高性能
        
    def check_boundary_collision(self, position: Tuple[int, int]) -> bool:
        """
        检查位置是否与游戏边界碰撞
        
        Args:
            position: 要检查的位置 (x, y)
            
        Returns:
            bool: 如果碰撞返回True，否则返回False
        """
        x, y = position
        
        # 检查是否超出游戏区域边界
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
            
        return False
    
    def check_self_collision(self, head_position: Tuple[int, int], 
                           body_segments: List[Tuple[int, int]]) -> bool:
        """
        检查蛇头是否与蛇身碰撞
        
        Args:
            head_position: 蛇头位置
            body_segments: 蛇身段列表（不包括头部）
            
        Returns:
            bool: 如果碰撞返回True，否则返回False
        """
        # 蛇头不能与蛇身任何部分重叠
        return head_position in body_segments
    
    def check_food_collision(self, snake_head: Tuple[int, int], 
                           food_position: Tuple[int, int]) -> bool:
        """
        检查蛇头是否与食物碰撞
        
        Args:
            snake_head: 蛇头位置
            food_position: 食物位置
            
        Returns:
            bool: 如果碰撞返回True，否则返回False
        """
        return snake_head == food_position
    
    def check_snake_collision(self, snake_segments: List[Tuple[int, int]]) -> Dict[str, bool]:
        """
        全面检查蛇的碰撞状态
        
        Args:
            snake_segments: 蛇的所有段（包括头部）
            
        Returns:
            Dict[str, bool]: 包含各种碰撞状态的字典
        """
        if not snake_segments:
            return {
                'boundary': False,
                'self': False,
                'any': False
            }
        
        head = snake_segments[0]
        body = snake_segments[1:] if len(snake_segments) > 1 else []
        
        # 检查边界碰撞
        boundary_collision = self.check_boundary_collision(head)
        
        # 检查自身碰撞
        self_collision = self.check_self_collision(head, body)
        
        # 任何碰撞
        any_collision = boundary_collision or self_collision
        
        return {
            'boundary': boundary_collision,
            'self': self_collision,
            'any': any_collision
        }
    
    def predict_collision(self, current_position: Tuple[int, int], 
                         direction: str, 
                         snake_segments: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        预测下一步移动是否会发生碰撞
        
        Args:
            current_position: 当前位置
            direction: 移动方向
            snake_segments: 蛇的所有段
            
        Returns:
            Dict[str, Any]: 预测结果，包括下一个位置和碰撞状态
        """
        # 计算下一个位置
        next_position = self.get_next_position(current_position, direction)
        
        # 创建预测的蛇段列表（头部移动到新位置）
        predicted_segments = [next_position] + snake_segments[:-1]
        
        # 检查碰撞
        collision_result = self.check_snake_collision(predicted_segments)
        
        return {
            'next_position': next_position,
            'collision': collision_result,
            'safe': not collision_result['any']
        }
    
    def get_next_position(self, current_position: Tuple[int, int], 
                         direction: str) -> Tuple[int, int]:
        """
        根据当前位置和方向计算下一个位置
        
        Args:
            current_position: 当前位置
            direction: 移动方向
            
        Returns:
            Tuple[int, int]: 下一个位置
        """
        x, y = current_position
        
        if direction == DIRECTION_UP:
            return (x, y - 1)
        elif direction == DIRECTION_DOWN:
            return (x, y + 1)
        elif direction == DIRECTION_LEFT:
            return (x - 1, y)
        elif direction == DIRECTION_RIGHT:
            return (x + 1, y)
        else:
            # 无效方向，返回当前位置
            return current_position
    
    def check_position_occupied(self, position: Tuple[int, int], 
                              snake_segments: List[Tuple[int, int]]) -> bool:
        """
        检查指定位置是否被蛇占据
        
        Args:
            position: 要检查的位置
            snake_segments: 蛇的所有段
            
        Returns:
            bool: 如果位置被占据返回True，否则返回False
        """
        return position in snake_segments
    
    def get_safe_positions(self, snake_segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        获取游戏区域内所有安全位置（不被蛇占据且在边界内）
        
        Args:
            snake_segments: 蛇的所有段
            
        Returns:
            List[Tuple[int, int]]: 安全位置列表
        """
        safe_positions = []
        
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                position = (x, y)
                if not self.check_position_occupied(position, snake_segments):
                    safe_positions.append(position)
        
        return safe_positions
    
    def check_multiple_food_collision(self, snake_head: Tuple[int, int], 
                                    food_positions: List[Tuple[int, int]]) -> List[int]:
        """
        检查蛇头与多个食物的碰撞
        
        Args:
            snake_head: 蛇头位置
            food_positions: 食物位置列表
            
        Returns:
            List[int]: 碰撞的食物索引列表
        """
        collided_indices = []
        
        for i, food_pos in enumerate(food_positions):
            if self.check_food_collision(snake_head, food_pos):
                collided_indices.append(i)
        
        return collided_indices
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        检查位置是否在有效的游戏区域内
        
        Args:
            position: 要检查的位置
            
        Returns:
            bool: 如果位置有效返回True，否则返回False
        """
        return not self.check_boundary_collision(position)
    
    def get_collision_direction(self, position: Tuple[int, int]) -> Optional[str]:
        """
        获取碰撞的方向（用于确定撞到了哪面墙）
        
        Args:
            position: 碰撞位置
            
        Returns:
            Optional[str]: 碰撞方向，如果没有碰撞返回None
        """
        x, y = position
        
        if x < 0:
            return DIRECTION_LEFT
        elif x >= self.grid_width:
            return DIRECTION_RIGHT
        elif y < 0:
            return DIRECTION_UP
        elif y >= self.grid_height:
            return DIRECTION_DOWN
        else:
            return None
    
    def calculate_collision_distance(self, start_position: Tuple[int, int], 
                                   direction: str, 
                                   snake_segments: List[Tuple[int, int]]) -> int:
        """
        计算从起始位置沿指定方向到碰撞点的距离
        
        Args:
            start_position: 起始位置
            direction: 移动方向
            snake_segments: 蛇的所有段
            
        Returns:
            int: 到碰撞点的距离（格子数）
        """
        current_pos = start_position
        distance = 0
        
        while True:
            next_pos = self.get_next_position(current_pos, direction)
            
            # 检查下一个位置是否会碰撞
            if (self.check_boundary_collision(next_pos) or 
                self.check_position_occupied(next_pos, snake_segments)):
                break
            
            current_pos = next_pos
            distance += 1
            
            # 防止无限循环
            if distance > self.grid_width + self.grid_height:
                break
        
        return distance
    
    def get_collision_info(self, snake_segments: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        获取详细的碰撞信息
        
        Args:
            snake_segments: 蛇的所有段
            
        Returns:
            Dict[str, Any]: 详细的碰撞信息
        """
        if not snake_segments:
            return {
                'has_collision': False,
                'collision_type': None,
                'collision_position': None,
                'collision_direction': None
            }
        
        head = snake_segments[0]
        collision_result = self.check_snake_collision(snake_segments)
        
        info = {
            'has_collision': collision_result['any'],
            'collision_type': None,
            'collision_position': head,
            'collision_direction': None
        }
        
        if collision_result['boundary']:
            info['collision_type'] = 'boundary'
            info['collision_direction'] = self.get_collision_direction(head)
        elif collision_result['self']:
            info['collision_type'] = 'self'
        
        return info
    
    def clear_cache(self):
        """清除碰撞检测缓存"""
        self.collision_cache.clear()


class AdvancedCollisionDetector(CollisionDetector):
    """
    高级碰撞检测器
    提供更复杂的碰撞检测功能，如区域碰撞、路径碰撞等
    """
    
    def __init__(self):
        super().__init__()
        self.collision_zones = {}  # 碰撞区域定义
    
    def add_collision_zone(self, zone_name: str, positions: List[Tuple[int, int]]):
        """
        添加自定义碰撞区域
        
        Args:
            zone_name: 区域名称
            positions: 区域包含的位置列表
        """
        self.collision_zones[zone_name] = set(positions)
    
    def check_zone_collision(self, position: Tuple[int, int], zone_name: str) -> bool:
        """
        检查位置是否与指定区域碰撞
        
        Args:
            position: 要检查的位置
            zone_name: 区域名称
            
        Returns:
            bool: 如果碰撞返回True，否则返回False
        """
        if zone_name not in self.collision_zones:
            return False
        
        return position in self.collision_zones[zone_name]
    
    def check_path_collision(self, start_position: Tuple[int, int], 
                           end_position: Tuple[int, int], 
                           snake_segments: List[Tuple[int, int]]) -> bool:
        """
        检查从起点到终点的路径是否有碰撞
        
        Args:
            start_position: 起始位置
            end_position: 结束位置
            snake_segments: 蛇的所有段
            
        Returns:
            bool: 如果路径上有碰撞返回True，否则返回False
        """
        # 简单的直线路径检测
        x1, y1 = start_position
        x2, y2 = end_position
        
        # 计算路径上的所有点
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x_step = 1 if x1 < x2 else -1
        y_step = 1 if y1 < y2 else -1
        
        current_x, current_y = x1, y1
        
        for _ in range(max(dx, dy) + 1):
            current_pos = (current_x, current_y)
            
            # 检查当前位置是否有碰撞
            if (self.check_boundary_collision(current_pos) or 
                self.check_position_occupied(current_pos, snake_segments)):
                return True
            
            # 移动到下一个位置
            if current_x != x2:
                current_x += x_step
            if current_y != y2:
                current_y += y_step
        
        return False
    
    def get_nearest_collision(self, position: Tuple[int, int], 
                            snake_segments: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        获取最近的碰撞点信息
        
        Args:
            position: 起始位置
            snake_segments: 蛇的所有段
            
        Returns:
            Dict[str, Any]: 最近碰撞点的信息
        """
        min_distance = float('inf')
        nearest_collision = None
        collision_type = None
        
        # 检查所有方向的碰撞距离
        for direction in DIRECTIONS:
            distance = self.calculate_collision_distance(position, direction, snake_segments)
            if distance < min_distance:
                min_distance = distance
                nearest_collision = self.get_next_position(position, direction)
                
                # 确定碰撞类型
                next_pos = position
                for _ in range(distance + 1):
                    next_pos = self.get_next_position(next_pos, direction)
                
                if self.check_boundary_collision(next_pos):
                    collision_type = 'boundary'
                elif self.check_position_occupied(next_pos, snake_segments):
                    collision_type = 'snake'
        
        return {
            'distance': min_distance,
            'position': nearest_collision,
            'type': collision_type
        }


# 全局碰撞检测器实例
collision_detector = CollisionDetector()
advanced_collision_detector = AdvancedCollisionDetector()


def check_collision(snake_segments: List[Tuple[int, int]]) -> Dict[str, bool]:
    """
    便捷函数：检查蛇的碰撞状态
    
    Args:
        snake_segments: 蛇的所有段
        
    Returns:
        Dict[str, bool]: 碰撞状态字典
    """
    return collision_detector.check_snake_collision(snake_segments)


def check_food_eaten(snake_head: Tuple[int, int], food_position: Tuple[int, int]) -> bool:
    """
    便捷函数：检查食物是否被吃掉
    
    Args:
        snake_head: 蛇头位置
        food_position: 食物位置
        
    Returns:
        bool: 如果食物被吃掉返回True，否则返回False
    """
    return collision_detector.check_food_collision(snake_head, food_position)


def get_safe_food_positions(snake_segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    便捷函数：获取安全的食物生成位置
    
    Args:
        snake_segments: 蛇的所有段
        
    Returns:
        List[Tuple[int, int]]: 安全位置列表
    """
    return collision_detector.get_safe_positions(snake_segments)


def predict_next_move(current_position: Tuple[int, int], 
                     direction: str, 
                     snake_segments: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    便捷函数：预测下一步移动的结果
    
    Args:
        current_position: 当前位置
        direction: 移动方向
        snake_segments: 蛇的所有段
        
    Returns:
        Dict[str, Any]: 预测结果
    """
    return collision_detector.predict_collision(current_position, direction, snake_segments)