"""
游戏状态管理器 (Game State Manager)
管理游戏的各种状态转换、数据持久化和状态验证
"""

import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from config.constants import *


class GameState(Enum):
    """游戏状态枚举"""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    SETTINGS = "settings"
    HIGH_SCORES = "high_scores"
    LOADING = "loading"
    EXITING = "exiting"


class Difficulty(Enum):
    """难度级别枚举"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CUSTOM = "custom"


@dataclass
class GameStats:
    """游戏统计数据"""
    score: int = 0
    high_score: int = 0
    level: int = 1
    food_eaten: int = 0
    time_played: float = 0.0
    games_played: int = 0
    total_score: int = 0
    average_score: float = 0.0
    best_time: float = 0.0
    snake_length: int = SNAKE_INITIAL_LENGTH
    
    def update_averages(self):
        """更新平均值统计"""
        if self.games_played > 0:
            self.average_score = self.total_score / self.games_played


@dataclass
class GameSettings:
    """游戏设置数据"""
    difficulty: str = "medium"
    sound_enabled: bool = True
    music_enabled: bool = True
    volume: float = 0.7
    fullscreen: bool = False
    show_grid: bool = True
    show_score: bool = True
    auto_pause: bool = True
    custom_speed: int = SNAKE_SPEED_MEDIUM
    theme: str = "classic"
    language: str = "zh_CN"


class GameStateManager:
    """游戏状态管理器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化游戏状态管理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.current_state = GameState.MENU
        self.previous_state = GameState.MENU
        self.state_history: List[GameState] = [GameState.MENU]
        
        # 游戏数据
        self.stats = GameStats()
        self.settings = GameSettings()
        self.session_start_time = time.time()
        self.game_start_time = 0.0
        self.pause_start_time = 0.0
        self.total_pause_time = 0.0
        
        # 状态变化回调
        self.state_change_callbacks: Dict[GameState, List[callable]] = {}
        
        # 确保数据目录存在
        self._ensure_data_directory()
        
        # 加载保存的数据
        self.load_game_data()
    
    def _ensure_data_directory(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_current_state(self) -> GameState:
        """获取当前游戏状态"""
        return self.current_state
    
    def get_previous_state(self) -> GameState:
        """获取上一个游戏状态"""
        return self.previous_state
    
    def set_state(self, new_state: GameState, force: bool = False) -> bool:
        """
        设置新的游戏状态
        
        Args:
            new_state: 新状态
            force: 是否强制切换状态
            
        Returns:
            是否成功切换状态
        """
        if not force and not self._is_valid_state_transition(self.current_state, new_state):
            return False
        
        # 保存状态历史
        self.previous_state = self.current_state
        self.state_history.append(new_state)
        
        # 处理状态退出逻辑
        self._on_state_exit(self.current_state)
        
        # 切换状态
        self.current_state = new_state
        
        # 处理状态进入逻辑
        self._on_state_enter(new_state)
        
        # 调用状态变化回调
        self._trigger_state_callbacks(new_state)
        
        return True
    
    def _is_valid_state_transition(self, from_state: GameState, to_state: GameState) -> bool:
        """
        检查状态转换是否有效
        
        Args:
            from_state: 源状态
            to_state: 目标状态
            
        Returns:
            是否为有效转换
        """
        # 定义有效的状态转换
        valid_transitions = {
            GameState.MENU: [GameState.PLAYING, GameState.SETTINGS, GameState.HIGH_SCORES, GameState.EXITING],
            GameState.PLAYING: [GameState.PAUSED, GameState.GAME_OVER, GameState.MENU],
            GameState.PAUSED: [GameState.PLAYING, GameState.MENU, GameState.SETTINGS],
            GameState.GAME_OVER: [GameState.MENU, GameState.PLAYING, GameState.HIGH_SCORES],
            GameState.SETTINGS: [GameState.MENU, GameState.PAUSED],
            GameState.HIGH_SCORES: [GameState.MENU],
            GameState.LOADING: [GameState.MENU, GameState.PLAYING],
            GameState.EXITING: []
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def _on_state_enter(self, state: GameState):
        """处理状态进入逻辑"""
        if state == GameState.PLAYING:
            self.game_start_time = time.time()
            self.total_pause_time = 0.0
        elif state == GameState.PAUSED:
            self.pause_start_time = time.time()
        elif state == GameState.GAME_OVER:
            self._update_game_stats()
    
    def _on_state_exit(self, state: GameState):
        """处理状态退出逻辑"""
        if state == GameState.PAUSED and self.pause_start_time > 0:
            self.total_pause_time += time.time() - self.pause_start_time
            self.pause_start_time = 0.0
    
    def _trigger_state_callbacks(self, state: GameState):
        """触发状态变化回调"""
        if state in self.state_change_callbacks:
            for callback in self.state_change_callbacks[state]:
                try:
                    callback(state)
                except Exception as e:
                    print(f"状态回调错误: {e}")
    
    def add_state_callback(self, state: GameState, callback: callable):
        """
        添加状态变化回调
        
        Args:
            state: 目标状态
            callback: 回调函数
        """
        if state not in self.state_change_callbacks:
            self.state_change_callbacks[state] = []
        self.state_change_callbacks[state].append(callback)
    
    def remove_state_callback(self, state: GameState, callback: callable):
        """
        移除状态变化回调
        
        Args:
            state: 目标状态
            callback: 回调函数
        """
        if state in self.state_change_callbacks:
            try:
                self.state_change_callbacks[state].remove(callback)
            except ValueError:
                pass
    
    def start_new_game(self, difficulty: str = "medium"):
        """
        开始新游戏
        
        Args:
            difficulty: 游戏难度
        """
        # 重置游戏统计
        self.stats.score = 0
        self.stats.level = 1
        self.stats.food_eaten = 0
        self.stats.snake_length = SNAKE_INITIAL_LENGTH
        
        # 设置难度
        self.settings.difficulty = difficulty
        
        # 切换到游戏状态
        self.set_state(GameState.PLAYING)
    
    def pause_game(self):
        """暂停游戏"""
        if self.current_state == GameState.PLAYING:
            self.set_state(GameState.PAUSED)
    
    def resume_game(self):
        """恢复游戏"""
        if self.current_state == GameState.PAUSED:
            self.set_state(GameState.PLAYING)
    
    def end_game(self):
        """结束游戏"""
        if self.current_state in [GameState.PLAYING, GameState.PAUSED]:
            self.set_state(GameState.GAME_OVER)
    
    def return_to_menu(self):
        """返回主菜单"""
        self.set_state(GameState.MENU)
    
    def update_score(self, points: int):
        """
        更新分数
        
        Args:
            points: 增加的分数
        """
        self.stats.score += points
        if self.stats.score > self.stats.high_score:
            self.stats.high_score = self.stats.score
    
    def update_food_eaten(self, count: int = 1):
        """
        更新吃到的食物数量
        
        Args:
            count: 食物数量
        """
        self.stats.food_eaten += count
        
        # 检查是否升级
        if self.stats.food_eaten % LEVEL_UP_FOOD_COUNT == 0:
            self.stats.level += 1
    
    def update_snake_length(self, length: int):
        """
        更新蛇的长度
        
        Args:
            length: 蛇的当前长度
        """
        self.stats.snake_length = length
    
    def get_game_time(self) -> float:
        """
        获取游戏时间（不包括暂停时间）
        
        Returns:
            游戏时间（秒）
        """
        if self.game_start_time == 0:
            return 0.0
        
        current_time = time.time()
        total_time = current_time - self.game_start_time
        
        # 减去暂停时间
        pause_time = self.total_pause_time
        if self.current_state == GameState.PAUSED and self.pause_start_time > 0:
            pause_time += current_time - self.pause_start_time
        
        return max(0.0, total_time - pause_time)
    
    def get_session_time(self) -> float:
        """
        获取会话时间
        
        Returns:
            会话时间（秒）
        """
        return time.time() - self.session_start_time
    
    def _update_game_stats(self):
        """更新游戏统计数据"""
        # 更新游戏时间
        self.stats.time_played = self.get_game_time()
        
        # 更新游戏次数和总分
        self.stats.games_played += 1
        self.stats.total_score += self.stats.score
        
        # 更新最佳时间
        if self.stats.best_time == 0 or self.stats.time_played < self.stats.best_time:
            self.stats.best_time = self.stats.time_played
        
        # 更新平均值
        self.stats.update_averages()
    
    def get_difficulty_settings(self) -> Dict[str, Any]:
        """
        获取当前难度设置
        
        Returns:
            难度设置字典
        """
        difficulty_map = {
            "easy": {
                "speed": SNAKE_SPEED_EASY,
                "food_score": FOOD_SCORE_BASE,
                "level_multiplier": 1.0
            },
            "medium": {
                "speed": SNAKE_SPEED_MEDIUM,
                "food_score": FOOD_SCORE_BASE,
                "level_multiplier": 1.2
            },
            "hard": {
                "speed": SNAKE_SPEED_HARD,
                "food_score": FOOD_SCORE_BASE,
                "level_multiplier": 1.5
            },
            "custom": {
                "speed": self.settings.custom_speed,
                "food_score": FOOD_SCORE_BASE,
                "level_multiplier": 1.0
            }
        }
        
        return difficulty_map.get(self.settings.difficulty, difficulty_map["medium"])
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        获取完整的状态信息
        
        Returns:
            状态信息字典
        """
        return {
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value,
            "stats": asdict(self.stats),
            "settings": asdict(self.settings),
            "game_time": self.get_game_time(),
            "session_time": self.get_session_time(),
            "difficulty_settings": self.get_difficulty_settings()
        }
    
    def save_game_data(self) -> bool:
        """
        保存游戏数据到文件
        
        Returns:
            是否保存成功
        """
        try:
            # 保存统计数据
            stats_file = os.path.join(self.data_dir, "game_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.stats), f, indent=2, ensure_ascii=False)
            
            # 保存设置数据
            settings_file = os.path.join(self.data_dir, "game_settings.json")
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.settings), f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"保存游戏数据失败: {e}")
            return False
    
    def load_game_data(self) -> bool:
        """
        从文件加载游戏数据
        
        Returns:
            是否加载成功
        """
        try:
            # 加载统计数据
            stats_file = os.path.join(self.data_dir, "game_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                    self.stats = GameStats(**stats_data)
            
            # 加载设置数据
            settings_file = os.path.join(self.data_dir, "game_settings.json")
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                    self.settings = GameSettings(**settings_data)
            
            return True
        except Exception as e:
            print(f"加载游戏数据失败: {e}")
            return False
    
    def reset_stats(self):
        """重置游戏统计数据"""
        self.stats = GameStats()
        self.stats.high_score = 0  # 保留最高分
    
    def reset_settings(self):
        """重置游戏设置"""
        self.settings = GameSettings()
    
    def export_data(self, file_path: str) -> bool:
        """
        导出游戏数据
        
        Args:
            file_path: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            export_data = {
                "stats": asdict(self.stats),
                "settings": asdict(self.settings),
                "export_time": time.time(),
                "version": "1.0"
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"导出数据失败: {e}")
            return False
    
    def import_data(self, file_path: str) -> bool:
        """
        导入游戏数据
        
        Args:
            file_path: 导入文件路径
            
        Returns:
            是否导入成功
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 验证数据格式
            if "stats" in import_data and "settings" in import_data:
                self.stats = GameStats(**import_data["stats"])
                self.settings = GameSettings(**import_data["settings"])
                return True
            
            return False
        except Exception as e:
            print(f"导入数据失败: {e}")
            return False


class AdvancedGameStateManager(GameStateManager):
    """高级游戏状态管理器"""
    
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir)
        
        # 高级功能
        self.auto_save_enabled = True
        self.auto_save_interval = 30.0  # 30秒自动保存
        self.last_auto_save = time.time()
        
        # 状态验证
        self.state_validation_enabled = True
        
        # 性能监控
        self.performance_monitoring = True
        self.state_change_times: Dict[str, float] = {}
    
    def update(self, current_time: float):
        """
        更新状态管理器
        
        Args:
            current_time: 当前时间
        """
        # 自动保存
        if self.auto_save_enabled:
            if current_time - self.last_auto_save >= self.auto_save_interval:
                self.save_game_data()
                self.last_auto_save = current_time
        
        # 性能监控
        if self.performance_monitoring:
            self._monitor_performance()
    
    def _monitor_performance(self):
        """监控性能指标"""
        current_time = time.time()
        state_key = f"{self.previous_state.value}_to_{self.current_state.value}"
        
        if state_key not in self.state_change_times:
            self.state_change_times[state_key] = current_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计字典
        """
        return {
            "state_changes": len(self.state_history),
            "average_session_time": self.get_session_time() / max(1, self.stats.games_played),
            "state_change_frequency": len(self.state_history) / max(1, self.get_session_time()),
            "memory_usage": self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        import sys
        return {
            "state_history_size": sys.getsizeof(self.state_history),
            "stats_size": sys.getsizeof(self.stats),
            "settings_size": sys.getsizeof(self.settings)
        }
    
    def validate_state_integrity(self) -> bool:
        """
        验证状态完整性
        
        Returns:
            状态是否完整
        """
        if not self.state_validation_enabled:
            return True
        
        try:
            # 验证状态枚举
            if not isinstance(self.current_state, GameState):
                return False
            
            # 验证统计数据
            if self.stats.score < 0 or self.stats.high_score < 0:
                return False
            
            # 验证设置数据
            if not (0.0 <= self.settings.volume <= 1.0):
                return False
            
            return True
        except Exception:
            return False
    
    def create_checkpoint(self) -> str:
        """
        创建游戏检查点
        
        Returns:
            检查点ID
        """
        checkpoint_id = f"checkpoint_{int(time.time())}"
        checkpoint_file = os.path.join(self.data_dir, f"{checkpoint_id}.json")
        
        checkpoint_data = {
            "id": checkpoint_id,
            "timestamp": time.time(),
            "state_info": self.get_state_info(),
            "state_history": [state.value for state in self.state_history[-10:]]  # 保存最近10个状态
        }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            return checkpoint_id
        except Exception as e:
            print(f"创建检查点失败: {e}")
            return ""
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        恢复游戏检查点
        
        Args:
            checkpoint_id: 检查点ID
            
        Returns:
            是否恢复成功
        """
        checkpoint_file = os.path.join(self.data_dir, f"{checkpoint_id}.json")
        
        try:
            if not os.path.exists(checkpoint_file):
                return False
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # 恢复状态信息
            state_info = checkpoint_data["state_info"]
            self.stats = GameStats(**state_info["stats"])
            self.settings = GameSettings(**state_info["settings"])
            
            # 恢复状态
            current_state = GameState(state_info["current_state"])
            self.set_state(current_state, force=True)
            
            return True
        except Exception as e:
            print(f"恢复检查点失败: {e}")
            return False