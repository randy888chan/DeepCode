"""
分数记录器 (Score Manager)
管理游戏分数、最高分记录、统计数据和排行榜功能
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
import threading

# 导入项目配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.constants import *


@dataclass
class ScoreRecord:
    """分数记录数据类"""
    score: int
    difficulty: str
    date: str
    duration: float  # 游戏时长（秒）
    snake_length: int
    food_eaten: int
    player_name: str = "Player"
    game_id: str = ""
    
    def __post_init__(self):
        if not self.game_id:
            self.game_id = f"{int(time.time())}_{self.score}"


@dataclass
class GameStatistics:
    """游戏统计数据类"""
    total_games: int = 0
    total_score: int = 0
    total_time: float = 0.0
    total_food_eaten: int = 0
    best_score: int = 0
    best_length: int = 0
    average_score: float = 0.0
    games_by_difficulty: Dict[str, int] = None
    
    def __post_init__(self):
        if self.games_by_difficulty is None:
            self.games_by_difficulty = {"easy": 0, "medium": 0, "hard": 0}


class ScoreManager:
    """分数管理器 - 处理分数记录、统计和排行榜"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化分数管理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        self.scores_file = os.path.join(data_dir, "scores.json")
        self.stats_file = os.path.join(data_dir, "statistics.json")
        self.db_file = os.path.join(data_dir, "scores.db")
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 初始化数据
        self.scores: List[ScoreRecord] = []
        self.statistics = GameStatistics()
        self.current_session_scores: List[int] = []
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 加载数据
        self.load_data()
        self.init_database()
    
    def init_database(self):
        """初始化SQLite数据库"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        score INTEGER NOT NULL,
                        difficulty TEXT NOT NULL,
                        date TEXT NOT NULL,
                        duration REAL NOT NULL,
                        snake_length INTEGER NOT NULL,
                        food_eaten INTEGER NOT NULL,
                        player_name TEXT DEFAULT 'Player',
                        game_id TEXT UNIQUE NOT NULL
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_stats (
                        date TEXT PRIMARY KEY,
                        games_played INTEGER DEFAULT 0,
                        total_score INTEGER DEFAULT 0,
                        best_score INTEGER DEFAULT 0,
                        total_time REAL DEFAULT 0.0
                    )
                ''')
                
                conn.commit()
        except Exception as e:
            print(f"数据库初始化错误: {e}")
    
    def add_score(self, score: int, difficulty: str, duration: float, 
                  snake_length: int, food_eaten: int, player_name: str = "Player") -> bool:
        """
        添加新的分数记录
        
        Args:
            score: 游戏分数
            difficulty: 难度级别
            duration: 游戏时长
            snake_length: 蛇的最终长度
            food_eaten: 吃掉的食物数量
            player_name: 玩家名称
            
        Returns:
            bool: 是否成功添加
        """
        try:
            with self._lock:
                # 创建分数记录
                record = ScoreRecord(
                    score=score,
                    difficulty=difficulty,
                    date=datetime.now().isoformat(),
                    duration=duration,
                    snake_length=snake_length,
                    food_eaten=food_eaten,
                    player_name=player_name
                )
                
                # 添加到内存列表
                self.scores.append(record)
                self.current_session_scores.append(score)
                
                # 更新统计数据
                self._update_statistics(record)
                
                # 保存到文件和数据库
                self.save_data()
                self._save_to_database(record)
                
                return True
                
        except Exception as e:
            print(f"添加分数记录错误: {e}")
            return False
    
    def _update_statistics(self, record: ScoreRecord):
        """更新统计数据"""
        stats = self.statistics
        
        # 基本统计
        stats.total_games += 1
        stats.total_score += record.score
        stats.total_time += record.duration
        stats.total_food_eaten += record.food_eaten
        
        # 最佳记录
        if record.score > stats.best_score:
            stats.best_score = record.score
        if record.snake_length > stats.best_length:
            stats.best_length = record.snake_length
        
        # 平均分数
        stats.average_score = stats.total_score / stats.total_games
        
        # 按难度统计
        if record.difficulty in stats.games_by_difficulty:
            stats.games_by_difficulty[record.difficulty] += 1
    
    def _save_to_database(self, record: ScoreRecord):
        """保存记录到数据库"""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO scores 
                    (score, difficulty, date, duration, snake_length, food_eaten, player_name, game_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.score, record.difficulty, record.date, record.duration,
                    record.snake_length, record.food_eaten, record.player_name, record.game_id
                ))
                
                # 更新每日统计
                today = datetime.now().strftime("%Y-%m-%d")
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_stats (date, games_played, total_score, best_score, total_time)
                    VALUES (?, 
                        COALESCE((SELECT games_played FROM daily_stats WHERE date = ?), 0) + 1,
                        COALESCE((SELECT total_score FROM daily_stats WHERE date = ?), 0) + ?,
                        MAX(COALESCE((SELECT best_score FROM daily_stats WHERE date = ?), 0), ?),
                        COALESCE((SELECT total_time FROM daily_stats WHERE date = ?), 0) + ?
                    )
                ''', (today, today, today, record.score, today, record.score, today, record.duration))
                
                conn.commit()
        except Exception as e:
            print(f"数据库保存错误: {e}")
    
    def get_high_scores(self, limit: int = 10, difficulty: str = None) -> List[ScoreRecord]:
        """
        获取最高分排行榜
        
        Args:
            limit: 返回记录数量限制
            difficulty: 难度筛选（可选）
            
        Returns:
            List[ScoreRecord]: 排序后的分数记录列表
        """
        filtered_scores = self.scores
        
        # 按难度筛选
        if difficulty:
            filtered_scores = [s for s in self.scores if s.difficulty == difficulty]
        
        # 按分数排序并限制数量
        return sorted(filtered_scores, key=lambda x: x.score, reverse=True)[:limit]
    
    def get_recent_scores(self, days: int = 7) -> List[ScoreRecord]:
        """
        获取最近几天的分数记录
        
        Args:
            days: 天数
            
        Returns:
            List[ScoreRecord]: 最近的分数记录
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_scores = []
        
        for score in self.scores:
            try:
                score_date = datetime.fromisoformat(score.date)
                if score_date >= cutoff_date:
                    recent_scores.append(score)
            except ValueError:
                continue
        
        return sorted(recent_scores, key=lambda x: x.date, reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取游戏统计信息
        
        Returns:
            Dict[str, Any]: 统计数据字典
        """
        stats_dict = asdict(self.statistics)
        
        # 添加额外统计信息
        stats_dict.update({
            "session_games": len(self.current_session_scores),
            "session_best": max(self.current_session_scores) if self.current_session_scores else 0,
            "session_average": sum(self.current_session_scores) / len(self.current_session_scores) if self.current_session_scores else 0,
            "total_records": len(self.scores),
            "last_played": self.scores[-1].date if self.scores else None
        })
        
        return stats_dict
    
    def get_daily_statistics(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        获取每日统计数据
        
        Args:
            days: 获取天数
            
        Returns:
            Dict[str, Dict[str, Any]]: 每日统计数据
        """
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT date, games_played, total_score, best_score, total_time
                    FROM daily_stats
                    WHERE date >= date('now', '-{} days')
                    ORDER BY date DESC
                '''.format(days))
                
                daily_stats = {}
                for row in cursor.fetchall():
                    date, games, total_score, best_score, total_time = row
                    daily_stats[date] = {
                        "games_played": games,
                        "total_score": total_score,
                        "best_score": best_score,
                        "total_time": total_time,
                        "average_score": total_score / games if games > 0 else 0
                    }
                
                return daily_stats
        except Exception as e:
            print(f"获取每日统计错误: {e}")
            return {}
    
    def is_new_high_score(self, score: int, difficulty: str = None) -> bool:
        """
        检查是否是新的最高分
        
        Args:
            score: 当前分数
            difficulty: 难度级别（可选）
            
        Returns:
            bool: 是否是新最高分
        """
        if difficulty:
            difficulty_scores = [s.score for s in self.scores if s.difficulty == difficulty]
            return not difficulty_scores or score > max(difficulty_scores)
        else:
            return score > self.statistics.best_score
    
    def clear_scores(self, confirm: bool = False) -> bool:
        """
        清除所有分数记录
        
        Args:
            confirm: 确认清除
            
        Returns:
            bool: 是否成功清除
        """
        if not confirm:
            return False
        
        try:
            with self._lock:
                self.scores.clear()
                self.statistics = GameStatistics()
                self.current_session_scores.clear()
                
                # 清除文件
                if os.path.exists(self.scores_file):
                    os.remove(self.scores_file)
                if os.path.exists(self.stats_file):
                    os.remove(self.stats_file)
                
                # 清除数据库
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM scores")
                    cursor.execute("DELETE FROM daily_stats")
                    conn.commit()
                
                return True
        except Exception as e:
            print(f"清除分数记录错误: {e}")
            return False
    
    def export_scores(self, filename: str = None) -> str:
        """
        导出分数记录到文件
        
        Args:
            filename: 导出文件名（可选）
            
        Returns:
            str: 导出文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scores_export_{timestamp}.json"
        
        export_path = os.path.join(self.data_dir, filename)
        
        try:
            export_data = {
                "export_date": datetime.now().isoformat(),
                "total_records": len(self.scores),
                "statistics": asdict(self.statistics),
                "scores": [asdict(score) for score in self.scores]
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return export_path
        except Exception as e:
            print(f"导出分数记录错误: {e}")
            return ""
    
    def import_scores(self, filename: str) -> bool:
        """
        从文件导入分数记录
        
        Args:
            filename: 导入文件路径
            
        Returns:
            bool: 是否成功导入
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 导入分数记录
            imported_scores = []
            for score_data in import_data.get("scores", []):
                record = ScoreRecord(**score_data)
                imported_scores.append(record)
            
            with self._lock:
                # 合并记录（避免重复）
                existing_ids = {score.game_id for score in self.scores}
                new_scores = [score for score in imported_scores if score.game_id not in existing_ids]
                
                self.scores.extend(new_scores)
                
                # 重新计算统计数据
                self._recalculate_statistics()
                
                # 保存数据
                self.save_data()
                
                # 保存到数据库
                for score in new_scores:
                    self._save_to_database(score)
            
            return True
        except Exception as e:
            print(f"导入分数记录错误: {e}")
            return False
    
    def _recalculate_statistics(self):
        """重新计算统计数据"""
        self.statistics = GameStatistics()
        for score in self.scores:
            self._update_statistics(score)
    
    def save_data(self):
        """保存数据到文件"""
        try:
            # 保存分数记录
            scores_data = [asdict(score) for score in self.scores]
            with open(self.scores_file, 'w', encoding='utf-8') as f:
                json.dump(scores_data, f, indent=2, ensure_ascii=False)
            
            # 保存统计数据
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.statistics), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"保存数据错误: {e}")
    
    def load_data(self):
        """从文件加载数据"""
        try:
            # 加载分数记录
            if os.path.exists(self.scores_file):
                with open(self.scores_file, 'r', encoding='utf-8') as f:
                    scores_data = json.load(f)
                self.scores = [ScoreRecord(**data) for data in scores_data]
            
            # 加载统计数据
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                self.statistics = GameStatistics(**stats_data)
            else:
                # 如果没有统计文件，重新计算
                self._recalculate_statistics()
                
        except Exception as e:
            print(f"加载数据错误: {e}")
            self.scores = []
            self.statistics = GameStatistics()


class AdvancedScoreManager(ScoreManager):
    """高级分数管理器 - 扩展功能"""
    
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir)
        self.achievements = self._load_achievements()
    
    def _load_achievements(self) -> Dict[str, bool]:
        """加载成就系统"""
        achievements_file = os.path.join(self.data_dir, "achievements.json")
        default_achievements = {
            "first_game": False,
            "score_100": False,
            "score_500": False,
            "score_1000": False,
            "length_20": False,
            "length_50": False,
            "speed_demon": False,  # 困难模式高分
            "marathon": False,     # 长时间游戏
            "perfectionist": False  # 连续高分
        }
        
        try:
            if os.path.exists(achievements_file):
                with open(achievements_file, 'r', encoding='utf-8') as f:
                    return {**default_achievements, **json.load(f)}
        except Exception as e:
            print(f"加载成就错误: {e}")
        
        return default_achievements
    
    def check_achievements(self, record: ScoreRecord) -> List[str]:
        """
        检查并解锁成就
        
        Args:
            record: 分数记录
            
        Returns:
            List[str]: 新解锁的成就列表
        """
        new_achievements = []
        
        # 检查各种成就条件
        achievement_checks = {
            "first_game": lambda: self.statistics.total_games == 1,
            "score_100": lambda: record.score >= 100,
            "score_500": lambda: record.score >= 500,
            "score_1000": lambda: record.score >= 1000,
            "length_20": lambda: record.snake_length >= 20,
            "length_50": lambda: record.snake_length >= 50,
            "speed_demon": lambda: record.difficulty == "hard" and record.score >= 300,
            "marathon": lambda: record.duration >= 600,  # 10分钟
            "perfectionist": lambda: len(self.current_session_scores) >= 3 and all(s >= 200 for s in self.current_session_scores[-3:])
        }
        
        for achievement, check_func in achievement_checks.items():
            if not self.achievements.get(achievement, False) and check_func():
                self.achievements[achievement] = True
                new_achievements.append(achievement)
        
        # 保存成就
        if new_achievements:
            self._save_achievements()
        
        return new_achievements
    
    def _save_achievements(self):
        """保存成就数据"""
        achievements_file = os.path.join(self.data_dir, "achievements.json")
        try:
            with open(achievements_file, 'w', encoding='utf-8') as f:
                json.dump(self.achievements, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存成就错误: {e}")
    
    def get_achievement_progress(self) -> Dict[str, Any]:
        """获取成就进度"""
        total_achievements = len(self.achievements)
        unlocked_count = sum(1 for unlocked in self.achievements.values() if unlocked)
        
        return {
            "total": total_achievements,
            "unlocked": unlocked_count,
            "progress": unlocked_count / total_achievements if total_achievements > 0 else 0,
            "achievements": self.achievements
        }


# 全局分数管理器实例
score_manager = None

def get_score_manager(advanced: bool = False) -> ScoreManager:
    """
    获取分数管理器实例（单例模式）
    
    Args:
        advanced: 是否使用高级功能
        
    Returns:
        ScoreManager: 分数管理器实例
    """
    global score_manager
    if score_manager is None:
        if advanced:
            score_manager = AdvancedScoreManager()
        else:
            score_manager = ScoreManager()
    return score_manager


if __name__ == "__main__":
    # 测试分数管理器
    print("测试分数管理器...")
    
    # 创建管理器实例
    manager = AdvancedScoreManager("test_data")
    
    # 添加测试分数
    test_scores = [
        (150, "easy", 120.5, 15, 10),
        (300, "medium", 180.0, 25, 20),
        (450, "hard", 240.5, 35, 30),
        (600, "medium", 300.0, 40, 35),
    ]
    
    for score, difficulty, duration, length, food in test_scores:
        success = manager.add_score(score, difficulty, duration, length, food)
        print(f"添加分数 {score} ({difficulty}): {'成功' if success else '失败'}")
    
    # 获取排行榜
    print("\n=== 最高分排行榜 ===")
    high_scores = manager.get_high_scores(5)
    for i, record in enumerate(high_scores, 1):
        print(f"{i}. {record.score}分 ({record.difficulty}) - {record.player_name}")
    
    # 获取统计信息
    print("\n=== 游戏统计 ===")
    stats = manager.get_statistics()
    print(f"总游戏数: {stats['total_games']}")
    print(f"最高分: {stats['best_score']}")
    print(f"平均分: {stats['average_score']:.1f}")
    print(f"总游戏时间: {stats['total_time']:.1f}秒")
    
    # 检查成就
    print("\n=== 成就系统 ===")
    progress = manager.get_achievement_progress()
    print(f"成就进度: {progress['unlocked']}/{progress['total']} ({progress['progress']*100:.1f}%)")
    
    print("\n分数管理器测试完成！")